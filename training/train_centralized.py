import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import load_split_data, get_dataloader
from models.model_factory import create_model, count_parameters, get_model_size_mb, compute_flops, measure_inference_latency
from evaluation.evaluate import evaluate_model


def train_centralized(
    model_name: str,
    config_path: str = "configs/config.yaml",
    quick: bool = False
) -> dict:
    """
    Centralized training pipeline for a single CNN architecture.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["dataset"].get("num_classes", 38)
    image_size = cfg["dataset"].get("image_size", 224)
    batch_size = cfg["training"].get("batch_size", 32)
    lr = cfg["training"].get("learning_rate", 0.001)
    weight_decay = cfg["training"].get("weight_decay", 0.0001)
    
    epochs = cfg["training"].get("quick_epochs", 2) if quick else cfg["training"].get("centralized_epochs", 8)
    num_workers = 0 if sys.platform == "win32" else cfg["training"].get("num_workers", 2)

    print(f"\n==================================================")
    print(f"  Centralized Training: {model_name.upper()}")
    print(f"  Device: {device} | Epochs: {epochs} | Quick Mode: {quick}")
    print(f"==================================================")

    # 1. Load Data
    train_samples = load_split_data("train", config_path)
    val_samples = load_split_data("val", config_path)
    test_samples = load_split_data("test", config_path)

    if quick:
        train_samples = train_samples[:600]
        val_samples = val_samples[:150]
        test_samples = test_samples[:150]
        print(f"[Quick Mode] Subsampled -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    train_loader = get_dataloader(train_samples, batch_size=batch_size, is_train=True, num_workers=num_workers, image_size=image_size)
    val_loader = get_dataloader(val_samples, batch_size=batch_size, is_train=False, num_workers=num_workers, image_size=image_size)
    test_loader = get_dataloader(test_samples, batch_size=batch_size, is_train=False, num_workers=num_workers, image_size=image_size)

    # 2. Instantiate Model
    model = create_model(model_name, num_classes=num_classes, pretrained=True)
    model.to(device)

    total_params, _ = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    gflops = compute_flops(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Tracking metrics per epoch
    history = []
    best_val_f1 = -1.0
    best_weights_path = f"results/checkpoints/centralized_{model_name}.pt"
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/centralized", exist_ok=True)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        # Checkpoint best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_weights_path)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "epoch_time_s": round(epoch_time, 2)
        })

    total_train_time = round(time.time() - start_total_time, 2)

    # 3. Final Test Evaluation (Load best model)
    print(f"\nEvaluating Best Checkpoint on Unseen Test Split...")
    model.load_state_dict(torch.load(best_weights_path))
    test_metrics, y_test_true, y_test_pred = evaluate_model(model, test_loader, criterion, device)
    latency_ms = measure_inference_latency(model, device)

    print(f"=== Final Test Results for {model_name.upper()} ===")
    print(f"  Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted F1:  {test_metrics['weighted_f1']:.4f}")
    print(f"  Total Train Time:  {total_train_time}s")
    print(f"  Inference Latency: {latency_ms}ms")

    # 4. Save per-epoch history CSV
    df_history = pd.DataFrame(history)
    history_csv = f"results/centralized/{model_name}.csv"
    df_history.to_csv(history_csv, index=False)
    print(f"[Saved History] -> {history_csv}")

    # Summary record for Table 3
    summary = {
        "Model": model_name,
        "Parameters (M)": round(total_params / 1e6, 2),
        "Size_MB": round(model_size_mb, 2),
        "GFLOPs": gflops,
        "Train_Loss": history[-1]["train_loss"],
        "Val_Loss": val_metrics["loss"],
        "Test_Accuracy": test_metrics["accuracy"],
        "Test_Macro_Precision": test_metrics["macro_precision"],
        "Test_Macro_Recall": test_metrics["macro_recall"],
        "Test_Macro_F1": test_metrics["macro_f1"],
        "Test_Weighted_F1": test_metrics["weighted_f1"],
        "Training_Time_s": total_train_time,
        "Inference_Latency_ms": latency_ms
    }

    # Update or create Table 3
    table3_csv = "results/tables/table3_centralized_performance.csv"
    table3_md = "results/tables/table3_centralized_performance.md"
    if os.path.exists(table3_csv):
        df_t3 = pd.read_csv(table3_csv)
        df_t3 = df_t3[df_t3["Model"] != model_name]
        df_t3 = pd.concat([df_t3, pd.DataFrame([summary])], ignore_index=True)
    else:
        df_t3 = pd.DataFrame([summary])

    df_t3.to_csv(table3_csv, index=False)
    with open(table3_md, "w") as f:
        f.write("# Table 3: Centralized Baseline Performance\n\n")
        f.write(df_t3.to_markdown(index=False))

    print(f"[Saved Table 3] -> {table3_csv}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick mode for debugging")
    args = parser.parse_args()

    train_centralized(model_name=args.model, config_path=args.config, quick=args.quick)
