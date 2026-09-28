import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data.dataset import load_split_data, get_dataloader
from models.model_factory import create_model


def plot_confusion_matrix_for_checkpoint(
    checkpoint_path: str,
    model_name: str,
    title: str,
    save_path: str,
    config_path: str = "configs/config.yaml"
):
    """
    Evaluates a saved checkpoint on the isolated test set and saves a confusion matrix.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found, skipping confusion matrix.")
        return

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["dataset"].get("num_classes", 38)
    image_size = cfg["dataset"].get("image_size", 224)

    # Load test data
    test_samples = load_split_data("test", config_path)
    num_workers = 0 if sys.platform == "win32" else 2
    test_loader = get_dataloader(test_samples, batch_size=64, is_train=False, num_workers=num_workers, image_size=image_size)

    # Load model
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    # Normalize by class true count
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        cbar_kws={'label': 'Recall / Normalized Rate'},
        xticklabels=[f"C{i}" for i in range(num_classes)],
        yticklabels=[f"C{i}" for i in range(num_classes)]
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class Index", fontsize=11)
    plt.ylabel("Ground Truth Class Index", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved Confusion Matrix] -> {save_path}")


def generate_selected_confusion_matrices(config_path: str = "configs/config.yaml"):
    """
    Section 26: Generate confusion matrices for:
    1. Best centralized model.
    2. Best FedAvg model under severe Non-IID (alpha=0.1).
    3. Best FedProx model under severe Non-IID (alpha=0.1).
    """
    print("\nGenerating Selected Confusion Matrices (Figure 9)...")
    plot_confusion_matrix_for_checkpoint(
        checkpoint_path="results/checkpoints/centralized_mobilenetv3.pt",
        model_name="mobilenetv3",
        title="Figure 9A: Centralized MobileNetV3 Normalized Confusion Matrix",
        save_path="results/figures/figure9a_cm_centralized.png",
        config_path=config_path
    )
    plot_confusion_matrix_for_checkpoint(
        checkpoint_path="results/checkpoints/fedavg_mobilenetv3_alpha_0.1.pt",
        model_name="mobilenetv3",
        title="Figure 9B: FedAvg MobileNetV3 (α=0.1) Normalized Confusion Matrix",
        save_path="results/figures/figure9b_cm_fedavg_severe.png",
        config_path=config_path
    )
    plot_confusion_matrix_for_checkpoint(
        checkpoint_path="results/checkpoints/fedprox_mobilenetv3_alpha_0.1_mu_0.01.pt",
        model_name="mobilenetv3",
        title="Figure 9C: FedProx MobileNetV3 (α=0.1, μ=0.01) Normalized Confusion Matrix",
        save_path="results/figures/figure9c_cm_fedprox_severe.png",
        config_path=config_path
    )


if __name__ == "__main__":
    generate_selected_confusion_matrices()
