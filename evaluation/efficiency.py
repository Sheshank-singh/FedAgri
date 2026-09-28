import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import time
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from models.model_factory import create_model, count_parameters, get_model_size_mb, compute_flops, measure_inference_latency



def calculate_communication_cost(
    model_size_mb: float,
    num_clients: int,
    num_rounds: int
) -> Dict[str, float]:
    """
    Section 19: Communication Cost calculation.
    For each round, each participating client downloads the global model
    and uploads its updated local model.
    Download volume per round: num_clients * model_size_mb
    Upload volume per round: num_clients * model_size_mb
    Total per round: 2 * num_clients * model_size_mb
    Total communication volume: 2 * num_clients * num_rounds * model_size_mb
    """
    download_per_round = num_clients * model_size_mb
    upload_per_round = num_clients * model_size_mb
    round_volume = download_per_round + upload_per_round
    total_volume = round_volume * num_rounds

    return {
        "model_size_mb": round(model_size_mb, 2),
        "participating_clients": num_clients,
        "rounds": num_rounds,
        "upload_volume_mb": round(upload_per_round * num_rounds, 2),
        "download_volume_mb": round(download_per_round * num_rounds, 2),
        "total_communication_mb": round(total_volume, 2),
        "total_communication_gb": round(total_volume / 1024.0, 3)
    }


def generate_table2_model_characteristics(
    config_path: str = "configs/config.yaml",
    save_csv: str = "results/tables/table2_model_characteristics.csv",
    save_md: str = "results/tables/table2_model_characteristics.md"
) -> pd.DataFrame:
    """
    Generates Table 2: Four model architecture characteristics.
    - Parameters
    - Model Size (MB)
    - GFLOPs
    - Inference Latency (ms)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", ["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"])
    num_classes = cfg["dataset"].get("num_classes", 38)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Profiling models on device: {device} (num_classes={num_classes})...")
    records = []

    for model_name in models:
        print(f"  -> Profiling {model_name}...")
        model = create_model(model_name, num_classes=num_classes, pretrained=False)
        total_params, trainable_params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        gflops = compute_flops(model)
        latency_ms = measure_inference_latency(model, device)

        records.append({
            "Model": model_name,
            "Parameters (M)": round(total_params / 1e6, 2),
            "Total_Params": total_params,
            "Size_MB": round(size_mb, 2),
            "GFLOPs": gflops,
            "Inference_Latency_ms": latency_ms
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv, index=False)
    with open(save_md, "w") as f:
        f.write("# Table 2: Model Architecture Characteristics\n\n")
        f.write(df.to_markdown(index=False))

    print(f"\n[Generated Table 2]")
    print(df.to_markdown(index=False))
    print(f"\nSaved Table 2 to {save_csv} and {save_md}")
    return df


if __name__ == "__main__":
    generate_table2_model_characteristics()
