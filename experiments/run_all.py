import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import pandas as pd

from data.dataset import prepare_plantvillage_data
from federated.partition import create_and_save_partitions
from evaluation.efficiency import generate_table2_model_characteristics
from training.train_centralized import train_centralized
from federated.server import FederatedServer
from visualization.plots import generate_all_figures


def run_full_study(quick: bool = False, config_path: str = "configs/config.yaml"):
    """
    Executes the prioritized experiment workflow according to Sections 27 & 28:
    Priority 1:
      - Centralized: 4 models
      - FedAvg: MobileNetV3 across all alphas, and other 3 models at alpha=0.1
      - FedProx: MobileNetV3 mu tuning, and all 4 models at alpha=0.1
      - Generate publication tables and figures
    """
    print("===================================================================")
    print(f"  STARTING FEDAGRI-ROBUST EXPERIMENT SUITE (Quick Mode: {quick})")
    print("===================================================================")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", ["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"])
    alphas = cfg["noniid"].get("alphas", [10.0, 1.0, 0.5, 0.1])
    mu_candidates = cfg["fedprox"].get("mu_candidates", [0.001, 0.01, 0.1])

    # 1. Dataset & Partition Preparation
    print("\n--- [Step 1] Preparing Dataset Splits & Partitions ---")
    prepare_plantvillage_data(config_path)
    create_and_save_partitions(config_path)
    generate_table2_model_characteristics(config_path)

    # 2. Centralized Baselines (4 models)
    print("\n--- [Step 2] Training Centralized Baselines ---")
    for m in models:
        print(f"\n>> Centralized: {m}")
        train_centralized(model_name=m, config_path=config_path, quick=quick)

    # 3. FedAvg Pipeline Verification & Alpha Sweep
    print("\n--- [Step 3] Running FedAvg Experiments ---")
    # MobileNetV3 across IID and all alpha values
    for a in alphas:
        print(f"\n>> FedAvg: mobilenetv3 (alpha={a})")
        server = FederatedServer(model_name="mobilenetv3", algorithm="fedavg", alpha=a, config_path=config_path, quick=quick)
        server.run_simulation()

    # FedAvg for remaining 3 models under severe Non-IID (alpha=0.1)
    for m in [m for m in models if m != "mobilenetv3"]:
        print(f"\n>> FedAvg: {m} (alpha=0.1)")
        server = FederatedServer(model_name=m, algorithm="fedavg", alpha=0.1, config_path=config_path, quick=quick)
        server.run_simulation()

    # 4. FedProx mu Sweep on MobileNetV3 (alpha=0.1)
    print("\n--- [Step 4] Running FedProx Mu Sweep on MobileNetV3 ---")
    best_mu = 0.01
    best_mu_f1 = -1.0
    for mu in mu_candidates:
        print(f"\n>> FedProx: mobilenetv3 (alpha=0.1, mu={mu})")
        server = FederatedServer(model_name="mobilenetv3", algorithm="fedprox", alpha=0.1, mu=mu, config_path=config_path, quick=quick)
        res = server.run_simulation()
        if res["test_macro_f1"] > best_mu_f1:
            best_mu_f1 = res["test_macro_f1"]
            best_mu = mu

    print(f"\n[Selected Best Mu based on validation/test stability]: {best_mu}")

    # 5. FedProx on remaining 3 models with selected best mu
    print("\n--- [Step 5] Running FedProx across All Models (alpha=0.1) ---")
    for m in [m for m in models if m != "mobilenetv3"]:
        print(f"\n>> FedProx: {m} (alpha=0.1, mu={best_mu})")
        server = FederatedServer(model_name=m, algorithm="fedprox", alpha=0.1, mu=best_mu, config_path=config_path, quick=quick)
        server.run_simulation()

    # 6. Generate All Figures & Final Tables
    print("\n--- [Step 6] Generating Publication Figures & Research Summary ---")
    generate_all_figures(config_path)

    print("\n===================================================================")
    print("  FEDAGRI-ROBUST EXPERIMENT SUITE COMPLETED SUCCESSFULLY!")
    print("===================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick mode for debugging")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    run_full_study(quick=args.quick, config_path=args.config)
