import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from federated.server import FederatedServer


def main():
    parser = argparse.ArgumentParser(description="Run FedProx Experiment")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"],
        help="Model architecture"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha parameter (default: 0.1 for severe Non-IID)")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal coefficient mu (e.g. 0.001, 0.01, 0.1)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--quick", action="store_true", help="Quick mode for debugging")
    args = parser.parse_args()

    server = FederatedServer(
        model_name=args.model,
        algorithm="fedprox",
        alpha=args.alpha,
        is_iid=False,
        mu=args.mu,
        config_path=args.config,
        quick=args.quick
    )
    summary = server.run_simulation()

    # Update Table 5: FedAvg vs FedProx under severe Non-IID (alpha=0.1)
    table5_csv = "results/tables/table5_fedavg_vs_fedprox.csv"
    table5_md = "results/tables/table5_fedavg_vs_fedprox.md"
    os.makedirs("results/tables", exist_ok=True)

    record = {
        "Model": summary["model"],
        "Alpha": str(summary["alpha"]),
        "Algorithm": "FedProx",
        "Mu": summary["mu"],
        "Accuracy": summary["test_accuracy"],
        "Macro_F1": summary["test_macro_f1"],
        "Weighted_F1": summary["test_weighted_f1"],
        "Mean_Client_Accuracy": summary["mean_client_accuracy"],
        "Worst_Client_Accuracy": summary["worst_client_accuracy"],
        "Client_Accuracy_Std": summary["client_accuracy_std"],
        "Time_s": summary["training_time_s"]
    }

    if os.path.exists(table5_csv):
        df_t5 = pd.read_csv(table5_csv)
        mask = (df_t5["Model"] == record["Model"]) & (df_t5["Algorithm"] == "FedProx") & (df_t5["Mu"] == record["Mu"])
        df_t5 = df_t5[~mask]
        df_t5 = pd.concat([df_t5, pd.DataFrame([record])], ignore_index=True)
    else:
        df_t5 = pd.DataFrame([record])

    df_t5.to_csv(table5_csv, index=False)
    with open(table5_md, "w") as f:
        f.write("# Table 5: FedAvg vs FedProx Robustness Comparison (Severe Non-IID Alpha=0.1)\n\n")
        f.write(df_t5.to_markdown(index=False))

    print(f"\n[Updated Table 5] -> {table5_csv}")


if __name__ == "__main__":
    main()
