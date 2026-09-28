import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import pandas as pd
from federated.server import FederatedServer


def main():
    parser = argparse.ArgumentParser(description="Run FedAvg Experiment")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"],
        help="Model architecture"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Dirichlet alpha parameter (e.g. 10.0, 1.0, 0.5, 0.1)")
    parser.add_argument("--iid", action="store_true", help="Run IID partition instead of Dirichlet Non-IID")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--quick", action="store_true", help="Quick mode for debugging")
    args = parser.parse_args()

    server = FederatedServer(
        model_name=args.model,
        algorithm="fedavg",
        alpha=args.alpha if not args.iid else None,
        is_iid=args.iid,
        config_path=args.config,
        quick=args.quick
    )
    summary = server.run_simulation()

    # Update Table 4: FedAvg under different Non-IID levels
    table4_csv = "results/tables/table4_fedavg_noniid.csv"
    table4_md = "results/tables/table4_fedavg_noniid.md"
    os.makedirs("results/tables", exist_ok=True)

    record = {
        "Model": summary["model"],
        "Alpha": "IID" if summary["alpha"] < 0 else str(summary["alpha"]),
        "FedAvg_Accuracy": summary["test_accuracy"],
        "FedAvg_Macro_F1": summary["test_macro_f1"],
        "FedAvg_Weighted_F1": summary["test_weighted_f1"],
        "Mean_Client_Accuracy": summary["mean_client_accuracy"],
        "Worst_Client_Accuracy": summary["worst_client_accuracy"],
        "Client_Accuracy_Std": summary["client_accuracy_std"],
        "Comm_MB": summary["total_communication_mb"],
        "Time_s": summary["training_time_s"]
    }

    if os.path.exists(table4_csv):
        df_t4 = pd.read_csv(table4_csv)
        mask = (df_t4["Model"] == record["Model"]) & (df_t4["Alpha"].astype(str) == str(record["Alpha"]))
        df_t4 = df_t4[~mask]
        df_t4 = pd.concat([df_t4, pd.DataFrame([record])], ignore_index=True)
    else:
        df_t4 = pd.DataFrame([record])

    df_t4.to_csv(table4_csv, index=False)
    with open(table4_md, "w") as f:
        f.write("# Table 4: FedAvg Performance under Varying Heterogeneity (Alpha)\n\n")
        f.write(df_t4.to_markdown(index=False))

    print(f"\n[Updated Table 4] -> {table4_csv}")


if __name__ == "__main__":
    main()
