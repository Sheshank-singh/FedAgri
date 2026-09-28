import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def set_plot_style():
    """Sets publication-quality matplotlib style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300
    })


def plot_figure1_system_architecture(save_path: str = "results/figures/figure1_system_architecture.png"):
    """
    Figure 1: Overall FedAgri-Robust Architecture Diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Draw boxes
    bbox_server = dict(boxstyle="round,pad=0.6", facecolor="#2b5c8f", edgecolor="black", alpha=0.9)
    bbox_client = dict(boxstyle="round,pad=0.5", facecolor="#43a2ca", edgecolor="black", alpha=0.8)
    bbox_data = dict(boxstyle="round,pad=0.4", facecolor="#a8ddb5", edgecolor="black", alpha=0.9)

    ax.text(0.5, 0.85, "FEDERATED SERVER\nGlobal Model Aggregation\n(FedAvg / FedProx)\nEvaluates on Isolated Validation Split", 
            ha="center", va="center", color="white", fontsize=12, fontweight="bold", bbox=bbox_server)

    client_positions = [0.15, 0.325, 0.5, 0.675, 0.85]
    for i, x in enumerate(client_positions):
        cid = i + 1
        # Client Model Box
        ax.text(x, 0.50, f"Client {cid}\nLocal CNN\n(w_k)", ha="center", va="center", color="white", fontsize=10, fontweight="bold", bbox=bbox_client)
        # Client Data Box
        ax.text(x, 0.20, f"Local Data\nDirichlet (α)\n[Raw Isolated]", ha="center", va="center", color="black", fontsize=9, bbox=bbox_data)

        # Arrows between data and client
        ax.annotate("", xy=(x, 0.42), xytext=(x, 0.28), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2ca25f"))
        # Arrows between client and server
        ax.annotate("", xy=(x, 0.58), xytext=(0.5, 0.74), arrowprops=dict(arrowstyle="<->", lw=1.5, color="#2b5c8f", linestyle="--"))

    ax.text(0.5, 0.05, "Backbones: MobileNetV3 | ResNet50 | EfficientNet-B0 | ConvNeXt-Tiny\nHeterogeneity: Dirichlet α ∈ {10.0, 1.0, 0.5, 0.1} | Optimization: FedAvg vs FedProx (μ)",
            ha="center", va="center", fontsize=11, style="italic", bbox=dict(boxstyle="square,pad=0.5", facecolor="#f0f0f0", edgecolor="gray"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 1 Saved] -> {save_path}")


def plot_figure3_centralized_comparison(
    table3_csv: str = "results/tables/table3_centralized_performance.csv",
    save_path: str = "results/figures/figure3_centralized_comparison.png"
):
    """
    Figure 3: Centralized Model Performance and Latency Comparison.
    """
    if not os.path.exists(table3_csv):
        print(f"Warning: {table3_csv} not found, skipping Figure 3.")
        return

    set_plot_style()
    df = pd.read_csv(table3_csv)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = "tab:blue"
    x = np.arange(len(df["Model"]))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df["Test_Macro_F1"], width, label="Test Macro-F1", color="#2b5c8f")
    ax1.set_ylabel("Macro-F1 Score", color="#2b5c8f", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="#2b5c8f")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in df["Model"]], fontweight="bold")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df["Inference_Latency_ms"], width, label="Latency (ms)", color="#e34a33")
    ax2.set_ylabel("Inference Latency (ms)", color="#e34a33", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="#e34a33")
    ax2.grid(False)

    plt.title("Figure 3: Centralized CNN Performance vs Single-Image Inference Latency", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 3 Saved] -> {save_path}")


def plot_figure4_fedavg_convergence(
    logs_dir: str = "results/logs",
    save_path: str = "results/figures/figure4_fedavg_convergence.png"
):
    """
    Figure 4: FedAvg Convergence for different alpha values.
    """
    set_plot_style()
    files = glob.glob(os.path.join(logs_dir, "fedavg_mobilenetv3_*_rounds.csv"))
    if not files:
        print(f"Warning: No round logs found in {logs_dir}, skipping Figure 4.")
        return

    plt.figure(figsize=(9, 5))
    for f in sorted(files):
        df = pd.read_csv(f)
        label_name = f.replace("\\", "/").split("/")[-1].replace("fedavg_mobilenetv3_", "").replace("_rounds.csv", "")
        plt.plot(df["round"], df["validation_macro_f1"], marker="o", linewidth=2, label=f"FedAvg ({label_name})")

    plt.title("Figure 4: MobileNetV3 FedAvg Convergence across Heterogeneity Levels", fontsize=13, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Validation Macro-F1", fontsize=12)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 4 Saved] -> {save_path}")


def plot_figure5_noniid_robustness(
    table4_csv: str = "results/tables/table4_fedavg_noniid.csv",
    save_path: str = "results/figures/figure5_noniid_robustness.png"
):
    """
    Figure 5: Performance vs Alpha (Non-IID Robustness curve).
    """
    if not os.path.exists(table4_csv):
        print(f"Warning: {table4_csv} not found, skipping Figure 5.")
        return

    set_plot_style()
    df = pd.read_csv(table4_csv)
    # Filter numeric alphas
    df_numeric = df[df["Alpha"] != "IID"].copy()
    df_numeric["Alpha"] = df_numeric["Alpha"].astype(float)
    df_numeric = df_numeric.sort_values(by="Alpha", ascending=True)

    plt.figure(figsize=(9, 5))
    for model_name, group in df_numeric.groupby("Model"):
        plt.plot(group["Alpha"], group["FedAvg_Macro_F1"], marker="s", linewidth=2.5, label=model_name.upper())

    plt.xscale("log")
    plt.title("Figure 5: CNN Backbone Robustness to Label Heterogeneity (Dirichlet α)", fontsize=13, fontweight="bold")
    plt.xlabel("Dirichlet Alpha α (Lower = Stronger Non-IID Heterogeneity)", fontsize=12)
    plt.ylabel("Test Macro-F1 Score", fontsize=12)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 5 Saved] -> {save_path}")


def plot_figure6_fedavg_vs_fedprox(
    logs_dir: str = "results/logs",
    save_path: str = "results/figures/figure6_fedavg_vs_fedprox_convergence.png"
):
    """
    Figure 6: FedAvg vs FedProx Convergence under severe Non-IID (alpha=0.1).
    """
    set_plot_style()
    f_avg = glob.glob(os.path.join(logs_dir, "*fedavg*alpha_0.1*_rounds.csv"))
    f_prox = glob.glob(os.path.join(logs_dir, "*fedprox*alpha_0.1*_rounds.csv"))

    if not f_avg and not f_prox:
        print(f"Warning: Alpha=0.1 round logs not found, skipping Figure 6.")
        return

    plt.figure(figsize=(9, 5))
    for f in f_avg[:2]:
        df = pd.read_csv(f)
        model = df["model"].iloc[0]
        plt.plot(df["round"], df["validation_macro_f1"], marker="o", linestyle="--", linewidth=2, label=f"{model.upper()} (FedAvg)")

    for f in f_prox[:2]:
        df = pd.read_csv(f)
        model = df["model"].iloc[0]
        mu = df["mu"].iloc[0]
        plt.plot(df["round"], df["validation_macro_f1"], marker="^", linewidth=2.5, label=f"{model.upper()} (FedProx, μ={mu})")

    plt.title("Figure 6: FedAvg vs FedProx Convergence under Severe Heterogeneity (α=0.1)", fontsize=13, fontweight="bold")
    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Validation Macro-F1", fontsize=12)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 6 Saved] -> {save_path}")


def plot_figure7_client_consistency(
    table5_csv: str = "results/tables/table5_fedavg_vs_fedprox.csv",
    save_path: str = "results/figures/figure7_client_consistency.png"
):
    """
    Figure 7: Client-level performance consistency (Mean vs Worst vs Std).
    """
    if not os.path.exists(table5_csv):
        print(f"Warning: {table5_csv} not found, skipping Figure 7.")
        return

    set_plot_style()
    df = pd.read_csv(table5_csv)

    plt.figure(figsize=(9, 5))
    x = np.arange(len(df["Model"]))
    width = 0.35

    plt.bar(x - width/2, df["Mean_Client_Accuracy"], width, label="Mean Client Accuracy", color="#41b6c4")
    plt.bar(x + width/2, df["Worst_Client_Accuracy"], width, label="Worst Client Accuracy", color="#253494")

    plt.title("Figure 7: Client-Level Performance Consistency under Severe Heterogeneity", fontsize=13, fontweight="bold")
    plt.xlabel("CNN Architecture", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(x, [m.upper() for m in df["Model"]], fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 7 Saved] -> {save_path}")


def plot_figure8_communication_tradeoff(
    table2_csv: str = "results/tables/table2_model_characteristics.csv",
    table3_csv: str = "results/tables/table3_centralized_performance.csv",
    save_path: str = "results/figures/figure8_communication_tradeoff.png"
):
    """
    Figure 8: Communication Volume vs Model Performance Trade-off.
    """
    if not os.path.exists(table2_csv) or not os.path.exists(table3_csv):
        print(f"Warning: Required tables not found, skipping Figure 8.")
        return

    set_plot_style()
    df2 = pd.read_csv(table2_csv)
    df3 = pd.read_csv(table3_csv)
    df3_clean = df3[["Model", "Test_Macro_F1"]]
    df_merged = pd.merge(df2, df3_clean, on="Model")

    # 5 clients, 20 rounds: total comm MB = 2 * 5 * 20 * Size_MB = 200 * Size_MB
    df_merged["Comm_GB"] = (200.0 * df_merged["Size_MB"]) / 1024.0



    plt.figure(figsize=(9, 5.5))
    sns.scatterplot(
        data=df_merged,
        x="Comm_GB",
        y="Test_Macro_F1",
        size="Parameters (M)",
        sizes=(100, 600),
        hue="Model",
        palette="tab10",
        legend=True
    )

    for _, row in df_merged.iterrows():
        plt.text(
            row["Comm_GB"] + 0.3,
            row["Test_Macro_F1"] - 0.01,
            f"{row['Model'].upper()}\n({row['Size_MB']} MB / {row['Inference_Latency_ms']} ms)",
            fontsize=9,
            fontweight="bold"
        )

    plt.title("Figure 8: Communication Cost (GB) vs Classification Performance Trade-Off", fontsize=13, fontweight="bold")
    plt.xlabel("Total Federated Communication Overhead (GB over 20 rounds, 5 clients)", fontsize=12)
    plt.ylabel("Test Macro-F1 Score", fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Figure 8 Saved] -> {save_path}")


def generate_table6_communication_efficiency(
    table2_csv: str = "results/tables/table2_model_characteristics.csv",
    save_csv: str = "results/tables/table6_communication_efficiency.csv",
    save_md: str = "results/tables/table6_communication_efficiency.md",
    num_clients: int = 5,
    num_rounds: int = 20
):
    """
    Table 6: Communication and Efficiency Comparison.
    """
    if not os.path.exists(table2_csv):
        return
    df2 = pd.read_csv(table2_csv)
    records = []
    for _, row in df2.iterrows():
        m_size = row["Size_MB"]
        upload_mb = num_clients * m_size * num_rounds
        download_mb = num_clients * m_size * num_rounds
        total_mb = upload_mb + download_mb
        records.append({
            "Model": row["Model"],
            "Parameters (M)": row["Parameters (M)"],
            "Size_MB": m_size,
            "GFLOPs": row["GFLOPs"],
            "Latency_ms": row["Inference_Latency_ms"],
            "Upload_Vol_MB": round(upload_mb, 2),
            "Download_Vol_MB": round(download_mb, 2),
            "Total_Comm_MB": round(total_mb, 2),
            "Total_Comm_GB": round(total_mb / 1024.0, 3)
        })

    df6 = pd.DataFrame(records)
    df6.to_csv(save_csv, index=False)
    with open(save_md, "w") as f:
        f.write("# Table 6: Communication Volume and Computational Efficiency Benchmarks\n\n")
        f.write(df6.to_markdown(index=False))
    print(f"[Table 6 Saved] -> {save_csv}")


def generate_all_figures(config_path: str = "configs/config.yaml"):
    """
    Generates Figures 1 to 8 and Table 6.
    """
    print("\nGenerating all publication figures...")
    plot_figure1_system_architecture()
    plot_figure3_centralized_comparison()
    plot_figure4_fedavg_convergence()
    plot_figure5_noniid_robustness()
    plot_figure6_fedavg_vs_fedprox()
    plot_figure7_client_consistency()
    plot_figure8_communication_tradeoff()
    generate_table6_communication_efficiency()
    print("All figures and Table 6 generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
