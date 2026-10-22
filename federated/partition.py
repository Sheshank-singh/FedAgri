import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from data.dataset import load_split_data


def partition_iid(
    train_samples: List[Tuple[str, int]],
    num_clients: int = 5,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Uniform random partitioning across clients (IID baseline).
    Returns mapping: client_id -> list of sample indices.
    """
    np.random.seed(seed)
    n_samples = len(train_samples)
    indices = np.random.permutation(n_samples)
    client_partitions = {}
    splits = np.array_split(indices, num_clients)
    for cid, split_idx in enumerate(splits):
        client_partitions[cid] = split_idx.tolist()
    return client_partitions


def partition_dirichlet(
    train_samples: List[Tuple[str, int]],
    num_classes: int = 38,
    num_clients: int = 5,
    alpha: float = 0.1,
    seed: int = 42,
    min_samples_per_client: int = 10
) -> Dict[int, List[int]]:
    """
    Dirichlet distribution partitioning for label-distribution heterogeneity.
    Conceptually: p_k ~ Dirichlet(alpha) for each class across clients, or for each client across classes.
    Here, for each class c, we sample proportions across K clients ~ Dirichlet(alpha).
    Guarantees:
    1. Uses training data only.
    2. No client overlap (each sample assigned to exactly one client).
    3. Preserves image-label correspondence.
    4. Reproducible with fixed seed.
    5. Ensures all clients receive at least min_samples_per_client.
    """
    np.random.seed(seed)
    n_samples = len(train_samples)
    labels = np.array([s[1] for s in train_samples])

    # Group sample indices by class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    while True:
        client_partitions = {cid: [] for cid in range(num_clients)}

        for c in range(num_classes):
            idx_c = class_indices[c]
            np.random.shuffle(idx_c)
            n_c = len(idx_c)

            # Sample client proportions for this class
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # Normalize to avoid numerical drift
            proportions = proportions / proportions.sum()
            # Compute split boundaries
            counts = (proportions * n_c).astype(int)
            # Allocate remainder to random client
            remainder = n_c - counts.sum()
            if remainder > 0:
                lucky_clients = np.random.choice(num_clients, remainder, replace=False)
                for lc in lucky_clients:
                    counts[lc] += 1

            start = 0
            for cid in range(num_clients):
                end = start + counts[cid]
                if end > start:
                    client_partitions[cid].extend(idx_c[start:end].tolist())
                start = end

        # Verify all clients have usable data
        min_client_size = min(len(client_partitions[cid]) for cid in range(num_clients))
        if min_client_size >= min_samples_per_client:
            break
        # Otherwise resample if a client was completely starved
        seed += 1
        np.random.seed(seed)

    return client_partitions


def create_and_save_partitions(
    config_path: str = "configs/config.yaml"
) -> Dict[str, str]:
    """
    Creates and saves partition JSON files for IID and all alpha values in config.
    Returns mapping from partition name to file path.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    num_classes = cfg["dataset"].get("num_classes", 38)
    num_clients = cfg["federated"].get("num_clients", 5)
    alphas = cfg["noniid"].get("alphas", [10.0, 1.0, 0.5, 0.1])
    processed_dir = cfg["dataset"].get("processed_dir", "data/processed")

    train_samples = load_split_data("train", config_path)
    partitions_dir = os.path.join(processed_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    saved_files = {}

    # 1. Create IID Partition
    iid_parts = partition_iid(train_samples, num_clients=num_clients, seed=seed)
    iid_file = os.path.join(partitions_dir, f"partition_iid_seed{seed}.json")
    with open(iid_file, "w") as f:
        json.dump(iid_parts, f)
    saved_files["iid"] = iid_file
    print(f"[Saved Partition] IID -> {iid_file}")

    # 2. Create Dirichlet Partitions for each alpha
    for alpha in alphas:
        alpha_parts = partition_dirichlet(
            train_samples,
            num_classes=num_classes,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed
        )
        alpha_file = os.path.join(partitions_dir, f"partition_alpha_{alpha}_seed{seed}.json")
        with open(alpha_file, "w") as f:
            json.dump(alpha_parts, f)
        saved_files[f"alpha_{alpha}"] = alpha_file
        print(f"[Saved Partition] alpha={alpha} -> {alpha_file}")

        # Visualize client distribution heatmap (Figure 2)
        visualize_partition_distribution(
            train_samples=train_samples,
            partitions=alpha_parts,
            num_classes=num_classes,
            num_clients=num_clients,
            title=f"Client Label Distribution (Dirichlet α = {alpha})",
            save_path=f"results/figures/client_distribution_alpha_{alpha}.png"
        )

    # Also visualize IID
    visualize_partition_distribution(
        train_samples=train_samples,
        partitions=iid_parts,
        num_classes=num_classes,
        num_clients=num_clients,
        title="Client Label Distribution (IID Uniform)",
        save_path="results/figures/client_distribution_iid.png"
    )

    return saved_files


def visualize_partition_distribution(
    train_samples: List[Tuple[str, int]],
    partitions: Dict[int, List[int]],
    num_classes: int,
    num_clients: int,
    title: str,
    save_path: str
):
    """
    Visualizes client vs class sample count matrix as a heatmap.
    """
    matrix = np.zeros((num_clients, num_classes), dtype=int)
    labels = np.array([s[1] for s in train_samples])

    for cid, sample_indices in partitions.items():
        client_labels = labels[sample_indices]
        for l in client_labels:
            matrix[int(cid), l] += 1

    plt.figure(figsize=(14, 5))
    sns.heatmap(
        matrix,
        cmap="YlGnBu",
        annot=False,
        cbar_kws={'label': 'Sample Count'},
        xticklabels=[f"C{c}" for c in range(num_classes)],
        yticklabels=[f"Client {cid}" for cid in range(num_clients)]
    )
    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Class Index (0 to 37)", fontsize=12)
    plt.ylabel("Federated Client", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [Saved Visualization] -> {save_path}")


def load_client_partition(
    alpha: Optional[float] = None,
    is_iid: bool = False,
    num_clients: Optional[int] = None,
    config_path: str = "configs/config.yaml"
) -> Dict[int, List[int]]:
    """
    Loads or generates client partition dictionary for federated training.
    Supports any arbitrary Dirichlet alpha parameter and client counts dynamically on-the-fly.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    default_clients = cfg["federated"].get("num_clients", 5)
    target_clients = int(num_clients) if num_clients is not None else default_clients
    processed_dir = cfg["dataset"].get("processed_dir", "data/processed")
    partitions_dir = os.path.join(processed_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)

    if is_iid or alpha is None:
        filename = f"partition_iid_seed{seed}.json" if target_clients == default_clients else f"partition_iid_clients_{target_clients}_seed{seed}.json"
    else:
        alpha_clean = round(float(alpha), 2)
        alpha_str = str(float(alpha_clean)) if alpha_clean.is_integer() else str(alpha_clean)
        filename = f"partition_alpha_{alpha_str}_seed{seed}.json" if target_clients == default_clients else f"partition_alpha_{alpha_str}_clients_{target_clients}_seed{seed}.json"

    filepath = os.path.join(partitions_dir, filename)

    if not os.path.exists(filepath):
        print(f"[Partition Engine] Partition file '{filename}' not found. Generating on-the-fly (alpha={alpha}, clients={target_clients})...")
        train_samples = load_split_data("train", config_path)
        num_classes = cfg["dataset"].get("num_classes", 38)

        if is_iid or alpha is None:
            parts = partition_iid(train_samples, num_clients=target_clients, seed=seed)
        else:
            parts = partition_dirichlet(
                train_samples,
                num_classes=num_classes,
                num_clients=target_clients,
                alpha=float(alpha),
                seed=seed
            )

        with open(filepath, "w") as f:
            json.dump(parts, f)
        print(f"[Partition Engine] Successfully saved new partition -> {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    # If the loaded partition doesn't have enough clients, re-generate on-the-fly
    if len(data) < target_clients:
        print(f"[Partition Engine] Partition had {len(data)} clients, needed {target_clients}. Regenerating...")
        train_samples = load_split_data("train", config_path)
        num_classes = cfg["dataset"].get("num_classes", 38)
        if is_iid or alpha is None:
            data = partition_iid(train_samples, num_clients=target_clients, seed=seed)
        else:
            data = partition_dirichlet(
                train_samples,
                num_classes=num_classes,
                num_clients=target_clients,
                alpha=float(alpha),
                seed=seed
            )
        with open(filepath, "w") as f:
            json.dump(data, f)

    return {int(k): v for k, v in data.items()}


if __name__ == "__main__":
    create_and_save_partitions()
