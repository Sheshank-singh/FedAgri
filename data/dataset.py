import os
import json
import random
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def get_transforms(image_size: int = 224, is_train: bool = True):
    """
    Standard preprocessing for ImageNet-pretrained CNNs.
    Train: Resize to 224x224, mild random rotation (15 deg), random horizontal flip, ImageNet norm.
    Val/Test: Resize to 224x224, ImageNet norm.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])


class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset for PlantVillage image classification.
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples  # List of (image_path, class_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_plantvillage_data(config_path: str = "configs/config.yaml") -> Dict:
    """
    Stage 1:
    1. Inspect all available classes.
    2. Print class counts.
    3. Support all 38 classes (or configured subset).
    4. Ensure adequate samples per class.
    5. Create a reproducible class mapping and save to classes.json.
    6. Generate fixed stratified split (70% train, 15% val, 15% test) with seed=42.
    7. Save splits and generate Table 1 statistics.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["dataset"]["raw_dir"]
    processed_dir = cfg["dataset"]["processed_dir"]
    selected_classes_cfg = cfg["dataset"]["selected_classes"]
    seed = cfg["seed"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "splits"), exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # 1. Inspect all valid class directories
    all_entries = sorted(os.listdir(raw_dir))
    all_class_counts = {}
    for entry in all_entries:
        entry_path = os.path.join(raw_dir, entry)
        if os.path.isdir(entry_path) and not entry.startswith('.'):
            valid_images = [
                f for f in os.listdir(entry_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(valid_images) > 0:
                all_class_counts[entry] = len(valid_images)

    print(f"=== Total Available PlantVillage Classes: {len(all_class_counts)} ===")
    for cls_name, count in all_class_counts.items():
        print(f"  {cls_name}: {count} images")

    # 2. Determine selected classes
    if selected_classes_cfg == "all" or (isinstance(selected_classes_cfg, list) and len(selected_classes_cfg) == 0):
        selected_classes = sorted(list(all_class_counts.keys()))
    elif isinstance(selected_classes_cfg, list):
        selected_classes = sorted(selected_classes_cfg)
        for cls_name in selected_classes:
            if cls_name not in all_class_counts:
                raise ValueError(f"Selected class '{cls_name}' not found in {raw_dir}")
    else:
        selected_classes = sorted(list(all_class_counts.keys()))

    num_classes = len(selected_classes)
    print(f"\nConfigured classes count: {num_classes}")

    # 3. Create reproducible class mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(selected_classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    mapping_path = os.path.join(processed_dir, "classes.json")
    with open(mapping_path, "w") as f:
        json.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f, indent=2)
    print(f"[Saved Class Mapping] -> {mapping_path}")

    # 4. Collect image samples for selected classes
    all_samples = []
    class_samples_count = {cls_name: 0 for cls_name in selected_classes}

    for cls_name in selected_classes:
        cls_dir = os.path.join(raw_dir, cls_name)
        images = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        class_samples_count[cls_name] = len(images)
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            all_samples.append((img_path, class_to_idx[cls_name]))

    print(f"\n=== Selected {num_classes} Classes Total Samples: {len(all_samples)} ===")

    # 5. Stratified 70% Train, 15% Val, 15% Test
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    # First split: 70% train, 30% temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels
    )

    # Second split: split 30% temp equally into 15% val and 15% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))

    print(f"\n=== Dataset Splits (Seed={seed}) ===")
    print(f"  Train samples: {len(train_samples)} ({len(train_samples)/len(all_samples):.1%})")
    print(f"  Validation samples: {len(val_samples)} ({len(val_samples)/len(all_samples):.1%})")
    print(f"  Test samples: {len(test_samples)} ({len(test_samples)/len(all_samples):.1%})")

    # Save splits
    splits_dir = os.path.join(processed_dir, "splits")
    with open(os.path.join(splits_dir, "train_split.json"), "w") as f:
        json.dump(train_samples, f)
    with open(os.path.join(splits_dir, "val_split.json"), "w") as f:
        json.dump(val_samples, f)
    with open(os.path.join(splits_dir, "test_split.json"), "w") as f:
        json.dump(test_samples, f)

    # 6. Generate Table 1: Class Statistics
    stats_records = []
    for cls_name, idx in class_to_idx.items():
        n_total = class_samples_count[cls_name]
        n_train = sum(1 for _, l in train_samples if l == idx)
        n_val = sum(1 for _, l in val_samples if l == idx)
        n_test = sum(1 for _, l in test_samples if l == idx)
        parts = cls_name.split("___")
        crop = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        status = "Healthy" if "healthy" in condition.lower() else "Diseased"

        stats_records.append({
            "Class_ID": idx,
            "Class_Name": cls_name,
            "Crop": crop,
            "Condition": condition,
            "Status": status,
            "Total_Samples": n_total,
            "Train_Samples (70%)": n_train,
            "Val_Samples (15%)": n_val,
            "Test_Samples (15%)": n_test
        })

    df_stats = pd.DataFrame(stats_records)
    table1_csv_path = "results/tables/table1_class_statistics.csv"
    table1_md_path = "results/tables/table1_class_statistics.md"
    df_stats.to_csv(table1_csv_path, index=False)
    with open(table1_md_path, "w") as f:
        f.write(f"# Table 1: PlantVillage Full {num_classes}-Class Statistics\n\n")
        f.write(df_stats.to_markdown(index=False))

    print(f"[Saved Table 1 CSV] -> {table1_csv_path}")
    print(f"[Saved Table 1 Markdown] -> {table1_md_path}")

    return {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "stats_df": df_stats
    }


def load_split_data(split_name: str, config_path: str = "configs/config.yaml") -> List[Tuple[str, int]]:
    """Load pre-saved split samples."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    processed_dir = cfg["dataset"]["processed_dir"]
    split_file = os.path.join(processed_dir, "splits", f"{split_name}_split.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}. Run prepare_plantvillage_data() first.")
    with open(split_file, "r") as f:
        samples = json.load(f)
    return [(s[0], int(s[1])) for s in samples]


def get_dataloader(
    samples: List[Tuple[str, int]],
    batch_size: int = 32,
    is_train: bool = True,
    num_workers: int = 2,
    image_size: int = 224
) -> DataLoader:
    """Create DataLoader with appropriate transforms."""
    transform = get_transforms(image_size=image_size, is_train=is_train)
    dataset = PlantVillageDataset(samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    prepare_plantvillage_data()
