import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import numpy as np

from evaluation.metrics import compute_classification_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluates model on dataloader.
    Returns:
      (metrics_dict, y_true, y_pred)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = round(total_loss / max(total_samples, 1), 4)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = avg_loss

    return metrics, y_true, y_pred
