import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple, Optional


def train_client_locally(
    model: nn.Module,
    train_loader: DataLoader,
    global_model_weights: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    local_epochs: int = 1,
    learning_rate: float = 0.001,
    mu: float = 0.0,
    weight_decay: float = 0.0001
) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
    """
    Performs local client training for FedAvg (mu = 0.0) or FedProx (mu > 0.0).
    
    Section 15: FedProx objective:
        Loss = L(w) + (mu / 2) * ||w - w_global||^2
    
    Returns:
        (updated_state_dict, num_samples, avg_train_loss, train_accuracy)
    """
    model.train()
    model.to(device)

    # Freeze reference copy of global weights on device if FedProx proximal term is enabled
    global_params = None
    if mu > 0.0 and global_model_weights is not None:
        global_params = {
            name: param.clone().detach().to(device)
            for name, param in global_model_weights.items()
        }

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for epoch in range(local_epochs):
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = targets.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, targets)

            # Add FedProx proximal penalty if mu > 0
            if mu > 0.0 and global_params is not None:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_params:
                        proximal_term += torch.sum((param - global_params[name]) ** 2)
                loss = ce_loss + (mu / 2.0) * proximal_term
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)

    # Move model weights back to cpu dictionary to save GPU VRAM
    updated_state_dict = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }

    return updated_state_dict, total_samples, avg_loss, acc
