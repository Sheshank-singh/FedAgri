import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def get_efficientnet_b0(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B0: Parameter/FLOP-efficient CNN baseline.
    Replaces the final linear layer in model.classifier[1].
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
