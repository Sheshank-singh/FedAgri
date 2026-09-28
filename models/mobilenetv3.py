import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_mobilenetv3(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    MobileNetV3-Small: Lightweight edge-oriented CNN baseline.
    Replaces the final linear classification layer in model.classifier[3].
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
