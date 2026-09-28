import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_resnet50(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    ResNet50: Conventional deep CNN baseline.
    Replaces the final fully connected layer model.fc.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
