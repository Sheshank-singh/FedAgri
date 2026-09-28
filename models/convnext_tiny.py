import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


def get_convnext_tiny(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    ConvNeXt-Tiny: Modern CNN architecture with stronger representation capacity.
    Replaces the final linear layer in model.classifier[2].
    """
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model
