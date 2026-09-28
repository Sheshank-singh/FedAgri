import time
import torch
import torch.nn as nn
from typing import Tuple, Dict

from models.mobilenetv3 import get_mobilenetv3
from models.resnet50 import get_resnet50
from models.efficientnet_b0 import get_efficientnet_b0
from models.convnext_tiny import get_convnext_tiny


MODEL_REGISTRY = {
    "mobilenetv3": get_mobilenetv3,
    "resnet50": get_resnet50,
    "efficientnet_b0": get_efficientnet_b0,
    "convnext_tiny": get_convnext_tiny,
}


def create_model(model_name: str, num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """
    Factory function to instantiate any of the four supported CNN backbones.
    """
    name_clean = model_name.lower().strip()
    if name_clean not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name_clean](num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculates parameter memory footprint in megabytes (FP32 = 4 bytes per param).
    """
    total_params = sum(p.numel() for p in model.parameters())
    return (total_params * 4.0) / (1024.0 * 1024.0)


def compute_flops(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> float:
    """
    Computes exact MACs/FLOPs using ptflops. Returns GFLOPs.
    """
    try:
        from ptflops import get_model_complexity_info
        model_eval = model.eval()
        macs, _ = get_model_complexity_info(
            model_eval,
            (input_size[1], input_size[2], input_size[3]),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        # FLOPs are typically ~2 * MACs
        gflops = (macs * 2.0) / 1e9
        return round(gflops, 4)
    except Exception as e:
        print(f"Warning: ptflops calculation error: {e}")
        return 0.0


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    warmup: int = 10,
    repetitions: int = 50
) -> float:
    """
    Measures average single-image inference latency in milliseconds (ms).
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size, device=device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repetitions):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
        else:
            start = time.time()
            for _ in range(repetitions):
                _ = model(dummy_input)
            elapsed = time.time() - start
            
    latency_ms = (elapsed / repetitions) * 1000.0
    return round(latency_ms, 3)


def get_architecture_profile(model_name: str, num_classes: int = 38, device: torch.device = None) -> Dict:
    """
    Profile an architecture for Table 2.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    total_params, trainable_params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    gflops = compute_flops(model)
    latency_ms = measure_inference_latency(model, device)
    
    return {
        "Model": model_name,
        "Parameters": total_params,
        "Trainable_Params": trainable_params,
        "Size_MB": round(size_mb, 2),
        "GFLOPs": gflops,
        "Inference_Latency_ms": latency_ms
    }
