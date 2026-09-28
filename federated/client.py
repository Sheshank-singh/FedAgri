import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    FLWR_AVAILABLE = True
except ImportError:
    FLWR_AVAILABLE = False

from data.dataset import PlantVillageDataset, get_dataloader
from models.model_factory import create_model
from training.local_train import train_client_locally
from evaluation.evaluate import evaluate_model


class VirtualFederatedClient:
    """
    Virtual Federated Client that holds its private local partition of training data.
    Raw images remain strictly local to this virtual client.
    """
    def __init__(
        self,
        client_id: int,
        client_samples: List[Tuple[str, int]],
        model_name: str,
        num_classes: int = 38,
        batch_size: int = 32,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        device: torch.device = None,
        image_size: int = 224
    ):
        self.client_id = client_id
        self.client_samples = client_samples
        self.num_samples = len(client_samples)
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create private local data loader
        num_workers = 0 if sys.platform == "win32" else 2
        self.train_loader = get_dataloader(
            self.client_samples,
            batch_size=self.batch_size,
            is_train=True,
            num_workers=num_workers,
            image_size=self.image_size
        )

        # Private local model instance
        self.model = create_model(self.model_name, num_classes=self.num_classes, pretrained=False)

    def fit(
        self,
        global_weights: Dict[str, torch.Tensor],
        mu: float = 0.0
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """
        Receives global weights, applies them, executes local training (FedAvg or FedProx),
        and returns the updated weights along with sample count and training metrics.
        """
        # Load received global weights
        self.model.load_state_dict(global_weights)

        # Train locally (with FedProx proximal term if mu > 0)
        updated_weights, num_samples, train_loss, train_acc = train_client_locally(
            model=self.model,
            train_loader=self.train_loader,
            global_model_weights=global_weights,
            device=self.device,
            local_epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            mu=mu
        )

        return updated_weights, num_samples, train_loss, train_acc

    def evaluate_local(
        self,
        weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Evaluates current or specified weights on the client's local data.
        """
        if weights is not None:
            self.model.load_state_dict(weights)
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        metrics, _, _ = evaluate_model(self.model, self.train_loader, criterion, self.device)
        return metrics


if FLWR_AVAILABLE:
    class FlowerFederatedClient(fl.client.NumPyClient):
        """
        Flower NumPyClient implementation for interoperability with Flower server simulation.
        """
        def __init__(self, virtual_client: VirtualFederatedClient, mu: float = 0.0):
            self.virtual_client = virtual_client
            self.mu = mu

        def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
            return [val.cpu().numpy() for _, val in self.virtual_client.model.state_dict().items()]

        def set_parameters(self, parameters: NDArrays):
            params_dict = zip(self.virtual_client.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.virtual_client.model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
            self.set_parameters(parameters)
            global_weights = copy.deepcopy(self.virtual_client.model.state_dict())
            updated_weights, num_samples, loss, acc = self.virtual_client.fit(global_weights, mu=self.mu)
            new_params = [val.cpu().numpy() for _, val in updated_weights.items()]
            return new_params, num_samples, {"train_loss": loss, "train_acc": acc}

        def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
            self.set_parameters(parameters)
            metrics = self.virtual_client.evaluate_local()
            return float(metrics["loss"]), self.virtual_client.num_samples, {"accuracy": float(metrics["accuracy"])}
