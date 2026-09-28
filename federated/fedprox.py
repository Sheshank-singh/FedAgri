import torch
from typing import Dict, List
from collections import OrderedDict
from federated.fedavg import aggregate_fedavg


def aggregate_fedprox(
    client_weights_list: List[Dict[str, torch.Tensor]],
    client_samples_list: List[int]
) -> Dict[str, torch.Tensor]:
    """
    FedProx aggregation function.
    Server aggregates client models via sample-weighted averaging,
    matching FedAvg aggregation while clients solved the proximal objective locally.
    """
    return aggregate_fedavg(client_weights_list, client_samples_list)


class FedProxManager:
    """
    Helper manager for FedProx hyperparameter configurations and verification.
    """
    def __init__(self, mu: float = 0.01):
        self.mu = float(mu)

    def get_mu(self) -> float:
        return self.mu

    def set_mu(self, mu: float):
        self.mu = float(mu)
