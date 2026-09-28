import copy
import torch
from typing import Dict, List
from collections import OrderedDict


def aggregate_fedavg(
    client_weights_list: List[Dict[str, torch.Tensor]],
    client_samples_list: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Standard weighted Federated Averaging:
        w_global = sum_{k=1}^K (n_k / N) * w_k
    where:
        - n_k = number of training samples on client k
        - N = sum(n_k) = total training samples across participating clients
        - w_k = client model parameters
    """
    total_samples = sum(client_samples_list)
    if total_samples == 0:
        raise ValueError("Total number of client samples is zero in FedAvg aggregation.")

    # Initialize aggregated weights container
    aggregated_weights = OrderedDict()
    first_client_dict = client_weights_list[0]

    for key in first_client_dict.keys():
        # Check if parameter is a floating-point weight or integer buffer (e.g. num_batches_tracked)
        if first_client_dict[key].is_floating_point():
            aggregated_weights[key] = torch.zeros_like(first_client_dict[key], dtype=first_client_dict[key].dtype)
            for k in range(len(client_weights_list)):
                weight_ratio = client_samples_list[k] / float(total_samples)
                aggregated_weights[key] += client_weights_list[k][key] * weight_ratio
        else:
            # Non-floating point buffers (e.g., batchnorm running counts) take majority / first client
            aggregated_weights[key] = copy.deepcopy(first_client_dict[key])

    return aggregated_weights
