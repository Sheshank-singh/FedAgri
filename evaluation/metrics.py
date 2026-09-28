import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard multi-class classification metrics:
    - Accuracy
    - Macro Precision, Macro Recall, Macro F1
    - Weighted F1
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4)
    }


def compute_client_robustness_metrics(client_accuracies: List[float]) -> Dict[str, float]:
    """
    Client-level performance consistency (Section 18):
    - Mean client accuracy: MeanAcc = mean(client accuracies)
    - Worst-client accuracy: WorstAcc = min(client accuracies)
    - Client performance standard deviation: StdAcc = std(client accuracies)
    """
    if not client_accuracies:
        return {"mean_client_accuracy": 0.0, "worst_client_accuracy": 0.0, "client_accuracy_std": 0.0}
    
    arr = np.array(client_accuracies, dtype=np.float64)
    return {
        "mean_client_accuracy": round(float(np.mean(arr)), 4),
        "worst_client_accuracy": round(float(np.min(arr)), 4),
        "client_accuracy_std": round(float(np.std(arr)), 4)
    }
