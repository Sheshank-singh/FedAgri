import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import time
import json
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from data.dataset import load_split_data, get_dataloader
from models.model_factory import create_model, count_parameters, get_model_size_mb, compute_flops, measure_inference_latency
from federated.client import VirtualFederatedClient
from federated.fedavg import aggregate_fedavg
from federated.fedprox import aggregate_fedprox
from federated.partition import load_client_partition
from evaluation.evaluate import evaluate_model
from evaluation.metrics import compute_client_robustness_metrics
from evaluation.efficiency import calculate_communication_cost


class FederatedServer:
    """
    Coordinates federated learning simulation across virtual clients.
    """
    def __init__(
        self,
        model_name: str,
        algorithm: str = "fedavg",
        alpha: Optional[float] = 0.1,
        is_iid: bool = False,
        mu: float = 0.0,
        config_path: str = "configs/config.yaml",
        quick: bool = False,
        callback: Optional[object] = None,
        stop_event: Optional[object] = None,
        run_id: Optional[str] = None,
        num_rounds: Optional[int] = None,
        num_clients: Optional[int] = None,
        local_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        client_lr: Optional[float] = None
    ):
        self.model_name = model_name
        self.algorithm = algorithm.lower()
        self.alpha = alpha
        self.is_iid = is_iid
        self.mu = mu if self.algorithm == "fedprox" else 0.0
        self.config_path = config_path
        self.quick = quick
        self.callback = callback
        self.stop_event = stop_event
        self.run_id = run_id

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.seed = self.cfg["seed"]
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = self.cfg["dataset"].get("num_classes", 38)
        self.image_size = self.cfg["dataset"].get("image_size", 224)

        if num_clients is not None:
            self.num_clients = num_clients
        elif quick:
            self.num_clients = self.cfg["federated"].get("quick_clients", 3)
        else:
            self.num_clients = self.cfg["federated"].get("num_clients", 5)

        if num_rounds is not None:
            self.num_rounds = num_rounds
        elif quick:
            self.num_rounds = self.cfg["federated"].get("quick_rounds", 3)
        else:
            self.num_rounds = self.cfg["federated"].get("num_rounds", 20)

        self.local_epochs = local_epochs if local_epochs is not None else self.cfg["federated"].get("local_epochs", 1)
        self.batch_size = batch_size if batch_size is not None else self.cfg["federated"].get("batch_size", 32)
        self.client_lr = client_lr if client_lr is not None else self.cfg["federated"].get("client_lr", 0.001)

        # 1. Initialize Global Model (Pretrained on ImageNet)
        self.global_model = create_model(self.model_name, num_classes=self.num_classes, pretrained=True)
        self.global_model.to(self.device)

        self.total_params, _ = count_parameters(self.global_model)
        self.model_size_mb = get_model_size_mb(self.global_model)
        self.gflops = compute_flops(self.global_model)

        # 2. Load Validation and Test Splits
        val_samples = load_split_data("val", self.config_path)
        test_samples = load_split_data("test", self.config_path)
        if quick:
            val_samples = val_samples[:150]
            test_samples = test_samples[:150]

        num_workers = 0 if sys.platform == "win32" else 2
        self.val_loader = get_dataloader(val_samples, batch_size=self.batch_size, is_train=False, num_workers=num_workers, image_size=self.image_size)
        self.test_loader = get_dataloader(test_samples, batch_size=self.batch_size, is_train=False, num_workers=num_workers, image_size=self.image_size)
        self.criterion = nn.CrossEntropyLoss()

        # 3. Setup Virtual Clients
        self.clients = self._setup_clients()

    def _setup_clients(self) -> List[VirtualFederatedClient]:
        train_samples = load_split_data("train", self.config_path)
        partition_dict = load_client_partition(alpha=self.alpha, is_iid=self.is_iid, config_path=self.config_path)

        clients = []
        for cid in range(self.num_clients):
            indices = partition_dict[cid]
            if self.quick:
                indices = indices[:200]
            c_samples = [train_samples[i] for i in indices]
            c = VirtualFederatedClient(
                client_id=cid,
                client_samples=c_samples,
                model_name=self.model_name,
                num_classes=self.num_classes,
                batch_size=self.batch_size,
                local_epochs=self.local_epochs,
                learning_rate=self.client_lr,
                device=self.device,
                image_size=self.image_size
            )
            clients.append(c)
        return clients

    def run_simulation(self) -> Dict:
        """
        Executes the federated rounds, performs aggregation, records convergence metrics,
        evaluates per-client consistency, and evaluates final performance on the test set.
        """
        algo_label = self.algorithm.upper()
        alpha_str = "IID" if self.is_iid else f"alpha={self.alpha}"
        mu_str = f"mu={self.mu}" if self.algorithm == "fedprox" else ""
        print(f"\n=======================================================")
        print(f"  Federated Experiment: {algo_label} | {self.model_name.upper()}")
        print(f"  Condition: {alpha_str} {mu_str} | Clients: {self.num_clients} | Rounds: {self.num_rounds}")
        print(f"=======================================================")

        os.makedirs("results/logs", exist_ok=True)
        os.makedirs("results/checkpoints", exist_ok=True)

        history = []
        best_val_f1 = -1.0
        if self.run_id:
            exp_id = self.run_id
        else:
            exp_id = f"{self.algorithm}_{self.model_name}_{'iid' if self.is_iid else f'alpha_{self.alpha}'}"
            if self.algorithm == "fedprox":
                exp_id += f"_mu_{self.mu}"
        checkpoint_path = f"results/checkpoints/{exp_id}.pt"

        total_sim_start = time.time()
        was_cancelled = False

        if self.callback and hasattr(self.callback, "on_experiment_start"):
            self.callback.on_experiment_start({
                "experiment": exp_id,
                "model": self.model_name,
                "algorithm": self.algorithm,
                "alpha": -1.0 if self.is_iid else float(self.alpha),
                "mu": float(self.mu),
                "num_rounds": self.num_rounds,
                "num_clients": self.num_clients,
                "batch_size": self.batch_size,
                "local_epochs": self.local_epochs,
                "client_lr": self.client_lr
            })

        for rnd in range(1, self.num_rounds + 1):
            if self.stop_event and getattr(self.stop_event, "is_set", lambda: False)():
                print(f"[Simulation Cancelled by User at Round {rnd}]")
                was_cancelled = True
                break

            round_start = time.time()
            if self.callback and hasattr(self.callback, "on_round_start"):
                self.callback.on_round_start(rnd, self.num_rounds)

            global_weights = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

            client_weights = []
            client_samples = []
            client_losses = []
            client_accs = []

            # 1. Virtual Clients Train Locally
            for client in self.clients:
                if self.stop_event and getattr(self.stop_event, "is_set", lambda: False)():
                    was_cancelled = True
                    break
                w_updated, n_k, loss_k, acc_k = client.fit(global_weights, mu=self.mu)
                client_weights.append(w_updated)
                client_samples.append(n_k)
                client_losses.append(loss_k)
                client_accs.append(acc_k)

                if self.callback and hasattr(self.callback, "on_client_fit_complete"):
                    self.callback.on_client_fit_complete(
                        round_num=rnd,
                        client_id=client.client_id,
                        loss=round(float(loss_k), 4),
                        accuracy=round(float(acc_k), 4),
                        sample_count=n_k
                    )

            if was_cancelled:
                break

            # 2. Server Aggregation (Weighted by client dataset size)
            if self.algorithm == "fedprox":
                aggregated = aggregate_fedprox(client_weights, client_samples)
            else:
                aggregated = aggregate_fedavg(client_weights, client_samples)

            self.global_model.load_state_dict(aggregated)
            if self.callback and hasattr(self.callback, "on_aggregation_complete"):
                self.callback.on_aggregation_complete(rnd, self.algorithm, self.mu)

            # 3. Global Model Validation Evaluation
            val_metrics, _, _ = evaluate_model(self.global_model, self.val_loader, self.criterion, self.device)

            # 4. Client-Level Performance Consistency (Section 18)
            client_eval_accs = []
            for client in self.clients:
                c_metrics = client.evaluate_local(weights=aggregated)
                client_eval_accs.append(c_metrics["accuracy"])

            robustness_metrics = compute_client_robustness_metrics(client_eval_accs)
            round_time = time.time() - round_start

            avg_train_loss = float(np.average(client_losses, weights=client_samples))

            print(f"Round {rnd:02d}/{self.num_rounds:02d} [{round_time:.1f}s] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['macro_f1']:.4f} | "
                  f"Client Acc (Mean/Worst/Std): {robustness_metrics['mean_client_accuracy']:.3f}/"
                  f"{robustness_metrics['worst_client_accuracy']:.3f}/{robustness_metrics['client_accuracy_std']:.3f}")

            # Checkpoint best global model
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                torch.save(self.global_model.state_dict(), checkpoint_path)

            round_entry = {
                "experiment": exp_id,
                "model": self.model_name,
                "algorithm": self.algorithm,
                "alpha": -1.0 if self.is_iid else float(self.alpha),
                "mu": float(self.mu),
                "round": rnd,
                "train_loss": round(avg_train_loss, 4),
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "mean_client_accuracy": robustness_metrics["mean_client_accuracy"],
                "worst_client_accuracy": robustness_metrics["worst_client_accuracy"],
                "client_accuracy_std": robustness_metrics["client_accuracy_std"],
                "round_time_s": round(round_time, 2)
            }
            history.append(round_entry)

            if self.callback and hasattr(self.callback, "on_round_complete"):
                self.callback.on_round_complete(round_entry)

        total_sim_time = round(time.time() - total_sim_start, 2)

        # 5. Final Test Evaluation on isolated test set
        print(f"\nEvaluating Global Model on Unseen Test Split...")
        if os.path.exists(checkpoint_path):
            self.global_model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        test_metrics, _, _ = evaluate_model(self.global_model, self.test_loader, self.criterion, self.device)
        latency_ms = measure_inference_latency(self.global_model, self.device)

        completed_rounds = len(history)
        comm_stats = calculate_communication_cost(
            model_size_mb=self.model_size_mb,
            num_clients=self.num_clients,
            num_rounds=max(completed_rounds, 1)
        )

        print(f"\n=== Final Federated Results for {exp_id} ===")
        print(f"  Test Accuracy:        {test_metrics['accuracy']:.4f}")
        print(f"  Test Macro F1:        {test_metrics['macro_f1']:.4f}")
        print(f"  Test Weighted F1:     {test_metrics['weighted_f1']:.4f}")
        print(f"  Total Simulation Time:{total_sim_time}s")
        print(f"  Total Comm Volume:    {comm_stats['total_communication_mb']} MB ({comm_stats['total_communication_gb']} GB)")

        # Save per-round CSV history
        df_history = pd.DataFrame(history)
        history_csv = f"results/logs/{exp_id}_rounds.csv"
        df_history.to_csv(history_csv, index=False)
        print(f"[Saved Round History] -> {history_csv}")

        summary = {
            "experiment": exp_id,
            "model": self.model_name,
            "algorithm": self.algorithm,
            "alpha": -1.0 if self.is_iid else float(self.alpha),
            "mu": float(self.mu),
            "rounds": self.num_rounds,
            "completed_rounds": completed_rounds,
            "was_cancelled": was_cancelled,
            "num_clients": self.num_clients,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "mean_client_accuracy": history[-1]["mean_client_accuracy"] if history else 0.0,
            "worst_client_accuracy": history[-1]["worst_client_accuracy"] if history else 0.0,
            "client_accuracy_std": history[-1]["client_accuracy_std"] if history else 0.0,
            "training_time_s": total_sim_time,
            "model_size_mb": self.model_size_mb,
            "total_communication_mb": comm_stats["total_communication_mb"],
            "gflops": self.gflops,
            "inference_latency_ms": latency_ms,
            "checkpoint_path": checkpoint_path if os.path.exists(checkpoint_path) else None
        }

        if self.callback and hasattr(self.callback, "on_simulation_complete"):
            self.callback.on_simulation_complete(summary)

        return summary
