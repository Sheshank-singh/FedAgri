import os
import sys
import time
import json
import uuid
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated.server import FederatedServer


@dataclass
class ClientRoundState:
    client_id: int
    round_num: int
    loss: float
    accuracy: float
    sample_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentJob:
    job_id: str
    experiment_id: str
    model_name: str
    algorithm: str
    alpha: float
    is_iid: bool
    mu: float
    num_clients: int
    num_rounds: int
    batch_size: int
    local_epochs: int
    client_lr: float
    quick: bool
    status: str  # "PENDING", "RUNNING", "COMPLETED", "CANCELLED", "FAILED"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_round: int = 0
    total_rounds: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    recent_client_states: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    log_events: List[Dict[str, Any]] = field(default_factory=list)


class ProgressCallbackBridge:
    """
    Bridge connecting FederatedServer simulation events to the orchestrator and WebSocket broadcast.
    """
    def __init__(self, orchestrator: "TrainingOrchestrator", job_id: str):
        self.orchestrator = orchestrator
        self.job_id = job_id

    def on_experiment_start(self, config: Dict[str, Any]):
        self.orchestrator.handle_experiment_start(self.job_id, config)

    def on_round_start(self, round_num: int, total_rounds: int):
        self.orchestrator.handle_round_start(self.job_id, round_num, total_rounds)

    def on_client_fit_complete(self, round_num: int, client_id: int, loss: float, accuracy: float, sample_count: int):
        self.orchestrator.handle_client_fit_complete(self.job_id, round_num, client_id, loss, accuracy, sample_count)

    def on_aggregation_complete(self, round_num: int, algorithm: str, mu: float):
        self.orchestrator.handle_aggregation_complete(self.job_id, round_num, algorithm, mu)

    def on_round_complete(self, round_data: Dict[str, Any]):
        self.orchestrator.handle_round_complete(self.job_id, round_data)

    def on_simulation_complete(self, summary: Dict[str, Any]):
        self.orchestrator.handle_simulation_complete(self.job_id, summary)


class TrainingOrchestrator:
    """
    Thread-safe orchestrator managing federated experiments and live WebSocket streaming.
    """
    def __init__(self):
        self.active_job: Optional[ExperimentJob] = None
        self.job_history: Dict[str, ExperimentJob] = {}
        self.lock = threading.Lock()
        self.stop_event: Optional[threading.Event] = None
        self.simulation_thread: Optional[threading.Thread] = None
        self.subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def add_subscriber(self, queue: asyncio.Queue):
        if queue not in self.subscribers:
            self.subscribers.append(queue)

    def remove_subscriber(self, queue: asyncio.Queue):
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    def broadcast_event_sync(self, event_type: str, data: Dict[str, Any]):
        """Thread-safe event broadcast from the training thread to async WebSocket queues."""
        payload = {
            "type": event_type,
            "job_id": self.active_job.job_id if self.active_job else None,
            "timestamp": time.time(),
            "data": data
        }

        # Also append to log events of the active job
        if self.active_job:
            self.active_job.log_events.append({
                "time": time.strftime("%H:%M:%S"),
                "type": event_type,
                "message": data.get("message", f"Event: {event_type}")
            })
            # Keep log events bounded
            if len(self.active_job.log_events) > 200:
                self.active_job.log_events = self.active_job.log_events[-200:]

        if self._loop and self._loop.is_running():
            for q in list(self.subscribers):
                try:
                    self._loop.call_soon_threadsafe(q.put_nowait, payload)
                except Exception as e:
                    pass

    def launch_experiment(self, config: Dict[str, Any]) -> ExperimentJob:
        with self.lock:
            if self.active_job and self.active_job.status in ("RUNNING", "PENDING"):
                raise RuntimeError(f"An experiment is already active (Job ID: {self.active_job.job_id}). Wait or stop it first.")

            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            model_name = config.get("model", "mobilenetv3")
            algorithm = config.get("algorithm", "fedavg").lower()
            alpha = float(config.get("alpha", 0.1))
            is_iid = bool(config.get("is_iid", False))
            mu = float(config.get("mu", 0.01 if algorithm == "fedprox" else 0.0))
            quick = bool(config.get("quick", False))
            num_clients = int(config.get("num_clients", 3 if quick else 5))
            num_rounds = int(config.get("num_rounds", 3 if quick else 20))
            batch_size = int(config.get("batch_size", 32))
            local_epochs = int(config.get("local_epochs", 1))
            client_lr = float(config.get("client_lr", 0.001))

            exp_id = f"{algorithm}_{model_name}_{'iid' if is_iid else f'alpha_{alpha}'}"
            if algorithm == "fedprox":
                exp_id += f"_mu_{mu}"
            exp_id += f"_{int(time.time())}"

            job = ExperimentJob(
                job_id=job_id,
                experiment_id=exp_id,
                model_name=model_name,
                algorithm=algorithm,
                alpha=alpha,
                is_iid=is_iid,
                mu=mu,
                num_clients=num_clients,
                num_rounds=num_rounds,
                batch_size=batch_size,
                local_epochs=local_epochs,
                client_lr=client_lr,
                quick=quick,
                status="PENDING",
                total_rounds=num_rounds
            )

            self.active_job = job
            self.job_history[job_id] = job
            self.stop_event = threading.Event()

            # Start worker thread
            self.simulation_thread = threading.Thread(
                target=self._run_simulation_worker,
                args=(job, self.stop_event),
                daemon=True,
                name=f"FedAgriWorker-{job_id}"
            )
            self.simulation_thread.start()

            return job

    def stop_active_experiment(self) -> bool:
        with self.lock:
            if not self.active_job or self.active_job.status != "RUNNING":
                return False
            if self.stop_event:
                self.stop_event.set()
                self.active_job.status = "CANCELLED"
                self.broadcast_event_sync("EXPERIMENT_CANCELLED", {
                    "message": "Experiment cancellation requested by user.",
                    "job_id": self.active_job.job_id
                })
                return True
            return False

    def get_active_job_dict(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            if not self.active_job:
                return None
            return asdict(self.active_job)

    def _run_simulation_worker(self, job: ExperimentJob, stop_event: threading.Event):
        try:
            job.status = "RUNNING"
            job.started_at = time.time()
            callback = ProgressCallbackBridge(self, job.job_id)

            server = FederatedServer(
                model_name=job.model_name,
                algorithm=job.algorithm,
                alpha=job.alpha if not job.is_iid else None,
                is_iid=job.is_iid,
                mu=job.mu,
                config_path="configs/config.yaml",
                quick=job.quick,
                callback=callback,
                stop_event=stop_event,
                run_id=job.experiment_id,
                num_rounds=job.num_rounds,
                num_clients=job.num_clients,
                local_epochs=job.local_epochs,
                batch_size=job.batch_size,
                client_lr=job.client_lr
            )

            summary = server.run_simulation()
            with self.lock:
                job.summary = summary
                job.completed_at = time.time()
                if stop_event.is_set():
                    job.status = "CANCELLED"
                else:
                    job.status = "COMPLETED"

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"[Orchestrator Worker Error]:\n{err_msg}")
            with self.lock:
                job.status = "FAILED"
                job.error = str(e)
                job.completed_at = time.time()

            self.broadcast_event_sync("EXPERIMENT_ERROR", {
                "message": f"Experiment failed: {str(e)}",
                "error": str(e)
            })

    # --- Callback Handlers from Training Thread ---

    def handle_experiment_start(self, job_id: str, config: Dict[str, Any]):
        self.broadcast_event_sync("EXPERIMENT_STARTED", {
            "message": f"Federated simulation started: {config['algorithm'].upper()} with {config['model'].upper()} across {config['num_clients']} clients.",
            "config": config
        })

    def handle_round_start(self, job_id: str, round_num: int, total_rounds: int):
        with self.lock:
            if self.active_job and self.active_job.job_id == job_id:
                self.active_job.current_round = round_num
                self.active_job.total_rounds = total_rounds

        self.broadcast_event_sync("ROUND_START", {
            "message": f"Starting Round {round_num}/{total_rounds}...",
            "round": round_num,
            "total_rounds": total_rounds
        })

    def handle_client_fit_complete(self, job_id: str, round_num: int, client_id: int, loss: float, accuracy: float, sample_count: int):
        with self.lock:
            if self.active_job and self.active_job.job_id == job_id:
                self.active_job.recent_client_states[client_id] = {
                    "client_id": client_id,
                    "round": round_num,
                    "loss": loss,
                    "accuracy": accuracy,
                    "samples": sample_count,
                    "updated_at": time.time()
                }

        self.broadcast_event_sync("CLIENT_FIT_COMPLETE", {
            "message": f"Client {client_id + 1} completed local training (Loss: {loss:.4f}, Acc: {accuracy:.3f}, Samples: {sample_count})",
            "round": round_num,
            "client_id": client_id,
            "loss": loss,
            "accuracy": accuracy,
            "sample_count": sample_count
        })

    def handle_aggregation_complete(self, job_id: str, round_num: int, algorithm: str, mu: float):
        algo_label = f"FedProx (μ={mu})" if algorithm == "fedprox" else "FedAvg (Sample-Weighted)"
        self.broadcast_event_sync("AGGREGATION_COMPLETE", {
            "message": f"Round {round_num} Aggregation completed using {algo_label}.",
            "round": round_num,
            "algorithm": algorithm,
            "mu": mu
        })

    def handle_round_complete(self, job_id: str, round_data: Dict[str, Any]):
        with self.lock:
            if self.active_job and self.active_job.job_id == job_id:
                self.active_job.history.append(round_data)

        self.broadcast_event_sync("ROUND_COMPLETE", {
            "message": f"Round {round_data['round']} complete | Val Acc: {round_data['validation_accuracy']:.4f} | F1: {round_data['validation_macro_f1']:.4f} | Worst Client: {round_data['worst_client_accuracy']:.4f}",
            "round_data": round_data
        })

    def handle_simulation_complete(self, job_id: str, summary: Dict[str, Any]):
        self.broadcast_event_sync("EXPERIMENT_COMPLETE", {
            "message": f"Federated Simulation Complete! Test Acc: {summary['test_accuracy']:.4f}, Macro-F1: {summary['test_macro_f1']:.4f}, Total Comm: {summary['total_communication_mb']} MB",
            "summary": summary
        })


# Global orchestrator singleton instance
orchestrator = TrainingOrchestrator()
