import os
import sys
import glob
import time
import json
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.orchestrator import orchestrator
from backend.inference import predict_leaf_disease, load_classes_mapping, REMEDIES
from models.model_factory import MODEL_REGISTRY, count_parameters, get_model_size_mb, compute_flops, measure_inference_latency, create_model


app = FastAPI(
    title="FedAgri API",
    description="Federated Learning Platform for Smart Agriculture — Research & Diagnostics",
    version="1.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    orchestrator.register_loop(asyncio.get_event_loop())
    print("[FedAgri Server] Asynchronous event loop registered with TrainingOrchestrator.")


# --- REST API Endpoints ---

@app.get("/api/health")
async def get_health():
    """Returns system status, GPU/CUDA hardware profile, and memory metrics."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Host System)"
    gpu_memory = None
    if cuda_available:
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
            gpu_memory = {
                "total_mb": round(total_mem, 1),
                "allocated_mb": round(allocated_mem, 1),
                "free_mb": round(total_mem - allocated_mem, 1)
            }
        except Exception:
            pass

    return {
        "status": "healthy",
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "gpu_memory": gpu_memory,
        "active_job": orchestrator.get_active_job_dict() is not None
    }


@app.get("/api/models")
async def get_models():
    """Returns architectural profiles, parameters, size in MB, and computational complexity."""
    table2_path = "results/tables/table2_model_characteristics.csv"
    if os.path.exists(table2_path):
        df = pd.read_csv(table2_path)
        return df.to_dict(orient="records")

    # Fallback to runtime calculation
    profiles = []
    descriptions = {
        "mobilenetv3": "Lightweight edge baseline with depthwise separable convolutions & squeeze-excitation blocks.",
        "resnet50": "Conventional deep residual network with bottleneck blocks and skip connections.",
        "efficientnet_b0": "Compound-scaled architecture balancing depth, width, and resolution efficiently.",
        "convnext_tiny": "Modernized pure convolutional network with 7x7 depthwise kernels and inverted bottlenecks."
    }

    for name in ["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"]:
        model = create_model(name, num_classes=38, pretrained=False)
        total_p, trainable_p = count_parameters(model)
        size_mb = get_model_size_mb(model)
        gflops = compute_flops(model)

        profiles.append({
            "Model": name,
            "Parameters (M)": round(total_p / 1e6, 2),
            "Total_Params": total_p,
            "Size_MB": round(size_mb, 2),
            "GFLOPs": gflops,
            "Description": descriptions.get(name, "")
        })
    return profiles


@app.get("/api/dataset/summary")
async def get_dataset_summary():
    """Returns dataset overview: 38 classes, sample count, and split breakdown."""
    classes_path = "data/processed/classes.json"
    splits_dir = "data/processed/splits"

    num_classes = 38
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            cdata = json.load(f)
            num_classes = len(cdata.get("class_to_idx", {}))

    train_count = 0
    val_count = 0
    test_count = 0

    if os.path.exists(os.path.join(splits_dir, "train_split.json")):
        with open(os.path.join(splits_dir, "train_split.json"), "r") as f:
            train_count = len(json.load(f))
    if os.path.exists(os.path.join(splits_dir, "val_split.json")):
        with open(os.path.join(splits_dir, "val_split.json"), "r") as f:
            val_count = len(json.load(f))
    if os.path.exists(os.path.join(splits_dir, "test_split.json")):
        with open(os.path.join(splits_dir, "test_split.json"), "r") as f:
            test_count = len(json.load(f))

    total_images = train_count + val_count + test_count if (train_count + val_count + test_count) > 0 else 54305

    crops = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper Bell", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]

    return {
        "name": "PlantVillage Crop Disease Dataset",
        "total_images": total_images,
        "num_classes": num_classes,
        "num_crops": len(crops),
        "crops": crops,
        "train_samples": train_count or 38013,
        "val_samples": val_count or 8146,
        "test_samples": test_count or 8146,
        "train_split": "70%",
        "val_split": "15%",
        "test_split": "15% (Strictly isolated)"
    }


@app.get("/api/dataset/classes")
async def get_dataset_classes():
    """Returns all 38 classes with crop categories, disease labels, and health indicators."""
    _, idx_to_class = load_classes_mapping()
    classes_list = []

    # Map sample assets if available
    sample_files = os.listdir("assets/samples") if os.path.exists("assets/samples") else []

    for idx, cls_raw in idx_to_class.items():
        parts = cls_raw.split("___")
        crop = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
        is_healthy = "healthy" in disease.lower()

        # Find matching sample
        sample_url = None
        for sf in sample_files:
            sf_clean = sf.lower().replace(".jpg", "").replace(".png", "")
            if any(term in sf_clean for term in cls_raw.lower().split("___")):
                sample_url = f"/assets/samples/{sf}"
                break

        classes_list.append({
            "idx": idx,
            "class_raw": cls_raw,
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
            "severity": REMEDIES.get(cls_raw, {}).get("severity", "Normal"),
            "sample_image": sample_url
        })

    return classes_list


@app.get("/api/dataset/samples")
async def get_sample_images():
    """Lists available test leaf samples for instant inference testing."""
    samples_dir = "assets/samples"
    if not os.path.exists(samples_dir):
        return []

    files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_data = []

    for f in sorted(files):
        # Format human-readable title
        clean_name = f.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").replace("_", " ").title()
        sample_data.append({
            "filename": f,
            "title": clean_name,
            "url": f"/assets/samples/{f}"
        })
    return sample_data


@app.get("/api/dataset/partitions")
async def get_partition_distribution(alpha: Optional[float] = Query(0.1), is_iid: bool = Query(False)):
    """Returns client sample allocations for Dirichlet non-IID partition visualization."""
    partitions_dir = "data/processed/partitions"
    fname = "partition_iid_seed42.json" if is_iid else f"partition_alpha_{alpha}_seed42.json"
    full_path = os.path.join(partitions_dir, fname)

    if not os.path.exists(full_path):
        # Look for default
        full_path = os.path.join(partitions_dir, "partition_alpha_0.1_seed42.json")

    if not os.path.exists(full_path):
        return {"error": "Partitions not generated yet."}

    with open(full_path, "r") as f:
        data = json.load(f)

    # Convert keys to integers
    client_counts = {int(k): len(v) for k, v in data.items()}
    return {
        "alpha": "IID" if is_iid else alpha,
        "num_clients": len(data),
        "client_sample_counts": client_counts,
        "total_partitioned_samples": sum(client_counts.values())
    }


@app.get("/api/experiments")
async def list_experiments():
    """Lists all completed experiments from isolated run folders and research baselines."""
    experiments = []
    seen_ids = set()

    # 1. First, discover from isolated timestamped run directories (results/runs/)
    runs_dir = "results/runs"
    if os.path.exists(runs_dir):
        folders = [f for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]
        folders.sort(reverse=True)
        for folder in folders:
            folder_path = os.path.join(runs_dir, folder)
            rounds_path = os.path.join(folder_path, "rounds.csv")
            summary_path = os.path.join(folder_path, "summary.json")
            config_path = os.path.join(folder_path, "config.json")
            ckpt_path = os.path.join(folder_path, "checkpoint.pt")

            if not os.path.exists(rounds_path) and not os.path.exists(summary_path):
                continue

            summary_data = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                except Exception:
                    pass

            config_data = {}
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                except Exception:
                    pass

            df = None
            if os.path.exists(rounds_path):
                try:
                    df = pd.read_csv(rounds_path)
                except Exception:
                    pass

            rounds_completed = len(df) if df is not None else summary_data.get("completed_rounds", 0)
            best_val_f1 = float(df["validation_macro_f1"].max()) if df is not None and "validation_macro_f1" in df else summary_data.get("test_macro_f1", 0.0)
            best_val_acc = float(df["validation_accuracy"].max()) if df is not None and "validation_accuracy" in df else summary_data.get("test_accuracy", 0.0)
            total_time = float(df["round_time_s"].sum()) if df is not None and "round_time_s" in df else summary_data.get("training_time_s", 0.0)

            model = config_data.get("model", summary_data.get("model", "unknown"))
            algo = config_data.get("algorithm", summary_data.get("algorithm", "fedavg")).upper()
            alpha_val = config_data.get("alpha", summary_data.get("alpha", -1.0))
            alpha_str = "IID" if float(alpha_val) < 0 else str(alpha_val)
            mu_val = float(config_data.get("mu", summary_data.get("mu", 0.0)))

            has_ckpt = os.path.exists(ckpt_path)
            seen_ids.add(folder)

            experiments.append({
                "experiment_id": folder,
                "run_folder": folder,
                "run_dir": folder_path.replace("\\", "/"),
                "is_isolated_run": True,
                "model": model,
                "algorithm": algo,
                "alpha": alpha_str,
                "mu": mu_val,
                "rounds_completed": rounds_completed,
                "val_accuracy": round(best_val_acc, 4),
                "val_macro_f1": round(best_val_f1, 4),
                "mean_client_accuracy": round(float(summary_data.get("mean_client_accuracy", 0.0)), 4),
                "worst_client_accuracy": round(float(summary_data.get("worst_client_accuracy", 0.0)), 4),
                "client_accuracy_std": round(float(summary_data.get("client_accuracy_std", 0.0)), 4),
                "total_time_s": round(total_time, 1),
                "has_checkpoint": has_ckpt,
                "checkpoint_name": f"runs/{folder}/checkpoint.pt" if has_ckpt else None,
                "checkpoint_download_url": f"/api/runs/{folder}/checkpoint" if has_ckpt else None
            })

    # 2. Add baseline research logs (results/logs/*_rounds.csv)
    log_files = sorted(glob.glob("results/logs/*_rounds.csv"))
    for log_path in log_files:
        try:
            exp_id = os.path.basename(log_path).replace("_rounds.csv", "")
            if exp_id in seen_ids:
                continue

            df = pd.read_csv(log_path)
            if len(df) == 0:
                continue
            last_row = df.iloc[-1]
            model = str(last_row.get("model", "unknown"))
            algo = str(last_row.get("algorithm", "fedavg")).upper()
            alpha_val = last_row.get("alpha", -1.0)
            alpha_str = "IID" if float(alpha_val) < 0 else str(alpha_val)
            mu_val = float(last_row.get("mu", 0.0))

            best_val_f1 = float(df["validation_macro_f1"].max()) if "validation_macro_f1" in df else 0.0
            best_val_acc = float(df["validation_accuracy"].max()) if "validation_accuracy" in df else 0.0
            total_time = float(df["round_time_s"].sum()) if "round_time_s" in df else 0.0

            ckpt_path = f"results/checkpoints/{exp_id}.pt"
            has_checkpoint = os.path.exists(ckpt_path)

            experiments.append({
                "experiment_id": exp_id,
                "run_folder": None,
                "run_dir": None,
                "is_isolated_run": False,
                "model": model,
                "algorithm": algo,
                "alpha": alpha_str,
                "mu": mu_val,
                "rounds_completed": len(df),
                "val_accuracy": round(best_val_acc, 4),
                "val_macro_f1": round(best_val_f1, 4),
                "mean_client_accuracy": round(float(last_row.get("mean_client_accuracy", 0.0)), 4),
                "worst_client_accuracy": round(float(last_row.get("worst_client_accuracy", 0.0)), 4),
                "client_accuracy_std": round(float(last_row.get("client_accuracy_std", 0.0)), 4),
                "total_time_s": round(total_time, 1),
                "has_checkpoint": has_checkpoint,
                "checkpoint_name": f"{exp_id}.pt" if has_checkpoint else None,
                "checkpoint_download_url": f"/api/checkpoints/download/{exp_id}.pt" if has_checkpoint else None
            })
        except Exception as e:
            print(f"Error parsing log {log_path}: {e}")

    return experiments


@app.get("/api/runs")
async def list_runs():
    """Lists all isolated experiment run directories from results/runs/ (never overwritten)."""
    runs_dir = "results/runs"
    if not os.path.exists(runs_dir):
        return []

    folders = [f for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]
    folders.sort(reverse=True)
    runs = []

    for folder in folders:
        folder_path = os.path.join(runs_dir, folder)
        config_path = os.path.join(folder_path, "config.json")
        summary_path = os.path.join(folder_path, "summary.json")
        rounds_path = os.path.join(folder_path, "rounds.csv")
        ckpt_path = os.path.join(folder_path, "checkpoint.pt")
        events_path = os.path.join(folder_path, "events.log")

        config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        summary_data = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
            except Exception:
                pass

        has_ckpt = os.path.exists(ckpt_path)
        ckpt_size_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2) if has_ckpt else 0.0

        rounds_count = 0
        if os.path.exists(rounds_path):
            try:
                df = pd.read_csv(rounds_path)
                rounds_count = len(df)
            except Exception:
                pass

        created_at = config_data.get("created_at", "")
        if not created_at and os.path.exists(folder_path):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(folder_path)))

        runs.append({
            "run_folder": folder,
            "run_dir": folder_path.replace("\\", "/"),
            "created_at": created_at,
            "model": config_data.get("model", summary_data.get("model", "unknown")),
            "algorithm": config_data.get("algorithm", summary_data.get("algorithm", "fedavg")).upper(),
            "alpha": config_data.get("alpha", summary_data.get("alpha", -1.0)),
            "is_iid": config_data.get("is_iid", False),
            "mu": config_data.get("mu", summary_data.get("mu", 0.0)),
            "rounds_completed": rounds_count or summary_data.get("completed_rounds", 0),
            "status": summary_data.get("status", "COMPLETED" if summary_data else "IN_PROGRESS"),
            "val_macro_f1": summary_data.get("test_macro_f1", 0.0),
            "val_accuracy": summary_data.get("test_accuracy", 0.0),
            "has_checkpoint": has_ckpt,
            "has_events_log": os.path.exists(events_path),
            "checkpoint_size_mb": ckpt_size_mb,
            "checkpoint_download_url": f"/api/runs/{folder}/checkpoint" if has_ckpt else None,
            "rounds_csv_url": f"/api/runs/{folder}/rounds" if os.path.exists(rounds_path) else None
        })

    return runs


@app.get("/api/runs/{run_folder}/checkpoint")
async def download_run_checkpoint(run_folder: str):
    """Downloads checkpoint.pt from an isolated timestamped run directory."""
    path = os.path.join("results", "runs", run_folder, "checkpoint.pt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Checkpoint file not found in run directory.")
    return FileResponse(path, filename=f"{run_folder}_checkpoint.pt", media_type="application/octet-stream")


@app.get("/api/runs/{run_folder}/rounds")
async def get_run_rounds(run_folder: str):
    """Returns round metrics from the isolated run directory."""
    path = os.path.join("results", "runs", run_folder, "rounds.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Rounds CSV not found in run directory.")
    df = pd.read_csv(path)
    return {
        "run_folder": run_folder,
        "rounds": df.to_dict(orient="records")
    }


@app.get("/api/experiments/active")
async def get_active_experiment():
    """Returns real-time status of the currently executing federated simulation."""
    job_dict = orchestrator.get_active_job_dict()
    if not job_dict:
        return {"active": False}
    return {"active": True, "job": job_dict}


@app.get("/api/experiments/{exp_id}")
async def get_experiment_details(exp_id: str):
    """Returns round-by-round convergence trajectory for a specific experiment or run directory."""
    # Check in results/runs/ first
    run_rounds_path = os.path.join("results", "runs", exp_id, "rounds.csv")
    if os.path.exists(run_rounds_path):
        df = pd.read_csv(run_rounds_path)
        return {
            "experiment_id": exp_id,
            "rounds": df.to_dict(orient="records")
        }

    # Check in results/logs/
    log_path = f"results/logs/{exp_id}_rounds.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        return {
            "experiment_id": exp_id,
            "rounds": df.to_dict(orient="records")
        }

    # Look for match in active job
    active = orchestrator.active_job
    if active and (active.experiment_id == exp_id or active.run_folder == exp_id):
        return {
            "experiment_id": exp_id,
            "rounds": active.history,
            "summary": active.summary
        }
    raise HTTPException(status_code=404, detail=f"Experiment '{exp_id}' not found.")


@app.post("/api/experiments/launch")
async def launch_experiment(config: Dict[str, Any]):
    """Launches a federated learning simulation asynchronously into an isolated run folder."""
    try:
        job = orchestrator.launch_experiment(config)
        return {
            "status": "launched",
            "job_id": job.job_id,
            "experiment_id": job.experiment_id,
            "run_folder": job.run_folder,
            "run_dir": job.run_dir,
            "config": config
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiments/stop")
async def stop_experiment():
    """Stops the active simulation cleanly between rounds."""
    stopped = orchestrator.stop_active_experiment()
    if not stopped:
        raise HTTPException(status_code=400, detail="No active running experiment to stop.")
    return {"status": "stopping_requested"}


@app.get("/api/checkpoints")
async def list_checkpoints():
    """Lists all saved PyTorch checkpoints across baseline checkouts and isolated run directories."""
    ckpts = []

    # 1. Isolated run directories (newest first)
    run_ckpts = sorted(glob.glob("results/runs/*/checkpoint.pt"), reverse=True)
    for f in run_ckpts:
        folder = os.path.basename(os.path.dirname(f))
        size_mb = round(os.path.getsize(f) / (1024 * 1024), 2)
        mtime = os.path.getmtime(f)
        ckpts.append({
            "filename": f"runs/{folder}/checkpoint.pt",
            "source": "isolated_run",
            "run_folder": folder,
            "display_name": f"Run: {folder}",
            "size_mb": size_mb,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
            "download_url": f"/api/runs/{folder}/checkpoint"
        })

    # 2. Baseline checkpoints directory
    ckpt_dir = "results/checkpoints"
    if os.path.exists(ckpt_dir):
        files = sorted(glob.glob(f"{ckpt_dir}/*.pt"))
        for f in files:
            fname = os.path.basename(f)
            size_mb = round(os.path.getsize(f) / (1024 * 1024), 2)
            mtime = os.path.getmtime(f)
            ckpts.append({
                "filename": fname,
                "source": "baseline_checkpoint",
                "run_folder": None,
                "display_name": fname.replace(".pt", ""),
                "size_mb": size_mb,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
                "download_url": f"/api/checkpoints/download/{fname}"
            })

    return ckpts


@app.get("/api/checkpoints/download/{filename:path}")
async def download_checkpoint(filename: str):
    """Downloads model weights file from legacy folder or isolated run directory."""
    if filename.startswith("runs/"):
        path = os.path.join("results", filename)
    else:
        path = os.path.join("results/checkpoints", filename)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Checkpoint file not found.")
    return FileResponse(path, filename=os.path.basename(path), media_type="application/octet-stream")


@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    checkpoint: str = Form("fedprox_mobilenetv3_alpha_0.1_mu_0.01.pt")
):
    """Diagnoses crop disease from uploaded leaf photo or sample asset."""
    image_bytes = None
    if file:
        image_bytes = await file.read()
    elif sample_filename:
        sample_path = os.path.join("assets/samples", sample_filename)
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image provided. Upload a file or select a sample image.")

    try:
        result = predict_leaf_disease(
            image_bytes=image_bytes,
            checkpoint_name=checkpoint,
            top_k=5
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/figures")
async def get_publication_figures():
    """Lists publication-quality scientific charts (Figures 1 to 9)."""
    fig_dir = "results/figures"
    if not os.path.exists(fig_dir):
        return []

    metadata = {
        "figure1_system_architecture.png": ("System Architecture", "High-level overview of FedAgri distributed topology and virtual clients."),
        "figure3_centralized_comparison.png": ("Centralized Baselines", "Comparison of Macro-F1 and inference latency across all four CNN backbones."),
        "figure4_fedavg_convergence.png": ("FedAvg Convergence", "Validation accuracy and loss trajectories under varying Dirichlet heterogeneity."),
        "figure5_noniid_robustness.png": ("Non-IID Robustness Degradation", "Degradation curves across alpha values highlighting backbone resilience."),
        "figure6_fedavg_vs_fedprox_convergence.png": ("FedAvg vs FedProx Convergence", "Comparative convergence stability under severe non-IID data."),
        "figure7_client_consistency.png": ("Client-Level Consistency", "Mean vs worst-client accuracy and standard deviation disparity."),
        "figure8_communication_tradeoff.png": ("Communication Trade-off", "Communication volume (MB) vs classification performance Pareto frontier."),
        "figure9a_cm_centralized.png": ("Confusion Matrix: Centralized", "Full 38-class normalized confusion matrix for centralized baseline."),
        "figure9b_cm_fedavg_severe.png": ("Confusion Matrix: FedAvg", "Confusion matrix under severe non-IID conditions demonstrating client drift."),
        "figure9c_cm_fedprox_severe.png": ("Confusion Matrix: FedProx", "Confusion matrix under FedProx showing mitigation of inter-class confusion.")
    }

    files = sorted(glob.glob(f"{fig_dir}/*.png"))
    figures = []
    for f in files:
        fname = os.path.basename(f)
        title, desc = metadata.get(fname, (fname.replace(".png", "").replace("_", " ").title(), "Scientific visualization figure."))
        figures.append({
            "filename": fname,
            "title": title,
            "description": desc,
            "url": f"/results/figures/{fname}"
        })
    return figures


# --- Real-Time WebSocket Endpoint ---

@app.websocket("/ws/training")
async def websocket_training_stream(websocket: WebSocket):
    """
    WebSocket endpoint streaming live training telemetry, client states,
    and aggregation events to active web clients.
    """
    await websocket.accept()
    queue = asyncio.Queue()
    orchestrator.add_subscriber(queue)

    # Send initial state immediately upon connection
    active_job = orchestrator.get_active_job_dict()
    await websocket.send_json({
        "type": "INITIAL_STATE",
        "timestamp": time.time(),
        "active_job": active_job
    })

    try:
        while True:
            # Wait for event from queue or client message
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        orchestrator.remove_subscriber(queue)
    except Exception:
        orchestrator.remove_subscriber(queue)


# --- Static Files Mounting ---

# 1. Mount leaf sample images
if os.path.exists("assets/samples"):
    app.mount("/assets/samples", StaticFiles(directory="assets/samples"), name="leaf_samples")

# 2. Mount publication scientific figures
if os.path.exists("results/figures"):
    app.mount("/results/figures", StaticFiles(directory="results/figures"), name="figures")

# 3. Mount React built JS/CSS bundle
if os.path.exists("frontend/dist/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="react_assets")
elif os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# 4. Fallback legacy routes
if os.path.exists("styles"):
    app.mount("/styles", StaticFiles(directory="styles"), name="styles")
if os.path.exists("js"):
    app.mount("/js", StaticFiles(directory="js"), name="js")


@app.get("/")
async def serve_index():
    """Serves the FedAgri modern React single-page application."""
    dist_index = "frontend/dist/index.html"
    if os.path.exists(dist_index):
        return FileResponse(dist_index)
    index_path = "index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FedAgri Frontend Loading...</h1>")


@app.get("/{full_path:path}")
async def catch_all_spa(full_path: str):
    """Catch-all route for single-page application routing."""
    # Check if exact file exists in frontend/dist
    dist_file = os.path.join("frontend/dist", full_path)
    if os.path.exists(dist_file) and os.path.isfile(dist_file):
        return FileResponse(dist_file)
    # Check if root index exists
    dist_index = "frontend/dist/index.html"
    if os.path.exists(dist_index):
        return FileResponse(dist_index)
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="Page not found")
