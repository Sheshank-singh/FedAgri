# FedAgri-Robust: Robust Federated Learning for Plant Disease Classification under Non-IID Data

**A Comparative Study of Lightweight and Conventional CNN Architectures**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/FL-Flower-FFD43B.svg)](https://flower.ai/)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-brightgreen.svg)](https://github.com/spMohanty/PlantVillage-Dataset)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Project Motivation

Modern precision agriculture increasingly relies on computer vision models to diagnose crop diseases early and minimize crop yield losses. However, training computer vision models centrally requires aggregating vast amounts of farm photos onto a central cloud server. In agricultural environments:
- **Data Privacy & Ownership**: Farms, growers, and agricultural cooperatives are reluctant to transmit private farm imagery.
- **Network Constraints**: Rural edge devices face bandwidth limitations, making transmitting thousands of high-resolution images costly or impractical.
- **Statistical Heterogeneity (Non-IID)**: Agricultural data across farms is inherently non-identically distributed. One farm in a temperate region may grow mostly Apples and Potatoes with early blight, whereas another farm in a tropical region may grow Tomatoes with curl virus.

**Federated Learning (FL)** addresses this dilemma by enabling edge clients to collaboratively train a shared global model while retaining raw imagery locally on client devices. Only model parameter updates are shared with the central server.

This project investigates how client data heterogeneity affects different CNN architectures, and evaluates whether **FedProx** provides empirical robustness improvements over standard **FedAvg** under severe Non-IID conditions.

---

## 2. Core Research Questions

- **RQ1**: How does increasing Non-IID client heterogeneity (controlled via Dirichlet distribution $\alpha$) affect federated plant disease classification accuracy and convergence?
- **RQ2**: How do different CNN architectures (lightweight vs. conventional deep vs. modern backbones) behave under federated Non-IID training?
- **RQ3**: Does FedProx improve robustness and stability over FedAvg under severe Non-IID conditions ($\alpha = 0.1$)?
- **RQ4**: What is the practical trade-off between classification performance, model parameter size, computational complexity (FLOPs), and communication overhead (MB transmitted)?

---

## 3. Research Hypotheses

- **H1**: Increasing client heterogeneity (decreasing $\alpha$) will negatively affect federated model performance and convergence stability across rounds.
- **H2**: The degradation impact of Non-IID data will differ significantly across CNN architectures due to differences in parameter scale and feature representation capacity.
- **H3**: FedProx will provide improved robustness and client-level performance consistency over FedAvg under severe Non-IID conditions ($\alpha = 0.1$).
- **H4**: Lightweight edge models such as **MobileNetV3** will provide a superior efficiency-to-communication trade-off compared to large models such as ConvNeXt-Tiny and ResNet50.

> *Note*: These hypotheses are formulated for empirical testing and are evaluated rigorously using experimental results.

---

## 4. Dataset: PlantVillage (38 Classes)

The study utilizes the complete, color-image **PlantVillage** dataset, comprising **54,305 images** distributed across **38 distinct healthy and diseased crop classes** spanning Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper Bell, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

### Dataset Splitting & Strict Isolation
- **Training Set (70%)**: 38,013 images used for centralized training and partitioned among simulated federated clients.
- **Validation Set (15%)**: 8,146 images used for hyperparameter evaluation, per-round tracking, and model checkpointing.
- **Test Set (15%)**: 8,146 images strictly isolated from client partitioning, local training, and hyperparameter tuning. It is evaluated only for final reported metrics.
- Fixed random seed: `seed = 42`.

---

## 5. Four Evaluated CNN Architectures

All four models are instantiated using ImageNet-pretrained weights, with the final classification linear layer customized to output 38 class logits:

1. **MobileNetV3-Small**: Lightweight edge baseline utilizing depthwise separable convolutions and squeeze-and-excitation blocks (1.56M parameters, 5.94 MB, 0.12 GFLOPs).
2. **ResNet50**: Conventional deep residual network utilizing bottleneck blocks and residual skip connections (23.59M parameters, 89.97 MB, 8.26 GFLOPs).
3. **EfficientNet-B0**: Parameter- and FLOP-efficient baseline scaled using compound coefficient scaling (4.06M parameters, 15.47 MB, 0.82 GFLOPs).
4. **ConvNeXt-Tiny**: Modernized pure convolutional architecture incorporating 7x7 depthwise convolutions and inverted bottleneck design (27.85M parameters, 106.24 MB, 8.98 GFLOPs).

---

## 6. Federated Learning Framework

### Virtual Client Simulation
5 virtual clients are simulated on a local GPU. Raw images remain strictly private to each virtual client. Each client receives a disjoint partition of the training split.

### FedAvg Aggregation
Standard sample-weighted Federated Averaging:
$$w_{global}^{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^{t+1}$$
where $n_k$ is client $k$'s training sample count, $N = \sum_k n_k$, and $w_k$ are the client's updated parameters.

### FedProx Optimization
In heterogeneous environments, local client models drift apart. FedProx adds a proximal regularization term to the local client loss:
$$\mathcal{L}_{FedProx}(w) = \mathcal{L}_{CE}(w; x, y) + \frac{\mu}{2} \|w - w_{global}\|^2$$
where $\mu$ controls the strength of the constraint restricting local weights from drifting too far from the global broadcast weights $w_{global}$.

### Non-IID Dirichlet Partitioning
Label distribution skew across clients is generated using a symmetric Dirichlet distribution:
$$p_k \sim \text{Dirichlet}(\alpha)$$
- $\alpha = 10.0$: Near-IID, balanced distribution.
- $\alpha = 1.0$: Moderate heterogeneity.
- $\alpha = 0.5$: Strong heterogeneity.
- $\alpha = 0.1$: Severe Non-IID heterogeneity (clients hold only a fraction of classes).
- **Experimental Fairness**: The exact same partition index files are loaded and reused across all four CNN backbones.

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (tested on NVIDIA RTX 4060 Ti 8GB)
- Conda package manager

```bash
# 1. Clone repository
git clone https://github.com/your-username/FedAgri-Robust.git
cd FedAgri-Robust

# 2. Create and activate conda environment
conda create -n fedagri python=3.10 -y
conda activate fedagri

# 3. Install PyTorch with CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install project requirements
pip install -r requirements.txt
```

---

## 8. Dataset Preparation

Execute the dataset preparation script to verify PlantVillage classes, generate reproducible 70/15/15 splits, and output `Table 1`:

```bash
python data/dataset.py
```

Generate and visualize Dirichlet Non-IID client partitions ($\alpha \in \{10.0, 1.0, 0.5, 0.1\}$):

```bash
python federated/partition.py
```

---

## 9. Execution Commands

### Launch the FedAgri Web Platform (Recommended)
Launch the unified interactive web interface and backend server with real-time telemetry:
```bash
python run_platform.py
```
Open **`http://localhost:8000`** in any web browser to:
- Configure and launch federated experiments with 1 click
- Stream real-time client training losses, validation accuracy, and worst-client robustness metrics
- Inspect 13+ benchmark experiments and multi-dimensional Pareto charts
- Run real-time plant disease diagnostic inference on leaf images with agronomic treatment prescriptions
- Explore the 38-class PlantVillage dataset and visualize Dirichlet non-IID label skew

---

### Command Line Interface (CLI) Execution

### Quick Verification Mode (`--quick`)
Test the entire pipeline in under 2 minutes (subsampled data, 3 clients, 3 rounds):
```bash
python experiments/run_all.py --quick
```

### Centralized Baselines
Train each model centrally:
```bash
python experiments/run_centralized.py --model mobilenetv3
python experiments/run_centralized.py --model resnet50
python experiments/run_centralized.py --model efficientnet_b0
python experiments/run_centralized.py --model convnext_tiny
```

### FedAvg Experiments
Run FedAvg under varying heterogeneity levels:
```bash
python experiments/run_fedavg.py --model mobilenetv3 --alpha 10.0
python experiments/run_fedavg.py --model mobilenetv3 --alpha 1.0
python experiments/run_fedavg.py --model mobilenetv3 --alpha 0.5
python experiments/run_fedavg.py --model mobilenetv3 --alpha 0.1
```

### FedProx Experiments
Run FedProx under severe Non-IID ($\alpha = 0.1$):
```bash
python experiments/run_fedprox.py --model mobilenetv3 --alpha 0.1 --mu 0.01
python experiments/run_fedprox.py --model resnet50 --alpha 0.1 --mu 0.01
python experiments/run_fedprox.py --model efficientnet_b0 --alpha 0.1 --mu 0.01
python experiments/run_fedprox.py --model convnext_tiny --alpha 0.1 --mu 0.01
```

### Generate Figures & Tables
Generate publication-quality plots and summary tables:
```bash
python visualization/plots.py
python visualization/confusion_matrix.py
```

---

## 10. Project Structure

```text
FedAgri-Robust/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml                          # Hyperparameter configuration
├── data/
│   ├── raw/PlantVillage-Dataset/            # 38 PlantVillage color class folders
│   ├── processed/                           # classes.json, splits, partitions
│   └── dataset.py                           # Dataset loader & splitter
├── models/
│   ├── mobilenetv3.py                       # MobileNetV3-Small
│   ├── resnet50.py                          # ResNet50
│   ├── efficientnet_b0.py                   # EfficientNet-B0
│   ├── convnext_tiny.py                     # ConvNeXt-Tiny
│   └── model_factory.py                     # Factory & profiling tools
├── training/
│   ├── train_centralized.py                 # Centralized trainer
│   └── local_train.py                       # Client trainer with FedProx term
├── federated/
│   ├── partition.py                         # Dirichlet partitioner & heatmaps
│   ├── client.py                            # Virtual & Flower client classes
│   ├── server.py                            # Federated simulation engine
│   ├── fedavg.py                            # Sample-weighted FedAvg
│   └── fedprox.py                           # FedProx aggregation & manager
├── evaluation/
│   ├── metrics.py                           # Accuracy, F1, client consistency
│   ├── evaluate.py                          # Evaluation loop
│   └── efficiency.py                        # Profiler & communication cost
├── visualization/
│   ├── plots.py                             # Figures 1 to 8 generator
│   └── confusion_matrix.py                  # Confusion matrices (Figure 9)
├── experiments/
│   ├── run_centralized.py                   # Centralized CLI
│   ├── run_fedavg.py                        # FedAvg CLI
│   ├── run_fedprox.py                       # FedProx CLI
│   └── run_all.py                           # Full suite CLI
├── results/
│   ├── figures/                             # Generated publication figures
│   ├── tables/                              # Tables 1 to 6 (CSV & Markdown)
│   ├── logs/                                # Per-round metrics logs
│   ├── checkpoints/                         # Model weights
│   └── research_summary.md                  # Comprehensive research report
└── notebooks/
    └── analysis.ipynb                       # Interactive analysis notebook
```

---

## 11. Privacy Terminology Notice

In this study, clients are simulated virtually on a single workstation. Raw training images remain strictly local to simulated client data loaders during federated training. No formal differential privacy (DP), cryptographic guarantees, or secure multi-party computation are claimed or implemented.

---

## 12. Reproducibility

Every experiment uses fixed random seeds (`seed = 42`). Client partition indices are saved to static JSON files (`partition_alpha_X_seed42.json`) to guarantee identical client distributions across all four CNN backbones.
