# FedAgri-Robust: Research Summary & Scientific Findings

**Title**: FedAgri-Robust: Robust Federated Learning for Plant Disease Classification under Non-IID Data — A Comparative Study of Lightweight and Conventional CNN Architectures

---

## 1. Executive Summary

This study investigates the interaction between statistical client heterogeneity (Dirichlet label non-IID skew), federated optimization algorithms (**FedAvg** vs. **FedProx**), and four distinct Convolutional Neural Network backbones (**MobileNetV3-Small**, **ResNet50**, **EfficientNet-B0**, **ConvNeXt-Tiny**) for distributed plant disease classification on the 38-class PlantVillage dataset (54,305 images).

All experiments were conducted with rigorous dataset splitting (70% train, 15% validation, 15% isolated test), reproducible random seeds (`seed=42`), identical client partition indices across architectures, and client data isolation.

---

## 2. Core Research Questions & Empirical Answers

### RQ1: How does increasing Non-IID client heterogeneity affect federated plant disease classification?
- **Finding**: As client heterogeneity increases from near-IID ($\alpha = 10.0$) to severe non-IID ($\alpha = 0.1$), global model accuracy and Macro-F1 degrade substantially under FedAvg.
- On MobileNetV3, Macro-F1 remains high under moderate heterogeneity ($\alpha=10.0 \to 99.68\%$, $\alpha=1.0 \to 99.42\%$, $\alpha=0.5 \to 99.21\%$), but degrades sharply to **92.98%** under severe heterogeneity ($\alpha=0.1$).
- More critically, client-level performance consistency deteriorates: under $\alpha=0.1$, the worst-client accuracy drops to **84.16%** and the standard deviation across clients increases by two orders of magnitude ($0.0006 \to 0.0583$).

### RQ2: How do different CNN architectures behave under federated Non-IID training?
- **Finding**: Backbones exhibit markedly different vulnerability to client drift under severe non-IID conditions ($\alpha=0.1$):
  1. **ResNet50** suffers catastrophic degradation under FedAvg (Macro-F1 drops to **77.05%**, and worst-client accuracy plummets to **31.76%** with client accuracy std = 0.2391).
  2. **EfficientNet-B0** achieves **87.16%** Macro-F1 (worst client = 63.26%, std = 0.1338).
  3. **ConvNeXt-Tiny** achieves **88.63%** Macro-F1 (worst client = 78.15%, std = 0.0810).
  4. **MobileNetV3** exhibits the highest intrinsic resilience under vanilla FedAvg, retaining **92.98%** Macro-F1 (worst client = 84.16%, std = 0.0583).
- Deeper conventional networks with larger parameter footprints experience severe client drift because local over-fitting to skewed label distributions causes divergence in gradient trajectories that cannot be reconciled by simple parameter averaging.

### RQ3: Does FedProx improve robustness over FedAvg under severe Non-IID conditions?
- **Finding**: **Yes, emphatically.** Incorporating the proximal regularization term ($\frac{\mu}{2}\|w - w_{global}\|^2$) with $\mu = 0.01$ stabilizes local client updates and dramatically mitigates client drift across all four architectures:
  - **ResNet50**: Macro-F1 surges from **77.05% to 96.06% (+19.01% gain)**, and worst-client accuracy surges from **31.76% to 81.59% (+49.83% gain)**.
  - **EfficientNet-B0**: Macro-F1 improves from **87.16% to 96.73% (+9.57% gain)**, and worst-client accuracy improves from **63.26% to 91.44% (+28.18% gain)**.
  - **ConvNeXt-Tiny**: Macro-F1 improves from **88.63% to 95.20% (+6.57% gain)**, and worst-client accuracy improves from **78.15% to 91.67% (+13.52% gain)**.
  - **MobileNetV3**: Macro-F1 improves from **92.98% to 95.36% (+2.38% gain)**, and worst-client accuracy improves from **84.16% to 90.85% (+6.69% gain)**.
- FedProx also reduces client performance disparity (std across clients drops from 0.2391 to 0.0681 for ResNet50, and from 0.1338 to 0.0302 for EfficientNet-B0).

### RQ4: What is the trade-off between classification performance, model size, computational complexity, and communication cost?
- **Finding**:
  - **MobileNetV3** requires only **1.56M parameters (5.94 MB)** and **1,188 MB (1.19 GB)** total communication over 20 rounds, while achieving **95.36% FedProx Macro-F1** with **3.4 ms** inference latency.
  - **ConvNeXt-Tiny** requires **27.85M parameters (106.24 MB)** and **21,248 MB (21.25 GB)** total communication (an **18-fold increase** in communication overhead) for a comparable **95.20% Macro-F1**.
  - **ResNet50** requires **23.59M parameters (89.97 MB)** and **17,994 MB (17.99 GB)** total communication for **96.06% Macro-F1**.
  - **EfficientNet-B0** represents an outstanding middle ground: **4.06M parameters (15.47 MB)**, **3.09 GB** total communication, and the highest overall FedProx Macro-F1 (**96.73%**).

---

## 3. Formal Hypothesis Evaluation

| Hypothesis | Description | Empirical Result | Evidence |
| :--- | :--- | :---: | :--- |
| **H1** | Increasing client heterogeneity negatively affects performance/convergence. | **SUPPORTED** | MobileNetV3 Macro-F1 drops monotonically from 99.68% ($\alpha=10$) $\to$ 99.42% ($\alpha=1.0$) $\to$ 99.21% ($\alpha=0.5$) $\to$ 92.98% ($\alpha=0.1$). Worst-client accuracy drops from 99.81% to 84.16%. |
| **H2** | Impact of Non-IID data differs across CNN architectures. | **SUPPORTED** | Under FedAvg ($\alpha=0.1$), ResNet50 degraded by 22.7% (Macro-F1 77.05%), EfficientNet-B0 by 12.6% (87.16%), ConvNeXt-Tiny by 11.0% (88.63%), while MobileNetV3 degraded by only 6.6% (92.98%). |
| **H3** | FedProx provides improved robustness over FedAvg under severe Non-IID. | **SUPPORTED** | FedProx ($\mu=0.01$) improved Macro-F1 by +19.01% on ResNet50, +9.57% on EfficientNet-B0, +6.57% on ConvNeXt-Tiny, and +2.38% on MobileNetV3, with massive gains in worst-client performance. |
| **H4** | Lightweight models provide superior efficiency/communication trade-offs. | **SUPPORTED** | MobileNetV3 achieves 95.36% F1 with only 1.19 GB communication and 3.4 ms latency, saving ~94.4% network traffic compared to ConvNeXt-Tiny (21.25 GB). EfficientNet-B0 provides the highest accuracy (96.73%) at only 3.09 GB. |

---

## 4. Key Experimental Tables

### Table 2: Model Architecture Profiles
| Model | Params (M) | Size (MB) | GFLOPs | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| MobileNetV3 | 1.56 | 5.94 | 0.12 | 3.40 |
| EfficientNet-B0 | 4.06 | 15.47 | 0.82 | 6.93 |
| ResNet50 | 23.59 | 89.97 | 8.26 | 5.89 |
| ConvNeXt-Tiny | 27.85 | 106.24 | 8.98 | 4.12 |

### Table 3: Centralized Baseline Performance (38 Classes)
| Model | Test Accuracy | Test Macro-F1 | Training Time (s) |
| :--- | :---: | :---: | :---: |
| MobileNetV3 | 99.71% | 99.53% | 574.4s |
| ResNet50 | 99.77% | 99.67% | 1346.9s |
| EfficientNet-B0 | 99.80% | 99.71% | 1259.7s |
| ConvNeXt-Tiny | 99.73% | 99.59% | 23055.6s |

### Table 5: FedAvg vs. FedProx under Severe Non-IID ($\alpha = 0.1$)
| Model | FedAvg F1 | FedProx F1 ($\mu=0.01$) | $\Delta$ F1 | FedAvg Worst Client | FedProx Worst Client | $\Delta$ Worst Client |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** | 92.98% | **95.36%** | **+2.38%** | 84.16% | **90.85%** | **+6.69%** |
| **ResNet50** | 77.05% | **96.06%** | **+19.01%** | 31.76% | **81.59%** | **+49.83%** |
| **EfficientNet-B0** | 87.16% | **96.73%** | **+9.57%** | 63.26% | **91.44%** | **+28.18%** |
| **ConvNeXt-Tiny** | 88.63% | **95.20%** | **+6.57%** | 78.15% | **91.67%** | **+13.52%** |

### Table 6: Communication Volume & Edge Complexity (20 Rounds, 5 Clients)
| Model | Total Communication | Upload Vol | Download Vol | GFLOPs | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3** | **1,188 MB (1.19 GB)** | 594 MB | 594 MB | **0.12** | **3.40 ms** |
| **EfficientNet-B0** | **3,094 MB (3.09 GB)** | 1,547 MB | 1,547 MB | **0.82** | 6.93 ms |
| **ResNet50** | **17,994 MB (17.99 GB)** | 8,997 MB | 8,997 MB | 8.26 | 5.89 ms |
| **ConvNeXt-Tiny** | **21,248 MB (21.25 GB)** | 10,624 MB | 10,624 MB | 8.98 | 4.12 ms |

---

## 5. Limitations & Future Work

### Limitations
1. **Virtual Simulation**: Virtual clients were simulated on a single GPU workstation rather than physical edge hardware (Raspberry Pi, Jetson).
2. **Network Dynamics**: Fixed communication model without packet loss, latency jitter, or straggler effects.
3. **Single Domain**: Experiments focused exclusively on PlantVillage laboratory conditions; in-field background clutter and variable lighting were not evaluated.

### Future Work
1. **In-field Domain Adaptation**: Extending FedAgri-Robust to real-world field datasets (e.g., PlantDoc) with noisy backgrounds.
2. **Communication Compression**: Implementing gradient quantization (e.g., INT8/FP16) or top-$k$ sparsification to further lower transmission overhead for large models.
3. **Asynchronous & Heterogeneous Compute**: Evaluating clients with varying local epoch capabilities under FedProx.
