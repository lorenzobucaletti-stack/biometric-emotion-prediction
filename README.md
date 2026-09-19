# Multimodal Biometric Emotion Recognition via Deep 1D-CNN and Explainable AI (SHAP)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep Learning framework for binary emotion recognition (positive vs. negative valence) based on psychophysiological biosignals from the POPANE benchmark dataset (1,157 subjects, 150 GB raw records). The project investigates the transition from a unimodal cardiac baseline (ECG) to multimodal architectures (ECG, Electrodermal Activity - EDA, and Affective features) through Early and Late Fusion, backed by model interpretability using SHAP (SHapley Additive exPlanations).

---

## Technical Highlights

- **Data Pipeline & Memory Optimization:** Implemented custom chunk-based lazy loading to bypass the 150 GB raw dataset memory barrier, windowing signals into uniform 1000-sample segments (1-second temporal resolution at 1000 Hz).
- **Subject-Wise Splitting:** Strict subject-independent partitioning (70% Train, 15% Validation, 15% Test) to eliminate data leakage and ensure true generalized physiological representations rather than patient-specific signatures.
- **Physical Undersampling vs. Loss Weighting:** Mitigated majority class bias (originally 14,400 positive vs. 4,500 negative instances) through physical undersampling, preventing neural networks from exploiting mathematical shortcuts ("lazy learning").
- **Independent Channel Normalization:** Per-channel Z-score standardization ($(\text{signal} - \mu) / \sigma$) to reconcile heterogeneous physical units (microvolts from ECG vs. microsiemens from EDA).
- **Explainable AI (XAI):** GradientExplainer-based SHAP analysis at the millisecond level, validating morphological feature extraction (QRS complexes) and identifying root causes of predictive failure in high-arousal scenarios.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MULTIMODAL PIPELINE                                    │
└────────────────────────────────────────────────────────────────────────────────────────┘

    [Raw CSV Files] ──> [Lazy Loading Index] ──> [Subject-Wise Split] ──> [Z-Score Norm]
                                                                                │
        ┌───────────────────────────────────────────────────────────────────────┴────────┐
        ▼                                                                               ▼
 ┌──────────────┐                                                                ┌──────────────┐
 │ EARLY FUSION │                                                                │ LATE FUSION  │
 └──────┬───────┘                                                                └──────┬───────┘
        │ Input-Level Concatenation (3, 1000)                                           │ 3 Independent Branches (1, 1000)
        ▼                                                                               ▼
 ┌──────────────┐                                                                ┌──────────────┐
 │ 3-Ch 1D-CNN  │ [Conv1d(3->18) -> BN -> ReLU -> MaxPool] x2                    │ Unimodal CNN │ x3 (Affect, ECG, EDA)
 └──────┬───────┘                                                                └──────┬───────┘
        ▼                                                                               ▼
 ┌──────────────┐                                                                ┌──────────────┐
 │ Dense Head   │ Linear(Flatten, 64) -> Dropout(0.4) -> Linear(64, 1)           │ Majority Vote│ Decision rule: sum(votes) >= 2
 └──────────────┘                                                                └──────────────┘
```

## Experimental Phases & Methodology

### Phase 1: Unimodal Cardiac Baseline (ECG)
- **Architecture:** 1D-CNN using a kernel size of 7, Batch Normalization, and adaptive pooling layers.
- **Iterative Refinement & Imbalance Handling:**
  - *Iteration 1 (Unbalanced Baseline):* Reached a nominal Accuracy of 78.11%, but exhibited severe majority class bias with a Class 0 Recall of only 0.11 (detecting only 480 True Negatives out of 4,500).
  - *Iteration 2–4 (Weighted BCE & Regularization):* Introduced class-weighted loss and L2 regularization to improve minority class sensitivity; however, loss-based checkpointing caused the network to over-predict the negative class to artificially minimize penalties.
  - *Iteration 5 (Definitive Baseline):* Implemented physical random undersampling combined with Macro F1-Score checkpointing, achieving an unbiased baseline with **62.73% Accuracy**, **0.63 Macro F1-Score**, and **0.6769 AUC-ROC**.
- **SHAP Interpretability:** Confirmed that the model correctly leverages QRS complex morphology for confident predictions, but fails to distinguish emotions with identical high arousal (e.g., threat vs. excitement) due to similar tachycardia profiles.

### Phase 2: Multimodal Integration (Early vs. Late Fusion)
- **Early Fusion (Input-Level):**
  - Concatenates Affect, ECG, and EDA into a 3-channel input tensor `(3, 1000)`.
  - Captures cross-modal temporal dependencies directly within the initial convolutional layers.
  - Achieved the highest performance across all configurations: **0.6630 Macro F1-Score**, **66.40% Accuracy**, and **0.7300 Class 0 Recall**.
- **Late Fusion (Decision-Level):**
  - Evaluated three parallel `UnimodalCNN` branches mediated by majority voting.
  - Suffered initial predictive collapse (24% accuracy) due to voting rigidity and learning rate conflicts.
  - Stabilized through independent per-branch optimizers and undersampling, converging to a **0.6300 Macro F1-Score**.

### Phase 3: Explainable AI & Feature Synergy (SHAP)
- **ECG as Temporal Anchor:** Provides localized, fast-acting morphological spikes aligned with heartbeats.
- **EDA as State Validator:** Supplies broader, tonic skin conductance attributions that resolve high-arousal ambiguities and eliminate false positives.
- **Affect Features:** Establishes a steady baseline probability profile across continuous recordings.

---

## Performance Benchmark

| Paradigm / Architecture | Input Dimension | Optimization & Strategy | Accuracy | Macro F1-Score | AUC-ROC | Class 0 Recall (Negative) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ECG Baseline (Unbalanced)** | $(1, 1000)$ | Standard BCE / Random Init | 78.11% | 0.5309 | 0.6980 | 0.1100 |
| **ECG Weighted Loss** | $(1, 1000)$ | Weighted BCE + BatchNorm | 70.07% | 0.6100 | 0.6600 | 0.4700 |
| **ECG Definitive Baseline** | $(1, 1000)$ | Undersampling + F1 Checkpoint | 62.73% | **0.6300** | 0.6769 | 0.6200 |
| **Late Fusion (Tribunal)** | $3 \times (1, 1000)$ | Undersampling + Majority Vote | 63.00% | **0.6300** | — | 0.6800 |
| **Early Fusion (Best Model)** | $(3, 1000)$ | Undersampling + LR Scheduler | **66.40%** | **0.6630** | **0.7291** | **0.7300** |

'''text
├── docs/
│   └── Report_ML.pdf                   # Complete scientific paper and technical report
├── src/
│   ├── common/                         # Shared utilities
│   │   ├── config.py                   # Master path definitions and hyperparameters
│   │   ├── create_index.py             # Raw data scanning and physical undersampling
│   │   ├── split_data.py               # Subject-wise 70/15/15 partitioner
│   │   └── run_total.py                # End-to-end pipeline orchestrator
│   ├── Fase1/                          # Phase 1: Unimodal ECG
│   │   ├── config_FASE1.py             # Hyperparameters & class mapping
│   │   ├── dataset_FASE1.py            # Lazy loading & per-window Z-score
│   │   ├── model_FASE1.py              # 1D-CNN PyTorch architecture
│   │   ├── train_FASE1.py              # Training loop with Macro-F1 checkpointing
│   │   ├── evaluate.py                 # Metric evaluation and confusion matrix
│   │   └── run_script_FASE1.py         # Phase 1 pipeline runner
│   ├── Fase2/                          # Phase 2: Multimodal Fusion
│   │   ├── early_fusion/               # 3-channel 1D-CNN (Affect, ECG, EDA)
│   │   │   ├── config_FASE2_early.py
│   │   │   ├── dataset_FASE2_early.py
│   │   │   ├── model_FASE2_early.py
│   │   │   ├── train_FASE2_early.py
│   │   │   └── test_FASE2_early.py
│   │   └── late_fusion/                # Parallel unimodal branches + Majority voting
│   │       ├── config_FASE2_late.py
│   │       ├── dataset_FASE2_late.py
│   │       ├── model_FASE2_late.py
│   │       ├── train_FASE2_late.py
│   │       └── test_FASE2_late.py
│   └── Fase3/                          # Phase 3: Explainability (SHAP)
│       ├── shap_fase1.py               # Single-lead ECG visual explanations
│       ├── shap_fase2_early.py         # 3-channel multimodal SHAP overlay
│       ├── shap_fase2_late.py          # Independent multi-judge SHAP attribution
│       ├── plots/                      # Exported visual explanations (Phase 1)
│       ├── plots_early_fusion/         # Exported visual explanations (Early Fusion)
│       └── plots_late_fusion/          # Exported visual explanations (Late Fusion)
└── README.md
