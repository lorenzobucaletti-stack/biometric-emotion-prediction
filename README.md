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
