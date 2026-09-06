# BODMAS Continual Learning Malware Detection System

An enterprise-grade, real-time PE (Portable Executable) malware detection and continual learning pipeline. This system monitors file downloads, extracts 2,381 static PE features, performs sub-millisecond neural network inference via an optimized C++ **ONNX Runtime engine** (with PyTorch fallback), captures low-confidence feedback, safely retrains aspirant candidates, subjects them to the **Dragon Gate (龙门 - Longmen)** trials, and executes zero-downtime atomic promotion.

---

## Architecture Overview

```text
                                  [ Incoming File Download ]
                                               │
                                               ▼
                               ┌──────────────────────────────────┐
                               │     pe_download_watchdog.py      │
                               │  - Sub-ms MZ header check        │
                               │  - Async worker queue            │
                               └────────────────┬─────────────────┘
                                                │ (Valid PE)
                                                ▼
                               ┌──────────────────────────────────┐
                               │     extraction/pe_extractor.py   │
                               │  - 2,381 raw static features     │
                               │  - 2,332 non-constant mask       │
                               │  - Top-250 feature selection     │
                               └────────────────┬─────────────────┘
                                                │
                                                ▼
                               ┌──────────────────────────────────┐
                               │     inference/predictor.py       │
                               │  - ONNX Runtime / PyTorch Engine │
                               │  - Latency: < 0.8 ms / sample    │
                               │  - Threshold: 0.55               │
                               └───────┬──────────────────┬───────┘
                                       │                  │
                 High-Confidence Result│                  │ Low-Confidence / Feedback
                                       ▼                  ▼
                            [ Safe / Malware Alert ]  [ data/feedback/pending/ ]
                                                          │
                                                          ▼ (Manual / Sandbox)
                                                     [ verify_sample.py ]
                                                          │
                                                          ▼
                                                     [ data/feedback/verified/ ]
                                                          │ (Threshold reached)
                                                          ▼
                                                     [ retrain.py ]
                                                          │
                                                          ▼ (Aspirant Candidate Born)
                                                     [ models/v{N}_Candidate/ ]
                                                          │
                                                          ▼
                                                     [ evaluate.py ] (🐉 Dragon Gate Trial)
                                                          │ (Conquered 4 Strict Trials)
                                                          ▼
                                                     [ promote.py ] (🐉 Evolve to Dragon)
                                                          │ (Light-Themed Modal Approval)
                                                          ▼
                                                     [ inference/v{N}_Promoted/ ]
                                                     [ ACTIVE_MODEL.json (Atomic) ]
```

---

## Key Components

### 1. Static Feature Extraction (`extraction/pe_extractor.py`)
- Extracts **2,381 raw static features** across PE headers, sections, imports, exports, resources, entropy distributions, and byte histograms using `LIEF` and `pefile`.
- Applies the production **2,332 non-constant feature mask**, **StandardScaler normalization**, and **Top-250 feature selection** matrices.

### 2. Real-Time Download Watchdog (`pe_download_watchdog.py`)
- Background filesystem watcher tracking the user's active Windows `Downloads` folder.
- Uses sub-millisecond magic byte probes (`< 0.05ms`) to filter non-PE files before reading.
- Handles browser lock cycles (`.crdownload`, `.part`, antivirus lock releases) with retry backoffs and asynchronous worker queues.

### 3. Production Inference Engine (`inference/predictor.py`)
- High-performance C++ **ONNX Runtime engine** (`malware_mlp_top250.onnx`) with sub-millisecond latency and automatic PyTorch fallback.
- Dynamically resolves active model weights via `inference/ACTIVE_MODEL.json`.
- Automatically routes ambiguous predictions to the feedback loop for review.

### 4. Continual Learning & The Dragon Gate (`training/`)
- **`feedback_collector.py`**: Serializes ambiguous or misclassified PE samples with full metadata and feature vectors.
- **`verify_sample.py`**: Interactive CLI tool to label pending feedback items (`0 = Benign`, `1 = Malware`).
- **`retrain.py`**: Fine-tunes weights from verified feedback using experience replay and births an aspirant candidate in `models/v{N}_Candidate/`.
- **`evaluate.py`**: **The Dragon Gate Trial (龙门 - Longmen)**: Rigorously challenges the aspirant candidate across 4 trials against the reigning Celestial Dragon. Writes `DRAGON_GATE_ASCENDED.json` or `DRAGON_GATE_FALLEN.json`.
- **`promote.py`**: **Dragon Gate Ascension**: Empowers the ascended candidate to evolve into the reigning Celestial Dragon (`inference/v{N}_Promoted/`) with zero-downtime atomic pointer updates (`ACTIVE_MODEL.json`). Includes light-themed modal approval and candidate audit tables (`--list`).

---

## 🐉 The 4 Trials of the Dragon Gate (龙门)

According to ancient lore, an aspirant serpent or carp swimming up the Yellow River must leap over the roaring **Dragon Gate** to transform into a Celestial Dragon. In this pipeline, an aspirant candidate model must conquer all 4 trials without regression before it can evolve:

| Trial | Mathematical Criterion | Strict Threshold | Gate Objective |
| :--- | :--- | :--- | :--- |
| **Trial 1: Recall** | `Recall(Aspirant) ≥ Recall(Dragon)` | Non-regression (≥ 0%) | Zero regression on malware detection coverage |
| **Trial 2: Precision** | `Precision(Aspirant) ≥ Precision(Dragon)` | Non-regression (≥ 0%) | Zero increase in false alarms on benign files |
| **Trial 3: F1-Score** | `F1-Score(Aspirant) > F1-Score(Dragon)` | Strict Improvement (>) | Candidate must provably outperform the active model |
| **Trial 4: PR-AUC** | `PR-AUC(Aspirant) ≥ PR-AUC(Dragon)` | Non-regression (≥ 0%) | Precision-Recall stability across all threshold bands |

> **🐉 Dragon Gate Rule of Law:** An aspirant candidate must conquer **all 4 trials simultaneously** to ascend. If even a single trial fails, the model remains in the mortal realm (`models/`) and cannot displace the reigning Celestial Dragon.

---

## Directory Structure

```text
.
├── .gitignore                       # Git ignore rules (excludes weights, bytecodes, artifacts)
├── README.md                        # Project documentation
├── requirements.txt                 # Project dependencies
├── pe_download_watchdog.py          # Real-time PE download monitor
├── extraction/
│   ├── pe_extractor.py              # 2,381-dimension PE static extractor
├── inference/
│   ├── ACTIVE_MODEL.json            # Atomic pointer to active production model
│   ├── predictor.py                 # Fast ONNX Runtime / PyTorch inference engine
│   └── v2_Promoted/                 # Current active production version (Reigning Dragon)
│       ├── malware_mlp_top250.onnx  # Optimized C++ ONNX Runtime graph
│       ├── malware_mlp_top250.pth   # PyTorch neural network weights
│       ├── scaler.pkl               # Fitted StandardScaler
│       ├── feature_mask.pkl         # 2,332 non-constant feature mask
│       ├── top250_positions.pkl     # Top 250 feature indices
│       ├── model_config.json        # Architecture & threshold config
│       └── version.json             # Version metadata
├── models/                          # Staged candidate model directories
│   ├── v1_Artifact/                 # Initial model release artifacts
│   └── v2_Artifact/                 # Retrained model release artifacts
├── data/
│   └── feedback/
│       ├── pending/                 # Unverified incoming feedback (.npz)
│       └── verified/                # Labeled samples ready for retraining (.npz)
└── training/
    ├── feedback_collector.py        # Feedback ingestion & persistence
    ├── verify_sample.py             # Feedback labeling & management utility
    ├── retrain.py                   # Continual fine-tuning trainer
    ├── evaluate.py                  # Benchmark & Dragon Gate evaluator
    └── promote.py                   # Dragon Gate Ascension & promotion script
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (optional, CPU fallback supported)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Start the Real-Time Downloads Watchdog
```bash
python pe_download_watchdog.py
```
*Monitors the Downloads directory in real time, extracts static features, and executes sub-millisecond ONNX Runtime inferences.*

### 2. Verify Feedback Samples
```bash
# Check status of feedback pool
python training/verify_sample.py --status

# Interactively verify pending samples
python training/verify_sample.py --interactive
```

### 3. Retrain & Challenge the Dragon Gate
```bash
# Standard Retraining (Trains candidate -> Dragon Gate Trial -> UI Modal)
python training/retrain.py

# Headless / Terminal only (skips GUI popup, uses CLI prompt)
python training/retrain.py --no-ui

# Non-interactive CI/CD (automatically promotes if gate passes without prompt)
python training/retrain.py --yes
```

### 4. Dragon Gate Candidate Audit & Manual Promotion
```bash
# Audit all candidate models in models/
python training/promote.py --list

# Manually challenge the Dragon Gate for a specific candidate
python training/evaluate.py --candidate 2

# Manually promote / evolve an ascended candidate (with light-themed UI dialog)
python training/promote.py --candidate 2
```

---

## Best Practices & Safety Notes

- **Atomic Pointer Swaps**: `ACTIVE_MODEL.json` uses atomic file replacement (`os.replace`) to guarantee zero inference downtime during promotions.
- **Experience Replay**: Verified feedback samples are mixed with baseline historical datasets to prevent catastrophic forgetting.
- **Strict Dragon Gate Gatekeeping**: Candidates that fail any of the 4 benchmark trials remain in the mortal realm as candidate artifacts and will never displace the active production Dragon.
