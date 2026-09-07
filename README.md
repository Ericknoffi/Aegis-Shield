# BODMAS // CONTINUAL MALWARE DEFENSE PIPELINE
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-101010?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Engine-ONNX](https://img.shields.io/badge/Engine-ONNX_Runtime_C++-101010?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Inference-Latency](https://img.shields.io/badge/Latency-<0.8ms-101010?style=flat-square&logo=speedtest&logoColor=00FF66)]()
[![Validation-Dragon_Gate](https://img.shields.io/badge/Protocol-龙门_Dragon_Gate-101010?style=flat-square&logo=matrix&logoColor=orange)]()
[![Safety-Atomic_Swap](https://img.shields.io/badge/Deployment-Zero--Downtime_Atomic-101010?style=flat-square)]()

An enterprise-grade, real-time PE malware detection and continual learning pipeline. Streams file system events, extracts 2,381 static PE features, executes sub-millisecond inference via an optimized **C++ ONNX Runtime engine** (with PyTorch fallback), captures low-confidence telemetry into feedback pools, safely retrains aspirant candidate models, tests them against the **Dragon Gate (龙门)** trial protocol, and deploys zero-downtime atomic pointer swaps.

```
[ SYSTEM TELEMETRY & SPECS ]
├─ Core Engine   : C++ ONNX Runtime (v1.16+) / PyTorch Fallback
├─ Feature Plane : 2,381-D Raw -> 2,332 Non-Const -> Top-250 Selected (LIEF + pefile)
├─ Latency       : ~0.74 ms / binary (Inference: 0.12ms | Probe: 0.04ms)
├─ Classification: Binary Threshold @ 0.55 [Benign: 0 | Malicious: 1]
├─ Safety Layer  : Atomic pointer swaps (`ACTIVE_MODEL.json`) + Experience Replay
└─ Gatekeeper    : 4-Trial Dragon Gate (龙门) Zero-Regression Protocol
```

---

## ⚡ Architecture Flow

```
[ Inbound Download ] ──> [ pe_download_watchdog.py ] (MZ Magic Probe < 0.05ms)
                               │ (Valid PE)
                               ▼
                        [ extraction/pe_extractor.py ] (2,381-D -> Top-250)
                               │
                               ▼
                        [ inference/predictor.py ] (ONNX Engine < 0.8ms)
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
     [ High Confidence ]             [ Low Confidence / Ambiguous ]
     (Alert / Allow)                          │
                                              ▼
                                     [ data/feedback/pending/ ]
                                     ├── [ benign/ ]  (Pred: Benign)
                                     └── [ malware/ ] (Pred: Malware)
                                              │
                                              ▼ (training/verify_sample.py <id> <0|1>)
                                     [ data/feedback/verified/ ]
                                     ├── [ benign/ ]  (True: 0)
                                     └── [ malware/ ] (True: 1)
                                              │
                                              ▼ (training/retrain.py)
                                     [ models/v{N}_Candidate/ ]
                                              │
                                              ▼ (training/evaluate.py)
                                     [ 🐉 DRAGON GATE TRIALS (龙门) ]
                                              │ (Pass: Recall + Prec + F1 + PR-AUC)
                                              ▼ (training/promote.py)
                                     [ inference/v{N}_Promoted/ ]
                                     [ ACTIVE_MODEL.json (Atomic) ]
```

---

## 🛠️ System Modules

| Component | Target File | Role & Performance Profile |
| :--- | :--- | :--- |
| **Download Watchdog** | `pe_download_watchdog.py` | Real-time filesystem observer. Filters non-PEs via `<0.05ms` MZ magic probe. Resolves `.crdownload`/`.part` lock contention via async backoff queues. |
| **Static Extractor** | `extraction/pe_extractor.py` | Generates 2,381 raw dimensions (PE headers, section entropy distributions, byte histograms, imports/exports). Applies 2,332 mask & Top-250 matrix. |
| **Inference Engine** | `inference/predictor.py` | Production ONNX Runtime engine (`malware_mlp_top250.onnx`). Sub-millisecond execution. Dynamic active-version resolution with PyTorch fallback. |
| **Feedback Ingestion** | `training/feedback_collector.py` | Serializes ambiguous samples (`0.40 ≤ P(malware) ≤ 0.70`) into compressed `.npz` feature bundles with contextual metadata. |
| **Sample Verifier** | `training/verify_sample.py` | CLI triage tool for sandboxed / manual labeling (`0 = Benign`, `1 = Malware`). Moves samples from `pending/` to `verified/`. |
| **Continual Retrainer**| `training/retrain.py` | Executes experience replay fine-tuning (blending verified feedback with historical baselines) and births candidate artifacts in `models/v{N}_Candidate/`. |
| **Gate Evaluator** | `training/evaluate.py` | **Dragon Gate Trial Engine**. Strict benchmarking against reigning production model. Writes `DRAGON_GATE_ASCENDED.json` or `DRAGON_GATE_FALLEN.json`. |
| **Promotion Core** | `training/promote.py` | Executes atomic switchover into `inference/v{N}_Promoted/`, updates `ACTIVE_MODEL.json` with zero-downtime, and provides UI approval modals. |

---

## 🐉 The Dragon Gate (龙门) Verification Protocol

A candidate model cannot displace the reigning production Dragon without conquering all 4 mathematical gates simultaneously:

```
┌───────┬───────────────────┬───────────────────────────────────────┬───────────────────────────────┐
│ TRIAL │ METRIC            │ MATHEMATICAL CRITERION                │ SYSTEM OBJECTIVE              │
├───────┼───────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│   1   │ Malware Recall    │ Recall(Aspirant)    ≥ Recall(Dragon)  │ Zero detection regression     │
│   2   │ Precision         │ Precision(Aspirant) ≥ Precision(Dragon)│ Zero false-positive inflation │
│   3   │ F1-Score          │ F1-Score(Aspirant)  > F1-Score(Dragon)│ Strict net intelligence gain  │
│   4   │ PR-AUC            │ PR-AUC(Aspirant)    ≥ PR-AUC(Dragon)  │ Cross-threshold stability     │
└───────┴───────────────────┴───────────────────────────────────────┴───────────────────────────────┘
```

> [!IMPORTANT]
> **Zero-Regression Rule**: If an aspirant candidate fails even 1 of the 4 trials, promotion is aborted immediately. The artifact is retained in `models/v{N}_Candidate/` for diagnostic review and will **never** touch production inference paths.

---

## 📂 Repository Matrix

```text
.
├── pe_download_watchdog.py          # Real-time PE download observer & inference hook
├── requirements.txt                 # Pinned dependencies (ONNX Runtime, PyTorch, LIEF, etc.)
├── extraction/
│   └── pe_extractor.py              # 2,381-D PE static feature extraction pipeline
├── inference/
│   ├── ACTIVE_MODEL.json            # Dynamic atomic pointer to active model version
│   ├── predictor.py                 # Low-latency ONNX Runtime / PyTorch inference engine
│   └── v2_Promoted/                 # Production Reigning Dragon (Active)
│       ├── malware_mlp_top250.onnx  # Optimized C++ ONNX graph
│       ├── malware_mlp_top250.pth   # PyTorch checkpoint weights
│       ├── scaler.pkl               # Fitted StandardScaler transform
│       ├── feature_mask.pkl         # 2,332 non-constant feature index mask
│       ├── top250_positions.pkl     # Top 250 feature selection index array
│       ├── model_config.json        # Hyperparameters and threshold bounds
│       └── version.json             # Build metadata & lineage tracking
├── models/                          # Staged candidate models & evaluation results
│   ├── v1_Artifact/                 # Baseline version release bundle
│   └── v2_Artifact/                 # Evaluated candidate version bundle
├── data/
│   └── feedback/
│       ├── pending/                 # Incoming feedback categorized by model prediction
│       │   ├── benign/              # Predicted Benign (.npz)
│       │   └── malware/             # Predicted Malware (.npz)
│       └── verified/                # Labeled ground truth staged for retraining
│           ├── benign/              # Confirmed Benign (0) (.npz)
│           └── malware/             # Confirmed Malware (1) (.npz)
└── training/
    ├── feedback_collector.py        # Ambiguity detection & sample serialization
    ├── two_step_verifier.py         # Authenticode & VirusTotal two-step verification engine
    ├── verify_sample.py             # Single-sample CLI labeling utility
    ├── batch_verify.py              # Bulk / Automated batch verification & dashboard
    ├── retrain.py                   # Continual fine-tuner with experience replay
    ├── evaluate.py                  # Dragon Gate benchmark evaluator
    └── promote.py                   # Atomic promotion engine with modal UI dialogs
```

---

## 🚀 Command Control Center

### 1. Installation
```bash
# Clone and enter workspace
cd "BODMAS"

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Live Monitoring & Inference
```bash
# Launch background downloads directory watcher
python pe_download_watchdog.py
```

### 3. Feedback Triage & Verification
```bash
# Display feedback dashboard (pending vs verified breakdown)
python training/batch_verify.py --status

# 🛡️ Two-Step Keyless Verification (Step 1: Authenticode -> Step 2: Windows Defender CLI)
python training/batch_verify.py --two-step

# Optional: Run with cloud VirusTotal API fallback
python training/batch_verify.py --two-step --vt-key <VIRUSTOTAL_API_KEY>

# Bulk-verify by predicted category (benign -> 0, malware -> 1)
python training/batch_verify.py --auto-predicted
python training/batch_verify.py --category benign --label 0
python training/batch_verify.py --category malware --label 1

# Single-sample manual verification (0 = Benign, 1 = Malware)
python training/verify_sample.py <sample_id> 0
python training/verify_sample.py <sample_id> 1
```

### 4. Continual Training & Dragon Gate Execution
```bash
# Standard workflow: Retrain candidate -> Challenge Dragon Gate -> UI Approval Modal
python training/retrain.py

# Terminal-only mode (headless CLI prompt instead of GUI modal)
python training/retrain.py --no-ui

# Headless CI/CD mode (auto-promotes upon trial victory without prompting)
python training/retrain.py --yes
```

### 5. Candidate Audits & Manual Operations
```bash
# List all staged candidates in models/ and active Dragon status
python training/promote.py --list

# Manually trigger the Dragon Gate trial for candidate version N
python training/evaluate.py --candidate 2

# Manually execute promotion with UI approval dialog for candidate N
python training/promote.py --candidate 2
```

---

## 🔒 Safety & Resilience Guarantees

<details>
<summary><b>View System Safety Guarantees</b></summary>

- **Atomic File Swaps (`os.replace`)**: Updates to `ACTIVE_MODEL.json` use POSIX/Win32 atomic replacement. Inference processes reading during an active deployment never observe half-written or corrupted state.
- **Experience Replay Memory**: During continual fine-tuning, verified feedback vectors are interleaved with historical baseline distributions to eliminate catastrophic forgetting.
- **Dual Inference Engine**: The system defaults to the high-throughput C++ ONNX Runtime runtime. If dynamic loading fails or specialized ops are required, `inference/predictor.py` automatically falls back to native PyTorch execution without crashing.
</details>

---

## 📜 Dataset Attribution

This project uses the **BODMAS** (Blue Hexagon Open Dataset for Malware Analysis) dataset.

**Original Authors:**
- Limin Yang
- Arridhana Ciptadi
- Ihar Laziuk
- Ali Ahmadzadeh
- Gang Wang

*Developed in collaboration with Blue Hexagon.*

- **Official Project Page:** [https://whyisyoung.github.io/BODMAS/](https://whyisyoung.github.io/BODMAS/)
- **Kaggle Distribution:** Obtained from the curated distribution by dhoogla: [https://www.kaggle.com/datasets/dhoogla/bodmas](https://www.kaggle.com/datasets/dhoogla/bodmas)

*All dataset ownership, original features, and research authorship remain with the original BODMAS authors and their associated project.*
