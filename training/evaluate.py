import os
import sys
import json
import glob
import warnings
import argparse

# Safe UTF-8 encoding for Windows terminals (handling Chinese characters and emoji)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INFERENCE_DIR = os.path.join(
    BASE_DIR,
    "inference"
)

FEEDBACK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "feedback"
)

VERIFIED_DIR = os.path.join(
    FEEDBACK_DIR,
    "verified"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# CONFIG
# ============================================================

RAW_FEATURES = 2381
PROCESSED_FEATURES = 2332
FINAL_FEATURES = 250

VAL_SIZE = 0.20
RANDOM_STATE = 42

THRESHOLD = 0.55

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL
# ============================================================

class MalwareMLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        return self.network(x).squeeze(1)


# ============================================================
# VERSION HELPERS
# ============================================================

def get_promoted_versions():

    versions = []

    for name in os.listdir(INFERENCE_DIR):

        if not name.startswith("v"):
            continue

        if not name.endswith("_Promoted"):
            continue

        try:

            number = int(
                name[1:].split("_")[0]
            )

            versions.append(number)

        except ValueError:

            continue

    return sorted(versions)


def get_current_promoted_version():

    versions = get_promoted_versions()

    if not versions:

        raise RuntimeError(
            "No promoted model found."
        )

    return versions[-1]


def get_candidate_versions():

    versions = set()

    if not os.path.exists(MODELS_DIR):
        return []

    for name in os.listdir(MODELS_DIR):
        full_path = os.path.join(MODELS_DIR, name)
        if not os.path.isdir(full_path):
            continue

        if not name.startswith("v"):
            continue

        try:
            raw_part = name[1:].split("_")[0]
            number = int(raw_part)
            versions.add(number)
        except ValueError:
            continue

    return sorted(list(versions))


def get_candidate_dir(candidate_version):
    """
    Finds the candidate folder in models/.
    Supports v{N}_Candidate, v{N}_Artifact, or any v{N} directory.
    """
    if not os.path.exists(MODELS_DIR):
        return os.path.join(MODELS_DIR, f"v{candidate_version}_Candidate")

    preferred = os.path.join(MODELS_DIR, f"v{candidate_version}_Candidate")
    if os.path.exists(preferred):
        return preferred

    for name in os.listdir(MODELS_DIR):
        full_path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(full_path) and name.startswith("v"):
            try:
                num = int(name[1:].split("_")[0])
                if num == candidate_version:
                    return full_path
            except ValueError:
                continue

    return preferred


def get_latest_candidate_version():

    versions = get_candidate_versions()

    if not versions:

        raise RuntimeError(
            "No candidate model found."
        )

    return versions[-1]


# ============================================================
# LOAD VERIFIED DATA
# ============================================================

def load_verified_feedback():

    files = sorted(
        glob.glob(
            os.path.join(
                VERIFIED_DIR,
                "*.npz"
            )
        )
    )

    if len(files) < 2:

        raise RuntimeError(
            "Not enough verified feedback samples."
        )

    X = []
    y = []

    for path in files:

        try:

            data = np.load(
                path,
                allow_pickle=False
            )

            features = np.asarray(
                data["features"],
                dtype=np.float32
            )

            metadata = json.loads(
                str(data["metadata"])
            )

            if "true_label" not in metadata:
                continue

            label = int(
                metadata["true_label"]
            )

            if features.shape != (RAW_FEATURES,):
                continue

            if label not in (0, 1):
                continue

            X.append(features)
            y.append(label)

        except Exception:

            continue

    if len(X) < 2:

        raise RuntimeError(
            "No valid verified samples found."
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    if len(np.unique(y)) < 2:

        raise RuntimeError(
            "Evaluation data must contain "
            "both benign and malware samples."
        )

    return X, y


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts(model_dir):

    model_path = os.path.join(
        model_dir,
        "malware_mlp_top250.pth"
    )

    scaler_path = os.path.join(
        model_dir,
        "scaler.pkl"
    )

    feature_mask_path = os.path.join(
        model_dir,
        "feature_mask.pkl"
    )

    top250_path = os.path.join(
        model_dir,
        "top250_positions.pkl"
    )

    for path in [
        model_path,
        scaler_path,
        feature_mask_path,
        top250_path
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing artifact:\n{path}"
            )

    scaler = joblib.load(
        scaler_path
    )

    feature_mask = joblib.load(
        feature_mask_path
    )

    top250_positions = joblib.load(
        top250_path
    )

    if isinstance(feature_mask, dict):

        non_constant_features = (
            feature_mask[
                "non_constant_features"
            ]
        )

    else:

        non_constant_features = feature_mask

    non_constant_positions = np.asarray(
        [
            int(name[1:]) - 1
            for name in non_constant_features
        ],
        dtype=np.int64
    )

    top250_positions = np.asarray(
        top250_positions,
        dtype=np.int64
    )

    return (
        model_path,
        scaler,
        non_constant_positions,
        top250_positions
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(
    X,
    scaler,
    non_constant_positions,
    top250_positions
):

    if X.shape[1] != RAW_FEATURES:

        raise ValueError(
            f"Expected {RAW_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    X = X[
        :,
        non_constant_positions
    ]

    X = scaler.transform(
        X
    )

    X = X[
        :,
        top250_positions
    ]

    return X.astype(
        np.float32
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):

    model = MalwareMLP(
        FINAL_FEATURES
    )

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X
):

    probabilities = []

    tensor = torch.from_numpy(
        X
    )

    for start in range(
        0,
        len(tensor),
        BATCH_SIZE
    ):

        batch = tensor[
            start:start + BATCH_SIZE
        ]

        batch = batch.to(
            DEVICE,
            non_blocking=True
        )

        with torch.no_grad():

            logits = model(
                batch
            )

            probs = torch.sigmoid(
                logits
            )

        probabilities.extend(
            probs.cpu().numpy()
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )

    predictions = (
        probabilities >= THRESHOLD
    ).astype(
        np.int64
    )

    return predictions, probabilities


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities
):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

    return metrics


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_dir,
    X_eval,
    y_eval
):

    (
        model_path,
        scaler,
        non_constant_positions,
        top250_positions
    ) = load_artifacts(
        model_dir
    )

    X_processed = preprocess(
        X_eval,
        scaler,
        non_constant_positions,
        top250_positions
    )

    model = load_model(
        model_path
    )

    predictions, probabilities = predict(
        model,
        X_processed
    )

    metrics = calculate_metrics(
        y_eval,
        predictions,
        probabilities
    )

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics
):

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Accuracy : {metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: {metrics['precision']:.6f}"
    )

    print(
        f"Recall   : {metrics['recall']:.6f}"
    )

    print(
        f"F1       : {metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC   : {metrics['pr_auc']:.6f}"
    )

    print()
    print("Confusion Matrix:")
    print(
        f"[[{metrics['tn']:4d} {metrics['fp']:4d}]"
    )
    print(
        f" [{metrics['fn']:4d} {metrics['tp']:4d}]]"
    )

    print()
    print(
        "TN:",
        metrics["tn"]
    )

    print(
        "FP:",
        metrics["fp"]
    )

    print(
        "FN:",
        metrics["fn"]
    )

    print(
        "TP:",
        metrics["tp"]
    )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# EVALUATE CANDIDATE PIPELINE
# ============================================================

def evaluate_candidate(candidate_version=None):

    print()
    print("=" * 70)
    print("CONTINUAL LEARNING MODEL EVALUATION")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # VERSIONS
    # --------------------------------------------------------

    current_version = (
        get_current_promoted_version()
    )

    if candidate_version is None:

        candidate_version = (
            get_latest_candidate_version()
        )

    current_dir = os.path.join(
        INFERENCE_DIR,
        f"v{current_version}_Promoted"
    )

    candidate_dir = get_candidate_dir(candidate_version)

    if not os.path.exists(candidate_dir):

        raise FileNotFoundError(
            f"Candidate directory not found:\n"
            f"{candidate_dir}"
        )

    print()
    print(
        "Current model :",
        f"V{current_version}"
    )

    print(
        "Candidate     :",
        f"V{candidate_version}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    X, y = load_verified_feedback()

    print()
    print(
        "Total verified samples:",
        len(X)
    )

    # --------------------------------------------------------
    # RECREATE SAME VALIDATION SPLIT
    #
    # This MUST match retrain.py:
    #
    # test_size=0.20
    # random_state=42
    # stratify=y
    # --------------------------------------------------------

    (
        X_train,
        X_eval,
        y_train,
        y_eval
    ) = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        "Evaluation samples:",
        len(X_eval)
    )

    print(
        "Evaluation benign:",
        int(np.sum(y_eval == 0))
    )

    print(
        "Evaluation malware:",
        int(np.sum(y_eval == 1))
    )

    # --------------------------------------------------------
    # EVALUATE CURRENT MODEL
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        f"EVALUATING V{current_version} PROMOTED MODEL"
    )
    print(
        "=" * 70
    )

    current_metrics = evaluate_model(
        current_dir,
        X_eval,
        y_eval
    )

    print_metrics(
        f"V{current_version} PROMOTED",
        current_metrics
    )

    # --------------------------------------------------------
    # EVALUATE CANDIDATE
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        f"EVALUATING V{candidate_version} CANDIDATE MODEL"
    )
    print(
        "=" * 70
    )

    candidate_metrics = evaluate_model(
        candidate_dir,
        X_eval,
        y_eval
    )

    print_metrics(
        f"V{candidate_version} CANDIDATE",
        candidate_metrics
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print("MODEL COMPARISON")
    print(
        "=" * 70
    )

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]

    print(
        f"{'Metric':<15}"
        f"{'Current':>14}"
        f"{'Candidate':>14}"
        f"{'Change':>14}"
    )

    print("-" * 57)

    changes = {}

    for metric in metric_names:

        current = current_metrics[
            metric
        ]

        candidate = candidate_metrics[
            metric
        ]

        change = (
            candidate -
            current
        )

        changes[metric] = change

        print(
            f"{metric:<15}"
            f"{current:>14.6f}"
            f"{candidate:>14.6f}"
            f"{change:>+14.6f}"
        )

    # --------------------------------------------------------
    # CONFUSION MATRIX COMPARISON
    # --------------------------------------------------------

    print()
    print(
        "Confusion Matrix Comparison"
    )

    print(
        f"{'':<12}"
        f"{'Current':>12}"
        f"{'Candidate':>12}"
    )

    for name in [
        "tn",
        "fp",
        "fn",
        "tp"
    ]:

        print(
            f"{name.upper():<12}"
            f"{current_metrics[name]:>12}"
            f"{candidate_metrics[name]:>12}"
        )

    # --------------------------------------------------------
    # PROMOTION GATE
    # --------------------------------------------------------

    #
    # Conservative production rule:
    #
    # 1. Recall must not decrease.
    # 2. Precision must not decrease.
    # 3. F1 must strictly improve.
    # 4. PR-AUC must not decrease.
    #
    # This means an identical candidate is NOT promoted.
    #

    recall_ok = (
        candidate_metrics["recall"]
        >=
        current_metrics["recall"]
    )

    precision_ok = (
        candidate_metrics["precision"]
        >=
        current_metrics["precision"]
    )

    f1_improved = (
        candidate_metrics["f1"]
        >
        current_metrics["f1"]
    )

    pr_auc_ok = (
        candidate_metrics["pr_auc"]
        >=
        current_metrics["pr_auc"]
    )

    promotion_pass = (
        recall_ok
        and precision_ok
        and f1_improved
        and pr_auc_ok
    )

    print()
    print(
        "=" * 70
    )
    print("🐉 THE DRAGON GATE TRIAL (龙门 - LONGMEN) 🐉")
    print(
        "=" * 70
    )

    print(
        "Trial 1: Recall >= current (No missed prey)  :",
        "✔ PASS" if recall_ok else "✖ FAIL"
    )

    print(
        "Trial 2: Precision >= current (No false roar):",
        "✔ PASS" if precision_ok else "✖ FAIL"
    )

    print(
        "Trial 3: F1 > current (Ascension of power)   :",
        "✔ PASS" if f1_improved else "✖ FAIL"
    )

    print(
        "Trial 4: PR-AUC >= current (Domain stability):",
        "✔ PASS" if pr_auc_ok else "✖ FAIL"
    )

    print()

    if promotion_pass:
        print(
            "🐉 RESULT: CANDIDATE LEAPED OVER THE DRAGON GATE (READY TO EVOLVE)!"
        )
    else:
        print(
            "🌊 RESULT: CANDIDATE FAILED THE DRAGON GATE (REMAINS MORTAL ASPIRANT)."
        )

    # --------------------------------------------------------
    # SAVE EVALUATION
    # --------------------------------------------------------

    gate_dict = {
        "recall_ok": recall_ok,
        "precision_ok": precision_ok,
        "f1_improved": f1_improved,
        "pr_auc_ok": pr_auc_ok,
        "passed": promotion_pass
    }

    evaluation = {

        "current_version":
            f"v{current_version}",

        "candidate_version":
            f"v{candidate_version}",

        "device":
            str(DEVICE),

        "threshold":
            THRESHOLD,

        "evaluation_samples":
            int(len(X_eval)),

        "current_metrics":
            current_metrics,

        "candidate_metrics":
            candidate_metrics,

        "changes":
            changes,

        "dragon_gate": gate_dict,
        "promotion_gate": gate_dict
    }

    output_path = os.path.join(
        candidate_dir,
        "evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=4
        )

    print()
    print(
        "Evaluation saved:"
    )

    print(
        output_path
    )

    # --------------------------------------------------------
    # UPDATE VERSION.JSON WITH DRAGON GATE STATUS TAG
    # --------------------------------------------------------

    version_file = os.path.join(
        candidate_dir,
        "version.json"
    )

    if os.path.exists(version_file):
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                v_data = json.load(f)

            v_data["dragon_gate_status"] = "ASCENDED" if promotion_pass else "FALLEN"
            v_data["gate_status"] = "PASSED" if promotion_pass else "FAILED"
            v_data["dragon_gate_details"] = gate_dict
            v_data["status"] = "dragon_candidate_ascended" if promotion_pass else "candidate_aspirant"

            with open(version_file, "w", encoding="utf-8") as f:
                json.dump(v_data, f, indent=4)
        except Exception:
            pass

    # --------------------------------------------------------
    # WRITE EXPLICIT DRAGON GATE TAG MARKER FILE
    # --------------------------------------------------------

    tag_file = os.path.join(
        candidate_dir,
        "DRAGON_GATE_ASCENDED.json" if promotion_pass else "DRAGON_GATE_FALLEN.json"
    )

    # Clean up opposite marker if it previously existed
    opposite_tag = os.path.join(
        candidate_dir,
        "DRAGON_GATE_FALLEN.json" if promotion_pass else "DRAGON_GATE_ASCENDED.json"
    )
    if os.path.exists(opposite_tag):
        try:
            os.remove(opposite_tag)
        except Exception:
            pass

    tag_summary = {
        "candidate_version": f"v{candidate_version}",
        "compared_against": f"v{current_version}",
        "dragon_gate": "ASCENDED (Ready to Evolve)" if promotion_pass else "FALLEN (Mortal Aspirant)",
        "recall_ok": recall_ok,
        "precision_ok": precision_ok,
        "f1_improved": f1_improved,
        "pr_auc_ok": pr_auc_ok
    }

    with open(tag_file, "w", encoding="utf-8") as f:
        json.dump(tag_summary, f, indent=4)

    return promotion_pass, evaluation


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Dragon Gate (Longmen) Model Evaluation Benchmark"
    )

    parser.add_argument(
        "--candidate",
        type=int,
        default=None,
        help="Candidate version number to test at Dragon Gate"
    )

    args = parser.parse_args()

    passed, evaluation = evaluate_candidate(args.candidate)

    print()
    print("=" * 70)
    print("DRAGON GATE TRIAL CONCLUDED")
    print("=" * 70)
    if passed:
        print("🐉 Candidate PASSED the Dragon Gate! Run promote.py to evolve into Dragon.")
    else:
        print("🌊 Candidate FAILED the Dragon Gate. Active Dragon (production) remains unchanged.")
    print("=" * 70)


if __name__ == "__main__":

    main()

