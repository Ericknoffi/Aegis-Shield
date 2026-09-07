import os
import sys
import json
import shutil
import warnings
import argparse
from datetime import datetime

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
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INFERENCE_DIR = os.path.join(BASE_DIR, "inference")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEEDBACK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "feedback",
    "verified"
)


# ============================================================
# CONFIG
# ============================================================

RAW_FEATURE_COUNT = 2381
PROCESSED_FEATURE_COUNT = 2332
FINAL_FEATURE_COUNT = 250

RETRAIN_THRESHOLD = 100

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

VAL_SIZE = 0.20
RANDOM_STATE = 42

THRESHOLD = 0.55


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):

        return self.network(x).squeeze(1)


# ============================================================
# VERSION HELPERS
# ============================================================

def get_version_number(name):

    """
    Converts:

        v2_Promoted -> 2
        v3_Candidate -> 3
    """

    try:

        return int(
            name.split("_")[0].replace("v", "")
        )

    except Exception:

        return -1


def find_latest_promoted():

    versions = []

    if not os.path.exists(INFERENCE_DIR):
        return None

    for name in os.listdir(INFERENCE_DIR):

        if not name.endswith("_Promoted"):
            continue

        number = get_version_number(name)

        if number >= 0:
            versions.append(
                (number, os.path.join(INFERENCE_DIR, name))
            )

    if not versions:
        return None

    versions.sort(key=lambda x: x[0])

    return versions[-1]


def get_next_version(current_version):

    return current_version + 1


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

def load_checkpoint(model, model_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


# ============================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================

def load_production_artifacts(model_dir):

    print()
    print("=" * 70)
    print("LOADING PRODUCTION ARTIFACTS")
    print("=" * 70)

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

    config_path = os.path.join(
        model_dir,
        "model_config.json"
    )

    required_files = [
        model_path,
        scaler_path,
        feature_mask_path,
        top250_path
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    scaler = joblib.load(scaler_path)

    feature_mask = joblib.load(
        feature_mask_path
    )

    top250_positions = joblib.load(
        top250_path
    )

    top250_positions = np.asarray(
        top250_positions,
        dtype=np.int64
    )

    if len(top250_positions) != FINAL_FEATURE_COUNT:

        raise ValueError(
            f"Expected {FINAL_FEATURE_COUNT} selected "
            f"features, got {len(top250_positions)}"
        )

    if scaler.n_features_in_ != PROCESSED_FEATURE_COUNT:

        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"expected {PROCESSED_FEATURE_COUNT}"
        )

    model = MalwareMLP(
        FINAL_FEATURE_COUNT
    ).to(DEVICE)

    model = load_checkpoint(
        model,
        model_path
    )

    model.eval()

    print("Model:", model_path)
    print("Scaler features:", scaler.n_features_in_)
    print("Selected features:", len(top250_positions))
    print("Device:", DEVICE)

    return {
        "model": model,
        "scaler": scaler,
        "feature_mask": feature_mask,
        "top250_positions": top250_positions
    }


# ============================================================
# LOAD VERIFIED FEEDBACK
# ============================================================

def load_verified_feedback():

    if not os.path.exists(FEEDBACK_DIR):

        return (
            np.empty(
                (0, RAW_FEATURE_COUNT),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            )
        )

    feature_rows = []
    labels = []

    files = []
    for root, _, filenames in os.walk(FEEDBACK_DIR):
        for f in filenames:
            if f.endswith(".npz"):
                files.append(os.path.join(root, f))
    files.sort()

    print()
    print("=" * 70)
    print("LOADING VERIFIED FEEDBACK")
    print("=" * 70)

    print("Verified files:", len(files))

    for path in files:
        filename = os.path.basename(path)

        try:

            # IMPORTANT:
            # Explicitly close NPZ on Windows.
            with np.load(path) as data:

                if "features" not in data:
                    print(
                        f"Skipping {filename}: "
                        f"missing features"
                    )
                    continue

                label = None
                if "true_label" in data:
                    label = int(
                        np.asarray(
                            data["true_label"]
                        ).item()
                    )
                elif "metadata" in data:
                    metadata_raw = data["metadata"]
                    if isinstance(metadata_raw, np.ndarray):
                        metadata_raw = metadata_raw.item()
                    metadata = json.loads(str(metadata_raw))
                    if "true_label" in metadata:
                        label = int(metadata["true_label"])

                if label is None:
                    print(
                        f"Skipping {filename}: "
                        f"missing true_label"
                    )
                    continue

                features = np.asarray(
                    data["features"],
                    dtype=np.float32
                ).copy()

            if features.shape != (
                RAW_FEATURE_COUNT,
            ):

                print(
                    f"Skipping {filename}: "
                    f"expected {RAW_FEATURE_COUNT} features, "
                    f"got {features.shape}"
                )

                continue

            if label not in (0, 1):

                print(
                    f"Skipping {filename}: "
                    f"invalid label {label}"
                )

                continue

            if not np.all(
                np.isfinite(features)
            ):

                print(
                    f"Skipping {filename}: "
                    f"non-finite feature values"
                )

                continue

            feature_rows.append(features)
            labels.append(label)

        except Exception as e:

            print(
                f"Skipping {filename}: {e}"
            )

    if not feature_rows:

        return (
            np.empty(
                (0, RAW_FEATURE_COUNT),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            )
        )

    X = np.stack(
        feature_rows
    ).astype(np.float32)

    y = np.asarray(
        labels,
        dtype=np.float32
    )

    print("Valid samples:", len(X))

    unique, counts = np.unique(
        y.astype(np.int64),
        return_counts=True
    )

    class_counts = dict(
        zip(
            unique.tolist(),
            counts.tolist()
        )
    )

    print("Class distribution:", class_counts)

    return X, y

# ============================================================
# ACTUAL PREPROCESSING
# ============================================================

def preprocess_feedback_with_mask(
    X_raw,
    feature_mask,
    scaler,
    top250_positions
):

    if X_raw.ndim != 2:

        raise ValueError(
            f"Expected 2D input, got {X_raw.ndim}D"
        )

    if X_raw.shape[1] != RAW_FEATURE_COUNT:

        raise ValueError(
            f"Expected {RAW_FEATURE_COUNT} raw features, "
            f"got {X_raw.shape[1]}"
        )

    non_constant_features = (
        feature_mask["non_constant_features"]
    )

    non_constant_positions = np.asarray(
        [
            int(name[1:]) - 1
            for name in non_constant_features
        ],
        dtype=np.int64
    )

    if len(non_constant_positions) != (
        PROCESSED_FEATURE_COUNT
    ):

        raise ValueError(
            "Feature mask mismatch: "
            f"expected {PROCESSED_FEATURE_COUNT} "
            f"non-constant features, "
            f"got {len(non_constant_positions)}"
        )

    # --------------------------------------------------------
    # STEP 1
    # Remove constant features
    # --------------------------------------------------------

    X_processed = X_raw[
        :,
        non_constant_positions
    ]

    if X_processed.shape[1] != (
        PROCESSED_FEATURE_COUNT
    ):

        raise ValueError(
            f"Processed shape mismatch: "
            f"{X_processed.shape}"
        )

    # --------------------------------------------------------
    # STEP 2
    # Apply production scaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X_processed
    )

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float32
    )

    if not np.all(
        np.isfinite(X_scaled)
    ):

        raise ValueError(
            "Scaler produced non-finite values."
        )

    # --------------------------------------------------------
    # STEP 3
    # Select exact top-250 positions
    # --------------------------------------------------------

    X_selected = X_scaled[
        :,
        top250_positions
    ]

    X_selected = np.asarray(
        X_selected,
        dtype=np.float32
    )

    if X_selected.shape[1] != FINAL_FEATURE_COUNT:

        raise ValueError(
            f"Final feature shape mismatch: "
            f"{X_selected.shape}"
        )

    if not np.all(
        np.isfinite(X_selected)
    ):

        raise ValueError(
            "Final features contain non-finite values."
        )

    return X_selected


# ============================================================
# DRY RUN
# ============================================================

def dry_run():

    print()
    print("=" * 70)
    print("RETRAIN DRY RUN")
    print("=" * 70)

    print()
    print("This mode will:")
    print("  [1] Load production artifacts")
    print("  [2] Load verified feedback")
    print("  [3] Validate raw feature dimensions")
    print("  [4] Apply production preprocessing")
    print("  [5] Validate the 250-feature model input")
    print("  [6] Check training feasibility")
    print()
    print("It will NOT:")
    print("  - train a model")
    print("  - create a candidate")
    print("  - modify V2")
    print("  - modify ACTIVE_MODEL.json")
    print("  - promote anything")
    print()

    latest = find_latest_promoted()

    if latest is None:

        raise RuntimeError(
            "No promoted production model found."
        )

    current_version, current_dir = latest

    print(
        f"Current production version: v{current_version}"
    )

    artifacts = load_production_artifacts(
        current_dir
    )

    X_raw, y = load_verified_feedback()

    sample_count = len(X_raw)

    print()
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print(
        "Verified samples:",
        sample_count
    )

    if sample_count == 0:

        print()
        print(
            "DRY RUN RESULT: FAIL"
        )

        print(
            "No verified samples are available."
        )

        return 1

    # --------------------------------------------------------
    # Preprocessing test
    # --------------------------------------------------------

    X_selected = preprocess_feedback_with_mask(
        X_raw,
        artifacts["feature_mask"],
        artifacts["scaler"],
        artifacts["top250_positions"]
    )

    print(
        "Final training shape:",
        X_selected.shape
    )

    print(
        "Finite values: OK"
    )

    # --------------------------------------------------------
    # Class information
    # --------------------------------------------------------

    class_0 = int(
        np.sum(y == 0)
    )

    class_1 = int(
        np.sum(y == 1)
    )

    print()
    print("Benign samples:", class_0)
    print("Malware samples:", class_1)

    # --------------------------------------------------------
    # Training feasibility
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING FEASIBILITY")
    print("=" * 70)

    if sample_count < RETRAIN_THRESHOLD:

        print(
            f"Not enough samples yet."
        )

        print(
            f"Required: {RETRAIN_THRESHOLD}"
        )

        print(
            f"Current:  {sample_count}"
        )

        print(
            f"Remaining: "
            f"{RETRAIN_THRESHOLD - sample_count}"
        )

        if class_0 == 0 or class_1 == 0:

            print(
                "Also requires both classes."
            )

        print()
        print(
            "DRY RUN RESULT: PASS"
        )

        print(
            "The feedback loading and preprocessing "
            "pipeline is working."
        )

        print(
            "No model was trained or modified."
        )

        return 0

    if class_0 == 0 or class_1 == 0:

        print(
            "DRY RUN RESULT: FAIL"
        )

        print(
            "Training requires both classes "
            "(Benign=0 and Malware=1)."
        )

        return 1

    # Check whether stratified split is possible.

    try:

        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_selected,
                y,
                test_size=VAL_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )

    except Exception as e:

        print(
            "DRY RUN RESULT: FAIL"
        )

        print(
            f"Stratified split failed: {e}"
        )

        return 1

    print(
        "Train shape:",
        X_train.shape
    )

    print(
        "Validation shape:",
        X_val.shape
    )

    print()
    print(
        "DRY RUN RESULT: PASS"
    )

    print(
        "The feedback dataset is large enough "
        "and structurally valid for training."
    )

    return 0


# ============================================================
# REAL RETRAIN
# ============================================================

def retrain():

    print()
    print("=" * 70)
    print("PRODUCTION FEEDBACK RETRAINING")
    print("=" * 70)

    latest = find_latest_promoted()

    if latest is None:

        raise RuntimeError(
            "No promoted model found."
        )

    current_version, current_dir = latest

    next_version = get_next_version(
        current_version
    )

    print(
        f"Current production: v{current_version}"
    )

    print(
        f"Candidate version:  v{next_version}"
    )

    artifacts = load_production_artifacts(
        current_dir
    )

    X_raw, y = load_verified_feedback()

    sample_count = len(X_raw)

    print()
    print(
        f"Verified samples available: {sample_count}"
    )

    if sample_count < RETRAIN_THRESHOLD:

        raise RuntimeError(
            f"Need at least {RETRAIN_THRESHOLD} "
            f"verified samples. "
            f"Only {sample_count} available."
        )

    class_0 = int(
        np.sum(y == 0)
    )

    class_1 = int(
        np.sum(y == 1)
    )

    if class_0 == 0 or class_1 == 0:

        raise RuntimeError(
            "Retraining requires both classes. "
            f"Benign={class_0}, Malware={class_1}"
        )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    X = preprocess_feedback_with_mask(
        X_raw,
        artifacts["feature_mask"],
        artifacts["scaler"],
        artifacts["top250_positions"]
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=VAL_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print()
    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MalwareMLP(
        FINAL_FEATURE_COUNT
    ).to(DEVICE)

    model = load_checkpoint(
        model,
        os.path.join(
            current_dir,
            "malware_mlp_top250.pth"
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    # --------------------------------------------------------
    # CANDIDATE DIRECTORY
    # --------------------------------------------------------

    candidate_dir = os.path.join(
        MODELS_DIR,
        f"v{next_version}_Candidate"
    )

    if os.path.exists(candidate_dir):

        print()
        print(
            f"Removing old candidate: "
            f"{candidate_dir}"
        )

        shutil.rmtree(
            candidate_dir
        )

    os.makedirs(
        candidate_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_loss = float("inf")

    history = []

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(
                DEVICE,
                non_blocking=True
            )

            batch_y = batch_y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                batch_X
            )

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            batch_size = batch_X.size(0)

            train_loss_sum += (
                loss.item() * batch_size
            )

            train_count += batch_size

        train_loss = (
            train_loss_sum /
            train_count
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(
                    DEVICE,
                    non_blocking=True
                )

                batch_y = batch_y.to(
                    DEVICE,
                    non_blocking=True
                )

                logits = model(
                    batch_X
                )

                loss = criterion(
                    logits,
                    batch_y
                )

                batch_size = batch_X.size(0)

                val_loss_sum += (
                    loss.item() * batch_size
                )

                val_count += batch_size

        val_loss = (
            val_loss_sum /
            val_count
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    candidate_dir,
                    "malware_mlp_top250.pth"
                )
            )

    # --------------------------------------------------------
    # EXPORT OPTIMIZED ONNX MODEL
    # --------------------------------------------------------

    try:
        best_pth = os.path.join(candidate_dir, "malware_mlp_top250.pth")
        best_onnx = os.path.join(candidate_dir, "malware_mlp_top250.onnx")
        export_model = MalwareMLP(FINAL_FEATURE_COUNT)
        export_model.load_state_dict(torch.load(best_pth, map_location="cpu", weights_only=True))
        export_model.eval()
        dummy_input = torch.randn(1, FINAL_FEATURE_COUNT, dtype=torch.float32)
        torch.onnx.export(
            export_model,
            dummy_input,
            best_onnx,
            input_names=["features"],
            output_names=["logits"],
            opset_version=18,
            dynamo=False
        )
        print("Exported candidate to ONNX Runtime format.")
    except Exception as onnx_err:
        print(f"Note: ONNX export skipped ({onnx_err})")

    # --------------------------------------------------------
    # COPY PREPROCESSING ARTIFACTS
    # --------------------------------------------------------

    for filename in [
        "scaler.pkl",
        "feature_mask.pkl",
        "top250_positions.pkl"
    ]:

        source = os.path.join(
            current_dir,
            filename
        )

        destination = os.path.join(
            candidate_dir,
            filename
        )

        shutil.copy2(
            source,
            destination
        )

    # --------------------------------------------------------
    # MODEL CONFIG
    # --------------------------------------------------------

    model_config = {
        "input_dim": FINAL_FEATURE_COUNT,
        "raw_feature_count": RAW_FEATURE_COUNT,
        "processed_feature_count": PROCESSED_FEATURE_COUNT,
        "selected_feature_count": FINAL_FEATURE_COUNT,
        "threshold": THRESHOLD,
        "architecture": "MalwareMLP",
        "device_used": str(DEVICE)
    }

    with open(
        os.path.join(
            candidate_dir,
            "model_config.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model_config,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # VERSION INFO
    # --------------------------------------------------------

    version_info = {
        "version": f"v{next_version}",
        "status": "candidate",
        "parent_version": f"v{current_version}",
        "created_at": datetime.now().isoformat(),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "verified_feedback_samples": int(sample_count),
        "benign_samples": class_0,
        "malware_samples": class_1,
        "best_validation_loss": float(best_val_loss)
    }

    with open(
        os.path.join(
            candidate_dir,
            "version.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            version_info,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    with open(
        os.path.join(
            candidate_dir,
            "training_history.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("🐉 RETRAINING COMPLETE: NEW ASPIRANT CANDIDATE BORN")
    print("=" * 70)

    print(
        f"Candidate: v{next_version}"
    )

    print(
        f"Location: {candidate_dir}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )
    return next_version, current_version


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="BODMAS production feedback retraining"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate feedback loading and "
            "preprocessing without training."
        )
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic post-training evaluation and promotion prompt."
    )

    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Use terminal prompt instead of GUI modal for promotion choice."
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Automatically promote candidate if it passes the gate without prompt."
    )

    args = parser.parse_args()

    print()
    print("BODMAS CONTINUAL LEARNING SYSTEM")

    print(
        "Device:",
        DEVICE
    )

    try:

        if args.dry_run:

            return dry_run()

        retrain_res = retrain()
        if retrain_res == 1 or not isinstance(retrain_res, tuple):
            return retrain_res

        next_version, current_version = retrain_res

        if args.no_eval:
            print()
            print("=" * 72)
            print(f"🐉 Aspirant Candidate v{next_version} born! Run evaluate.py / promote.py to face Dragon Gate.")
            print("=" * 72)
            return 0

        # ----------------------------------------------------
        # DRAGON GATE AUTO-EVALUATION
        # ----------------------------------------------------
        try:
            from evaluate import evaluate_candidate
            from promote import show_promotion_dialog, promote
        except ImportError:
            from training.evaluate import evaluate_candidate
            from training.promote import show_promotion_dialog, promote

        print()
        print("🐉" + "=" * 70 + "🐉")
        print("          SUMMONING THE DRAGON GATE TRIAL (龙门 - LONGMEN)...          ")
        print("🐉" + "=" * 70 + "🐉")

        passed, evaluation = evaluate_candidate(next_version)

        # ----------------------------------------------------
        # DRAGON GATE RESULT
        # ----------------------------------------------------
        if not passed:
            print()
            print("🌊" + "=" * 70 + "🌊")
            print("     DRAGON GATE TRIAL: FAILED (REMAINS MORTAL ASPIRANT)     ")
            print("🌊" + "=" * 70 + "🌊")
            print(f"Aspirant candidate v{next_version} could not conquer the Dragon Gate.")
            print(f"Reigning Celestial Dragon continues reigning: v{current_version}")
            print(f"Candidate preserved in mortal realm: models/v{next_version}_Candidate/")
            print("=" * 72)
            return 0

        # Dragon Gate Passed -> Give Choice to User via UI / CLI
        print()
        print("🐉" + "=" * 70 + "🐉")
        print("      DRAGON GATE TRIAL: ASCENDED! (WORTHY TO EVOLVE)       ")
        print("🐉" + "=" * 70 + "🐉")

        if args.yes:
            approved = True
        elif args.no_ui:
            try:
                choice = input(
                    f"\nAspirant v{next_version} passed all 4 trials! Evolve into Celestial Dragon? [Y/n]: "
                ).strip().lower()
                approved = choice in ("y", "yes", "")
            except Exception:
                approved = False
        else:
            print("\nDisplaying Dragon Gate Ascension confirmation dialog...")
            approved = show_promotion_dialog(
                next_version,
                current_version,
                evaluation
            )

        if approved:
            print("\n🐉 Channeling Dragon Gate energy... Evolving into Celestial Dragon...")
            promoted_dir = promote(next_version, current_version)
            print()
            print("🐉" + "=" * 70 + "🐉")
            print("            EVOLUTION COMPLETE: CELESTIAL DRAGON BORN            ")
            print("🐉" + "=" * 70 + "🐉")
            print(f"Reigning Celestial Dragon (Active Production): v{next_version}")
            print(f"Dragon Temple (Model Location): {promoted_dir}")
            print(f"Previous dragon (v{current_version}) preserved safely for instant rollback.")
            print("🐉" + "=" * 70 + "🐉")
        else:
            print()
            print("=" * 72)
            print("ASCENSION DEFERRED: KEPT IN MORTAL REALM")
            print("=" * 72)
            print(f"Aspirant v{next_version} preserved in models/v{next_version}_Candidate/")
            print(f"Reigning Celestial Dragon remains: v{current_version}")
            print("=" * 72)

        return 0

    except Exception as e:

        print()
        print("=" * 70)
        print("RETRAINING FAILED")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )

        print()
        print(
            "Production model was NOT modified."
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )