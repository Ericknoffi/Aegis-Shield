import os
import json
import uuid
from datetime import datetime

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEEDBACK_DIR = os.path.join(
    BASE_DIR,
    "data",
    "feedback"
)

PENDING_DIR = os.path.join(
    FEEDBACK_DIR,
    "pending"
)

PENDING_BENIGN_DIR = os.path.join(
    PENDING_DIR,
    "benign"
)

PENDING_MALWARE_DIR = os.path.join(
    PENDING_DIR,
    "malware"
)

VERIFIED_DIR = os.path.join(
    FEEDBACK_DIR,
    "verified"
)

VERIFIED_BENIGN_DIR = os.path.join(
    VERIFIED_DIR,
    "benign"
)

VERIFIED_MALWARE_DIR = os.path.join(
    VERIFIED_DIR,
    "malware"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for d in [
    PENDING_DIR,
    PENDING_BENIGN_DIR,
    PENDING_MALWARE_DIR,
    VERIFIED_DIR,
    VERIFIED_BENIGN_DIR,
    VERIFIED_MALWARE_DIR
]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# CONSTANTS
# ============================================================

RAW_FEATURE_COUNT = 2381


# ============================================================
# FEEDBACK COLLECTOR
# ============================================================

class FeedbackCollector:

    # ========================================================
    # SAVE PREDICTION (Subdivided into benign / malware)
    # ========================================================

    def save_prediction(
        self,
        raw_features,
        prediction,
        probability,
        threshold,
        sha256=None,
        file_path=None,
        file_name=None
    ):

        raw_features = np.asarray(
            raw_features,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if raw_features.ndim != 1:
            raise ValueError(
                "raw_features must be a 1D array."
            )

        if len(raw_features) != RAW_FEATURE_COUNT:
            raise ValueError(
                f"Expected {RAW_FEATURE_COUNT} features, "
                f"got {len(raw_features)}"
            )

        # ----------------------------------------------------
        # Generate ID & Determine Category Subfolder
        # ----------------------------------------------------

        sample_id = uuid.uuid4().hex

        is_malware = (
            str(prediction).lower() in ["malware", "1", "true"]
            or float(probability) >= float(threshold)
        )
        category = "malware" if is_malware else "benign"
        target_dir = PENDING_MALWARE_DIR if is_malware else PENDING_BENIGN_DIR

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "sample_id": sample_id,
            "prediction": str(prediction),
            "probability": float(probability),
            "threshold": float(threshold),
            "predicted_category": category,
            "sha256": str(sha256) if sha256 else None,
            "file_path": str(file_path) if file_path else None,
            "file_name": str(file_name) if file_name else None,
            "timestamp": datetime.now().isoformat(),
            "raw_feature_count": RAW_FEATURE_COUNT,
            "verified": False
        }

        # ----------------------------------------------------
        # Save to specific pending subfolder
        # ----------------------------------------------------

        output_file = os.path.join(
            target_dir,
            f"{sample_id}.npz"
        )

        np.savez_compressed(
            output_file,
            features=raw_features,
            metadata=json.dumps(metadata)
        )

        return {
            "sample_id": sample_id,
            "status": "pending",
            "category": category,
            "prediction": prediction,
            "probability": float(probability),
            "filepath": output_file
        }


    # ========================================================
    # FIND PENDING FILE (Helper across subdirectories)
    # ========================================================

    def find_pending_file(self, sample_id):
        clean_id = sample_id.removesuffix(".npz")
        candidate_paths = [
            os.path.join(PENDING_BENIGN_DIR, f"{clean_id}.npz"),
            os.path.join(PENDING_MALWARE_DIR, f"{clean_id}.npz"),
            os.path.join(PENDING_DIR, f"{clean_id}.npz")
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                return p
        return None


    # ========================================================
    # VERIFY SAMPLE (Subdivided into verified/benign or malware)
    # ========================================================

    def verify_sample(
        self,
        sample_id,
        true_label
    ):

        # ----------------------------------------------------
        # Validate label
        # ----------------------------------------------------

        if true_label not in [0, 1]:
            raise ValueError(
                "true_label must be 0 or 1."
            )

        clean_id = sample_id.removesuffix(".npz")

        # ----------------------------------------------------
        # Locate Pending File
        # ----------------------------------------------------

        pending_file = self.find_pending_file(clean_id)

        if not pending_file:
            raise FileNotFoundError(
                f"Pending sample not found: {sample_id} "
                f"(searched in pending/benign and pending/malware)"
            )

        # ----------------------------------------------------
        # Load Sample Data
        # ----------------------------------------------------

        with np.load(
            pending_file,
            allow_pickle=False
        ) as data:

            features = np.asarray(
                data["features"],
                dtype=np.float32
            ).copy()

            metadata_raw = data["metadata"]

            if isinstance(
                metadata_raw,
                np.ndarray
            ):
                metadata_raw = metadata_raw.item()

            metadata = json.loads(
                str(metadata_raw)
            )

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if features.ndim != 1 or len(features) != RAW_FEATURE_COUNT:
            raise ValueError(
                f"Stored sample has invalid features. Expected {RAW_FEATURE_COUNT}."
            )

        # ----------------------------------------------------
        # Update metadata
        # ----------------------------------------------------

        verified_category = "malware" if true_label == 1 else "benign"
        target_verified_dir = (
            VERIFIED_MALWARE_DIR if true_label == 1 else VERIFIED_BENIGN_DIR
        )
        verified_file = os.path.join(
            target_verified_dir,
            f"{clean_id}.npz"
        )

        metadata["true_label"] = int(true_label)
        metadata["verified_category"] = verified_category
        metadata["verified"] = True
        metadata["verified_timestamp"] = datetime.now().isoformat()

        # ----------------------------------------------------
        # Write verified file safely
        # ----------------------------------------------------

        temp_file = verified_file + ".tmp"

        np.savez_compressed(
            temp_file,
            features=features,
            metadata=json.dumps(metadata)
        )

        actual_temp_file = temp_file
        if not os.path.exists(actual_temp_file):
            possible_file = temp_file + ".npz"
            if os.path.exists(possible_file):
                actual_temp_file = possible_file

        os.replace(
            actual_temp_file,
            verified_file
        )

        # Remove the pending sample
        os.remove(pending_file)

        return {
            "sample_id": clean_id,
            "true_label": int(true_label),
            "category": verified_category,
            "verified": True,
            "filepath": verified_file
        }


    # ========================================================
    # COUNT PENDING (Recursively across subdirectories)
    # ========================================================

    def count_pending(self):
        if not os.path.exists(PENDING_DIR):
            return 0

        count = 0
        for root, _, files in os.walk(PENDING_DIR):
            count += len([f for f in files if f.endswith(".npz")])
        return count


    # ========================================================
    # COUNT VERIFIED (Recursively across subdirectories)
    # ========================================================

    def count_verified(self):
        if not os.path.exists(VERIFIED_DIR):
            return 0

        count = 0
        for root, _, files in os.walk(VERIFIED_DIR):
            count += len([f for f in files if f.endswith(".npz")])
        return count


    # ========================================================
    # BUILD VERIFIED DATASET (Walks benign and malware subfolders)
    # ========================================================

    def build_verified_dataset(self):
        X = []
        y = []

        if not os.path.exists(VERIFIED_DIR):
            return (
                np.empty((0, RAW_FEATURE_COUNT), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        for root, _, filenames in os.walk(VERIFIED_DIR):
            for filename in filenames:
                if not filename.endswith(".npz"):
                    continue

                filepath = os.path.join(root, filename)

                try:
                    with np.load(filepath, allow_pickle=False) as data:
                        features = np.asarray(data["features"], dtype=np.float32).copy()
                        metadata_raw = data["metadata"]

                        if isinstance(metadata_raw, np.ndarray):
                            metadata_raw = metadata_raw.item()

                        metadata = json.loads(str(metadata_raw))

                    if "true_label" not in metadata:
                        continue

                    label = int(metadata["true_label"])
                    if label not in [0, 1]:
                        continue

                    if features.ndim != 1 or len(features) != RAW_FEATURE_COUNT:
                        continue

                    X.append(features)
                    y.append(label)

                except Exception as e:
                    print(f"Skipping {filename}: {e}")

        if not X:
            return (
                np.empty((0, RAW_FEATURE_COUNT), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32)
        )