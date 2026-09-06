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

VERIFIED_DIR = os.path.join(
    FEEDBACK_DIR,
    "verified"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    PENDING_DIR,
    exist_ok=True
)

os.makedirs(
    VERIFIED_DIR,
    exist_ok=True
)


# ============================================================
# CONSTANTS
# ============================================================

RAW_FEATURE_COUNT = 2381


# ============================================================
# FEEDBACK COLLECTOR
# ============================================================

class FeedbackCollector:

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    def save_prediction(
        self,
        raw_features,
        prediction,
        probability,
        threshold
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
        # Generate ID
        # ----------------------------------------------------

        sample_id = uuid.uuid4().hex

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = {
            "sample_id": sample_id,
            "prediction": str(prediction),
            "probability": float(probability),
            "threshold": float(threshold),
            "timestamp": datetime.now().isoformat(),
            "raw_feature_count": RAW_FEATURE_COUNT,
            "verified": False
        }

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_file = os.path.join(
            PENDING_DIR,
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
            "prediction": prediction,
            "probability": float(probability)
        }


    # ========================================================
    # VERIFY SAMPLE
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

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        pending_file = os.path.join(
            PENDING_DIR,
            f"{sample_id}.npz"
        )

        verified_file = os.path.join(
            VERIFIED_DIR,
            f"{sample_id}.npz"
        )

        if not os.path.exists(
            pending_file
        ):

            raise FileNotFoundError(
                f"Pending sample not found: "
                f"{sample_id}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Explicitly close np.load() before touching the
        # original file.
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

        # At this point the npz file is CLOSED.

        # ----------------------------------------------------
        # Validate features
        # ----------------------------------------------------

        if features.ndim != 1:

            raise ValueError(
                "Stored feature vector is not 1D."
            )

        if len(features) != RAW_FEATURE_COUNT:

            raise ValueError(
                f"Stored sample has "
                f"{len(features)} features. "
                f"Expected {RAW_FEATURE_COUNT}."
            )

        # ----------------------------------------------------
        # Update metadata
        # ----------------------------------------------------

        metadata["true_label"] = int(
            true_label
        )

        metadata["verified"] = True

        metadata["verified_timestamp"] = (
            datetime.now().isoformat()
        )

        # ----------------------------------------------------
        # Write verified file
        #
        # Write to temporary file first so that an interrupted
        # operation doesn't create a corrupt feedback sample.
        # ----------------------------------------------------

        temp_file = verified_file + ".tmp"

        np.savez_compressed(
            temp_file,
            features=features,
            metadata=json.dumps(metadata)
        )

        # np.savez_compressed may append .npz when the supplied
        # filename doesn't already end with .npz.
        actual_temp_file = temp_file

        if not os.path.exists(
            actual_temp_file
        ):

            possible_file = (
                temp_file + ".npz"
            )

            if os.path.exists(
                possible_file
            ):

                actual_temp_file = possible_file

        # ----------------------------------------------------
        # Replace existing verified sample if necessary
        # ----------------------------------------------------

        os.replace(
            actual_temp_file,
            verified_file
        )

        # ----------------------------------------------------
        # Remove pending sample
        # ----------------------------------------------------

        os.remove(
            pending_file
        )

        return {
            "sample_id": sample_id,
            "true_label": int(true_label),
            "verified": True
        }


    # ========================================================
    # COUNT PENDING
    # ========================================================

    def count_pending(self):

        if not os.path.exists(
            PENDING_DIR
        ):

            return 0

        return len(
            [
                f
                for f in os.listdir(PENDING_DIR)
                if f.endswith(".npz")
            ]
        )


    # ========================================================
    # COUNT VERIFIED
    # ========================================================

    def count_verified(self):

        if not os.path.exists(
            VERIFIED_DIR
        ):

            return 0

        return len(
            [
                f
                for f in os.listdir(VERIFIED_DIR)
                if f.endswith(".npz")
            ]
        )


    # ========================================================
    # BUILD VERIFIED DATASET
    # ========================================================

    def build_verified_dataset(self):

        X = []
        y = []

        if not os.path.exists(
            VERIFIED_DIR
        ):

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

        for filename in os.listdir(
            VERIFIED_DIR
        ):

            if not filename.endswith(".npz"):
                continue

            filepath = os.path.join(
                VERIFIED_DIR,
                filename
            )

            try:

                # IMPORTANT:
                # Explicitly close the NPZ file.
                with np.load(
                    filepath,
                    allow_pickle=False
                ) as data:

                    features = np.asarray(
                        data["features"],
                        dtype=np.float32
                    ).copy()

                    metadata_raw = data[
                        "metadata"
                    ]

                    if isinstance(
                        metadata_raw,
                        np.ndarray
                    ):

                        metadata_raw = (
                            metadata_raw.item()
                        )

                    metadata = json.loads(
                        str(metadata_raw)
                    )

                # --------------------------------------------
                # Validate label
                # --------------------------------------------

                if "true_label" not in metadata:

                    continue

                label = int(
                    metadata["true_label"]
                )

                if label not in [0, 1]:

                    continue

                # --------------------------------------------
                # Validate features
                # --------------------------------------------

                if (
                    features.ndim != 1
                    or len(features)
                    != RAW_FEATURE_COUNT
                ):

                    continue

                X.append(
                    features
                )

                y.append(
                    label
                )

            except Exception as e:

                print(
                    f"Skipping {filename}: {e}"
                )

        if not X:

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

        return (
            np.asarray(
                X,
                dtype=np.float32
            ),
            np.asarray(
                y,
                dtype=np.float32
            )
        )