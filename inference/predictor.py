import os
import json
import warnings
import joblib
import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

try:
    import onnxruntime as ort
    # Suppress verbose ONNX Runtime logging
    ort.set_default_logger_severity(3)
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


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

ACTIVE_MODEL_FILE = os.path.join(
    INFERENCE_DIR,
    "ACTIVE_MODEL.json"
)


# ============================================================
# CONSTANTS
# ============================================================

RAW_FEATURE_COUNT = 2381
PROCESSED_FEATURE_COUNT = 2332
FINAL_FEATURE_COUNT = 250

DEFAULT_THRESHOLD = 0.55


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
# FIND LATEST PROMOTED MODEL
# ============================================================

def find_latest_promoted_model():

    promoted_versions = []

    for name in os.listdir(INFERENCE_DIR):

        if not name.endswith("_Promoted"):
            continue

        path = os.path.join(
            INFERENCE_DIR,
            name
        )

        if not os.path.isdir(path):
            continue

        try:
            version_number = int(
                name.replace("v", "")
                    .replace("_Promoted", "")
            )

            promoted_versions.append(
                (version_number, path)
            )

        except ValueError:
            continue

    if not promoted_versions:

        raise FileNotFoundError(
            "No promoted model found inside inference/"
        )

    promoted_versions.sort(
        key=lambda x: x[0]
    )

    return promoted_versions[-1][1]


# ============================================================
# GET ACTIVE MODEL DIRECTORY
# ============================================================

def get_active_model_dir():

    # If ACTIVE_MODEL.json exists, use it
    if os.path.exists(ACTIVE_MODEL_FILE):

        with open(
            ACTIVE_MODEL_FILE,
            "r"
        ) as f:

            active_config = json.load(f)

        model_path = active_config.get(
            "model_path"
        )

        if model_path:

            if not os.path.isabs(model_path):

                model_path = os.path.join(
                    BASE_DIR,
                    model_path
                )

            model_path = os.path.abspath(
                model_path
            )

            if os.path.isdir(model_path):

                return model_path

            raise FileNotFoundError(
                f"Active model directory does not exist:\n"
                f"{model_path}"
            )

    # No ACTIVE_MODEL.json yet.
    # Fall back to latest promoted model.
    return os.path.abspath(
        find_latest_promoted_model()
    )


# ============================================================
# PREDICTOR
# ============================================================

class MalwarePredictor:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None
        self.scaler = None

        self.non_constant_positions = None
        self.top250_positions = None

        self.model_dir = None
        self.model_version = None
        self.threshold = DEFAULT_THRESHOLD

        self.load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(self):

        self.model_dir = get_active_model_dir()

        print()
        print("=" * 60)
        print("Loading production malware detector")
        print("=" * 60)

        print(
            "Model directory:"
        )
        print(
            self.model_dir
        )

        # ----------------------------------------------------
        # Load model configuration
        # ----------------------------------------------------

        config_file = os.path.join(
            self.model_dir,
            "model_config.json"
        )

        if not os.path.exists(config_file):

            raise FileNotFoundError(
                f"Missing model_config.json:\n"
                f"{config_file}"
            )

        with open(
            config_file,
            "r"
        ) as f:

            config = json.load(f)

        input_dim = config.get(
            "input_dim",
            FINAL_FEATURE_COUNT
        )

        self.threshold = config.get(
            "threshold",
            DEFAULT_THRESHOLD
        )

        # ----------------------------------------------------
        # Load version
        # ----------------------------------------------------

        version_file = os.path.join(
            self.model_dir,
            "version.json"
        )

        if os.path.exists(version_file):

            with open(
                version_file,
                "r"
            ) as f:

                version_info = json.load(f)

            self.model_version = version_info.get(
                "version",
                os.path.basename(self.model_dir)
            )

        else:

            self.model_version = os.path.basename(
                self.model_dir
            )

        # ----------------------------------------------------
        # Clean version name if necessary
        # ----------------------------------------------------

        if self.model_version.endswith(
            "_Promoted"
        ):

            self.model_version = (
                self.model_version
                .replace("_Promoted", "")
            )

        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        scaler_file = os.path.join(
            self.model_dir,
            "scaler.pkl"
        )

        if not os.path.exists(scaler_file):

            raise FileNotFoundError(
                f"Missing scaler.pkl:\n"
                f"{scaler_file}"
            )

        self.scaler = joblib.load(
            scaler_file
        )

        # ----------------------------------------------------
        # Load feature mask
        # ----------------------------------------------------

        feature_mask_file = os.path.join(
            self.model_dir,
            "feature_mask.pkl"
        )

        if not os.path.exists(
            feature_mask_file
        ):

            raise FileNotFoundError(
                f"Missing feature_mask.pkl:\n"
                f"{feature_mask_file}"
            )

        feature_mask = joblib.load(
            feature_mask_file
        )

        non_constant_features = feature_mask[
            "non_constant_features"
        ]

        self.non_constant_positions = np.asarray(
            [
                int(name[1:]) - 1
                for name in non_constant_features
            ],
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Load top 250 positions
        # ----------------------------------------------------

        top250_file = os.path.join(
            self.model_dir,
            "top250_positions.pkl"
        )

        if not os.path.exists(
            top250_file
        ):

            raise FileNotFoundError(
                f"Missing top250_positions.pkl:\n"
                f"{top250_file}"
            )

        self.top250_positions = np.asarray(
            joblib.load(top250_file),
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Validate preprocessing
        # ----------------------------------------------------

        if len(
            self.non_constant_positions
        ) != PROCESSED_FEATURE_COUNT:

            raise ValueError(
                "Feature mask mismatch. "
                f"Expected {PROCESSED_FEATURE_COUNT}, "
                f"got {len(self.non_constant_positions)}"
            )

        if len(
            self.top250_positions
        ) != FINAL_FEATURE_COUNT:

            raise ValueError(
                "Top-250 feature mismatch. "
                f"Expected {FINAL_FEATURE_COUNT}, "
                f"got {len(self.top250_positions)}"
            )

        scaler_features = getattr(
            self.scaler,
            "n_features_in_",
            None
        )

        if scaler_features != PROCESSED_FEATURE_COUNT:

            raise ValueError(
                "Scaler feature mismatch. "
                f"Expected {PROCESSED_FEATURE_COUNT}, "
                f"got {scaler_features}"
            )

        # ----------------------------------------------------
        # Load neural network (ONNX Runtime or PyTorch)
        # ----------------------------------------------------

        onnx_file = os.path.join(
            self.model_dir,
            "malware_mlp_top250.onnx"
        )

        model_file = os.path.join(
            self.model_dir,
            "malware_mlp_top250.pth"
        )

        self.onnx_session = None
        self.backend = None

        if ONNX_AVAILABLE and os.path.exists(onnx_file):
            try:
                # Fast C++ ONNX Runtime Engine
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = 2
                self.onnx_session = ort.InferenceSession(
                    onnx_file,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                self.backend = "ONNX Runtime (Ultra-Fast C++ Engine)"
            except Exception as e:
                self.onnx_session = None

        # Fallback to PyTorch if ONNX is not used
        if self.onnx_session is None:
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"Missing model file:\n{model_file}"
                )

            self.model = MalwareMLP(
                input_dim=input_dim
            )

            checkpoint = torch.load(
                model_file,
                map_location=self.device
            )

            # Support multiple checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model.load_state_dict(
                state_dict
            )

            self.model.to(
                self.device
            )

            self.model.eval()
            self.backend = f"PyTorch ({str(self.device).upper()})"

        # ----------------------------------------------------
        # Validate input dimension
        # ----------------------------------------------------

        if input_dim != FINAL_FEATURE_COUNT:

            raise ValueError(
                f"Model input dimension must be "
                f"{FINAL_FEATURE_COUNT}, "
                f"got {input_dim}"
            )

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        print(
            "Version:",
            self.model_version
        )

        print(
            "Backend:",
            self.backend
        )

        print(
            "Raw features:",
            RAW_FEATURE_COUNT
        )

        print(
            "Processed features:",
            PROCESSED_FEATURE_COUNT
        )

        print(
            "Selected features:",
            FINAL_FEATURE_COUNT
        )

        print(
            "Threshold:",
            self.threshold
        )

        print("=" * 60)
        print()


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        raw_features
    ):

        raw_features = np.asarray(
            raw_features,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if raw_features.ndim != 1:

            raise ValueError(
                f"Expected 1D feature vector, "
                f"got shape {raw_features.shape}"
            )

        if len(raw_features) != RAW_FEATURE_COUNT:

            raise ValueError(
                f"Expected {RAW_FEATURE_COUNT} raw features, "
                f"got {len(raw_features)}"
            )

        # ----------------------------------------------------
        # Remove constant features
        # ----------------------------------------------------

        processed = raw_features[
            self.non_constant_positions
        ]

        # ----------------------------------------------------
        # StandardScaler
        # ----------------------------------------------------

        processed = self.scaler.transform(
            processed.reshape(
                1,
                -1
            )
        )

        # ----------------------------------------------------
        # Select top 250
        # ----------------------------------------------------

        processed = processed[
            0,
            self.top250_positions
        ]

        return processed.astype(np.float32)


    # ========================================================
    # SINGLE PREDICTION
    # ========================================================

    def predict_sample(
        self,
        raw_features
    ):

        x = self.preprocess(
            raw_features
        )

        if self.onnx_session is not None:
            # ONNX Runtime Fast Path
            input_tensor = np.expand_dims(x, axis=0)
            logits = self.onnx_session.run(
                None,
                {self.onnx_input_name: input_tensor}
            )[0]
            logit_val = float(logits.flatten()[0])
            probability = float(1.0 / (1.0 + np.exp(-logit_val)))
        else:
            # PyTorch Fallback Path
            tensor = torch.tensor(
                x,
                dtype=torch.float32,
                device=self.device
            )
            with torch.no_grad():
                logits = self.model(
                    tensor.unsqueeze(0)
                )
                probability = torch.sigmoid(
                    logits
                ).item()

        prediction = (
            "Malware"
            if probability >= self.threshold
            else "Benign"
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "threshold": self.threshold,
            "model_version": self.model_version,
            "backend": self.backend
        }


    # ========================================================
    # COMPATIBILITY PREDICT METHOD
    # ========================================================

    def predict(
        self,
        raw_features
    ):

        # Check whether a new model was promoted
        self.reload_if_changed()

        return self.predict_sample(
            raw_features
        )


    # ========================================================
    # BATCH PREDICTION
    # ========================================================

    def predict_batch(
        self,
        raw_features_batch
    ):

        raw_features_batch = np.asarray(
            raw_features_batch,
            dtype=np.float32
        )

        if raw_features_batch.ndim != 2:

            raise ValueError(
                "Batch input must be 2D."
            )

        if (
            raw_features_batch.shape[1]
            != RAW_FEATURE_COUNT
        ):

            raise ValueError(
                f"Expected {RAW_FEATURE_COUNT} "
                f"features per sample, "
                f"got {raw_features_batch.shape[1]}"
            )

        # ----------------------------------------------------
        # Remove constant features
        # ----------------------------------------------------

        processed = raw_features_batch[
            :,
            self.non_constant_positions
        ]

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        processed = self.scaler.transform(
            processed
        )

        # ----------------------------------------------------
        # Top 250
        # ----------------------------------------------------

        processed = processed[
            :,
            self.top250_positions
        ]

        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        x = torch.tensor(
            processed,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            logits = self.model(x)

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

        predictions = np.where(
            probabilities >= self.threshold,
            "Malware",
            "Benign"
        )

        return [
            {
                "prediction": prediction,
                "probability": float(probability),
                "threshold": self.threshold,
                "model_version": self.model_version
            }

            for prediction, probability
            in zip(
                predictions,
                probabilities
            )
        ]


    # ========================================================
    # RELOAD IF PRODUCTION MODEL CHANGED
    # ========================================================

    def reload_if_changed(self):

        current_model_dir = get_active_model_dir()

        current_model_dir = os.path.abspath(
            current_model_dir
        )

        if current_model_dir != self.model_dir:

            print()
            print("=" * 60)
            print("NEW PRODUCTION MODEL DETECTED")
            print("=" * 60)

            print(
                "Old:",
                self.model_dir
            )

            print(
                "New:",
                current_model_dir
            )

            self.load_model()

            return True

        return False


# ============================================================
# GLOBAL PREDICTOR
# ============================================================

predictor = MalwarePredictor()


# ============================================================
# SIMPLE FUNCTION WRAPPER
# ============================================================

def predict(
    raw_features
):

    predictor.reload_if_changed()

    return predictor.predict_sample(
        raw_features
    )