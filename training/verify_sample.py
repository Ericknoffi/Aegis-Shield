import sys
import os
import json
import subprocess

from feedback_collector import FeedbackCollector


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAINING_DIR = os.path.join(
    BASE_DIR,
    "training"
)

STATE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "feedback",
    "retrain_state.json"
)


# ============================================================
# RETRAINING CONFIG
# ============================================================

RETRAIN_THRESHOLD = 100


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_retrain_state():

    if not os.path.exists(STATE_FILE):

        return {
            "last_processed_verified_count": 0
        }

    try:

        with open(
            STATE_FILE,
            "r"
        ) as f:

            return json.load(f)

    except Exception:

        return {
            "last_processed_verified_count": 0
        }


def save_retrain_state(
    verified_count
):

    os.makedirs(
        os.path.dirname(STATE_FILE),
        exist_ok=True
    )

    state = {
        "last_processed_verified_count": verified_count
    }

    temp_file = STATE_FILE + ".tmp"

    with open(
        temp_file,
        "w"
    ) as f:

        json.dump(
            state,
            f,
            indent=4
        )

    os.replace(
        temp_file,
        STATE_FILE
    )


# ============================================================
# AUTOMATIC TRAINING PIPELINE
# ============================================================

def run_training_pipeline():

    print()
    print("=" * 60)
    print("AUTOMATIC RETRAINING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Retrain
    # --------------------------------------------------------

    print()
    print("[1/3] Retraining candidate model...")

    retrain_script = os.path.join(
        TRAINING_DIR,
        "retrain.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            retrain_script
        ],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print()
        print("Retraining failed.")
        print("Production model remains unchanged.")

        return False

    # --------------------------------------------------------
    # Step 2: Evaluate
    # --------------------------------------------------------

    print()
    print("[2/3] Evaluating candidate model...")

    evaluate_script = os.path.join(
        TRAINING_DIR,
        "evaluate.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            evaluate_script
        ],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print()
        print("Evaluation failed.")
        print("Production model remains unchanged.")

        return False

    # --------------------------------------------------------
    # Step 3: Promotion
    # --------------------------------------------------------

    print()
    print("[3/3] Checking promotion gate...")

    promote_script = os.path.join(
        TRAINING_DIR,
        "promote.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            promote_script
        ],
        cwd=BASE_DIR
    )

    if result.returncode != 0:

        print()
        print("Promotion failed.")
        print("Production model remains unchanged.")

        return False

    print()
    print("=" * 60)
    print("AUTOMATIC PIPELINE FINISHED")
    print("=" * 60)

    return True


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) != 3:

    print("Usage:")
    print(
        "python training\\verify_sample.py "
        "<sample_id> <label>"
    )

    print()
    print("label:")
    print("  0 = Benign")
    print("  1 = Malware")

    sys.exit(1)


sample_id = sys.argv[1]


# ============================================================
# VALIDATE LABEL
# ============================================================

try:

    true_label = int(
        sys.argv[2]
    )

except ValueError:

    print("Label must be 0 or 1.")
    sys.exit(1)


if true_label not in [0, 1]:

    print("Label must be 0 or 1.")
    sys.exit(1)


# ============================================================
# VERIFY SAMPLE
# ============================================================

collector = FeedbackCollector()

try:

    result = collector.verify_sample(
        sample_id=sample_id,
        true_label=true_label
    )

except Exception as e:

    print()
    print("Verification failed:")
    print(e)

    sys.exit(1)


print()
print("=" * 60)
print("SAMPLE VERIFIED")
print("=" * 60)

print(
    "Sample ID:",
    sample_id
)

print(
    "True label:",
    "Malware" if true_label == 1 else "Benign"
)

print(
    "Verified samples:",
    collector.count_verified()
)

print("=" * 60)


# ============================================================
# CHECK RETRAINING THRESHOLD
# ============================================================

verified_count = collector.count_verified()

state = load_retrain_state()

last_processed_count = state.get(
    "last_processed_verified_count",
    0
)

new_verified_samples = (
    verified_count -
    last_processed_count
)

print()
print(
    "New verified samples since last training:",
    new_verified_samples
)

print(
    "Retraining threshold:",
    RETRAIN_THRESHOLD
)


# ============================================================
# TRIGGER RETRAINING
# ============================================================

if new_verified_samples >= RETRAIN_THRESHOLD:

    print()
    print(
        "Retraining threshold reached."
    )

    pipeline_success = run_training_pipeline()

    # --------------------------------------------------------
    # Mark this batch as processed regardless of whether
    # promotion succeeded.
    #
    # The verified samples remain available as replay data.
    # --------------------------------------------------------

    save_retrain_state(
        verified_count
    )

    if pipeline_success:

        print()
        print(
            "Training cycle completed."
        )

    else:

        print()
        print(
            "Training cycle did not produce "
            "a new production model."
        )

else:

    remaining = (
        RETRAIN_THRESHOLD -
        new_verified_samples
    )

    print()
    print(
        f"{remaining} more verified samples "
        "needed before automatic retraining."
    )