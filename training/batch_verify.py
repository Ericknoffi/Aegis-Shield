"""
Batch Verification Utility for BODMAS Continual Learning Pipeline.
------------------------------------------------------------------
Automates bulk labeling of pending feedback samples (0 = Benign, 1 = Malware)
without requiring manual one-by-one commands.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from training.feedback_collector import (
        FeedbackCollector,
        PENDING_DIR,
        PENDING_BENIGN_DIR,
        PENDING_MALWARE_DIR,
        VERIFIED_DIR,
        VERIFIED_BENIGN_DIR,
        VERIFIED_MALWARE_DIR
    )
except ImportError:
    from feedback_collector import (
        FeedbackCollector,
        PENDING_DIR,
        PENDING_BENIGN_DIR,
        PENDING_MALWARE_DIR,
        VERIFIED_DIR,
        VERIFIED_BENIGN_DIR,
        VERIFIED_MALWARE_DIR
    )

from training.verify_sample import load_retrain_state, save_retrain_state, run_training_pipeline, RETRAIN_THRESHOLD


def list_npz_in_dir(directory: str) -> list[str]:
    """Return all sample filenames in a directory."""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".npz")]


def show_status(collector: FeedbackCollector):
    """Print a clean dashboard of pending and verified feedback pools."""
    pending_benign = len(list_npz_in_dir(PENDING_BENIGN_DIR))
    pending_malware = len(list_npz_in_dir(PENDING_MALWARE_DIR))
    total_pending = collector.count_pending()

    verified_benign = len(list_npz_in_dir(VERIFIED_BENIGN_DIR))
    verified_malware = len(list_npz_in_dir(VERIFIED_MALWARE_DIR))
    total_verified = collector.count_verified()

    state = load_retrain_state()
    last_processed = state.get("last_processed_verified_count", 0)
    new_verified = total_verified - last_processed

    print("\n" + "=" * 65)
    print("BODMAS FEEDBACK POOL DASHBOARD")
    print("=" * 65)
    print("  [PENDING SAMPLES] (Awaiting Ground-Truth Verification)")
    print(f"    [+] Predicted Benign : {pending_benign}")
    print(f"    [!] Predicted Malware: {pending_malware}")
    print(f"    [*] Total Pending    : {total_pending}")
    print("-" * 65)
    print("  [VERIFIED SAMPLES] (Ready for Continual Retraining)")
    print(f"    [+] Confirmed Benign (0) : {verified_benign}")
    print(f"    [!] Confirmed Malware (1): {verified_malware}")
    print(f"    [*] Total Verified       : {total_verified}")
    print("-" * 65)
    print(f"  [*] New Verified since last retrain : {new_verified} / {RETRAIN_THRESHOLD}")
    if new_verified >= RETRAIN_THRESHOLD:
        print("  [>] STATUS: Retraining threshold REACHED! Retrain can trigger.")
    else:
        print(f"  [>] STATUS: {RETRAIN_THRESHOLD - new_verified} more verified samples needed for auto-retrain.")
    print("=" * 65 + "\n")


def batch_verify_category(collector: FeedbackCollector, category: str, true_label: int, limit: int = None) -> int:
    """Verify all samples in a specific pending category (benign or malware)."""
    source_dir = PENDING_MALWARE_DIR if category == "malware" else PENDING_BENIGN_DIR
    files = list_npz_in_dir(source_dir)

    if limit and limit > 0:
        files = files[:limit]

    if not files:
        print(f"[*] No pending samples found in {category} folder.")
        return 0

    print(f"\n[*] Processing {len(files)} samples from 'pending/{category}' -> Label: {true_label} ({'Malware' if true_label == 1 else 'Benign'})...")

    success_count = 0
    for filename in files:
        sample_id = filename.removesuffix(".npz")
        try:
            collector.verify_sample(sample_id=sample_id, true_label=true_label)
            success_count += 1
        except Exception as e:
            print(f"  [!] Failed to verify {sample_id}: {e}")

    print(f"[+] Successfully verified and moved {success_count} / {len(files)} samples to 'verified/{'malware' if true_label == 1 else 'benign'}'.")
    return success_count


def batch_auto_predicted(collector: FeedbackCollector) -> int:
    """Auto-approve all pending files according to their predicted folder."""
    total_processed = 0
    # 1. Process pending benign -> true_label 0
    total_processed += batch_verify_category(collector, category="benign", true_label=0)
    # 2. Process pending malware -> true_label 1
    total_processed += batch_verify_category(collector, category="malware", true_label=1)
    return total_processed


def check_and_trigger_retrain(collector: FeedbackCollector):
    """Check if verified count crossed threshold and trigger retraining pipeline."""
    verified_count = collector.count_verified()
    state = load_retrain_state()
    last_processed = state.get("last_processed_verified_count", 0)
    new_verified = verified_count - last_processed

    if new_verified >= RETRAIN_THRESHOLD:
        print("\n" + "=" * 65)
        print("[!] RETRAINING THRESHOLD REACHED - LAUNCHING PIPELINE")
        print("=" * 65)
        pipeline_success = run_training_pipeline()
        save_retrain_state(verified_count)
        if pipeline_success:
            print("[+] Continual learning pipeline completed successfully!")
        else:
            print("[*] Pipeline completed. Active model preserved.")
    else:
        print(f"[*] {RETRAIN_THRESHOLD - new_verified} more verified samples required for automatic retraining.")


def main():
    parser = argparse.ArgumentParser(
        description="BODMAS Batch Verification & Bulk Labeling Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/batch_verify.py --status
  python training/batch_verify.py --two-step
  python training/batch_verify.py --two-step --vt-key <VIRUSTOTAL_API_KEY>
  python training/batch_verify.py --auto-predicted
  python training/batch_verify.py --category benign --label 0
  python training/batch_verify.py --category malware --label 1
        """
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Display summary dashboard of pending and verified feedback pools"
    )
    parser.add_argument(
        "--two-step",
        action="store_true",
        help="Run Two-Step Verification (Step 1: Authenticode Signature -> Step 2: VirusTotal Threat-Intel)"
    )
    parser.add_argument(
        "--vt-key",
        type=str,
        default=None,
        help="Optional VirusTotal API key for Step 2 threat-intel queries (or set VIRUSTOTAL_API_KEY env)"
    )
    parser.add_argument(
        "--auto-predicted",
        action="store_true",
        help="Auto-approve all pending files based on their predicted category (benign -> 0, malware -> 1)"
    )
    parser.add_argument(
        "--category",
        choices=["benign", "malware"],
        help="Filter pending samples by predicted category to bulk-verify"
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        help="Ground-truth label to assign (0 = Benign, 1 = Malware)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Bulk-verify ALL pending samples (requires --label)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of samples to process in this batch"
    )

    args = parser.parse_args()
    collector = FeedbackCollector()

    # Default action if no arguments provided: display status
    if len(sys.argv) == 1 or args.status:
        show_status(collector)
        return

    processed = 0

    if args.two_step:
        try:
            from training.two_step_verifier import TwoStepVerifier
        except ImportError:
            from two_step_verifier import TwoStepVerifier

        verifier = TwoStepVerifier(vt_api_key=args.vt_key)
        stats = verifier.run_batch_two_step()
        processed = stats["verified_benign"] + stats["verified_malware"]

    elif args.auto_predicted:
        processed = batch_auto_predicted(collector)

    elif args.category:
        if args.label is None:
            print("[!] Error: --category requires --label (0 or 1).")
            sys.exit(1)
        processed = batch_verify_category(collector, category=args.category, true_label=args.label, limit=args.limit)

    elif args.all:
        if args.label is None:
            print("[!] Error: --all requires --label (0 or 1).")
            sys.exit(1)
        processed += batch_verify_category(collector, category="benign", true_label=args.label, limit=args.limit)
        processed += batch_verify_category(collector, category="malware", true_label=args.label, limit=args.limit)

    else:
        parser.print_help()
        return

    if processed > 0:
        check_and_trigger_retrain(collector)


if __name__ == "__main__":
    main()
