import os
import sys
import json
import shutil
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

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# HELPERS
# ============================================================

def get_promoted_versions():

    versions = []

    if not os.path.exists(INFERENCE_DIR):
        return versions

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
        if not name.startswith("v"):
            continue

        try:
            raw_part = name[1:].split("_")[0].replace(".7z", "").replace(".zip", "")
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
# VALIDATE CANDIDATE
# ============================================================

def validate_candidate(
    candidate_version,
    current_version
):

    candidate_dir = get_candidate_dir(candidate_version)

    evaluation_path = os.path.join(
        candidate_dir,
        "evaluation.json"
    )

    version_path = os.path.join(
        candidate_dir,
        "version.json"
    )

    model_path = os.path.join(
        candidate_dir,
        "malware_mlp_top250.pth"
    )

    required_files = [
        evaluation_path,
        version_path,
        model_path
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required candidate file missing:\n"
                f"{path}"
            )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    with open(
        evaluation_path,
        "r",
        encoding="utf-8"
    ) as f:

        evaluation = json.load(f)

    evaluated_current = evaluation.get(
        "current_version"
    )

    evaluated_candidate = evaluation.get(
        "candidate_version"
    )

    if evaluated_current != f"v{current_version}":

        raise RuntimeError(
            "Evaluation was performed against "
            "a different promoted model.\n\n"
            f"Current promoted model : v{current_version}\n"
            f"Evaluation current     : {evaluated_current}"
        )

    if evaluated_candidate != f"v{candidate_version}":

        raise RuntimeError(
            "Evaluation belongs to a different "
            "candidate version."
        )

    # Check dragon_gate or fallback to promotion_gate for backwards compatibility
    dragon_gate = evaluation.get("dragon_gate") or evaluation.get("promotion_gate")

    if dragon_gate is None:

        raise RuntimeError(
            "dragon_gate / promotion_gate missing from evaluation.json."
        )

    passed = dragon_gate.get(
        "passed",
        False
    )

    if not passed:

        return False, evaluation

    # --------------------------------------------------------
    # VERSION METADATA
    # --------------------------------------------------------

    with open(
        version_path,
        "r",
        encoding="utf-8"
    ) as f:

        version_metadata = json.load(f)

    parent_version = version_metadata.get(
        "parent_version"
    )

    if parent_version != f"v{current_version}":

        raise RuntimeError(
            "Candidate parent version does not "
            "match the current production model.\n\n"
            f"Expected: v{current_version}\n"
            f"Found   : {parent_version}"
        )

    return True, evaluation


# ============================================================
# PROMOTE
# ============================================================

def promote(
    candidate_version,
    current_version
):

    candidate_dir = get_candidate_dir(candidate_version)

    promoted_dir = os.path.join(
        INFERENCE_DIR,
        f"v{candidate_version}_Promoted"
    )

    current_dir = os.path.join(
        INFERENCE_DIR,
        f"v{current_version}_Promoted"
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    if os.path.exists(promoted_dir):

        raise RuntimeError(
            f"Promoted directory already exists:\n"
            f"{promoted_dir}"
        )

    if not os.path.exists(candidate_dir):

        raise FileNotFoundError(
            f"Candidate directory not found:\n"
            f"{candidate_dir}"
        )

    if not os.path.exists(current_dir):

        raise FileNotFoundError(
            f"Current production directory not found:\n"
            f"{current_dir}"
        )

    # --------------------------------------------------------
    # COPY CANDIDATE
    # --------------------------------------------------------

    shutil.copytree(
        candidate_dir,
        promoted_dir
    )

    # --------------------------------------------------------
    # UPDATE VERSION STATUS
    # --------------------------------------------------------

    version_path = os.path.join(
        promoted_dir,
        "version.json"
    )

    with open(
        version_path,
        "r",
        encoding="utf-8"
    ) as f:

        version_metadata = json.load(f)

    version_metadata["status"] = "evolved_dragon"
    version_metadata["evolution"] = "CELESTIAL_DRAGON"
    version_metadata["dragon_gate_ascended"] = True

    version_metadata[
        "promoted_at"
    ] = datetime.now().isoformat()

    version_metadata[
        "previous_production_version"
    ] = f"v{current_version}"

    with open(
        version_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            version_metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # CREATE ACTIVE MODEL POINTER (REIGNING DRAGON)
    # --------------------------------------------------------

    active_path = os.path.join(
        INFERENCE_DIR,
        "ACTIVE_MODEL.json"
    )

    active_metadata = {

        "active_version":
            f"v{candidate_version}",

        "model_directory":
            f"v{candidate_version}_Promoted",

        "previous_version":
            f"v{current_version}",

        "status":
            "CELESTIAL_DRAGON_LIVE",

        "updated_at":
            datetime.now().isoformat()
    }

    temporary_active_path = (
        active_path + ".tmp"
    )

    with open(
        temporary_active_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            active_metadata,
            f,
            indent=4
        )

    # Atomic replacement of pointer
    os.replace(
        temporary_active_path,
        active_path
    )

    # --------------------------------------------------------
    # UPDATE CANDIDATE DIRECTORY WITH DRAGON ASCENDED TAG
    # --------------------------------------------------------

    candidate_version_file = os.path.join(
        candidate_dir,
        "version.json"
    )

    if os.path.exists(candidate_version_file):
        try:
            with open(candidate_version_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)

            c_data["deployment_status"] = "DRAGON_ASCENDED"
            c_data["evolution"] = "CELESTIAL_DRAGON"
            c_data["dragon_gate_status"] = "ASCENDED"
            c_data["promoted_at"] = datetime.now().isoformat()

            with open(candidate_version_file, "w", encoding="utf-8") as f:
                json.dump(c_data, f, indent=4)
        except Exception:
            pass

    # Write marker files (both DRAGON_ASCENDED.json and PROMOTED.json for backwards compatibility)
    dragon_tag = os.path.join(candidate_dir, "DRAGON_ASCENDED.json")
    with open(dragon_tag, "w", encoding="utf-8") as f:
        json.dump({
            "version": f"v{candidate_version}",
            "status": "DRAGON_ASCENDED",
            "evolution": "CELESTIAL_DRAGON",
            "dragon_gate_trial": "PASSED",
            "promoted_at": datetime.now().isoformat(),
            "previous_reigning_dragon": f"v{current_version}"
        }, f, indent=4)

    promoted_tag = os.path.join(candidate_dir, "PROMOTED.json")
    with open(promoted_tag, "w", encoding="utf-8") as f:
        json.dump({
            "version": f"v{candidate_version}",
            "status": "PROMOTED",
            "promoted_at": datetime.now().isoformat(),
            "previous_version": f"v{current_version}"
        }, f, indent=4)

    return promoted_dir


import unicodedata

def get_display_width(s):
    """Calculates terminal visual column width accounting for wide emoji & CJK characters."""
    width = 0
    for char in s:
        ea = unicodedata.east_asian_width(char)
        if ea in ('W', 'F') or ord(char) > 0x1F000 or ord(char) in (0x23F3, 0x2714, 0x2716):
            width += 2
        else:
            width += 1
    return width

def pad_cell(s, target_width, align="left"):
    """Pads a string with spaces based on its visual terminal display width."""
    cur_w = get_display_width(s)
    pad = max(0, target_width - cur_w)
    if align == "right":
        return (" " * pad) + s
    return s + (" " * pad)


# ============================================================
# LIST CANDIDATES (DRAGON GATE AUDIT)
# ============================================================

def list_candidates():
    """
    Displays a clear, perfectly aligned summary table of all candidate models in models/
    with their Dragon Gate Trial status (ASCENDED/FALLEN) and Evolution Realm.
    """
    col1_w = 18
    col2_w = 24
    col3_w = 24
    col4_w = 22

    hdr = pad_cell("Aspirant Model", col1_w) + pad_cell("Dragon Gate Trial", col2_w) + pad_cell("Evolution / Realm", col3_w) + pad_cell("Created At", col4_w) + "F1-Score"
    total_w = get_display_width(hdr)

    print()
    print("=" * total_w)
    banner = "🐉 THE DRAGON GATE (龙门) - ASPIRANT CANDIDATE AUDIT 🐉"
    banner_pad = max(0, (total_w - get_display_width(banner)) // 2)
    print(" " * banner_pad + banner)
    print("=" * total_w)

    if not os.path.exists(MODELS_DIR):
        print("No models directory found.")
        return

    versions = get_candidate_versions()
    if not versions:
        print("No candidate models found in models/ directory.")
        return

    try:
        current_promoted = f"v{get_current_promoted_version()}"
    except Exception:
        current_promoted = "None"

    print(f"Reigning Celestial Dragon (Active Production): {current_promoted}\n")
    print(hdr)
    print("-" * total_w)

    for v in versions:
        cand_dir = get_candidate_dir(v)
        v_file = os.path.join(cand_dir, "version.json")
        eval_file = os.path.join(cand_dir, "evaluation.json")
        metrics_file = os.path.join(cand_dir, "metrics.json")

        gate_status = "⏳ UNTESTED"
        deploy_status = "Mortal Realm"
        created_at = "Unknown"
        f1_score = "N/A"

        # Check directory timestamp if available
        if os.path.exists(cand_dir):
            try:
                mtime = os.path.getmtime(cand_dir)
                created_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        # Check if this model is the currently promoted production dragon
        if f"v{v}" == current_promoted or os.path.exists(os.path.join(INFERENCE_DIR, f"v{v}_Promoted")):
            deploy_status = "👑 DRAGON (LIVE)"
            gate_status = "🐉 ASCENDED"

        if os.path.exists(v_file):
            try:
                with open(v_file, "r", encoding="utf-8") as f:
                    v_meta = json.load(f)
                if "created_at" in v_meta:
                    created_at = v_meta["created_at"][:19].replace("T", " ")
                d_stat = v_meta.get("deployment_status", v_meta.get("status", "NOT_PROMOTED"))
                if d_stat in ("PROMOTED", "DRAGON_ASCENDED", "candidate_promoted", "evolved_dragon"):
                    deploy_status = "👑 DRAGON (LIVE)"
                    gate_status = "🐉 ASCENDED"
                if "dragon_gate_status" in v_meta:
                    dg = v_meta["dragon_gate_status"]
                    gate_status = "🐉 ASCENDED" if dg == "ASCENDED" else "🌊 FALLEN"
                elif "gate_status" in v_meta:
                    gate_status = "🐉 ASCENDED" if v_meta["gate_status"] == "PASSED" else "🌊 FALLEN"
                if "f1" in v_meta and f1_score == "N/A":
                    f1_score = f"{float(v_meta['f1']):.4f}"
            except Exception:
                pass

        if os.path.exists(eval_file):
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    e_meta = json.load(f)
                gate_data = e_meta.get("dragon_gate") or e_meta.get("promotion_gate", {})
                gate_passed = gate_data.get("passed", False)
                gate_status = "🐉 ASCENDED" if gate_passed else "🌊 FALLEN"
                f1_val = e_meta.get("candidate_metrics", {}).get("f1", None)
                if f1_val is not None:
                    f1_score = f"{f1_val:.4f}"
            except Exception:
                pass

        if f1_score == "N/A" and os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    m_meta = json.load(f)
                if "f1" in m_meta and m_meta["f1"] is not None:
                    f1_score = f"{float(m_meta['f1']):.4f}"
            except Exception:
                pass

        if os.path.exists(os.path.join(cand_dir, "DRAGON_ASCENDED.json")) or os.path.exists(os.path.join(cand_dir, "PROMOTED.json")):
            deploy_status = "👑 DRAGON (LIVE)"
            gate_status = "🐉 ASCENDED"

        folder_name = os.path.basename(cand_dir).replace(".7z", "").replace(".zip", "")
        display_name = f"v{v}" if folder_name in (f"v{v}_Candidate", f"v{v}") else f"v{v} ({folder_name})"

        row = pad_cell(display_name, col1_w) + pad_cell(gate_status, col2_w) + pad_cell(deploy_status, col3_w) + pad_cell(created_at, col4_w) + f1_score
        print(row)

    print("=" * total_w)


# ============================================================
# UI PROMOTION DIALOG (DRAGON GATE - LIGHT THEME)
# ============================================================

def show_promotion_dialog(
    candidate_version,
    current_version,
    evaluation
):
    """
    Shows a clean, modern light-themed GUI modal dialog when the candidate leaps over the Dragon Gate,
    giving the user the choice to Evolve to Dragon or Keep current production model.
    Falls back to terminal interactive prompt if GUI is unavailable.
    """
    user_choice = {"promote": False}

    try:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        root.title("🐉 Dragon Gate Ascension (龙门) - Model Evolution")
        root.geometry("660x540")
        root.minsize(620, 500)
        root.configure(bg="#F8FAFC")

        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (660 // 2)
        y = (root.winfo_screenheight() // 2) - (540 // 2)
        root.geometry(f"660x540+{x}+{y}")

        # Header Frame (White Card)
        header_frame = tk.Frame(root, bg="#FFFFFF", pady=18, padx=24, highlightbackground="#E2E8F0", highlightthickness=1)
        header_frame.pack(fill="x")

        # Status badge / title
        title_lbl = tk.Label(
            header_frame,
            text=f"🐉 Aspirant Model v{candidate_version} Leaped Over The Dragon Gate!",
            font=("Segoe UI", 12, "bold"),
            fg="#059669",
            bg="#FFFFFF"
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header_frame,
            text=f"Reigning Dragon: v{current_version}  ➔  Ascendant Dragon: v{candidate_version}",
            font=("Segoe UI", 9),
            fg="#64748B",
            bg="#FFFFFF"
        )
        sub_lbl.pack(anchor="w", pady=(4, 0))

        # Content Frame
        content_frame = tk.Frame(root, bg="#F8FAFC", padx=24, pady=16)
        content_frame.pack(fill="both", expand=True)

        prompt_lbl = tk.Label(
            content_frame,
            text="The aspirant has triumphed in all 4 trials of the Dragon Gate without regression.\nWould you like to evolve this candidate into the reigning Celestial Dragon for production?",
            font=("Segoe UI", 9),
            fg="#334155",
            bg="#F8FAFC",
            justify="left"
        )
        prompt_lbl.pack(anchor="w", pady=(0, 12))

        # Metrics Comparison Table (Card with border)
        table_card = tk.Frame(content_frame, bg="#FFFFFF", padx=12, pady=10, highlightbackground="#CBD5E1", highlightthickness=1)
        table_card.pack(fill="both", expand=True)

        # Table Header
        headers = ["Trial / Metric", f"v{current_version} (Current Dragon)", f"v{candidate_version} (Ascendant)", "Delta / Change"]
        col_widths = [16, 20, 18, 16]

        for col_idx, (header, width) in enumerate(zip(headers, col_widths)):
            lbl = tk.Label(
                table_card,
                text=header,
                font=("Segoe UI", 9, "bold"),
                fg="#0F172A",
                bg="#F1F5F9",
                width=width,
                padx=6,
                pady=6,
                anchor="w" if col_idx == 0 else "e"
            )
            lbl.grid(row=0, column=col_idx, sticky="ew", pady=(0, 4))

        curr_m = evaluation.get("current_metrics", {})
        cand_m = evaluation.get("candidate_metrics", {})
        metric_rows = [
            ("F1-Score", "f1"),
            ("Recall", "recall"),
            ("Precision", "precision"),
            ("PR-AUC", "pr_auc"),
            ("Accuracy", "accuracy")
        ]

        for row_idx, (display_name, key) in enumerate(metric_rows, start=1):
            curr_val = curr_m.get(key, 0.0)
            cand_val = cand_m.get(key, 0.0)
            delta = cand_val - curr_val

            delta_str = f"{delta:+.4f}"
            row_bg = "#FFFFFF" if row_idx % 2 != 0 else "#F8FAFC"
            delta_color = "#059669" if delta > 0 else ("#0F172A" if delta == 0 else "#DC2626")

            tk.Label(
                table_card, text=display_name, font=("Segoe UI", 9),
                fg="#334155", bg=row_bg, padx=6, pady=4, anchor="w"
            ).grid(row=row_idx, column=0, sticky="ew")

            tk.Label(
                table_card, text=f"{curr_val:.4f}", font=("Segoe UI", 9),
                fg="#64748B", bg=row_bg, padx=6, pady=4, anchor="e"
            ).grid(row=row_idx, column=1, sticky="ew")

            tk.Label(
                table_card, text=f"{cand_val:.4f}", font=("Segoe UI", 9, "bold"),
                fg="#0F172A", bg=row_bg, padx=6, pady=4, anchor="e"
            ).grid(row=row_idx, column=2, sticky="ew")

            tk.Label(
                table_card, text=delta_str, font=("Segoe UI", 9, "bold"),
                fg=delta_color, bg=row_bg, padx=6, pady=4, anchor="e"
            ).grid(row=row_idx, column=3, sticky="ew")

        # Footer Action Bar
        btn_frame = tk.Frame(root, bg="#FFFFFF", padx=24, pady=14, highlightbackground="#E2E8F0", highlightthickness=1)
        btn_frame.pack(fill="x")

        def on_promote():
            user_choice["promote"] = True
            root.destroy()

        def on_cancel():
            user_choice["promote"] = False
            root.destroy()

        promote_btn = tk.Button(
            btn_frame,
            text=f"🐉 Evolve to Dragon (Promote v{candidate_version})",
            font=("Segoe UI", 9, "bold"),
            bg="#059669",
            fg="#FFFFFF",
            activebackground="#047857",
            activeforeground="#FFFFFF",
            relief="flat",
            padx=16,
            pady=7,
            cursor="hand2",
            command=on_promote
        )
        promote_btn.pack(side="right", padx=(10, 0))

        cancel_btn = tk.Button(
            btn_frame,
            text=f"Keep in Mortal Realm (v{current_version})",
            font=("Segoe UI", 9),
            bg="#F1F5F9",
            fg="#475569",
            activebackground="#E2E8F0",
            activeforeground="#1E293B",
            highlightbackground="#CBD5E1",
            highlightthickness=1,
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=on_cancel
        )
        cancel_btn.pack(side="right")

        # Ensure window is front and active
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        root.mainloop()

        return user_choice["promote"]

    except Exception:
        # Fallback to interactive CLI prompt if Tkinter fails
        print()
        print("-" * 75)
        print(f"🐉 DRAGON GATE ASCENSION: Aspirant v{candidate_version} passed all 4 trials.")
        print("-" * 75)
        try:
            choice = input(
                f"Evolve aspirant v{candidate_version} into Celestial Dragon (Production)? [Y/n]: "
            ).strip().lower()
            return choice in ("y", "yes", "")
        except Exception:
            return False


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Promote/Evolve a validated continual-learning "
            "aspirant model through the Dragon Gate."
        )
    )

    parser.add_argument(
        "--candidate",
        type=int,
        default=None,
        help=(
            "Candidate version number. "
            "If omitted, latest candidate is used."
        )
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Automatically approve promotion without prompt."
    )

    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Use terminal confirmation instead of GUI dialog."
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all candidate models with their Dragon Gate trial status (ASCENDED/FALLEN)."
    )

    args = parser.parse_args()

    if args.list:
        list_candidates()
        return

    print()
    print("🐉" + "=" * 70 + "🐉")
    print("           THE DRAGON GATE ASCENSION (龙门 - LONGMEN)           ")
    print("🐉" + "=" * 70 + "🐉")

    # --------------------------------------------------------
    # CURRENT MODEL
    # --------------------------------------------------------

    current_version = (
        get_current_promoted_version()
    )

    # --------------------------------------------------------
    # CANDIDATE
    # --------------------------------------------------------

    if args.candidate is None:

        candidate_version = (
            get_latest_candidate_version()
        )

    else:

        candidate_version = args.candidate

    print()
    print(
        "Reigning Celestial Dragon :",
        f"v{current_version}"
    )

    print(
        "Aspirant Candidate        :",
        f"v{candidate_version}"
    )

    # --------------------------------------------------------
    # VALIDATE AT THE DRAGON GATE
    # --------------------------------------------------------

    print()
    print(
        "Summoning Dragon Gate trial verification..."
    )

    passed, evaluation = validate_candidate(
        candidate_version,
        current_version
    )

    gate = evaluation.get("dragon_gate") or evaluation.get("promotion_gate", {})

    # --------------------------------------------------------
    # REJECT (FAILED DRAGON GATE)
    # --------------------------------------------------------

    if not passed:

        print()
        print("🌊" + "=" * 70 + "🌊")
        print("     DRAGON GATE TRIAL: FAILED (REMAINS MORTAL ASPIRANT)     ")
        print("🌊" + "=" * 70 + "🌊")

        print()
        print(
            f"v{candidate_version} could not leap over the Dragon Gate."
        )

        print()
        print(
            "Reigning Celestial Dragon continues protecting production:"
        )

        print(
            f"v{current_version}"
        )

        print()
        print(
            "Dragon Gate Trials Breakdown:"
        )

        print(
            "Trial 1: Recall >= Reigning Dragon   :",
            gate.get("recall_ok")
        )

        print(
            "Trial 2: Precision >= Reigning Dragon:",
            gate.get("precision_ok")
        )

        print(
            "Trial 3: F1-Score > Reigning Dragon  :",
            gate.get("f1_improved")
        )

        print(
            "Trial 4: PR-AUC >= Reigning Dragon   :",
            gate.get("pr_auc_ok")
        )

        print()
        print("=" * 72)

        return

    # --------------------------------------------------------
    # GATE PASSED -> USER CHOICE (UI / CLI)
    # --------------------------------------------------------

    print()
    print("🐉" + "=" * 70 + "🐉")
    print("      DRAGON GATE TRIAL: ASCENDED! (WORTHY TO EVOLVE)       ")
    print("🐉" + "=" * 70 + "🐉")
    print(f"Aspirant candidate v{candidate_version} conquered all 4 trials.")

    if not args.yes:
        if args.no_ui:
            try:
                choice = input(
                    f"\nEvolve aspirant v{candidate_version} into Celestial Dragon (Production)? [Y/n]: "
                ).strip().lower()
                approved = choice in ("y", "yes", "")
            except Exception:
                approved = False
        else:
            print("\nDisplaying Dragon Gate Ascension confirmation dialog...")
            approved = show_promotion_dialog(
                candidate_version,
                current_version,
                evaluation
            )

        if not approved:
            print()
            print("=" * 72)
            print("ASCENSION DEFERRED: KEPT IN MORTAL REALM")
            print("=" * 72)
            print(f"Aspirant v{candidate_version} remains preserved in models/v{candidate_version}_Candidate/")
            print(f"Reigning Celestial Dragon remains: v{current_version}")
            print("=" * 72)
            return

    # --------------------------------------------------------
    # EXECUTE EVOLUTION / PROMOTION
    # --------------------------------------------------------

    print()
    print("🐉 Channeling Dragon Gate energy... Evolving into Celestial Dragon...")

    promoted_dir = promote(
        candidate_version,
        current_version
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print()
    print("🐉" + "=" * 70 + "🐉")
    print("            EVOLUTION COMPLETE: CELESTIAL DRAGON BORN            ")
    print("🐉" + "=" * 70 + "🐉")

    print()
    print(
        "Previous Dragon :",
        f"v{current_version}"
    )

    print(
        "Reigning Dragon :",
        f"v{candidate_version} (Active Production)"
    )

    print()
    print(
        "Dragon Temple (Model Location):"
    )

    print(
        promoted_dir
    )

    print()
    print(
        "ACTIVE_MODEL.json pointer updated."
    )

    print()
    print(
        f"v{current_version} is peacefully preserved in inference/v{current_version}_Promoted/."
    )

    print(
        "Instant rollback to prior Dragon remains possible at any time."
    )

    print()
    print("🐉" + "=" * 70 + "🐉")


if __name__ == "__main__":

    main()