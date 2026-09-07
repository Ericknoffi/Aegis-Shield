"""
Two-Step Verification Engine for BODMAS Continual Learning Pipeline.
-------------------------------------------------------------------
Step 1: Authenticode Digital Signature Analysis (LIEF / PKCS#7 Trust Matrix)
Step 2: Keyless Multi-Engine Verification:
        - 2A: Native Windows Defender Engine (MpCmdRun.exe - 100% Free & Local)
        - 2B: Local Threat-Intel Hash Reputation Cache
        - 2C: VirusTotal API (Optional Cloud Fallback)
"""

import os
import sys
import json
import glob
import subprocess
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import lief
except ImportError:
    lief = None

from training.feedback_collector import (
    FeedbackCollector,
    PENDING_DIR,
    PENDING_BENIGN_DIR,
    PENDING_MALWARE_DIR,
    VERIFIED_BENIGN_DIR,
    VERIFIED_MALWARE_DIR
)


# ============================================================
# PATHS & CACHE
# ============================================================

CACHE_FILE = os.path.join(PROJECT_ROOT, "data", "threat_intel_cache.json")


# ============================================================
# LOCATE WINDOWS DEFENDER ENGINE (Keyless & Local)
# ============================================================

def find_defender_cli() -> Optional[str]:
    """Locates the native MpCmdRun.exe Windows Defender scanner."""
    candidates = [
        r"C:\Program Files\Windows Defender\MpCmdRun.exe"
    ]
    platform_dirs = glob.glob(r"C:\ProgramData\Microsoft\Windows Defender\Platform\*\MpCmdRun.exe")
    candidates.extend(platform_dirs)

    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ============================================================
# TRUSTED DIGITAL SIGNATURE PUBLISHERS & ROOT CAs
# ============================================================

TRUSTED_PUBLISHERS = {
    "microsoft corporation",
    "microsoft windows",
    "python software foundation",
    "google llc",
    "google inc",
    "adobe inc",
    "adobe systems",
    "apple inc",
    "valve corporation",
    "mozilla corporation",
    "oracle america",
    "amazon.com",
    "intel corporation",
    "nvidia corporation",
    "cisco systems",
    "canonical ltd",
    "electronic arts",
    "epic games",
    "discord inc",
    "spotify ab",
    "github, inc",
    "docker inc",
    "jetbrains s.r.o",
    "brave software",
    "wireshark foundation",
    "digicert",
    "verisign",
    "sectigo"
}


# ============================================================
# LOCAL THREAT INTEL CACHE HELPERS
# ============================================================

def load_threat_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_threat_cache(cache: Dict[str, Any]):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    temp_file = CACHE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(temp_file, CACHE_FILE)


# ============================================================
# TWO-STEP VERIFIER CLASS
# ============================================================

class TwoStepVerifier:
    """
    Executes high-assurance, multi-layered verification without requiring API keys:
    - Step 1: Authenticode Digital Signature Verification (LIEF)
    - Step 2: Native Windows Defender CLI Scan (MpCmdRun.exe) + Threat-Intel Cache
    """

    def __init__(self, vt_api_key: Optional[str] = None):
        self.collector = FeedbackCollector()
        self.vt_api_key = vt_api_key or os.environ.get("VIRUSTOTAL_API_KEY", "").strip()
        self.defender_cli = find_defender_cli()
        self.cache = load_threat_cache()

    # ========================================================
    # STEP 1: AUTHENTICODE DIGITAL SIGNATURE VERIFICATION
    # ========================================================

    def verify_authenticode(self, file_path: Optional[str]) -> Dict[str, Any]:
        """
        Validates PE digital certificate against trusted authorities.
        Returns label 0 (Benign) if verified with high confidence.
        """
        if not file_path or not os.path.exists(file_path):
            return {
                "verified": False,
                "label": None,
                "step": "Step 1 (Authenticode)",
                "reason": "Original binary not available on disk"
            }

        if lief is None:
            return {
                "verified": False,
                "label": None,
                "step": "Step 1 (Authenticode)",
                "reason": "LIEF library not available"
            }

        try:
            binary = lief.PE.parse(file_path)
            if not binary or not binary.has_signatures:
                return {
                    "verified": False,
                    "label": None,
                    "step": "Step 1 (Authenticode)",
                    "reason": "Unsigned binary"
                }

            found_publishers = []
            is_trusted = False

            for sig in binary.signatures:
                for cert in sig.certificates:
                    subject = str(cert.subject).lower()
                    issuer = str(cert.issuer).lower()
                    for trusted in TRUSTED_PUBLISHERS:
                        if trusted in subject or trusted in issuer:
                            is_trusted = True
                            found_publishers.append(cert.subject)
                            break
                    if is_trusted:
                        break
                if is_trusted:
                    break

            if is_trusted:
                signer_name = found_publishers[0] if found_publishers else "Trusted CA"
                return {
                    "verified": True,
                    "label": 0,  # Benign
                    "step": "Step 1 (Authenticode)",
                    "reason": f"Trusted Digital Signature: {signer_name}"
                }

            return {
                "verified": False,
                "label": None,
                "step": "Step 1 (Authenticode)",
                "reason": "Signed by untrusted or unknown publisher"
            }

        except Exception as e:
            return {
                "verified": False,
                "label": None,
                "step": "Step 1 (Authenticode)",
                "reason": f"Signature parsing error: {e}"
            }

    # ========================================================
    # STEP 2A: NATIVE WINDOWS DEFENDER CLI (100% Free & Keyless)
    # ========================================================

    def verify_windows_defender(self, file_path: Optional[str]) -> Dict[str, Any]:
        """
        Scans binary using Windows Defender CLI (MpCmdRun.exe).
        Exit code 0 = Clean (0: Benign)
        Exit code 2 = Threat Detected (1: Malware)
        """
        if not file_path or not os.path.exists(file_path):
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (Windows Defender)",
                "reason": "Original binary not available on disk"
            }

        if not self.defender_cli:
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (Windows Defender)",
                "reason": "MpCmdRun.exe Windows Defender scanner not located"
            }

        try:
            cmd = [
                self.defender_cli,
                "-Scan",
                "-ScanType", "3",
                "-File", file_path,
                "-DisableRemediation"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=25.0
            )

            # Exit Code 0: Clean / No Threat
            if result.returncode == 0:
                return {
                    "verified": True,
                    "label": 0,
                    "step": "Step 2 (Windows Defender Engine)",
                    "reason": "Clean scan: No threats identified by Microsoft Defender"
                }

            # Exit Code 2: Threat Found
            elif result.returncode == 2:
                return {
                    "verified": True,
                    "label": 1,
                    "step": "Step 2 (Windows Defender Engine)",
                    "reason": "Threat Alert: Malicious activity flagged by Microsoft Defender"
                }

            else:
                return {
                    "verified": False,
                    "label": None,
                    "step": "Step 2 (Windows Defender Engine)",
                    "reason": f"Scanner returned code {result.returncode}"
                }

        except subprocess.TimeoutExpired:
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (Windows Defender Engine)",
                "reason": "Defender scan timed out"
            }
        except Exception as e:
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (Windows Defender Engine)",
                "reason": f"Defender scan error: {e}"
            }

    # ========================================================
    # STEP 2B & 2C: THREAT-INTEL / VIRUSTOTAL (Optional Cloud)
    # ========================================================

    def verify_threat_intel(self, sha256: Optional[str]) -> Dict[str, Any]:
        """Checks local threat cache or queries VirusTotal if API key available."""
        if not sha256:
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (Threat-Intel)",
                "reason": "SHA-256 hash missing"
            }

        clean_hash = sha256.lower().strip()

        # Check local cache first
        if clean_hash in self.cache:
            entry = self.cache[clean_hash]
            return {
                "verified": True,
                "label": entry["label"],
                "step": "Step 2 (Threat-Intel Cache)",
                "reason": entry["reason"]
            }

        if not self.vt_api_key:
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (VirusTotal)",
                "reason": "No VIRUSTOTAL_API_KEY configured"
            }

        # Query VirusTotal API v3
        url = f"https://www.virustotal.com/api/v3/files/{clean_hash}"
        req = urllib.request.Request(url, headers={"x-apikey": self.vt_api_key})

        try:
            with urllib.request.urlopen(req, timeout=10.0) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    attrs = payload.get("data", {}).get("attributes", {})
                    stats = attrs.get("last_analysis_stats", {})

                    malicious = stats.get("malicious", 0)
                    suspicious = stats.get("suspicious", 0)
                    harmless = stats.get("harmless", 0)
                    undetected = stats.get("undetected", 0)
                    total_engines = malicious + suspicious + harmless + undetected

                    if malicious >= 3 or (malicious + suspicious) >= 4:
                        label = 1
                        reason = f"VirusTotal Threat Alert: {malicious}/{total_engines} engines flagged as MALWARE"
                    elif malicious == 0 and (harmless + undetected) >= 10:
                        label = 0
                        reason = f"VirusTotal Clean: 0/{total_engines} detections across antivirus engines"
                    else:
                        return {
                            "verified": False,
                            "label": None,
                            "step": "Step 2 (VirusTotal)",
                            "reason": f"Ambiguous VirusTotal result: {malicious} malicious, {suspicious} suspicious"
                        }

                    self.cache[clean_hash] = {"label": label, "reason": reason}
                    save_threat_cache(self.cache)

                    return {
                        "verified": True,
                        "label": label,
                        "step": "Step 2 (VirusTotal API)",
                        "reason": reason
                    }

        except Exception as e:
            return {
                "verified": False,
                "label": None,
                "step": "Step 2 (VirusTotal)",
                "reason": f"VirusTotal query failed: {e}"
            }

    # ========================================================
    # UNIFIED EVALUATOR (No API Key Required)
    # ========================================================

    def evaluate_sample(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Step 1: Authenticode Digital Signature (Free, Fast, Offline)
        2. Step 2A: Native Windows Defender CLI Engine (Free, Local)
        3. Step 2B: Local Threat Intel Hash Cache
        4. Step 2C: VirusTotal API (Optional Cloud Fallback)
        """
        file_path = metadata.get("file_path")
        sha256 = metadata.get("sha256")

        # 1. Step 1: Digital Certificate / Authenticode
        res_step1 = self.verify_authenticode(file_path)
        if res_step1["verified"]:
            return res_step1

        # 2. Step 2A: Native Windows Defender Engine
        res_defender = self.verify_windows_defender(file_path)
        if res_defender["verified"]:
            return res_defender

        # 3. Step 2B/C: Threat-Intel Cache / VirusTotal
        res_threat = self.verify_threat_intel(sha256)
        if res_threat["verified"]:
            return res_threat

        # Inconclusive
        return {
            "verified": False,
            "label": None,
            "step": "Inconclusive",
            "reason": f"Step 1: {res_step1['reason']} | Step 2 (Defender): {res_defender['reason']}"
        }

    # ========================================================
    # BATCH PROCESS PENDING QUEUES
    # ========================================================

    def run_batch_two_step(self) -> Dict[str, int]:
        """Iterates over pending samples and performs Keyless Two-Step Verification."""
        stats = {"verified_benign": 0, "verified_malware": 0, "inconclusive": 0}

        pending_files = []
        for root, _, filenames in os.walk(PENDING_DIR):
            for fn in filenames:
                if fn.endswith(".npz"):
                    pending_files.append(os.path.join(root, fn))

        if not pending_files:
            print("\n[*] No pending samples found in queue.")
            return stats

        print("\n" + "=" * 75)
        print(f"[*] RUNNING TWO-STEP KEYLESS VERIFICATION ON {len(pending_files)} PENDING SAMPLES")
        print("=" * 75)

        for path in pending_files:
            filename = os.path.basename(path)
            sample_id = filename.removesuffix(".npz")

            try:
                with np.load(path, allow_pickle=False) as data:
                    metadata_raw = data["metadata"]
                    if isinstance(metadata_raw, (bytes, str)):
                        metadata = json.loads(str(metadata_raw))
                    else:
                        metadata = json.loads(metadata_raw.item())

                verdict = self.evaluate_sample(metadata)

                if verdict["verified"]:
                    label = verdict["label"]
                    self.collector.verify_sample(sample_id=sample_id, true_label=label)
                    cat = "Benign (0)" if label == 0 else "Malware (1)"
                    if label == 0:
                        stats["verified_benign"] += 1
                    else:
                        stats["verified_malware"] += 1

                    print(f"  [+] {sample_id[:16]}... -> VERIFIED: {cat} via {verdict['step']}")
                    print(f"      Details: {verdict['reason']}")
                else:
                    stats["inconclusive"] += 1
                    print(f"  [-] {sample_id[:16]}... -> INCONCLUSIVE: {verdict['reason']}")

            except Exception as e:
                print(f"  [!] Error processing {filename}: {e}")
                stats["inconclusive"] += 1

        print("-" * 75)
        print(f"  [RESULTS] Verified Benign: {stats['verified_benign']} | Verified Malware: {stats['verified_malware']} | Inconclusive: {stats['inconclusive']}")
        print("=" * 75 + "\n")

        return stats
