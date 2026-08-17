"""
attack_identifier.py - Deterministic Rule-Based Baseline Engine for Log Correlation
MS Thesis: Multi-Source Heterogeneous Log Correlation using Reinforcement Learning

Implements a state-of-the-art heuristic sliding-window correlation engine that links
firewall reconnaissance / web exploitation triggers to subsequent endpoint executions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix


def run_rule_engine(
    df: pd.DataFrame,
    time_window_seconds: float = 600.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes a deterministic temporal correlation rule engine:
    - Maintains a state table of active suspicious source IPs / target hosts.
    - If a perimeter event matches scanning / web exploit patterns, registers the IP in the tracker.
    - If an endpoint event occurs within Delta T <= time_window_seconds involving the tracked IP/host
      with execution/privilege escalation signatures, flags both as correlated attacks.
    """
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
    predictions = np.zeros(len(df_sorted), dtype=int)

    # State tracker: { ip_address: [(timestamp, event_index, stage_type)] }
    active_threat_state: Dict[str, list] = {}

    # Recon / Web Exploit Signatures
    recon_web_signatures = {
        "ET_SCAN_NMAP_SYN_PROBE", "ET_SCAN_PORT_SWEEP",
        "ET_WEB_EXPLOIT_SQLI_PAYLOAD", "ET_WEB_RCE_UNSERIALIZE_POST",
        "ET_WEB_FILE_UPLOAD_PHP_SHELL", "ET_CVE_2023_AUTH_BYPASS_EXPLOIT",
        "BLOCKED_PORT_SCAN_NOISE"
    }

    # Endpoint Execution / Elevation Signatures
    endpoint_signatures = {
        "SIG_REVERSE_SHELL_EXECUTION", "SIG_UNAUTHORIZED_SUDO_ELEVATION",
        "SIG_CREDENTIAL_ACCESS_SHADOW", "SIG_MIMIKATZ_LSASS_ACCESS",
        "SIG_CRON_PERSISTENCE_WRITE"
    }

    for idx, row in df_sorted.iterrows():
        current_time = row["timestamp"]
        src_ip = str(row["src_ip"])
        dst_ip = str(row["dst_ip"])
        rule_hit = str(row["rule_hit"])
        source_type = int(row["source_type"])

        # Prune expired threats from active state outside sliding window
        for ip in list(active_threat_state.keys()):
            active_threat_state[ip] = [
                (t, e_idx, s_type)
                for (t, e_idx, s_type) in active_threat_state[ip]
                if (current_time - t).total_seconds() <= time_window_seconds
            ]
            if not active_threat_state[ip]:
                del active_threat_state[ip]

        # Case 1: Firewall Perimeter Recon / Exploit Detection
        if source_type == 0:
            if rule_hit in recon_web_signatures or "SCAN" in rule_hit or "EXPLOIT" in rule_hit:
                if src_ip not in active_threat_state:
                    active_threat_state[src_ip] = []
                active_threat_state[src_ip].append((current_time, idx, "PERIMETER"))
                # Speculatively flag perimeter threat
                predictions[idx] = 1

        # Case 2: Endpoint Audit / Host Execution Event
        elif source_type == 1:
            # Check if associated with an active tracked attacker IP or destination host
            is_correlated = False
            if src_ip in active_threat_state:
                is_correlated = True
            
            # Check if matching known high-risk execution signatures
            if rule_hit in endpoint_signatures or "REVERSE_SHELL" in rule_hit or "PRIVILEGE" in rule_hit:
                predictions[idx] = 1
                if is_correlated:
                    # Retroactively ensure all prior causal triggers within window are flagged
                    for (t, e_idx, s_type) in active_threat_state.get(src_ip, []):
                        predictions[e_idx] = 1
            elif is_correlated and row["action"] in ["PROCESS_SPAWN", "CLI_COMMAND", "APP_LAUNCH"]:
                predictions[idx] = 1

    targets = df_sorted["label"].values.astype(int)

    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    cm = confusion_matrix(targets, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "far": float(far),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

    return predictions, metrics


def evaluate_baseline(metadata_path: str = "data/processed/metadata.parquet") -> Dict[str, Any]:
    """
    Loads preprocessed metadata parquet and runs the deterministic correlation rule engine.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run preprocessing first.")

    print(f"[*] Loading processed metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    print("[*] Executing Deterministic Sliding-Window Rule Engine (Delta T <= 600s)...")
    predictions, metrics = run_rule_engine(df, time_window_seconds=600.0)

    print("\n=======================================================")
    print("      DETERMINISTIC RULE ENGINE (BASELINE) RESULTS     ")
    print("=======================================================")
    print(classification_report(df["label"].values, predictions, target_names=["Benign", "Attack"]))
    print(f"  Precision       : {metrics['precision']:.4f}")
    print(f"  Recall          : {metrics['recall']:.4f}")
    print(f"  F1-Score        : {metrics['f1_score']:.4f}")
    print(f"  False Alarm Rate: {metrics['far']:.4f}")
    print(f"  Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    print("=======================================================\n")

    return metrics


if __name__ == "__main__":
    evaluate_baseline()
