"""
preprocessing.py - 3-Tier Multi-Source Schema Harmonization & Feature Engineering
MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks
Candidate: Haaziq Rasool (Bahria University Islamabad)

Harmonizes 3 distinct heterogeneous log streams:
- source_type 0: Firewall Logs (Perimeter edge)
- source_type 1: Web / WAF Logs (Application tier: SQLi, XSS, RCE, Uploads)
- source_type 2: Endpoint Logs (Host auditd & process execution)

Outputs:
- data/processed/features.npy (Normalized multi-feature matrix for RL environment)
- data/processed/targets.npy (Binary ground truth labels: 0=Benign, 1=Attack)
- data/processed/metadata.parquet (Enriched tabular DataFrame preserving raw fields & timestamps)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Tuple, Dict


def load_and_harmonize_logs(
    firewall_path: str = "data/synthetic/firewall_logs.csv",
    web_waf_path: str = "data/synthetic/web_waf_logs.csv",
    endpoint_path: str = "data/synthetic/endpoint_logs.csv"
) -> pd.DataFrame:
    """
    Ingests 3 disparate log streams, maps them into a unified multi-source schema,
    assigns source_type (0=Firewall, 1=Web/WAF, 2=Endpoint), and sorts globally chronologically.
    """
    for p in [firewall_path, web_waf_path, endpoint_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required log file not found at: {p}")

    print(f"[*] Ingesting 3-tier log feeds from {firewall_path}, {web_waf_path}, and {endpoint_path}...")
    df_fw = pd.read_csv(firewall_path)
    df_web = pd.read_csv(web_waf_path)
    df_ep = pd.read_csv(endpoint_path)

    # 1. Standardize Firewall Log Schema (source_type = 0)
    df_fw_norm = pd.DataFrame({
        "timestamp": pd.to_datetime(df_fw["timestamp"]),
        "source_type": 0,  # 0 = Firewall
        "src_ip": df_fw["src_ip"].astype(str),
        "dst_ip": df_fw["dst_ip"].astype(str),
        "port": df_fw["dst_port"].fillna(0).astype(int),
        "protocol": df_fw["protocol"].fillna("UNKNOWN").astype(str),
        "action": df_fw["action"].fillna("UNKNOWN").astype(str),
        "rule_hit": df_fw["rule_hit"].fillna("UNKNOWN").astype(str),
        "bytes_transferred": df_fw["bytes_transferred"].fillna(0).astype(float),
        "campaign_id": df_fw["campaign_id"].fillna("BENIGN").astype(str),
        "attack_stage": df_fw["attack_stage"].fillna(0).astype(int),
        "label": df_fw["label"].fillna(0).astype(int)
    })

    # 2. Standardize Web / WAF Log Schema (source_type = 1)
    df_web_norm = pd.DataFrame({
        "timestamp": pd.to_datetime(df_web["timestamp"]),
        "source_type": 1,  # 1 = Web / WAF
        "src_ip": df_web["client_ip"].astype(str),
        "dst_ip": df_web["server_ip"].astype(str),
        "port": 80,  # Web traffic maps to HTTP/HTTPS ports
        "protocol": df_web["http_method"].fillna("GET").astype(str),
        "action": df_web["attack_type"].fillna("BENIGN").astype(str),
        "rule_hit": df_web["waf_rule_hit"].fillna("UNKNOWN").astype(str),
        "bytes_transferred": df_web["bytes_sent"].fillna(0).astype(float),
        "campaign_id": df_web["campaign_id"].fillna("BENIGN").astype(str),
        "attack_stage": df_web["attack_stage"].fillna(0).astype(int),
        "label": df_web["label"].fillna(0).astype(int)
    })

    # 3. Standardize Endpoint Log Schema (source_type = 2)
    df_ep_norm = pd.DataFrame({
        "timestamp": pd.to_datetime(df_ep["timestamp"]),
        "source_type": 2,  # 2 = Endpoint
        "src_ip": df_ep["src_ip"].astype(str),
        "dst_ip": df_ep["host_ip"].astype(str),
        "port": 0,
        "protocol": df_ep["process_name"].fillna("UNKNOWN").astype(str),
        "action": df_ep["action"].fillna("UNKNOWN").astype(str),
        "rule_hit": df_ep["rule_hit"].fillna("UNKNOWN").astype(str),
        "bytes_transferred": 0.0,
        "campaign_id": df_ep["campaign_id"].fillna("BENIGN").astype(str),
        "attack_stage": df_ep["attack_stage"].fillna(0).astype(int),
        "label": df_ep["label"].fillna(0).astype(int)
    })

    # Merge and sort globally chronologically
    df_merged = pd.concat([df_fw_norm, df_web_norm, df_ep_norm], ignore_index=True)
    df_merged = df_merged.sort_values(by="timestamp").reset_index(drop=True)

    print(f"[+] Harmonized {len(df_merged):,} total multi-source log events across 3 heterogeneous tiers.")
    return df_merged


def engineer_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict]:
    """
    Computes inter-arrival time deltas, encodes categorical attributes, and scales features to [0, 1].
    """
    print("[*] Engineering temporal and relational correlation features...")
    df_feat = df.copy()

    # 1. Relative Inter-Event Time Delta (Delta t_i = t_i - t_{i-1}) in seconds
    time_diffs = df_feat["timestamp"].diff().dt.total_seconds().fillna(0.0)
    time_diffs = np.clip(time_diffs, 0.0, 3600.0)
    df_feat["time_delta"] = time_diffs

    # 2. Categorical Encoders
    encoders = {}
    categorical_cols = ["src_ip", "dst_ip", "protocol", "action", "rule_hit"]
    for col in categorical_cols:
        le = LabelEncoder()
        df_feat[f"{col}_enc"] = le.fit_transform(df_feat[col].astype(str))
        encoders[col] = le

    # 3. Selected Feature Matrix for State Representation
    feature_cols = [
        "source_type",
        "time_delta",
        "src_ip_enc",
        "dst_ip_enc",
        "port",
        "protocol_enc",
        "action_enc",
        "rule_hit_enc",
        "bytes_transferred"
    ]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_feat[feature_cols].values)
    targets = df_feat["label"].values.astype(np.int64)

    return scaled_features, targets, df_feat, encoders


def process_and_save(
    firewall_path: str = "data/synthetic/firewall_logs.csv",
    web_waf_path: str = "data/synthetic/web_waf_logs.csv",
    endpoint_path: str = "data/synthetic/endpoint_logs.csv",
    output_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Executes the 3-tier harmonization, feature extraction, and disk persistence.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_merged = load_and_harmonize_logs(firewall_path, web_waf_path, endpoint_path)
    features, targets, df_metadata, _ = engineer_features(df_merged)

    feat_path = os.path.join(output_dir, "features.npy")
    target_path = os.path.join(output_dir, "targets.npy")
    meta_path = os.path.join(output_dir, "metadata.parquet")

    np.save(feat_path, features)
    np.save(target_path, targets)
    df_metadata.to_parquet(meta_path, index=False)

    num_total = len(targets)
    num_attacks = int(np.sum(targets))
    num_benign = num_total - num_attacks
    attack_ratio = (num_attacks / num_total) * 100

    print("\n=======================================================")
    print("      3-TIER PREPROCESSING & HARMONIZATION SUMMARY     ")
    print("=======================================================")
    print(f"  Processed Features Shape : {features.shape}")
    print(f"  Targets Shape            : {targets.shape}")
    print(f"  Total Multi-Source Logs  : {num_total:,}")
    print(f"    - Firewall Logs (Tier 0): {len(df_metadata[df_metadata['source_type'] == 0]):,}")
    print(f"    - Web/WAF Logs  (Tier 1): {len(df_metadata[df_metadata['source_type'] == 1]):,}")
    print(f"    - Endpoint Logs (Tier 2): {len(df_metadata[df_metadata['source_type'] == 2]):,}")
    print(f"  Benign Events (Class 0)  : {num_benign:,} ({100 - attack_ratio:.2f}%)")
    print(f"  Attack Events (Class 1)  : {num_attacks:,} ({attack_ratio:.2f}%)")
    print(f"  Saved Features to        : {feat_path}")
    print(f"  Saved Targets to         : {target_path}")
    print(f"  Saved Metadata to        : {meta_path}")
    print("=======================================================\n")

    return features, targets, df_metadata


if __name__ == "__main__":
    process_and_save()
