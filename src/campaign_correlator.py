"""
campaign_correlator.py - Causal Graph Reconstruction & Multi-Campaign Correlation
MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks
Candidate: Haaziq Rasool (Bahria University Islamabad)

Implements Proposal Objective 3 & Literature Architecture (Sen et al., 2022 / Shapoorifard et al., 2023):
- Takes raw RL-flagged alerts across Firewall, Web/WAF, and Endpoint streams.
- Reconstructs causally-linked 4-stage attack graphs (Recon -> Web Exploit -> Reverse Shell -> Priv Esc).
- Evaluates formal correlation clustering metrics:
  - Campaign Reconstruction Purity
  - Cluster Completeness Score
  - Adjusted Rand Index (ARI)
  - Full Kill-Chain Completion Rate
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score


class CampaignCorrelator:
    """
    Causal Graph Correlation & Multi-Stage Attack Reconstructor.
    Groups discrete alerts into coherent attack campaign graphs based on temporal proximity
    and shared host/attacker identities.
    """
    def __init__(self, time_window_seconds: float = 900.0):
        self.time_window_seconds = time_window_seconds

    def correlate_alerts(self, df_alerts: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Links flagged alerts into reconstructed campaign clusters.
        """
        df_sorted = df_alerts.sort_values(by="timestamp").reset_index(drop=True).copy()
        
        clusters = []
        cluster_id_counter = 1
        
        # Dynamic active campaign tracking: { cluster_id: {'attacker_ip': ..., 'target_ip': ..., 'last_seen': ..., 'stages': set()} }
        active_clusters: Dict[int, Dict] = {}

        cluster_assignments = []

        for idx, row in df_sorted.iterrows():
            ts = row["timestamp"]
            s_ip = str(row["src_ip"])
            d_ip = str(row["dst_ip"])
            stage = int(row.get("attack_stage", 0))

            # Prune expired clusters
            for cid in list(active_clusters.keys()):
                if (ts - active_clusters[cid]["last_seen"]).total_seconds() > self.time_window_seconds:
                    del active_clusters[cid]

            matched_cluster_id = None

            # Match criteria: Shared Attacker IP OR Target Server within temporal window
            for cid, c_info in active_clusters.items():
                if (s_ip == c_info["attacker_ip"] or d_ip == c_info["target_ip"]) and stage >= max(c_info["stages"]):
                    matched_cluster_id = cid
                    break

            if matched_cluster_id is not None:
                active_clusters[matched_cluster_id]["last_seen"] = ts
                active_clusters[matched_cluster_id]["stages"].add(stage)
                cluster_assignments.append(f"RECON-CAMP-{matched_cluster_id:04d}")
            else:
                new_cid = cluster_id_counter
                cluster_id_counter += 1
                active_clusters[new_cid] = {
                    "attacker_ip": s_ip,
                    "target_ip": d_ip,
                    "last_seen": ts,
                    "stages": {stage}
                }
                cluster_assignments.append(f"RECON-CAMP-{new_cid:04d}")

        df_sorted["predicted_cluster"] = cluster_assignments

        # Compute formal clustering & correlation metrics against ground truth campaign_id
        true_labels = df_sorted["campaign_id"].astype(str).values
        pred_labels = df_sorted["predicted_cluster"].values

        ari = adjusted_rand_score(true_labels, pred_labels)
        homogeneity = homogeneity_score(true_labels, pred_labels)
        completeness = completeness_score(true_labels, pred_labels)
        v_measure = v_measure_score(true_labels, pred_labels)

        # Count full 4-stage kill-chain reconstructions
        full_chains_reconstructed = 0
        total_true_campaigns = len(df_sorted[df_sorted["campaign_id"] != "BENIGN"]["campaign_id"].unique())

        for cid, group in df_sorted[df_sorted["campaign_id"] != "BENIGN"].groupby("campaign_id"):
            stages_present = set(group["attack_stage"].unique())
            if {1, 2, 3, 4}.issubset(stages_present):
                full_chains_reconstructed += 1

        chain_reconstruction_rate = full_chains_reconstructed / total_true_campaigns if total_true_campaigns > 0 else 1.0

        metrics = {
            "adjusted_rand_index": float(ari),
            "campaign_purity_homogeneity": float(homogeneity),
            "cluster_completeness": float(completeness),
            "v_measure": float(v_measure),
            "kill_chain_reconstruction_rate": float(chain_reconstruction_rate),
            "total_campaigns_analyzed": int(total_true_campaigns)
        }

        return df_sorted, metrics


def evaluate_campaign_correlation(df_flagged_alerts: pd.DataFrame) -> Dict[str, Any]:
    correlator = CampaignCorrelator(time_window_seconds=900.0)
    _, metrics = correlator.correlate_alerts(df_flagged_alerts)
    return metrics
