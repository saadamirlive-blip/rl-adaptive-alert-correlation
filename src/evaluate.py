"""
evaluate.py - Comprehensive Multi-Baseline, Multi-RL, Latency & Public Benchmark Evaluation Suite
MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks
Candidate: Haaziq Rasool (Bahria University Islamabad)
Supervisor: Dr. Hafiz Ishfaq Ahmad

Benchmarks:
1. Deterministic Rule Engine (10-min Sliding Window Heuristic Baseline)
2. Supervised Random Forest (Standard Supervised ML Baseline)
3. Unsupervised Isolation Forest (Standard Anomaly Detection Baseline)
4. Proposed Deep Q-Network (DQN - Value-Based RL)
5. Proposed Proximal Policy Optimization (PPO - Policy Gradient RL)
6. Multi-Stage Campaign Causal Graph Reconstruction (ARI, Homogeneity, Completeness)
7. Streaming Per-Event Inference Latency & Throughput Benchmark
8. Public Benchmark Cross-Domain Validation (CICIDS2017 + ModSecurity)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env import LogCorrelationEnv
from src.attack_identifier import run_rule_engine
from src.train_agent import StandaloneDQN, StandalonePPO
from src.campaign_correlator import CampaignCorrelator
from src.ingest_public_dataset import run_cross_domain_evaluation


def calculate_mttd(
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    ground_truth_path: str = "data/synthetic/ground_truth.jsonl"
) -> float:
    """
    Computes Mean Time to Detection (MTTD) in seconds from campaign start time.
    """
    df_eval = df_test.copy().reset_index(drop=True)
    df_eval["pred"] = predictions

    campaign_starts = {}
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                cid = rec["campaign_id"]
                st = pd.to_datetime(rec["start_time"])
                campaign_starts[cid] = st

    time_to_detections = []
    attack_groups = df_eval[df_eval["campaign_id"] != "BENIGN"].groupby("campaign_id")

    for cid, group in attack_groups:
        detected_stages = group[group["pred"] == 1]
        if not detected_stages.empty:
            earliest_detected_time = detected_stages["timestamp"].min()
            t_start = campaign_starts.get(cid, group["timestamp"].min())
            ttd_seconds = (earliest_detected_time - t_start).total_seconds()
            time_to_detections.append(max(0.0, ttd_seconds))
        else:
            time_to_detections.append(600.0)

    if not time_to_detections:
        return 0.0

    return float(np.mean(time_to_detections))


def compute_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray, mttd: float) -> Dict[str, Any]:
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "model": name,
        "precision": float(p),
        "recall": float(r),
        "f1_score": float(f1),
        "far": float(far),
        "mttd": float(mttd),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def measure_inference_latency(agent: Any, test_features: np.ndarray, num_runs: int = 5000) -> Dict[str, float]:
    """
    Measures per-event streaming correlation inference time in microseconds (μs) and events/sec throughput.
    """
    samples = test_features[:min(len(test_features), num_runs)]
    # Warmup
    for i in range(min(50, len(samples))):
        _ = agent.predict(samples[i], deterministic=True)

    start_t = time.perf_counter()
    for i in range(len(samples)):
        _ = agent.predict(samples[i], deterministic=True)
    end_t = time.perf_counter()

    total_time = end_t - start_t
    avg_latency_us = (total_time / len(samples)) * 1_000_000
    throughput_eps = len(samples) / total_time

    return {
        "avg_latency_microseconds": float(avg_latency_us),
        "throughput_events_per_sec": float(throughput_eps),
        "evaluated_samples": len(samples)
    }


def evaluate_all(
    features_path: str = "data/processed/features.npy",
    targets_path: str = "data/processed/targets.npy",
    metadata_path: str = "data/processed/metadata.parquet",
    dqn_model_path: str = "models/dqn_agent.zip",
    ppo_model_path: str = "models/ppo_agent.zip",
    ground_truth_path: str = "data/synthetic/ground_truth.jsonl",
    results_dir: str = "results"
) -> Dict[str, Any]:
    os.makedirs(results_dir, exist_ok=True)

    print(f"[*] Ingesting test slice from {features_path} and {metadata_path}...")
    features = np.load(features_path)
    targets = np.load(targets_path)
    df_meta = pd.read_parquet(metadata_path)

    # 80/20 Hold-out Test Partition
    split_idx = int(0.8 * len(features))
    train_features, train_targets = features[:split_idx], targets[:split_idx]
    test_features, test_targets = features[split_idx:], targets[split_idx:]
    df_test = df_meta.iloc[split_idx:].reset_index(drop=True)

    print(f"[+] Hold-out test samples: {len(test_features):,} (Attacks: {int(np.sum(test_targets))}, Benign: {int(len(test_targets) - np.sum(test_targets))})")

    # =========================================================================
    # Baseline 1: Deterministic Heuristic Rule Engine
    # =========================================================================
    print("\n[*] [1/5] Evaluating Naive Rule Engine Baseline...")
    rule_preds, _ = run_rule_engine(df_test, time_window_seconds=600.0)
    rule_mttd = calculate_mttd(df_test, rule_preds, ground_truth_path)
    rule_metrics = compute_metrics("Naive Rule Engine (Baseline)", test_targets, rule_preds, rule_mttd)

    # =========================================================================
    # Baseline 2: Supervised Random Forest Classifier
    # =========================================================================
    print("[*] [2/5] Training & Evaluating Supervised Random Forest Baseline...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(train_features, train_targets)
    rf_preds = rf.predict(test_features)
    rf_mttd = calculate_mttd(df_test, rf_preds, ground_truth_path)
    rf_metrics = compute_metrics("Supervised Random Forest", test_targets, rf_preds, rf_mttd)

    # =========================================================================
    # Baseline 3: Unsupervised Isolation Forest (Anomaly Detection)
    # =========================================================================
    print("[*] [3/5] Evaluating Unsupervised Isolation Forest Baseline...")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(train_features)
    iso_raw = iso.predict(test_features)
    iso_preds = np.where(iso_raw == -1, 1, 0)
    iso_mttd = calculate_mttd(df_test, iso_preds, ground_truth_path)
    iso_metrics = compute_metrics("Isolation Forest (Anomaly)", test_targets, iso_preds, iso_mttd)

    # =========================================================================
    # Model 4: Proposed Deep Q-Network (DQN Agent)
    # =========================================================================
    print(f"[*] [4/5] Evaluating Proposed Deep Q-Network (DQN) from {dqn_model_path}...")
    dqn_agent = StandaloneDQN.load(dqn_model_path)
    test_env = LogCorrelationEnv(test_features, test_targets)
    dqn_preds = []
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = dqn_agent.predict(obs, deterministic=True)
        dqn_preds.append(int(action))
        obs, _, term, trunc, _ = test_env.step(int(action))
        done = term or trunc

    dqn_preds = np.array(dqn_preds, dtype=int)
    dqn_mttd = calculate_mttd(df_test, dqn_preds, ground_truth_path)
    dqn_metrics = compute_metrics("Proposed RL Agent (DQN)", test_targets, dqn_preds, dqn_mttd)

    # =========================================================================
    # Model 5: Proposed Proximal Policy Optimization (PPO Agent)
    # =========================================================================
    print(f"[*] [5/5] Evaluating Proposed Proximal Policy Optimization (PPO) from {ppo_model_path}...")
    ppo_agent = StandalonePPO.load(ppo_model_path)
    test_env_ppo = LogCorrelationEnv(test_features, test_targets)
    ppo_preds = []
    obs, _ = test_env_ppo.reset()
    done = False
    while not done:
        action, _ = ppo_agent.predict(obs, deterministic=True)
        ppo_preds.append(int(action))
        obs, _, term, trunc, _ = test_env_ppo.step(int(action))
        done = term or trunc

    ppo_preds = np.array(ppo_preds, dtype=int)
    ppo_mttd = calculate_mttd(df_test, ppo_preds, ground_truth_path)
    ppo_metrics = compute_metrics("Proposed RL Agent (PPO)", test_targets, ppo_preds, ppo_mttd)

    # =========================================================================
    # Latency & Throughput Benchmark
    # =========================================================================
    print("\n[*] Benchmarking Per-Event Streaming Inference Latency...")
    latency_dqn = measure_inference_latency(dqn_agent, test_features)
    latency_ppo = measure_inference_latency(ppo_agent, test_features)
    print(f"    - DQN Latency: {latency_dqn['avg_latency_microseconds']:.2f} us/event ({latency_dqn['throughput_events_per_sec']:,.0f} events/sec)")
    print(f"    - PPO Latency: {latency_ppo['avg_latency_microseconds']:.2f} us/event ({latency_ppo['throughput_events_per_sec']:,.0f} events/sec)")

    # =========================================================================
    # Causal Graph Campaign Reconstruction
    # =========================================================================
    print("\n[*] Evaluating Multi-Stage Campaign Causal Graph Reconstruction...")
    correlator = CampaignCorrelator(time_window_seconds=900.0)
    df_flagged = df_test[dqn_preds == 1].copy()
    _, cluster_metrics = correlator.correlate_alerts(df_flagged)

    # =========================================================================
    # Cross-Domain Public Dataset Evaluation (CICIDS2017 + ModSecurity)
    # =========================================================================
    public_eval_metrics = run_cross_domain_evaluation(dqn_model_path)

    # =========================================================================
    # Print Master Comparative Benchmark Table
    # =========================================================================
    print("\n" + "=" * 118)
    print("         COMPREHENSIVE BENCHMARK: RULE ENGINE vs ML BASELINES vs DEEP Q-NETWORK vs PPO")
    print("=" * 118)
    header = f"{'Evaluation Metric':<24} | {'Rule Engine':<14} | {'Random Forest':<15} | {'Isolation Forest':<17} | {'Proposed DQN':<15} | {'Proposed PPO':<15}"
    print(header)
    print("-" * 118)
    print(f"{'Precision':<24} | {rule_metrics['precision']:<14.4f} | {rf_metrics['precision']:<15.4f} | {iso_metrics['precision']:<17.4f} | {dqn_metrics['precision']:<15.4f} | {ppo_metrics['precision']:<15.4f}")
    print(f"{'Recall (Sensitivity)':<24} | {rule_metrics['recall']:<14.4f} | {rf_metrics['recall']:<15.4f} | {iso_metrics['recall']:<17.4f} | {dqn_metrics['recall']:<15.4f} | {ppo_metrics['recall']:<15.4f}")
    print(f"{'F1-Score':<24} | {rule_metrics['f1_score']:<14.4f} | {rf_metrics['f1_score']:<15.4f} | {iso_metrics['f1_score']:<17.4f} | {dqn_metrics['f1_score']:<15.4f} | {ppo_metrics['f1_score']:<15.4f}")
    print(f"{'False Alarm Rate (FAR)':<24} | {rule_metrics['far']:<14.4f} | {rf_metrics['far']:<15.4f} | {iso_metrics['far']:<17.4f} | {dqn_metrics['far']:<15.4f} | {ppo_metrics['far']:<15.4f}")
    rule_mttd_str = f"{rule_metrics['mttd']:.1f} s"
    rf_mttd_str = f"{rf_metrics['mttd']:.1f} s"
    iso_mttd_str = f"{iso_metrics['mttd']:.1f} s"
    dqn_mttd_str = f"{dqn_metrics['mttd']:.1f} s"
    ppo_mttd_str = f"{ppo_metrics['mttd']:.1f} s"
    print(f"{'Mean Time to Detect':<24} | {rule_mttd_str:<14} | {rf_mttd_str:<15} | {iso_mttd_str:<17} | {dqn_mttd_str:<15} | {ppo_mttd_str:<15}")
    print(f"{'False Positives (FP)':<24} | {rule_metrics['fp']:<14} | {rf_metrics['fp']:<15} | {iso_metrics['fp']:<17} | {dqn_metrics['fp']:<15} | {ppo_metrics['fp']:<15}")
    print(f"{'False Negatives (FN)':<24} | {rule_metrics['fn']:<14} | {rf_metrics['fn']:<15} | {iso_metrics['fn']:<17} | {dqn_metrics['fn']:<15} | {ppo_metrics['fn']:<15}")
    print("=" * 118)

    # Generate Publication-Grade 4-Panel Plot
    plot_path = os.path.join(results_dir, "model_comparison.png")
    generate_publication_plots(
        [rule_metrics, rf_metrics, iso_metrics, dqn_metrics, ppo_metrics],
        cluster_metrics,
        latency_dqn,
        public_eval_metrics,
        plot_path
    )

    # Save JSON records
    all_results = {
        "rule_engine_baseline": rule_metrics,
        "random_forest_baseline": rf_metrics,
        "isolation_forest_baseline": iso_metrics,
        "proposed_dqn_agent": dqn_metrics,
        "proposed_ppo_agent": ppo_metrics,
        "latency_and_throughput": {
            "dqn": latency_dqn,
            "ppo": latency_ppo
        },
        "campaign_reconstruction_metrics": cluster_metrics,
        "public_benchmark_cross_domain": public_eval_metrics
    }
    json_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] Saved comprehensive evaluation metrics to: {json_path}")
    return all_results


def generate_publication_plots(
    models_metrics: List[Dict[str, Any]],
    cluster_metrics: Dict[str, Any],
    latency_data: Dict[str, float],
    public_eval_data: Dict[str, float],
    save_path: str
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.0), dpi=300)

    model_names = ["Rule Engine", "Random Forest", "Isolation Forest", "DQN (Proposed)", "PPO (Proposed)"]

    # Panel 1: F1-Score & Precision
    precisions = [m["precision"] for m in models_metrics]
    f1_scores = [m["f1_score"] for m in models_metrics]
    x = np.arange(len(model_names))
    w = 0.35
    axes[0].bar(x - w/2, precisions, w, label="Precision", color="#5C6BC0", edgecolor="black", alpha=0.9)
    axes[0].bar(x + w/2, f1_scores, w, label="F1-Score", color="#26A69A", edgecolor="black", alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=25, ha="right", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Score [0.0 - 1.0]", fontsize=10, fontweight="bold")
    axes[0].set_title("(a) Precision & F1-Score Comparison", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)

    # Panel 2: False Positive Count (Alert Fatigue)
    fps = [m["fp"] for m in models_metrics]
    colors_fp = ["#D9534F", "#337AB7", "#F0AD4E", "#2E7D32", "#43A047"]
    bars_fp = axes[1].bar(model_names, fps, color=colors_fp, edgecolor="black", alpha=0.9, width=0.55)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=25, ha="right", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("False Positives (Alert Count)", fontsize=10, fontweight="bold")
    axes[1].set_title("(b) Alert Fatigue Reduction", fontsize=11, fontweight="bold")
    for bar in bars_fp:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{int(yval)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 3: Causal Graph & Campaign Clustering Quality
    c_labels = ["ARI", "Purity\n(Homogeneity)", "Completeness", "Kill-Chain\nRate"]
    c_vals = [
        cluster_metrics["adjusted_rand_index"],
        cluster_metrics["campaign_purity_homogeneity"],
        cluster_metrics["cluster_completeness"],
        cluster_metrics["kill_chain_reconstruction_rate"]
    ]
    axes[2].bar(c_labels, c_vals, color=["#8E24AA", "#3949AB", "#00897B", "#7CB342"], edgecolor="black", alpha=0.9, width=0.5)
    axes[2].set_title("(c) Causal Graph Reconstruction", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Metric Value [0.0 - 1.0]", fontsize=10, fontweight="bold")
    axes[2].set_ylim(0, 1.15)
    for i, v in enumerate(c_vals):
        axes[2].text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 4: Public Benchmark Transferability (CICIDS2017 + ModSecurity)
    p_labels = ["Public\nPrecision", "Public\nRecall", "Public\nF1-Score"]
    p_vals = [public_eval_data["public_precision"], public_eval_data["public_recall"], public_eval_data["public_f1"]]
    axes[3].bar(p_labels, p_vals, color=["#FB8C00", "#00ACC1", "#43A047"], edgecolor="black", alpha=0.9, width=0.45)
    axes[3].set_title("(d) Cross-Domain Public Validation", fontsize=11, fontweight="bold")
    axes[3].set_ylabel("Transfer Score [0.0 - 1.0]", fontsize=10, fontweight="bold")
    axes[3].set_ylim(0, 1.15)
    for i, v in enumerate(p_vals):
        axes[3].text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Publication-grade 4-panel comparison plot saved to: {save_path}")


if __name__ == "__main__":
    evaluate_all()
