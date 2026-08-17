# Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RL: DQN & PPO](https://img.shields.io/badge/RL-DQN%20%7C%20PPO-success.svg)](https://stable-baselines3.readthedocs.io/)

Production-grade research implementation for the MS Thesis:  
**"Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks"**  
*Candidate:* Haaziq Rasool | *Department of Computer Science, Bahria University Islamabad* | *Supervisor:* Dr. Hafiz Ishfaq Ahmad

---

## 🚀 Key Experimental Findings & Benchmarks

Evaluated on an unseen **20% hold-out test set** ($N = 6,550$ multi-tier events: 125 attack stages across 30 active campaigns, 6,425 benign background logs):

| Evaluation Metric | Naive Rule Engine | Isolation Forest | Supervised Random Forest | Proposed RL (DQN) | Proposed RL (PPO) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Precision** | 100.00% | 29.49% | 100.00% | **100.00%** | 2.95% |
| **Recall (Sensitivity)** | 40.00% | 73.60% | 100.00% | **100.00%** | 80.00% |
| **F1-Score** | 0.5714 | 0.4211 | 1.0000 | **1.0000** | 0.0570 |
| **False Alarm Rate (FAR)**| 0.00% | 3.42% | 0.00% | **0.00%** | 51.13% |
| **False Positives (FP)** | 0 | 220 False Alarms | 0 | **0 False Alarms** | 3,285 |
| **False Negatives (FN)** | 75 Missed Attacks | 33 Missed | 0 | **0 Missed Attacks** | 25 |
| **Per-Event Latency** | — | — | 4.8 μs | **13.6 μs** | 18.4 μs |
| **Streaming Throughput** | — | — | 208,000 eps | **73,344 eps** | 54,376 eps |

### Cross-Domain & Stealth Robustness
* **Public Benchmark Transferability (CICIDS2017 + ModSecurity):** Zero-shot cross-domain ingestion achieved **96.33% detection recall** through the unified 3-tier schema normalizer.
* **Adversarial Stealth Evasion Resilience:** Under deliberate 1-hour inter-stage delays ($\Delta t = 3,600\,\text{s}$), fixed-window rule engines degraded from 100% to **16.7% recall**, while the RL agent retained **100.0% recall**.

---

## 📁 Repository Structure

```
├── data/
│   ├── synthetic/
│   │   ├── firewall_logs.csv         # Tier 1: Perimeter firewall telemetry
│   │   ├── web_waf_logs.csv          # Tier 2: Web & ModSecurity WAF logs
│   │   ├── endpoint_logs.csv         # Tier 3: Host auditd process execution
│   │   └── ground_truth.jsonl        # 150 4-stage campaign causal chains
│   ├── processed/
│   │   ├── features.npy              # Normalized feature matrix (32,750, 9)
│   │   ├── targets.npy               # Binary labels (0=Benign, 1=Attack)
│   │   └── metadata.parquet          # Tabular dataset preserving raw fields
│   └── public_benchmark/             # Ingested CICIDS2017 & ModSecurity validation samples
├── models/
│   ├── web_attack_classifier.joblib  # Stage 3 Supervised Web NLP Classifier
│   ├── dqn_agent.zip                 # Stage 4 Value-Based Deep Q-Network
│   └── ppo_agent.zip                 # Stage 4 Policy-Gradient PPO Agent
├── results/
│   ├── model_comparison.png          # IEEE-grade 4-panel comparative plot
│   ├── adversarial_stealth_benchmark.png # Delay-scaling evasion curve
│   └── evaluation_metrics.json       # JSON benchmark records
├── src/
│   ├── __init__.py
│   ├── gen_synthetic.py              # 3-tier multi-source telemetry generator
│   ├── preprocessing.py              # Schema harmonization & Delta t engine
│   ├── web_attack_classifier.py      # Stage 3 Supervised Web NLP identifier
│   ├── env.py                        # Custom Gymnasium LogCorrelationEnv
│   ├── train_agent.py                # Dual RL training pipeline (DQN + PPO)
│   ├── attack_identifier.py          # Deterministic sliding-window baseline
│   ├── campaign_correlator.py        # Causal graph reconstruction & clustering
│   ├── ingest_public_dataset.py      # Public benchmark schema alignment
│   ├── adversarial_eval.py           # Stealth delay stress-testing suite
│   └── evaluate.py                   # Master multi-model comparative evaluator
├── requirements.txt
├── run_pipeline.sh                   # 1-Click Linux execution script
├── generate_thesis_doc.py            # Word document generator
└── MS_Thesis_Log_Correlation_RL_Guide.docx # Comprehensive Defense Guide
```

---

## 🚀 How to Run the End-to-End Pipeline

```bash
# Step 1: Generate 3-Tier Multi-Source Synthetic Dataset (32,750 events)
python src/gen_synthetic.py

# Step 2: Harmonize Schemas and Preprocess Features
python src/preprocessing.py

# Step 3: Train Stage 3 Supervised Web Attack Identifier
python src/web_attack_classifier.py

# Step 4: Train Dual RL Agents (DQN + PPO)
python src/train_agent.py

# Step 5: Run Master Evaluation (All 5 Baselines, Latency, Public Datasets)
python src/evaluate.py

# Step 6: Run Adversarial Stealth Evasion Stress Test
python src/adversarial_eval.py

# Step 7: Re-generate Thesis Word Guide
python generate_thesis_doc.py
```
