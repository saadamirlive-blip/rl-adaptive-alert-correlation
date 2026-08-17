#!/usr/bin/env bash
# ==============================================================================
# MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation
# Linux Automated End-to-End Pipeline Script
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "======================================================================"
echo " Starting Full 3-Tier Multi-Source Log Correlation Pipeline"
echo "======================================================================"

# 1. Check Python & Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[*] Virtual environment not found. Creating .venv..."
    python3 -m venv .venv
fi

echo "[*] Activating virtual environment..."
source .venv/bin/activate

echo "[*] Installing / Verifying requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 2. Step 1: Generate 3-Tier Synthetic Multi-Source Logs
echo ""
echo "----------------------------------------------------------------------"
echo "[STEP 1/6] Generating 3-Tier Heterogeneous Logs & Attack Campaigns"
echo "----------------------------------------------------------------------"
python3 src/gen_synthetic.py

# 3. Step 2: Harmonize and Preprocess Data
echo ""
echo "----------------------------------------------------------------------"
echo "[STEP 2/6] Harmonizing Schemas & Engineering Temporal Features"
echo "----------------------------------------------------------------------"
python3 src/preprocessing.py

# 4. Step 3: Train Stage 3 Supervised Web Attack Identifier
echo ""
echo "----------------------------------------------------------------------"
echo "[STEP 3/6] Training Stage 3 Supervised NLP Web Attack Identifier"
echo "----------------------------------------------------------------------"
python3 src/web_attack_classifier.py

# 5. Step 4: Train Dual RL (DQN + PPO) Agents
echo ""
echo "----------------------------------------------------------------------"
echo "[STEP 4/6] Training Dual RL Agents (Deep Q-Network + PPO)"
echo "----------------------------------------------------------------------"
python3 src/train_agent.py

# 6. Step 5: Evaluate All Baselines, Latency, & Public Datasets
echo ""
echo "----------------------------------------------------------------------"
echo "[STEP 5/6] Comprehensive Evaluation & Publication Plot Generation"
echo "----------------------------------------------------------------------"
python3 src/evaluate.py

# 7. Step 6: Adversarial Delay Evasion Stress Test
echo ""
echo "----------------------------------------------------------------------"
echo "[STEP 6/6] Executing Adversarial Stealth Delay Stress Testing"
echo "----------------------------------------------------------------------"
python3 src/adversarial_eval.py

echo ""
echo "======================================================================"
echo " Pipeline Execution Complete!"
echo " Results & Artifacts saved to:"
echo "   - Master Plot:       results/model_comparison.png"
echo "   - Adversarial Plot:  results/adversarial_stealth_benchmark.png"
echo "   - JSON Records:      results/evaluation_metrics.json"
echo "   - Defense Guide:     MS_Thesis_Log_Correlation_RL_Guide.docx"
echo "======================================================================"
