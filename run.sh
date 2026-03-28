#!/usr/bin/env bash
# ============================================================
# run.sh — Reproducibility script for Manufacturing Defect Detection
# Team: Baguio, Bernarte, De Castro, Doton
# Intelligent Systems Final Project
# ============================================================
set -e

echo "=== Step 1: Install dependencies ==="
pip install -r requirements.txt

echo "=== Step 2: Download dataset ==="
python data/download_data.py

echo "=== Step 3: Run EDA Notebook ==="
jupyter nbconvert --to notebook --execute \
  notebooks/experiments/EDA_Notebook.ipynb \
  --output notebooks/experiments/EDA_Notebook_out.ipynb

echo "=== Step 4: Train CNN ==="
jupyter nbconvert --to notebook --execute \
  notebooks/experiments/CNN_Experiment.ipynb \
  --output notebooks/experiments/CNN_Experiment_out.ipynb

echo "=== Step 5: Train RL Agent ==="
jupyter nbconvert --to notebook --execute \
  notebooks/experiments/RL_Notebook.ipynb \
  --output notebooks/experiments/RL_Notebook_out.ipynb

echo ""
echo "=== Done! Results saved to notebooks/experiments/results/ ==="
echo "  cnn_metrics.json    — CNN test macro-F1 and per-epoch metrics"
echo "  rl_metrics.json     — RL episode rewards and hyperparameters"
echo "  cnn_confusion_matrix.png"
echo "  cnn_learning_curves.png"
echo "  rl_learning_curve.png"
echo "  rl_qtable_heatmap.png"
