# Model Card — Manufacturing Defect Detection (Vision)

| | |
|---|---|
| **Version** | 0.2 — Week 2 Baseline |
| **Course** | INTELSY-CS 303 |
| **Team** | Baguio, Eriel Ben L. · Doton, John Harold D. · Bernarte, Karl Shane Y. · De Castro, Juan Carlo C. |
| **GitHub** | https://github.com/Haruuowo/INTELSYS-FIINALSPROJ-giT |

---

## 1. Model Overview

Binary image classifier that detects manufacturing surface defects in images.
Outputs: `Positive` (defect present) or `Negative` (no defect) + confidence score.

The system is designed as a decision-support tool for quality assurance engineers,
not an autonomous replacement for human inspection.

---

## 2. Model Details

### 2.1 Baseline Models (Week 2 — Current)

| Field | Value |
|-------|-------|
| Feature extraction | HOG — 9 orientations · 8×8 cells · 2×2 block norm |
| Input size | 128×128 px · RGB |
| HOG feature dim | 8,100 per image |
| Classifier A | Random Forest · 200 trees · class_weight=balanced · StandardScaler |
| Classifier B | SVM · RBF kernel · C=10 · class_weight=balanced · StandardScaler |
| Decision threshold | 0.35 (tuned to maximise True Positive detection) |
| Framework | scikit-learn |

### 2.2 Target Model (Week 3)

| Field | Value |
|-------|-------|
| Architecture | MobileNetV2 (fine-tuned) |
| Input size | 224×224 px · RGB · ImageNet normalisation |
| Framework | PyTorch |

---

## 3. Dataset

| Field | Value |
|-------|-------|
| Positive sources | Bridge_Crack_Image (train + val) · CrackForest · MT_Blowhole · MT_Break · MT_Crack · MT_Fray · DeepPCB (defective) |
| Negative sources | Magnetic-Tile-Defect MT_Free · DeepPCB (non-defective) |
| Split | 70% train / 15% val / 15% test · stratified · seed=42 |
| Class balancing | class_weight=balanced applied to both classifiers |
| PII | None — structural and industrial surfaces only |

---

## 4. Evaluation Results — Test Set (Week 2 Baseline)

| Model | Accuracy | Macro F1 | True Positive | Missed Defects | False Alarm Rate |
|-------|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 0.9853 | 0.9534 | 492 / 500 | 8 | 0.0000 |
| SVM (RBF) | 0.9890 | 0.9644 | 494 / 500 | 6 | 0.0000 |

### CNN Results (Week 3 — To Be Filled)

| Model | Val Accuracy | Val Macro F1 | Test Macro F1 | FAR |
|-------|:---:|:---:|:---:|:---:|
| MobileNetV2 | — | — | — | — |

---

## 5. Success Criteria

| Metric | Target | Random Forest | SVM (RBF) |
|--------|--------|:---:|:---:|
| Macro F1 | ≥ 0.85 | 0.9534 ✓ | 0.9644 ✓ |
| False-Alarm Rate | ≤ 5% | 0.00% ✓ | 0.00% ✓ |

Both baseline models exceed all project targets.

---

## 6. Limitations

- Trained on 1,000 image subset (SAMPLE_LIMIT=1000) — run on full dataset before final submission
- HOG features miss subtle texture patterns detectable by deep learning (CNN planned for Week 3)
- Negative class is significantly smaller than Positive — compensated via class_weight=balanced
- Not validated for real-time or embedded deployment

---

## 7. Intended Use

Decision-support tool for quality assurance engineers inspecting manufactured products
for surface defects. Not for fully autonomous production-line control.

---

## 8. Ethics Summary

| Risk | Mitigation |
|------|-----------|
| Worker surveillance | Pipeline restricted to surface images only — no biometric data |
| Economic displacement | System framed as decision-support, inspectors retain final authority |
| Missed defects | Threshold tuned to 0.35 to favour recall over precision |
| False alarms | Monitored explicitly — currently 0% on test set |

---

## 9. Version History

| Version | Week | Changes |
|---------|------|---------|
| 0.1 | Week 1 | Proposal · dataset identified · architecture planned |
| 0.2 | Week 2 | Data pipeline · HOG baselines · model card started |
| 0.9 | Week 3 (planned) | CNN training · Grad-CAM · full evaluation |
| 1.0 | Week 3 (final) | Final report · demo · v1.0 release |