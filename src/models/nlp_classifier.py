"""
nlp_classifier.py
=================
NLP scaffold for Manufacturing Defect Detection (Week 2)
Holy Angel University — INTELSYS Final Project AY2526

Pipeline:
  1. Synthetic defect label corpus (stand-in until real metadata is available)
  2. TF-IDF vectorization of defect label text
  3. Logistic Regression classifier
  4. Macro-F1 evaluation + confusion matrix

Authors: Baguio, Doton, Bernarte, De Castro
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Synthetic defect label corpus
#    These mimic the kind of short text descriptions a QA technician or
#    automated metadata tagger would produce per inspection event.
#    Replace with real CSV metadata once dataset is acquired.
# ---------------------------------------------------------------------------

DEFECT_CORPUS = [
    # --- crack ---
    ("hairline crack along weld seam", "crack"),
    ("surface crack detected near edge", "crack"),
    ("deep crack visible under UV light", "crack"),
    ("micro crack on structural surface", "crack"),
    ("longitudinal crack on base material", "crack"),
    ("transverse crack with oxide residue", "crack"),
    ("crack propagation observed post-stress", "crack"),
    ("fatigue crack near bolt hole", "crack"),
    ("crack with rust staining", "crack"),
    ("branching crack network on panel", "crack"),
    # --- scratch ---
    ("linear scratch across painted surface", "scratch"),
    ("deep scratch exposing base metal", "scratch"),
    ("tool mark scratch pattern", "scratch"),
    ("scratch cluster from handling damage", "scratch"),
    ("fine scratch on polished surface", "scratch"),
    ("parallel scratches from abrasive contact", "scratch"),
    ("scratch visible under raking light", "scratch"),
    ("diagonal scratch on component face", "scratch"),
    ("scratch with raised edges burr", "scratch"),
    ("surface scratch from conveyor belt contact", "scratch"),
    # --- dent ---
    ("minor dent on outer panel", "dent"),
    ("impact dent with paint chipping", "dent"),
    ("sharp dent at corner junction", "dent"),
    ("dent from mechanical impact event", "dent"),
    ("dent cluster on flat surface area", "dent"),
    ("concave dent without surface fracture", "dent"),
    ("dent deformation on thin sheet metal", "dent"),
    ("dent near weld zone observed", "dent"),
    ("pressure dent from assembly clamp", "dent"),
    ("dent with surrounding stress marks", "dent"),
    # --- hole ---
    ("unwanted hole in substrate material", "hole"),
    ("punched hole misalignment detected", "hole"),
    ("corrosion hole through sheet metal", "hole"),
    ("drilled hole with rough burr edges", "hole"),
    ("void hole from casting defect", "hole"),
    ("hole diameter out of tolerance spec", "hole"),
    ("elongated hole from stress fracture", "hole"),
    ("pinhole array in coating layer", "hole"),
    ("hole cluster from acid pitting", "hole"),
    ("through-hole with delamination around edge", "hole"),
    # --- good (no defect) ---
    ("surface inspection passed no defects found", "good"),
    ("clean surface within all tolerance limits", "good"),
    ("no anomaly detected on product surface", "good"),
    ("uniform texture confirmed defect free", "good"),
    ("passed QA visual inspection cycle", "good"),
    ("surface quality meets specification standard", "good"),
    ("smooth finish observed no irregularities", "good"),
    ("product accepted after full inspection", "good"),
    ("no visible anomaly under structured light", "good"),
    ("inspection result nominal product approved", "good"),
]

LABELS = ["crack", "scratch", "dent", "hole", "good"]


def build_dataset(corpus):
    texts = [item[0] for item in corpus]
    labels = [item[1] for item in corpus]
    return texts, labels


def build_pipeline():
    """Return a sklearn Pipeline: TF-IDF → Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def evaluate(pipeline, X_test, y_test, label_names):
    """Print classification report and return macro-F1."""
    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=label_names)
    return macro_f1, report, y_pred


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("NLP Classifier — Confusion Matrix\n(TF-IDF + Logistic Regression)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved → {save_path}")


def plot_cv_scores(cv_scores, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = list(range(1, len(cv_scores) + 1))
    ax.bar(folds, cv_scores, color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean F1 = {cv_scores.mean():.3f}")
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("5-Fold Cross-Validation — Macro F1\n(NLP Scaffold)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] CV scores plot saved → {save_path}")


def main():
    print("=" * 60)
    print("  NLP Scaffold — Manufacturing Defect Classifier")
    print("  TF-IDF + Logistic Regression | Macro-F1 Logging")
    print("=" * 60)

    # Build dataset
    texts, labels = build_dataset(DEFECT_CORPUS)
    print(f"\n[✓] Dataset loaded: {len(texts)} samples, {len(set(labels))} classes")
    for lbl in LABELS:
        count = labels.count(lbl)
        print(f"     {lbl:>10s}: {count} samples")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"\n[✓] Split — Train: {len(X_train)}, Test: {len(X_test)}")

    # Build & train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[✓] Pipeline trained: TF-IDF (1-2 grams, 500 features) + LogisticRegression")

    # Evaluate
    macro_f1, report, y_pred = evaluate(pipeline, X_test, y_test, LABELS)
    print(f"\n{'─'*60}")
    print(f"  MACRO F1-SCORE : {macro_f1:.4f}")
    print(f"{'─'*60}")
    print("\nClassification Report:\n")
    print(report)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="f1_macro")
    print(f"[✓] 5-Fold CV Macro-F1 scores: {cv_scores.round(3)}")
    print(f"    Mean: {cv_scores.mean():.4f}  |  Std: {cv_scores.std():.4f}")

    # Save plots
    import os
    os.makedirs("screenshots", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, LABELS, "screenshots/nlp_confusion_matrix.png")
    plot_cv_scores(cv_scores, "screenshots/nlp_cv_scores.png")

    # Log metrics to file
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/nlp_metrics.txt", "w") as f:
        f.write(f"NLP Scaffold — Macro F1 Log\n")
        f.write(f"{'='*40}\n")
        f.write(f"Test Macro F1   : {macro_f1:.4f}\n")
        f.write(f"CV Mean F1      : {cv_scores.mean():.4f}\n")
        f.write(f"CV Std F1       : {cv_scores.std():.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")
    print("[✓] Metrics logged → data/processed/nlp_metrics.txt")

    print("\n[✓] NLP Scaffold complete. Target F1 ≥ 0.85 — check plots for details.")
    return macro_f1


if __name__ == "__main__":
    main()
