import pathlib, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog
from skimage import exposure

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE  = (128, 128)
SEED      = 42
CLASSES   = ["Negative", "Positive"]

HOG_PARAMS = dict(
    orientations    = 9,
    pixels_per_cell = (8, 8),
    cells_per_block = (2, 2),
    channel_axis    = -1,
)

np.random.seed(SEED)

# ── Optional limit ────────────────────────────────────────────────────────────
# Set to an integer (e.g. 1000) to run on a subset for quick testing.
# Set to None to use the full dataset
SAMPLE_LIMIT = 1000


# Extract HOG feature vector from a single image file
def extract_hog(filepath: str) -> np.ndarray:
    img = Image.open(filepath).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return hog(arr, **HOG_PARAMS)


# Load all images from a split CSV and return HOG feature matrix X and label vector y
# If SAMPLE_LIMIT is set, samples that many images per class for faster testing
def load_features(csv_path: pathlib.Path, split_name: str):
    df = pd.read_csv(csv_path)

    if SAMPLE_LIMIT is not None:
        # Sample equally from each class to preserve balance
        sampled = []
        per_class = SAMPLE_LIMIT // len(CLASSES)
        for cls in CLASSES:
            cls_df = df[df["label"] == cls]
            n = min(per_class, len(cls_df))
            sampled.append(cls_df.sample(n, random_state=SEED))
        df = pd.concat(sampled).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"  [LIMIT] Using {len(df)} images ({per_class} per class) out of full split")

    X, y = [], []
    t0 = time.time()
    for _, row in df.iterrows():
        try:
            X.append(extract_hog(row["filepath"]))
            y.append(int(row["label_idx"]))
        except Exception as e:
            print(f"  [WARN] Skipping {row['filepath']}: {e}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"  {split_name:5s}: {len(X):4d} samples | "
          f"HOG dim={X.shape[1]:,} | {time.time()-t0:.1f}s")
    return X, y


# Define Random Forest and SVM pipelines with standard scaling and class balancing
# class_weight="balanced" compensates for unequal Positive/Negative counts mathematically
def build_models():
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2,
                class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            ))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf", C=10.0, gamma="scale",
                class_weight="balanced",
                probability=True, random_state=SEED,
            ))
        ]),
    }


# Train a model, evaluate on val and test splits, and return all logged metrics
def evaluate_model(model, X_train, y_train, X_val, y_val,
                   X_test, y_test, name: str) -> dict:
    print(f"\n  Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    results = {"model": name, "train_time_s": round(time.time() - t0, 2)}

    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        far   = ((preds == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)

        results[f"{split_name}_accuracy"]         = round(accuracy_score(y, preds), 4)
        results[f"{split_name}_macro_f1"]         = round(f1_score(y, preds, average="macro"), 4)
        results[f"{split_name}_f1_positive"]      = round(f1_score(y, preds, pos_label=1, average="binary"), 4)
        results[f"{split_name}_f1_negative"]      = round(f1_score(y, preds, pos_label=0, average="binary"), 4)
        results[f"{split_name}_false_alarm_rate"] = round(far, 4)
        results[f"{split_name}_proba"]            = proba
        results[f"{split_name}_preds"]            = preds
        results[f"{split_name}_true"]             = y

        print(f"\n    [{split_name.upper()}] {name}")
        print(classification_report(y, preds, target_names=CLASSES, digits=4))
        print(f"    False-Alarm Rate: {far:.4f}  (target <= 0.05)")

    return results


# Save confusion matrices for all models side by side
def plot_confusion_matrices(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — Test Set", fontweight="bold")
    for ax, res in zip(axes, all_results):
        cm = confusion_matrix(res["test_true"], res["test_preds"])
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{res['model']}\nMacro F1={res['test_macro_f1']:.4f} | Acc={res['test_accuracy']:.4f}")
    plt.tight_layout()
    out = RESULTS_DIR / "baseline_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {out}")


# Save ROC curves for all models on the test set
def plot_roc_curves(all_results):
    fig, ax = plt.subplots(figsize=(7, 5))
    for res, color in zip(all_results, ["#2980b9", "#c0392b"]):
        fpr, tpr, _ = roc_curve(res["test_true"], res["test_proba"])
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{res['model']}  (AUC={auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.axvline(0.05, ls=":", color="grey", lw=1, label="FAR target=0.05")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "baseline_roc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  ROC curve saved:        {out}")


# Save HOG visualisation samples for both classes
def plot_hog_samples(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("HOG Feature Visualisation", fontweight="bold")
    for row_idx, cls in enumerate(CLASSES):
        samples = df[df["label"] == cls].sample(2, random_state=SEED)
        for col_base, (_, s) in enumerate(samples.iterrows()):
            img = np.array(Image.open(s["filepath"]).convert("RGB")
                           .resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32) / 255.0
            _, hog_img = hog(img, **HOG_PARAMS, visualize=True)
            axes[row_idx][col_base * 2].imshow(img)
            axes[row_idx][col_base * 2].set_title(cls, fontsize=9)
            axes[row_idx][col_base * 2].axis("off")
            axes[row_idx][col_base * 2 + 1].imshow(exposure.equalize_hist(hog_img), cmap="gray")
            axes[row_idx][col_base * 2 + 1].set_title("HOG", fontsize=9)
            axes[row_idx][col_base * 2 + 1].axis("off")
    plt.tight_layout()
    out = RESULTS_DIR / "baseline_hog_samples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  HOG samples saved:      {out}")


# Save all metrics to CSV and JSON, print summary table, and check against project targets
def save_metrics(all_results):
    metric_keys = [
        "train_time_s",
        "val_accuracy",  "val_macro_f1",  "val_f1_positive",  "val_false_alarm_rate",
        "test_accuracy", "test_macro_f1", "test_f1_positive", "test_false_alarm_rate",
    ]
    rows = [{"model": r["model"], **{k: r.get(k, "N/A") for k in metric_keys}}
            for r in all_results]
    df = pd.DataFrame(rows)

    df.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    df.to_json(RESULTS_DIR / "baseline_metrics.json", orient="records", indent=2)

    print(f"\n{'='*65}\n  BASELINE METRICS SUMMARY\n{'='*65}")
    print(df.to_string(index=False))
    print(f"{'='*65}")
    print("\nTARGET CHECK:")
    for res in all_results:
        f1_ok  = "PASS" if res["test_macro_f1"]         >= 0.85 else "FAIL"
        far_ok = "PASS" if res["test_false_alarm_rate"] <= 0.05 else "FAIL"
        print(f"  {res['model']:20s} | F1={res['test_macro_f1']:.4f} [{f1_ok}]  |  "
              f"FAR={res['test_false_alarm_rate']:.4f} [{far_ok}]")


# Main: extract HOG features, train RF and SVM, evaluate, save all plots and metrics
def main():
    print(f"\n{'='*65}\n  HOG BASELINE — Random Forest + SVM\n{'='*65}")

    print("\n[1/4] Extracting HOG features...")
    X_train, y_train = load_features(DATA_DIR / "train_split.csv", "Train")
    X_val,   y_val   = load_features(DATA_DIR / "val_split.csv",   "Val")
    X_test,  y_test  = load_features(DATA_DIR / "test_split.csv",  "Test")

    print("\n[2/4] Generating HOG visualisation...")
    plot_hog_samples(DATA_DIR / "train_split.csv")

    print("\n[3/4] Training and evaluating models...")
    all_results = []
    for name, model in build_models().items():
        res = evaluate_model(model, X_train, y_train,
                             X_val, y_val, X_test, y_test, name)
        all_results.append(res)

    print("\n[4/4] Saving plots and metrics...")
    plot_confusion_matrices(all_results)
    plot_roc_curves(all_results)
    save_metrics(all_results)

    print(f"\n{'='*65}\n  All outputs saved to results/\n{'='*65}\n")


if __name__ == "__main__":
    main()