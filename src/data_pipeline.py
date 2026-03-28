import pathlib
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR    = pathlib.Path(__file__).parent.parent
OUT_DIR     = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

ARCHIVE = pathlib.Path.home() / "Downloads" / "archive"

# All Positive folders (defective images)
POSITIVE_DIRS = [
    ARCHIVE / "Bridge_Crack_Image" / "DBCC_Training_Data_Set" / "train",
    ARCHIVE / "Bridge_Crack_Image" / "DBCC_Training_Data_Set" / "val",
    ARCHIVE / "CrackForest" / "image",
    ARCHIVE / "Magnetic-Tile-Defect" / "MT_Blowhole",
    ARCHIVE / "Magnetic-Tile-Defect" / "MT_Break",
    ARCHIVE / "Magnetic-Tile-Defect" / "MT_Crack",
    ARCHIVE / "Magnetic-Tile-Defect" / "MT_Fray",
]

# All Negative folders (defect-free images)
NEGATIVE_DIRS = [
    ARCHIVE / "Magnetic-Tile-Defect" / "MT_Free" / "Imgs",
]

# DeepPCB: scan all group folders, split by _not suffix
DEEPPCB_ROOT = ARCHIVE / "DeepPCB" / "PCBData"

TARGET_SIZE = (227, 227)
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
SEED        = 42
TRAIN_RATIO = 0.70
CLASSES     = ["Negative", "Positive"]

random.seed(SEED)
np.random.seed(SEED)

# Recursively collect all valid image files from a folder
def collect_images(folder: pathlib.Path) -> list:
    return [
        str(f) for f in sorted(folder.rglob("*"))
        if f.suffix.lower() in VALID_EXTS and f.is_file()
    ]

# Discover images from all sources, label them, and print a summary of class distribution
def discover_images() -> pd.DataFrame:
    records = []

    # ── Positive: explicit defect folders ────────────────────────────────────
    for folder in POSITIVE_DIRS:
        if not folder.exists():
            print(f"  [WARN] Skipping missing folder: {folder}")
            continue
        files = collect_images(folder)
        records += [{"filepath": f, "label": "Positive", "label_idx": 1}
                    for f in files]

    # ── DeepPCB: group folders split by _not suffix ───────────────────────────
    if DEEPPCB_ROOT.exists():
        for group_folder in sorted(DEEPPCB_ROOT.iterdir()):
            if not group_folder.is_dir():
                continue
            for sub in sorted(group_folder.iterdir()):
                if not sub.is_dir():
                    continue
                files = collect_images(sub)
                # Folders ending in _not are defect-free (Negative)
                if sub.name.endswith("_not"):
                    records += [{"filepath": f, "label": "Negative", "label_idx": 0}
                                for f in files]
                else:
                    records += [{"filepath": f, "label": "Positive", "label_idx": 1}
                                for f in files]
    else:
        print(f"  [WARN] DeepPCB folder not found: {DEEPPCB_ROOT}")

    # ── Negative: explicit clean folders ─────────────────────────────────────
    for folder in NEGATIVE_DIRS:
        if not folder.exists():
            print(f"  [WARN] Skipping missing folder: {folder}")
            continue
        files = collect_images(folder)
        records += [{"filepath": f, "label": "Negative", "label_idx": 0}
                    for f in files]

    df  = pd.DataFrame(records)
    sep = "=" * 50
    print(f"\n{sep}\n  Dataset sources\n{sep}")
    for cls in CLASSES:
        print(f"  {cls:12s}: {(df['label'] == cls).sum():,} images")
    print(f"  {'Total':12s}: {len(df):,} images\n{sep}\n")
    return df

# Load + preprocess image: open → convert to RGB → resize → normalize to [0, 1]
def load_image(filepath: str, size: tuple = TARGET_SIZE,
               normalize: bool = True) -> np.ndarray:
    img = Image.open(filepath).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    if normalize:
        arr /= 255.0
    return arr

# Stratified split into train/val/test sets, ensuring class balance in each split and reproducibility with a fixed random seed
def split_dataframe(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_RATIO),
        stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_df["label"], random_state=SEED
    )

    print("Split summary (stratified):")
    for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos = (part["label"] == "Positive").sum()
        neg = (part["label"] == "Negative").sum()
        print(f"  {name:5s}: {len(part):5,} ({len(part)/len(df)*100:4.1f}%) "
              f"| Pos={pos} Neg={neg}")
    print()
    return train_df, val_df, test_df

# Save splits as CSV files and write a config JSON for reproducibility and future reference
def save_splits(train_df, val_df, test_df, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(out_dir / f"{name}_split.csv", index=False)
    print(f"Splits saved → {out_dir.resolve()}")
    print("  train_split.csv · val_split.csv · test_split.csv")

    config = {
        "dataset":     "Manufacturing Defect Detection — combined sources",
        "sources": {
            "Positive": [str(d) for d in POSITIVE_DIRS] + ["DeepPCB group subfolders (non _not)"],
            "Negative": [str(d) for d in NEGATIVE_DIRS] + ["DeepPCB group subfolders (_not)"],
        },
        "target_size": list(TARGET_SIZE),
        "normalize":   True,
        "norm_range":  "[0, 1] — divide by 255",
        "split":       {"train": 0.70, "val": 0.15, "test": 0.15},
        "stratified":  True,
        "seed":        SEED,
        "classes":     {i: c for i, c in enumerate(CLASSES)},
    }
    with open(out_dir / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  pipeline_config.json\n")

# Batch loading function to read CSV manifest and return arrays (X, y) for a split
def load_split_as_arrays(csv_path, size: tuple = TARGET_SIZE,
                         normalize: bool = True, verbose: bool = True):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        try:
            X.append(load_image(row["filepath"], size=size, normalize=normalize))
            y.append(int(row["label_idx"]))
        except Exception as e:
            if verbose:
                print(f"  [WARN] Skipping {row['filepath']}: {e}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    if verbose:
        print(f"  Loaded {len(X)} images — shape={X.shape}")
    return X, y

# Summary plot for class distribution and split sizes using Matplotlib, saved as a PNG file in the results directory
def _bar_chart(ax, labels, values, colors, title):
    ax.bar(labels, values, color=colors, edgecolor="black", width=0.5)
    ax.set_title(title)
    ax.set_ylabel("Count")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", fontweight="bold")

def plot_summary(df, train_df, val_df, test_df, results_dir: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Data Pipeline Summary — Manufacturing Defect Detection", fontweight="bold")

    counts = df["label"].value_counts()
    _bar_chart(axes[0], counts.index, counts.values,
               ["#27ae60", "#c0392b"], "Class Distribution")

    _bar_chart(axes[1],
               ["Train (70%)", "Val (15%)", "Test (15%)"],
               [len(train_df), len(val_df), len(test_df)],
               ["#2980b9", "#8e44ad", "#e67e22"], "Images per Split")

    plt.tight_layout()
    out = results_dir / "data_pipeline_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}\n")

# Main function to run the entire pipeline: discover images, split, save CSVs, and plot summary
def run_pipeline():
    sep = "=" * 50
    print(f"\n{sep}\n  MANUFACTURING DEFECT DETECTION — DATA PIPELINE\n{sep}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df                        = discover_images()
    train_df, val_df, test_df = split_dataframe(df)
    save_splits(train_df, val_df, test_df, OUT_DIR)
    plot_summary(df, train_df, val_df, test_df, RESULTS_DIR)

    print(f"{sep}\n  Done. Files written:")
    for f in ["data/train_split.csv", "data/val_split.csv", "data/test_split.csv",
              "data/pipeline_config.json", "results/data_pipeline_summary.png"]:
        print(f"    {f}")
    print(f"{sep}\n")
    return train_df, val_df, test_df

if __name__ == "__main__":
    run_pipeline()