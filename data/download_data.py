#!/usr/bin/env python3
"""
Data Download Script for Manufacturing Defect Detection Project

This script downloads the MVTec Anomaly Detection Dataset, which is used for
training and evaluating defect detection models.

Dataset License: CC BY-NC-SA 4.0
The MVTec AD dataset is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
Please refer to the dataset website for full license details.
"""

import os
import urllib.request
from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm


def download_file(url: str, destination: str) -> None:
    """
    Download a file from URL to destination with progress bar.

    Args:
        url: URL to download from
        destination: Local path to save the file
    """
    print(f"Downloading from {url}...")

    with urllib.request.urlopen(url) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        block_size = 8192

        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                file.write(buffer)
                pbar.update(len(buffer))

    print(f"Downloaded to {destination}")


def download_mvtec_dataset() -> None:
    """
    Download the MVTec Anomaly Detection Dataset.

    Note: The MVTec AD dataset requires manual download due to licensing requirements.
    Please visit https://www.mvtec.com/company/research/datasets/mvtec-ad and download
    the dataset manually, then place the zip file in data/raw/ and run this script
    to extract it.
    """
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "mvtec_anomaly_detection.zip"

    if zip_path.exists():
        print("MVTec dataset zip file found. Extracting...")
        extract_path = raw_dir / "mvtec_ad"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted to {extract_path}")
    else:
        print("MVTec AD dataset requires manual download.")
        print("Please:")
        print("1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print("2. Download the dataset (requires free registration)")
        print("3. Save the zip file as: data/raw/mvtec_anomaly_detection.zip")
        print("4. Run this script again to extract the dataset")
        print("\nDataset License: CC BY-NC-SA 4.0")


def download_placeholder_kaggle_dataset() -> None:
    """
    Placeholder function for future Kaggle dataset support.

    This function can be extended to download datasets from Kaggle
    using the Kaggle API when additional datasets are needed.
    """
    print("Kaggle dataset download not implemented yet.")
    print("To add Kaggle support:")
    print("1. Install kaggle package: pip install kaggle")
    print("2. Set up Kaggle API credentials")
    print("3. Use kaggle.api.competition_download_files() or similar")


def main() -> None:
    """Main function to download all required datasets."""
    print("Starting data download for Manufacturing Defect Detection project...")

    # Create data directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directories created: {raw_dir}, {processed_dir}")

    # Download datasets
    download_mvtec_dataset()
    download_placeholder_kaggle_dataset()

    print("Data download completed!")


if __name__ == "__main__":
    main()