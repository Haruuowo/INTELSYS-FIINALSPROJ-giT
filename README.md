# Manufacturing Defect Detection (Vision)

This project implements a manufacturing defect detection system using computer vision techniques. It employs a CNN classifier and light segmentation model for classifying and segmenting surface defects, with anomaly detection serving as a baseline approach. The system is designed to identify various types of manufacturing defects in industrial products using deep learning methods.

## Team Members

- Doton, John Harold D.
- De Castro, Juan Carlo C.
- Baguio, Eriel Ben L.
- Bernarte, Karl Shane Y.

## Project Structure

```
docs/           # Documentation and reports
data/           # Data directory
  ├── raw/      # Raw downloaded datasets
  └── processed/# Processed data for training
src/            # Source code
  ├── models/   # Model architectures
  ├── utils/    # Utility functions
  └── training/ # Training scripts
notebooks/      # Jupyter notebooks for experimentation
experiments/    # Experiment results and metrics
  └── results/  # PNG images and JSON metrics files
models/         # Saved model checkpoints
```

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/manufacturing-defect-detection.git
   cd manufacturing-defect-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Alternative: Conda Environment
If you prefer using conda:
```bash
conda env create -f environment.yml
conda activate defect-detection
```

## Tutorial

### Getting Started with Notebooks
1. Navigate to the `notebooks/` folder.
2. Open any `.ipynb` file in Jupyter Notebook or JupyterLab.
3. Follow the cells to run experiments.

### Running Experiments
1. Place your experiment scripts in appropriate folders.
2. Save results (PNG images and JSON metrics) in `experiments/results/`.
3. For RL agent experiments, ensure `rl_metrics.json` and `cnn_metrics.json` are saved there.

### Viewing Results
- Check `experiments/results/` for output images and metrics files.
- Use notebooks to visualize and analyze results.

## Data Download

To download the required datasets, run the data download script:

```bash
python data/download_data.py
```

This script will download the MVTec Anomaly Detection Dataset and save it to `data/raw/`.

## Usage

1. Download and prepare the data as described above.
2. Run training scripts from the `src/` directory.
3. Use notebooks in `notebooks/` for experimentation and visualization.

## GitHub Project Board

Track our progress on the [GitHub Project Board](https://github.com/yourusername/manufacturing-defect-detection/projects/1).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
