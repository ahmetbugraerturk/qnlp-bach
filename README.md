# Quantum Natural Language Processing for Sentence Classification

## PHYS 450 — Quantum Computing Project (Spring 2026)

**Author:** Ahmet Buğra Ertürk

---

# Overview

This project explores and compares **classical** and **quantum-inspired** approaches for sentence classification.

The repository contains:

* A **classical LSTM baseline** implemented in PyTorch
* A **QNLP pipeline** based on:

  * `lambeq`
  * DisCoCat sentence diagrams
  * PennyLane quantum circuit simulation
* Dataset generation and preprocessing utilities
* SLURM batch scripts for HPC execution
* Analysis and visualization scripts for experiments and scaling behavior

The project supports three datasets:

| Dataset     | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| `synthetic` | Artificial grammar-based sentence dataset                      |
| `sst2`      | Stanford Sentiment Treebank binary sentiment dataset           |
| `bach`      | Bach chorale sequences converted into NLP-like token sequences |

---

# Repository Structure

```text
phys450_proj/
├── README.md
├── requirements.txt
├── .gitignore
│
├── bash/
│   ├── run_lstm.sh
│   ├── run_qnlp.sh
│   ├── run_grid_search.sh
│   └── run_analyze.sh
│
├── scripts/
│   ├── data_utils/
│   │   ├── sentence_data_utils.py
│   │   ├── bach_dataset_generator.py
│   │   └── bach_qnlp_sentiment_dataset.json
│   │
│   ├── models/
│   │   ├── lstm.py
│   │   └── qnlp.py
│   │
│   └── analyze/
│       ├── analyze_grid.py
│       ├── timing_and_scaling.py
│       ├── generate_plots.py
│       └── generate_diagrams.py
│    
└── results/
    ├── example_s2c/
    │   ├── <dataset>_<label>_diagram.png
    │   ├── <dataset>_<label>_circuit_<n_layer>layer.png
    │   └── <dataset>_<label>_qiskit_<n_layer>layer.png
    │
    ├── grid_analysis_results/
    │   ├── aggregated_stats.csv
    │   ├── all_raw_results.csv
    │   └── detailed_analysis_<dataset>.png
    │
    ├── output_<model>/job_<id>/
    │   ├── config.json
    │   ├── best_model.pth
    │   ├── history.json
    │   └── training_metrics.png
    │
    ├── report_figures/
    │   ├── curves_synthetic_n10_seed<seed>.png
    │   ├── f1_vs_trainsize_<dataset>.png
    │   ├── lodd_bach_n50_seed<seed>.png
    │   ├── overfit_gap_comparison.png
    │   ├── scaling_analysis.png
    │   └── variance_comparison_n10.png
    │
    └── timing_results.csv
```

---

# Environment Setup

The project was primarily developed on the Koç University Valar HPC cluster.

## Create Conda Environment

```bash
conda create -n bach_qnlp python=3.10
conda activate bach_qnlp
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Core Technologies

| Component        | Library              |
| ---------------- | -------------------- |
| Deep Learning    | PyTorch              |
| Quantum Circuits | PennyLane            |
| QNLP Framework   | lambeq               |
| NLP Utilities    | HuggingFace datasets |
| Music Processing | music21              |
| Plotting         | matplotlib           |

---

# Running Experiments

## Classical LSTM Model

Run from the repository root:

```bash
python scripts/models/lstm.py \
    --data_source bach \
    --num_train 50 \
    --num_val 10 \
    --num_test 30 \
    --embed_dim 16 \
    --hidden_dim 32 \
    --epochs 100 \
    --learning_rate 0.01 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --batch_size 16 \
    --seed 42 \
    --output_dir output_classical/run_01
```

---

## Quantum QNLP Model

```bash
python scripts/models/qnlp.py \
    --data_source bach \
    --num_train 50 \
    --num_val 10 \
    --num_test 30 \
    --n_layers 2 \
    --epochs 100 \
    --learning_rate 0.1 \
    --weight_decay 1e-4 \
    --seed 42 \
    --output_dir output_qnlp/run_01
```

---

# Supported Dataset Options

Both training scripts support:

```text
bach
sst2
synthetic
```

Example:

```bash
python scripts/models/lstm.py --data_source sst2
```

---

# Running on SLURM (HPC)

The `bash/` directory contains batch scripts for automated execution.

## Important: Update Environment Paths

Before running any SLURM job scripts, you must edit the hardcoded Python environment paths inside the `run_*.sh` files.

The bash scripts currently contain a line similar to:

```bash
ENV=${ENV:-"/home/aerturk23/.conda/envs/bach_qnlp/bin/python"}
```

This path is specific to the original development environment on the Koç University Valar cluster.

You should replace it with the Python executable path of your own Conda environment.

Example:

```bash
ENV=${ENV:-"/home/YOUR_USERNAME/.conda/envs/YOUR_ENV_NAME/bin/python"}
```

To find your correct Python path after activating your environment:

```bash
which python
```

Copy the returned path and replace the hardcoded path in the SLURM scripts.

Files that usually require editing:

```text
bash/run_lstm.sh
bash/run_qnlp.sh
bash/run_grid_search.sh
bash/run_analyze.sh
```

Example workflow:

```bash
conda activate bach_qnlp
which python
```

Possible output:

```text
/home/username/.conda/envs/bach_qnlp/bin/python
```

Then update the scripts accordingly.

If these paths are not updated correctly, SLURM jobs may fail due to:

* Missing Python interpreters
* Missing installed packages
* Incorrect Conda environments
* Permission issues on HPC systems

## LSTM

```bash
sbatch bash/run_lstm.sh sst2 100 10 30 0.5 2026
```

## QNLP

```bash
sbatch bash/run_qnlp.sh bach 50 10 30 2 42
```

---

# SLURM Script Arguments

| Position | Meaning                                   |
| -------- | ----------------------------------------- |
| 1        | Dataset name                              |
| 2        | Training size                             |
| 3        | Validation size                           |
| 4        | Test size                                 |
| 5        | Dropout (LSTM) or number of layers (QNLP) |
| 6        | Random seed                               |

---

# Output Structure

Each run creates an output directory containing:

```text
output_<model>/job_<id>/
├── config.json
├── best_model.pth
├── history.json
└── training_metrics.png
```

| File                   | Description                 |
| ---------------------- | --------------------------- |
| `config.json`          | Hyperparameters used        |
| `best_model.pth`       | Best model checkpoint       |
| `history.json`         | Training/validation metrics |
| `training_metrics.png` | Accuracy and loss plots     |

---

# Grid Search

Grid search experiments can be launched with:

```bash
bash/run_grid_search.sh
```

The grid search script automatically evaluates multiple experiment configurations across datasets, training sizes, random seeds, and model hyperparameters.

Grid search configuration:

```text
DATA_SOURCES = ("bach", "sst2", "synthetic")
TRAIN_SIZES = (10, 50, 100, 250, 500)
SEEDS = (42, 100, 2026)
LAYERS = (1, 2)
DROPOUTS = (0.2, 0.5)
VAL_SIZE = 50
TEST_SIZE = 100
```

SLURM log outputs are stored in:

```text
output_grid_master/job_<id>/
├── output.log
└── output.err
```

Each individual experiment configuration creates a separate training output directory:

```text
output_<model>/job_<id>/
├── config.json
├── best_model.pth
├── history.json
└── training_metrics.png
```

| File | Description |
|---|---|
| `config.json` | Hyperparameters and experiment configuration |
| `best_model.pth` | Best saved model checkpoint |
| `history.json` | Training and validation metrics across epochs |
| `training_metrics.png` | Accuracy/loss visualization for the run |

The grid search framework enables systematic comparison across:

* Classical LSTM and QNLP models
* Multiple datasets
* Different training set sizes
* Random seeds
* QNLP circuit depths
* LSTM regularization settings

The analysis script aggregates all experiment outputs and computes summary statistics.

```bash
bash/run_analyze.sh
```
---

# Analysis and Visualization

The `scripts/analyze/` directory contains utilities for aggregating experiment results, generating publication-quality figures, studying runtime scaling behavior, and visualizing QNLP sentence structures.

These scripts are intended for post-processing after training experiments complete.

Running:

```bash
bash/run_analyze.sh
```

creates a full analysis directory:

```text
output_analyze/job_<id>/
├── logs/
│   ├── timing.log
│   ├── diagrams.log
│   ├── analyze.log
│   └── plots.log
│
└── results/
    ├── example_s2c/
    │   ├── <dataset>_<label>_diagram.png
    │   ├── <dataset>_<label>_circuit_<n_layer>layer.png
    │   └── <dataset>_<label>_qiskit_<n_layer>layer.png
    │
    ├── grid_analysis_results/
    │   ├── aggregated_stats.csv
    │   ├── all_raw_results.csv
    │   └── detailed_analysis_<dataset>.png
    │
    ├── report_figures/
    │   ├── curves_synthetic_n10_seed<seed>.png
    │   ├── f1_vs_trainsize_<dataset>.png
    │   ├── lodd_bach_n50_seed<seed>.png
    │   ├── overfit_gap_comparison.png
    │   ├── scaling_analysis.png
    │   └── variance_comparison_n10.png
    │
    └── timing_results.csv
```

---

## Timing and Scaling Analysis

```bash
python scripts/analyze/timing_and_scaling.py
```

This script analyzes how computational cost changes as the problem size increases.

Main functionality:

* Measures runtime growth with increasing sentence length
* Studies vocabulary size effects
* Compares classical LSTM training time against QNLP simulation time
* Evaluates scaling bottlenecks of quantum circuit simulation
* Aggregates timing statistics across multiple runs

Generated outputs:

```text
results/
├── timing_results.csv
└── report_figures/
    └── scaling_analysis.png
```

| Output | Description |
|---|---|
| `timing_results.csv` | Raw timing measurements collected across experiments |
| `scaling_analysis.png` | Visualization of runtime scaling behavior |

The scaling plots highlight one of the central observations of the project:

> QNLP simulation cost grows significantly faster than classical NLP models as sentence complexity increases.

---

## Grid Search Analysis

```bash
python scripts/analyze/analyze_grid.py
```

This script aggregates all grid-search experiment outputs and computes summary statistics.

The analysis combines results across:

* Multiple datasets
* Different training set sizes
* Multiple random seeds
* Different QNLP layer counts
* Different LSTM dropout configurations

Grid search configuration:

```text
DATA_SOURCES = ("bach", "sst2", "synthetic")
TRAIN_SIZES = (10, 50, 100, 250, 500)
SEEDS = (42, 100, 2026)
LAYERS = (1, 2)
DROPOUTS = (0.2, 0.5)
VAL_SIZE = 50
TEST_SIZE = 100
```

Generated outputs:

```text
results/grid_analysis_results/
├── aggregated_stats.csv
├── all_raw_results.csv
└── detailed_analysis_<dataset>.png
```

| Output | Description |
|---|---|
| `aggregated_stats.csv` | Mean/std performance statistics grouped by configuration |
| `all_raw_results.csv` | Raw metrics collected from all experiment runs |
| `detailed_analysis_<dataset>.png` | Dataset-specific visual summaries |

These analyses help compare:

* QNLP vs classical model performance
* Generalization behavior
* Sensitivity to training size
* Variance across random seeds
* Scaling trends across datasets

---

## Plot Generation

```bash
python scripts/analyze/generate_plots.py
```

This script generates publication-style plots from training logs and aggregated experiment outputs.

Main functionality:

* Reads saved `history.json` files
* Extracts training and validation metrics
* Produces visual summaries for experiments
* Generates report-ready figures
* Compares model behavior across datasets and configurations

Generated outputs:

```text
results/report_figures/
├── curves_synthetic_n10_seed<seed>.png
├── f1_vs_trainsize_<dataset>.png
├── lodd_bach_n50_seed<seed>.png
├── overfit_gap_comparison.png
├── scaling_analysis.png
└── variance_comparison_n10.png
```

| Figure | Description |
|---|---|
| `curves_synthetic_n10_seed<seed>.png` | Training curves for synthetic dataset experiments |
| `f1_vs_trainsize_<dataset>.png` | F1-score as a function of training set size |
| `lodd_bach_n50_seed<seed>.png` | Bach dataset learning dynamics visualization |
| `overfit_gap_comparison.png` | Comparison of train-validation overfitting gaps |
| `scaling_analysis.png` | Runtime and complexity scaling visualization |
| `variance_comparison_n10.png` | Performance variance across seeds/configurations |

These figures are primarily intended for:

* Final reports
* Research presentations
* Comparative model analysis
* Studying training stability and scalability

---

## Diagram Generation

```bash
python scripts/analyze/generate_diagrams.py
```

This script visualizes the compositional structure of QNLP pipelines.

Main functionality:

* Converts sentences into DisCoCat diagrams
* Generates lambeq compositional structures
* Draws PennyLane quantum circuits
* Exports Qiskit-compatible circuit diagrams
* Produces visualizations used in reports and presentations

Generated outputs:

```text
results/example_s2c/
├── <dataset>_<label>_diagram.png
├── <dataset>_<label>_circuit_<n_layer>layer.png
└── <dataset>_<label>_qiskit_<n_layer>layer.png
```

| Figure | Description |
|---|---|
| `<dataset>_<label>_diagram.png` | DisCoCat grammatical sentence structure |
| `<dataset>_<label>_circuit_<n_layer>layer.png` | PennyLane quantum circuit representation |
| `<dataset>_<label>_qiskit_<n_layer>layer.png` | Qiskit-rendered quantum circuit visualization |

These diagrams help demonstrate how grammatical sentence structures are transformed into compositional quantum circuits within the lambeq framework.

---

# Bach Dataset Generation

If the Bach dataset JSON file is missing, regenerate it with:

```bash
python scripts/data_utils/bach_dataset_generator.py
```

This downloads Bach chorales using `music21`, converts them into symbolic token sequences, and saves the processed dataset.

---

# Configuration Files

Both training scripts support loading hyperparameters from a JSON configuration file.

Example:

```bash
python scripts/models/lstm.py --config_file config.json
```

Example configuration:

```json
{
    "data_source": "bach",
    "num_train": 50,
    "num_val": 10,
    "num_test": 30,
    "epochs": 100,
    "learning_rate": 0.01,
    "seed": 42,
    "output_dir": "output_classical/config_run"
}
```

---

# Important Notes

## lambeq Parser Limitation

The original `BobcatParser` requires downloading remote model weights from Quantinuum servers.

Due to remote access issues during development, the project instead uses:

* `cups_reader`
* parser-free sentence structures

This avoids external network dependencies.

---

## QNLP Scaling Limitation

Quantum circuit simulation scales exponentially with sentence length.

Practical limits:

* Approximately 5 tokens per sentence
* Density matrix simulation complexity grows exponentially

This becomes the primary computational bottleneck in QNLP experiments.

---

# Research Goals

The project investigates:

* Whether QNLP models can learn meaningful sentence representations
* Runtime scaling behavior of quantum simulations
* Performance comparisons between classical and quantum-inspired NLP pipelines
* Structural advantages and limitations of compositional quantum language models

---

# Future Improvements

Potential future extensions include:

* Real quantum hardware execution
* Tensor-network circuit compression
* Transformer-based classical baselines
* Improved diagram ansätze
* Hybrid quantum-classical embeddings
* Longer sentence handling

---

# License

This project was developed for academic and research purposes as part of the PHYS 450 Quantum Computing course.
