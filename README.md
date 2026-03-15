# Hierarchical Deep Learning and Classical Baselines for Music Genre Classification

**Authors:** B. Naishadha · P. V. S. Kaushik Reddy · M. Chandralekha · Rahul Raj M  
**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai  
**Course:** 22AIE311 – Software Engineering (Project Based)  

---

## Overview

This repository implements a complete **music genre classification pipeline** on the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), comparing:

- **Classical ML baselines** — SVM (RBF kernel) and Random Forest on MFCC features
- **Flat CNN-LSTM** — mel-spectrogram-based deep model for 10-way classification
- **Hierarchical LSTM Tree** — 7-node divide-and-conquer architecture following the strong/mild genre taxonomy

### Genre Taxonomy

```
Root
├── STRONG: hiphop, metal, pop, rock, reggae
│   ├── sub-strong1: hiphop, metal, rock   → LSTM3a (3-way)
│   └── sub-strong2: pop, reggae           → LSTM3b (2-way)
└── MILD: jazz, disco, country, classical, blues
    ├── sub-mild1: disco, country          → LSTM3c (2-way)
    └── sub-mild2: jazz, classical, blues  → LSTM3d (3-way)
```

### Results Summary

| Model | Input | Test Accuracy |
|---|---|---|
| SVM (RBF, C=10) | MFCC mean/std (40-dim) | **68.0%** |
| Random Forest (300 trees) | MFCC mean/std (40-dim) | 63.0% |
| Flat CNN-LSTM | Log-mel spectrogram (128×130) | 54.5% |
| LSTM1 — strong vs. mild | Log-mel spectrogram | 74.5% |
| LSTM2a — sub-strong split | Log-mel spectrogram | 81.0% |
| LSTM2b — sub-mild split | Log-mel spectrogram | 76.0% |
| LSTM3a — hiphop/metal/rock | Log-mel spectrogram | 70.0% |
| LSTM3b — pop/reggae | Log-mel spectrogram | **87.5%** |
| LSTM3c — disco/country | Log-mel spectrogram | 85.0% |
| LSTM3d — jazz/classical/blues | Log-mel spectrogram | 76.7% |

---

## Repository Structure

```
music-genre-classification-hierarchical-lstm/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── preprocessor.py           # Audio loading, MFCC & mel-spectrogram extraction
│   ├── models.py                 # CNN-LSTM architecture builder
│   ├── hierarchy.py              # Hierarchical LSTM tree: training & inference
│   ├── classifiers.py            # SVM and Random Forest wrappers
│   └── evaluator.py              # Metrics, plots, confusion matrices, ROC curves
│
├── notebooks/
│   ├── 01_preprocessing.ipynb    # Data loading & feature extraction walkthrough
│   ├── 02_classical_baselines.ipynb   # SVM & Random Forest training & evaluation
│   ├── 03_cnn_lstm_flat.ipynb    # Flat CNN-LSTM training & evaluation
│   ├── 04_hierarchical_lstm.ipynb     # Full hierarchical tree training & evaluation
│   └── 05_full_pipeline.ipynb    # End-to-end pipeline in one notebook
│
├── data/
│   ├── features_30_sec.csv       # Pre-extracted features for 30-sec clips (1000 rows)
│   └── features_3_sec.csv        # Pre-extracted features for 3-sec clips (9990 rows)
│
├── results/                      # Auto-generated plots & model outputs
│   └── .gitkeep
│
└── tests/
    └── test_preprocessor.py      # Unit tests for preprocessing functions
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/amrita-cse-aie/music-genre-classification-hierarchical-lstm.git
cd music-genre-classification-hierarchical-lstm
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset

Download the GTZAN dataset from Kaggle:
```
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
```

Unzip and place under:
```
data/genres_original/<genre>/<audio_files>.wav
```

Or in **Google Colab**, mount your Drive and set:
```python
DATA_PATH = '/content/drive/MyDrive/GTZAN/genres_original/'
```

The pre-extracted feature CSVs (`features_30_sec.csv`, `features_3_sec.csv`) are already included in `data/` — these are sufficient to run all classical ML experiments without needing the raw audio.

---

## Quick Start

### Run SVM & Random Forest (no GPU needed, uses CSV features)

```python
from src.classifiers import train_svm, train_random_forest
from src.evaluator import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/features_30_sec.csv')
feature_cols = [c for c in df.columns if c not in ['filename', 'length', 'label']]
X = df[feature_cols].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

svm_model, svm_scaler = train_svm(X_train, y_train)
evaluate_model(svm_model, X_test, y_test, svm_scaler, model_name='SVM')
```

### Run Hierarchical Inference on a new audio file

```python
from src.preprocessor import compute_melspectrogram, load_audio
from src.hierarchy import HierarchicalLSTMTree

# Load pre-trained tree (after running notebook 04)
tree = HierarchicalLSTMTree.load('results/hierarchical_tree/')
audio = load_audio('path/to/your/song.wav')
spec = compute_melspectrogram(audio)
result = tree.predict(spec)
print(result)
# {'genre': 'jazz', 'leaf_prob': 0.82, 'top_level': 'mild', 'subgroup': 'sub-mild2'}
```

---

## Reproducibility

All random seeds are fixed:
```python
import numpy as np, tensorflow as tf, random
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
```

---

## Citation

```
B. Naishadha, P. V. S. Kaushik Reddy, M. Chandralekha, Rahul Raj M,
"Hierarchical Deep Learning and Classical Baselines for Music Genre Classification",
Amrita Vishwa Vidyapeetham, 2026.
```
