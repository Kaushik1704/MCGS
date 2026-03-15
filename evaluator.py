"""
evaluator.py
------------
Evaluation utilities: classification reports, confusion matrix heatmaps,
F1-score bar charts, multiclass ROC curves, and training history plots.

All figures are saved to the results/ directory (auto-created if missing).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize

from .preprocessor import GENRES

RESULTS_DIR = 'results'


def _ensure_results():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Predict helpers ───────────────────────────────────────────────────────────

def predict_sklearn(model, X_test: np.ndarray,
                    scaler=None) -> np.ndarray:
    """Get predictions from a sklearn model (Pipeline or raw classifier)."""
    return model.predict(X_test)


def predict_keras(model, X_test: np.ndarray) -> np.ndarray:
    """Get hard predictions from a Keras model."""
    probs = model.predict(X_test, verbose=0)
    return np.argmax(probs, axis=1)


# ── Classification Report ─────────────────────────────────────────────────────

def print_classification_report(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  genre_names: list[str] = GENRES,
                                  model_name: str = 'Model') -> str:
    """Print and return a scikit-learn classification report.

    Args:
        y_true:       Ground-truth integer labels.
        y_pred:       Predicted integer labels.
        genre_names:  Class names (default all 10 GTZAN genres).
        model_name:   Display name for the header line.

    Returns:
        Report string.
    """
    report = classification_report(y_true, y_pred,
                                   target_names=genre_names,
                                   digits=3)
    acc = (y_true == y_pred).mean()
    print(f"\n{'='*55}")
    print(f" {model_name}  —  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print('='*55)
    print(report)
    return report


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          genre_names: list[str] = GENRES,
                          model_name: str = 'Model',
                          save: bool = True,
                          cmap: str = 'Blues') -> None:
    """Plot and optionally save a confusion-matrix heatmap.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        genre_names: Tick labels.
        model_name:  Title and filename stem.
        save:        Write PNG to results/ if True.
        cmap:        Matplotlib colormap name.
    """
    _ensure_results()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=genre_names, yticklabels=genre_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted Genre', fontsize=12)
    ax.set_ylabel('True Genre',      fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, pad=12)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR,
                            f'confusion_matrix_{model_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.show()


# ── F1 Score Bar Chart ────────────────────────────────────────────────────────

def plot_f1_scores(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   genre_names: list[str] = GENRES,
                   model_name: str = 'Model',
                   save: bool = True) -> None:
    """Plot a per-class F1-score bar chart.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        genre_names: Class names.
        model_name:  Title and filename stem.
        save:        Write PNG to results/ if True.
    """
    _ensure_results()
    from sklearn.metrics import f1_score
    f1s = f1_score(y_true, y_pred, average=None)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(genre_names, f1s, color=plt.cm.tab10.colors[:len(genre_names)])
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xlabel('Genre',    fontsize=12)
    ax.set_title(f'Per-Class F1 Scores — {model_name}', fontsize=14)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR,
                            f'f1_scores_{model_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.show()


# ── Multiclass ROC Curves ─────────────────────────────────────────────────────

def plot_roc_curves(model, X_test: np.ndarray,
                   y_test: np.ndarray,
                   genre_names: list[str] = GENRES,
                   model_name: str = 'CNN-LSTM',
                   is_keras: bool = True,
                   save: bool = True) -> None:
    """Plot one-vs-rest ROC curves for each class plus macro / micro averages.

    Args:
        model:       Trained model (Keras or sklearn with predict_proba).
        X_test:      Test feature array.
        y_test:      Integer label array.
        genre_names: Class names.
        model_name:  Title and filename stem.
        is_keras:    True → use model.predict(); False → model.predict_proba().
        save:        Write PNG to results/ if True.
    """
    _ensure_results()
    n_classes = len(genre_names)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))

    if is_keras:
        y_score = model.predict(X_test, verbose=0)
    else:
        y_score = model.predict_proba(X_test)

    # Per-class ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
    roc_auc['macro'] = auc(all_fpr, mean_tpr)

    # Micro average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(fpr['micro'], tpr['micro'],
            label=f'micro-avg (AUC={roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=2)
    ax.plot(fpr['macro'], tpr['macro'],
            label=f'macro-avg (AUC={roc_auc["macro"]:.2f})',
            color='navy',    linestyle=':', linewidth=2)

    colors = plt.cm.tab10.colors
    for i, (name, color) in enumerate(zip(genre_names, colors)):
        ax.plot(fpr[i], tpr[i], color=color, linewidth=1.5,
                label=f'{name} (AUC={roc_auc[i]:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1,
            label='Random Classifier (AUC=0.50)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'Multiclass ROC Curves — {model_name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR,
                            f'roc_curves_{model_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.show()


# ── Training History ──────────────────────────────────────────────────────────

def plot_training_history(history, model_name: str = 'CNN-LSTM',
                          save: bool = True) -> None:
    """Plot training and validation accuracy/loss curves.

    Args:
        history:    Keras History object returned by model.fit().
        model_name: Title and filename stem.
        save:       Write PNG to results/ if True.
    """
    _ensure_results()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'{model_name} — Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{model_name} — Loss', fontsize=13)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR,
                            f'training_history_{model_name.replace(" ", "_")}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.show()


# ── Node Accuracy Bar Chart ───────────────────────────────────────────────────

def plot_node_accuracies(node_accuracies: dict[str, float],
                         save: bool = True) -> None:
    """Bar chart of all seven hierarchical LSTM node accuracies.

    Args:
        node_accuracies: dict mapping node name → test accuracy float.
        save:            Write PNG to results/ if True.
    """
    _ensure_results()
    names = list(node_accuracies.keys())
    accs  = [node_accuracies[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=plt.cm.Set2.colors[:len(names)])
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.68, color='red', linestyle='--', linewidth=1,
               label='SVM flat accuracy (68%)')
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Hierarchical LSTM Node Accuracies', fontsize=14)
    ax.legend()
    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, 'hierarchical_node_accuracies.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved → {path}")
    plt.show()


# ── Convenience wrapper ───────────────────────────────────────────────────────

def evaluate_sklearn_model(model, X_test: np.ndarray,
                            y_test: np.ndarray,
                            model_name: str = 'Model') -> None:
    """Run full evaluation suite for an sklearn model (report + CM + F1)."""
    y_pred = model.predict(X_test)
    print_classification_report(y_test, y_pred, model_name=model_name)
    plot_confusion_matrix(y_test, y_pred, model_name=model_name)
    plot_f1_scores(y_test, y_pred, model_name=model_name)


def evaluate_keras_model(model, X_test: np.ndarray,
                          y_test: np.ndarray,
                          model_name: str = 'CNN-LSTM') -> None:
    """Run full evaluation suite for a Keras model (report + CM + F1 + ROC)."""
    y_pred = predict_keras(model, X_test)
    print_classification_report(y_test, y_pred, model_name=model_name)
    plot_confusion_matrix(y_test, y_pred, model_name=model_name)
    plot_f1_scores(y_test, y_pred, model_name=model_name)
    plot_roc_curves(model, X_test, y_test, model_name=model_name)
