"""
hierarchy.py
------------
Hierarchical LSTM tree for music genre classification.

Taxonomy (from Tang et al. & conference paper):
  Level 0  — LSTM1   : strong vs. mild  (binary, all 1000 tracks)
  Level 1a — LSTM2a  : sub-strong1 vs. sub-strong2  (strong subset)
  Level 1b — LSTM2b  : sub-mild1   vs. sub-mild2    (mild subset)
  Level 2a — LSTM3a  : hiphop / metal / rock         (sub-strong1)
  Level 2b — LSTM3b  : pop / reggae                  (sub-strong2)
  Level 2c — LSTM3c  : disco / country               (sub-mild1)
  Level 2d — LSTM3d  : jazz / classical / blues      (sub-mild2)

Each node uses the same CNN-LSTM backbone (build_cnn_lstm) but with
n_classes matching the node's local classification problem.
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

from .models import build_cnn_lstm, train_model
from .preprocessor import GENRES, GENRE_TO_IDX, IDX_TO_GENRE

# Fix seeds
tf.random.set_seed(42)
np.random.seed(42)

# ── Taxonomy Definition ───────────────────────────────────────────────────────

TAXONOMY = {
    # Top level
    'strong': ['hiphop', 'metal', 'pop', 'rock', 'reggae'],
    'mild':   ['jazz', 'disco', 'country', 'classical', 'blues'],
    # Mid level
    'sub_strong1': ['hiphop', 'metal', 'rock'],
    'sub_strong2': ['pop', 'reggae'],
    'sub_mild1':   ['disco', 'country'],
    'sub_mild2':   ['jazz', 'classical', 'blues'],
}

# Node definitions: (node_name, parent_group, local_classes)
NODES = [
    # (name,      genres_subset,                                n_classes)
    ('lstm1',   TAXONOMY['strong'] + TAXONOMY['mild'],         2),   # strong vs mild
    ('lstm2a',  TAXONOMY['sub_strong1'] + TAXONOMY['sub_strong2'], 2),  # ss1 vs ss2
    ('lstm2b',  TAXONOMY['sub_mild1']   + TAXONOMY['sub_mild2'],   2),  # sm1 vs sm2
    ('lstm3a',  TAXONOMY['sub_strong1'],                       3),
    ('lstm3b',  TAXONOMY['sub_strong2'],                       2),
    ('lstm3c',  TAXONOMY['sub_mild1'],                         2),
    ('lstm3d',  TAXONOMY['sub_mild2'],                         3),
]

# Maps each node's local label indices back to genre strings
NODE_LABELS = {
    'lstm1':  ['strong', 'mild'],
    'lstm2a': ['sub_strong1', 'sub_strong2'],
    'lstm2b': ['sub_mild1',   'sub_mild2'],
    'lstm3a': TAXONOMY['sub_strong1'],
    'lstm3b': TAXONOMY['sub_strong2'],
    'lstm3c': TAXONOMY['sub_mild1'],
    'lstm3d': TAXONOMY['sub_mild2'],
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _subset(X: np.ndarray, y: np.ndarray,
            genre_list: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract the subset of (X, y) belonging to genre_list and remap labels
    to 0 … len(genre_list)-1.

    Args:
        X:          Spectrogram array (N, 128, 130, 1).
        y:          Integer genre labels (N,) in GTZAN 0-9 encoding.
        genre_list: List of genre strings to keep.

    Returns:
        X_sub, y_sub (remapped), local_to_global (dict: local_idx → genre str)
    """
    global_indices = np.array([GENRE_TO_IDX[g] for g in genre_list])
    mask = np.isin(y, global_indices)
    X_sub = X[mask]
    y_global = y[mask]

    # Remap to 0..n-1 preserving order defined by genre_list
    global_to_local = {GENRE_TO_IDX[g]: i for i, g in enumerate(genre_list)}
    y_local = np.array([global_to_local[lbl] for lbl in y_global], dtype=np.int32)
    local_to_genre = {i: g for i, g in enumerate(genre_list)}

    return X_sub, y_local, local_to_genre


def _top_label(X: np.ndarray, y_global: np.ndarray) -> np.ndarray:
    """Convert global genre labels to binary strong(0) / mild(1) labels."""
    strong_idx = set(GENRE_TO_IDX[g] for g in TAXONOMY['strong'])
    return np.array([0 if lbl in strong_idx else 1
                     for lbl in y_global], dtype=np.int32)


def _mid_strong_label(X: np.ndarray, y_global: np.ndarray) -> np.ndarray:
    """Convert strong-genre labels to sub_strong1(0) / sub_strong2(1)."""
    ss1_idx = set(GENRE_TO_IDX[g] for g in TAXONOMY['sub_strong1'])
    return np.array([0 if lbl in ss1_idx else 1
                     for lbl in y_global], dtype=np.int32)


def _mid_mild_label(X: np.ndarray, y_global: np.ndarray) -> np.ndarray:
    """Convert mild-genre labels to sub_mild1(0) / sub_mild2(1)."""
    sm1_idx = set(GENRE_TO_IDX[g] for g in TAXONOMY['sub_mild1'])
    return np.array([0 if lbl in sm1_idx else 1
                     for lbl in y_global], dtype=np.int32)


# ── HierarchicalLSTMTree ──────────────────────────────────────────────────────

class HierarchicalLSTMTree:
    """Train, evaluate, save, load, and run inference through the full
    hierarchical LSTM tree.

    Usage:
        tree = HierarchicalLSTMTree()
        tree.train(X_all, y_all)
        result = tree.predict(spectrogram)   # shape (128, 130, 1)
        tree.save('results/tree/')
        tree = HierarchicalLSTMTree.load('results/tree/')
    """

    def __init__(self, epochs: int = 50, batch_size: int = 16,
                 patience: int = 5, val_size: float = 0.2):
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.val_size   = val_size
        self.node_models: dict[str, tf.keras.Model] = {}
        self.node_accuracies: dict[str, float] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def _train_node(self, name: str, X: np.ndarray, y: np.ndarray,
                    n_classes: int) -> float:
        """Train a single node and return its test accuracy."""
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.val_size,
            stratify=y, random_state=42
        )
        X_tr2, X_val, y_tr2, y_val = train_test_split(
            X_tr, y_tr, test_size=self.val_size,
            stratify=y_tr, random_state=42
        )
        model = build_cnn_lstm(n_classes=n_classes)
        train_model(model, X_tr2, y_tr2, X_val, y_val,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    patience=self.patience)
        _, acc = model.evaluate(X_te, y_te, verbose=0)
        print(f"  [{name}] test accuracy = {acc:.4f}")
        self.node_models[name] = model
        self.node_accuracies[name] = float(acc)
        return acc

    def train(self, X_all: np.ndarray, y_all: np.ndarray) -> dict[str, float]:
        """Train all seven LSTM nodes.

        Args:
            X_all: Full spectrogram array (1000, 128, 130, 1).
            y_all: Full integer label array (1000,) in 0-9 encoding.

        Returns:
            dict mapping node name → test accuracy.
        """
        print("\n=== Training LSTM1: strong vs. mild ===")
        y_top = _top_label(X_all, y_all)
        self._train_node('lstm1', X_all, y_top, n_classes=2)

        # Strong subtree
        strong_mask = np.isin(y_all, [GENRE_TO_IDX[g]
                                      for g in TAXONOMY['strong']])
        X_strong, y_strong_global = X_all[strong_mask], y_all[strong_mask]

        print("\n=== Training LSTM2a: sub-strong1 vs. sub-strong2 ===")
        y_mid_s = _mid_strong_label(X_strong, y_strong_global)
        self._train_node('lstm2a', X_strong, y_mid_s, n_classes=2)

        print("\n=== Training LSTM3a: hiphop / metal / rock ===")
        X_ss1, y_ss1, _ = _subset(X_all, y_all, TAXONOMY['sub_strong1'])
        self._train_node('lstm3a', X_ss1, y_ss1, n_classes=3)

        print("\n=== Training LSTM3b: pop / reggae ===")
        X_ss2, y_ss2, _ = _subset(X_all, y_all, TAXONOMY['sub_strong2'])
        self._train_node('lstm3b', X_ss2, y_ss2, n_classes=2)

        # Mild subtree
        mild_mask = np.isin(y_all, [GENRE_TO_IDX[g]
                                    for g in TAXONOMY['mild']])
        X_mild, y_mild_global = X_all[mild_mask], y_all[mild_mask]

        print("\n=== Training LSTM2b: sub-mild1 vs. sub-mild2 ===")
        y_mid_m = _mid_mild_label(X_mild, y_mild_global)
        self._train_node('lstm2b', X_mild, y_mid_m, n_classes=2)

        print("\n=== Training LSTM3c: disco / country ===")
        X_sm1, y_sm1, _ = _subset(X_all, y_all, TAXONOMY['sub_mild1'])
        self._train_node('lstm3c', X_sm1, y_sm1, n_classes=2)

        print("\n=== Training LSTM3d: jazz / classical / blues ===")
        X_sm2, y_sm2, _ = _subset(X_all, y_all, TAXONOMY['sub_mild2'])
        self._train_node('lstm3d', X_sm2, y_sm2, n_classes=3)

        print("\n=== Node Accuracy Summary ===")
        for node, acc in self.node_accuracies.items():
            print(f"  {node:8s}  {acc:.4f}  ({acc*100:.1f}%)")

        return self.node_accuracies

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, spec: np.ndarray) -> dict:
        """Run hierarchical inference on a single mel-spectrogram.

        Args:
            spec: Spectrogram of shape (128, 130, 1) or (1, 128, 130, 1).

        Returns:
            dict with keys:
              'genre'       — predicted leaf genre string
              'top_level'   — 'strong' or 'mild'
              'subgroup'    — e.g. 'sub_strong1'
              'leaf_prob'   — confidence at the leaf node
              'top_probs'   — [p_strong, p_mild]
        """
        if spec.ndim == 3:
            spec = spec[np.newaxis, ...]  # add batch dim

        # ── Level 0: strong vs. mild ──────────────────────────────────────
        top_probs = self.node_models['lstm1'].predict(spec, verbose=0)[0]
        top_idx   = int(np.argmax(top_probs))
        top_label = 'strong' if top_idx == 0 else 'mild'

        if top_label == 'strong':
            # ── Level 1 strong: sub_strong1 vs. sub_strong2 ───────────────
            mid_probs = self.node_models['lstm2a'].predict(spec, verbose=0)[0]
            mid_idx   = int(np.argmax(mid_probs))
            subgroup  = 'sub_strong1' if mid_idx == 0 else 'sub_strong2'

            if subgroup == 'sub_strong1':
                leaf_probs = self.node_models['lstm3a'].predict(spec, verbose=0)[0]
                leaf_genres = TAXONOMY['sub_strong1']
            else:
                leaf_probs = self.node_models['lstm3b'].predict(spec, verbose=0)[0]
                leaf_genres = TAXONOMY['sub_strong2']

        else:
            # ── Level 1 mild: sub_mild1 vs. sub_mild2 ────────────────────
            mid_probs = self.node_models['lstm2b'].predict(spec, verbose=0)[0]
            mid_idx   = int(np.argmax(mid_probs))
            subgroup  = 'sub_mild1' if mid_idx == 0 else 'sub_mild2'

            if subgroup == 'sub_mild1':
                leaf_probs = self.node_models['lstm3c'].predict(spec, verbose=0)[0]
                leaf_genres = TAXONOMY['sub_mild1']
            else:
                leaf_probs = self.node_models['lstm3d'].predict(spec, verbose=0)[0]
                leaf_genres = TAXONOMY['sub_mild2']

        leaf_idx  = int(np.argmax(leaf_probs))
        genre     = leaf_genres[leaf_idx]
        leaf_prob = float(leaf_probs[leaf_idx])

        return {
            'genre':     genre,
            'top_level': top_label,
            'subgroup':  subgroup,
            'leaf_prob': leaf_prob,
            'top_probs': top_probs.tolist(),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: str) -> None:
        """Save all node models to save_dir/<node_name>.keras"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        for name, model in self.node_models.items():
            path = os.path.join(save_dir, f'{name}.keras')
            model.save(path)
        # Save accuracies
        import json
        with open(os.path.join(save_dir, 'accuracies.json'), 'w') as f:
            json.dump(self.node_accuracies, f, indent=2)
        print(f"Tree saved to {save_dir}/")

    @classmethod
    def load(cls, save_dir: str) -> 'HierarchicalLSTMTree':
        """Load a previously saved HierarchicalLSTMTree."""
        import json
        tree = cls()
        for node_name in ['lstm1', 'lstm2a', 'lstm2b',
                          'lstm3a', 'lstm3b', 'lstm3c', 'lstm3d']:
            path = os.path.join(save_dir, f'{node_name}.keras')
            if os.path.exists(path):
                tree.node_models[node_name] = tf.keras.models.load_model(path)
        acc_path = os.path.join(save_dir, 'accuracies.json')
        if os.path.exists(acc_path):
            with open(acc_path) as f:
                tree.node_accuracies = json.load(f)
        print(f"Tree loaded from {save_dir}/")
        return tree
