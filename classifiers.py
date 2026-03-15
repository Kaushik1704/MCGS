"""
classifiers.py
--------------
SVM (RBF kernel) and Random Forest classifiers for MFCC-based music genre
classification.  Both models replicate the exact configurations described in
the conference paper.

  SVM  : RBF kernel, C=10, gamma='scale', z-score normalization of features.
  RF   : 300 trees, unlimited depth, n_jobs=-1 (parallel over all CPU cores).
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# ── SVM ───────────────────────────────────────────────────────────────────────

def build_svm(C: float = 10.0, gamma: str = 'scale') -> Pipeline:
    """Return a scikit-learn Pipeline: StandardScaler → SVC(RBF).

    Args:
        C:     Regularization parameter (default 10).
        gamma: Kernel coefficient (default 'scale').

    Returns:
        Untrained sklearn Pipeline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=C, gamma=gamma,
                       probability=True, random_state=42))
    ])


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              C: float = 10.0, gamma: str = 'scale') -> Pipeline:
    """Fit and return the SVM pipeline.

    Args:
        X_train: MFCC feature matrix (N, 40).
        y_train: Integer genre labels (N,).

    Returns:
        Trained sklearn Pipeline.
    """
    model = build_svm(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model


# ── Random Forest ─────────────────────────────────────────────────────────────

def build_random_forest(n_estimators: int = 300) -> RandomForestClassifier:
    """Return an untrained Random Forest classifier.

    Args:
        n_estimators: Number of trees (default 300).

    Returns:
        Untrained RandomForestClassifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 300) -> RandomForestClassifier:
    """Fit and return the Random Forest classifier.

    Args:
        X_train:      MFCC feature matrix (N, 40).
        y_train:      Integer genre labels (N,).
        n_estimators: Number of trees (default 300).

    Returns:
        Trained RandomForestClassifier.
    """
    model = build_random_forest(n_estimators)
    model.fit(X_train, y_train)
    return model


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Serialize a sklearn model to disk using joblib."""
    joblib.dump(model, path)
    print(f"Saved → {path}")


def load_model(path: str):
    """Load a joblib-serialized sklearn model."""
    return joblib.load(path)


# ── Cross-validation helper ───────────────────────────────────────────────────

def cross_validate_svm(X: np.ndarray, y: np.ndarray,
                       cv: int = 5, C: float = 10.0) -> dict:
    """Run k-fold cross-validation on the SVM and return mean ± std accuracy.

    Args:
        X:  Full MFCC feature matrix.
        y:  Full label array.
        cv: Number of folds (default 5).
        C:  Regularization parameter.

    Returns:
        dict with keys 'mean', 'std', 'scores'.
    """
    from sklearn.model_selection import cross_val_score
    model = build_svm(C=C)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    result = {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    print(f"SVM {cv}-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    return result
