"""
models.py
---------
CNN-LSTM model architecture for mel-spectrogram-based music genre
classification (both flat 10-way and hierarchical node variants).

Architecture (from the conference paper):
  Input  : (128, 130, 1)
  Block 1: Conv2D(32, 3×3, ReLU, same) → MaxPool2D(2×2)
  Block 2: Conv2D(64, 3×3, ReLU, same) → MaxPool2D(2×2)
  Block 3: Conv2D(128, 3×3, ReLU, same) → MaxPool2D(2×2)
  Reshape: (batch, 16, 2048)   [16 time steps × 2048 features]
  LSTM   : 128 units, dropout=0.5
  Output : Dense(n_classes, softmax)

Training config:
  Optimizer : Adam (lr=1e-3)
  Loss      : sparse_categorical_crossentropy
  Batch     : 16
  Callbacks : EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
  Validation: 20 % of training split
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# Fix seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ── Model Builder ─────────────────────────────────────────────────────────────

def build_cnn_lstm(input_shape: tuple = (128, 130, 1),
                   n_classes: int = 10,
                   lstm_units: int = 128,
                   dropout: float = 0.5,
                   learning_rate: float = 1e-3) -> tf.keras.Model:
    """Build and compile the CNN-LSTM model.

    Args:
        input_shape:   Shape of one spectrogram sample (H, W, C).
        n_classes:     Number of output classes.
        lstm_units:    Number of LSTM hidden units.
        dropout:       Dropout rate applied inside the LSTM.
        learning_rate: Adam learning rate.

    Returns:
        Compiled tf.keras.Model (~1.2 M parameters for 10-class variant).
    """
    inp = layers.Input(shape=input_shape, name='mel_input')

    # ── CNN backbone ──────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      name='conv1')(inp)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      name='conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)

    # After 3 × (2×2) max-pool: spatial dims reduced by 2³ = 8
    # Input (128, 130) → (16, 16) with 128 filters
    # Permute to (batch, time, freq, channels) then flatten freq×ch per timestep
    x = layers.Permute((2, 1, 3), name='permute')(x)  # (batch, W', H', C)
    _, T, F, C = x.shape
    x = layers.Reshape((T, F * C), name='reshape')(x)  # (batch, 16, 2048)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    x = layers.LSTM(lstm_units, dropout=dropout, name='lstm')(x)

    # ── Output ────────────────────────────────────────────────────────────────
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = models.Model(inp, out, name=f'CNN_LSTM_{n_classes}class')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Training Wrapper ──────────────────────────────────────────────────────────

def train_model(model: tf.keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray,   y_val: np.ndarray,
                epochs: int = 50,
                batch_size: int = 16,
                patience: int = 5) -> tf.keras.callbacks.History:
    """Train the model with early stopping.

    Args:
        model:      Compiled tf.keras.Model.
        X_train:    Training spectrograms (N, 128, 130, 1).
        y_train:    Training integer labels (N,).
        X_val:      Validation spectrograms.
        y_val:      Validation integer labels.
        epochs:     Maximum training epochs (default 50).
        batch_size: Mini-batch size (default 16).
        patience:   Early-stopping patience on val_accuracy (default 5).

    Returns:
        Keras History object.
    """
    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def train_val_split(X: np.ndarray, y: np.ndarray,
                    val_size: float = 0.2,
                    random_state: int = 42):
    """Split training data into train / validation sets.

    Args:
        X:            Feature array.
        y:            Label array.
        val_size:     Fraction for validation (default 0.2).
        random_state: Seed.

    Returns:
        X_tr, X_val, y_tr, y_val
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=val_size,
                            stratify=y, random_state=random_state)
