"""
preprocessor.py
---------------
Audio loading, resampling, MFCC feature extraction, and log-mel spectrogram
computation for the GTZAN music genre classification pipeline.

All functions follow the exact configuration used in the conference paper:
  - Sample rate  : 22,050 Hz  (mono)
  - Clip length  : 30 seconds (661,794 samples)
  - MFCC         : 20 coefficients → 40-dim vector (mean + std)
  - Mel-spec     : 128 bands, FFT=2048, hop=512, normalized → (128, 130, 1)
"""

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22_050
CLIP_DURATION = 30                          # seconds
CLIP_SAMPLES  = SAMPLE_RATE * CLIP_DURATION # 661,794
N_MFCC        = 20
N_MELS        = 128
N_FFT         = 2048
HOP_LENGTH    = 512
TIME_FRAMES   = 130                         # fixed mel-spec width after pad/trunc

GENRES = ['blues', 'classical', 'country', 'disco',
          'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as mono at the target sample rate and pad/truncate to
    exactly CLIP_SAMPLES (30 seconds).

    Args:
        file_path: Path to the .wav file.
        sr:        Target sample rate (default 22,050 Hz).

    Returns:
        1-D float32 numpy array of length CLIP_SAMPLES.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)), mode='constant')
    else:
        y = y[:CLIP_SAMPLES]

    return y.astype(np.float32)


# ── MFCC Features (for classical ML) ─────────────────────────────────────────

def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC) -> np.ndarray:
    """Extract a 40-dimensional MFCC feature vector (mean + std over time).

    Args:
        y:      Audio signal (mono, float32).
        sr:     Sample rate.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        1-D float32 array of shape (2 * n_mfcc,).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    return np.concatenate([mfcc.mean(axis=1),
                           mfcc.std(axis=1)], axis=0).astype(np.float32)


def build_mfcc_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Walk data_dir/GENRE/*.wav and build the full MFCC feature matrix.

    Args:
        data_dir: Root directory containing genre sub-folders.

    Returns:
        X: float32 array of shape (N, 40).
        y: int32  array of shape (N,)  — integer genre labels.
    """
    X_list, y_list = [], []
    data_path = Path(data_dir)

    for genre in GENRES:
        genre_dir = data_path / genre
        if not genre_dir.exists():
            print(f"[WARNING] Genre folder not found: {genre_dir}")
            continue
        wav_files = sorted(genre_dir.glob('*.wav'))
        for wav in tqdm(wav_files, desc=f'MFCC {genre}', leave=False):
            try:
                y_audio = load_audio(str(wav))
                feat = extract_mfcc(y_audio)
                X_list.append(feat)
                y_list.append(GENRE_TO_IDX[genre])
            except Exception as e:
                print(f"[SKIP] {wav.name}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def mfcc_from_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the pre-extracted MFCC features from the provided Kaggle CSV
    (features_30_sec.csv or features_3_sec.csv).

    The CSV has columns: filename, length, chroma_stft_mean, ..., mfcc20_var, label.
    We use columns mfcc1_mean … mfcc20_var (40 columns) as features.

    Args:
        csv_path: Path to the features CSV.

    Returns:
        X: float32 array of shape (N, 40).
        y: int32  array of shape (N,).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    mfcc_cols = [c for c in df.columns
                 if c.startswith('mfcc') and (c.endswith('_mean') or c.endswith('_var'))]
    mfcc_cols = sorted(mfcc_cols, key=lambda c: (int(c.split('_')[0][4:]),
                                                   0 if c.endswith('mean') else 1))
    X = df[mfcc_cols].values.astype(np.float32)
    y = np.array([GENRE_TO_IDX[g] for g in df['label']], dtype=np.int32)
    return X, y


# ── Mel-Spectrogram Features (for deep models) ────────────────────────────────

def compute_melspectrogram(y: np.ndarray,
                           sr: int = SAMPLE_RATE,
                           n_mels: int = N_MELS,
                           n_fft: int = N_FFT,
                           hop_length: int = HOP_LENGTH,
                           time_frames: int = TIME_FRAMES) -> np.ndarray:
    """Compute a normalized log-mel spectrogram with fixed time dimension.

    Processing steps:
      1. Compute mel-spectrogram power.
      2. Convert to dB scale.
      3. Normalize to zero mean / unit variance.
      4. Pad or truncate time axis to `time_frames`.
      5. Expand channel dimension.

    Args:
        y:           Audio signal (mono, float32).
        sr:          Sample rate.
        n_mels:      Number of mel bands.
        n_fft:       FFT window size.
        hop_length:  Hop length between frames.
        time_frames: Fixed number of time frames (default 130).

    Returns:
        float32 array of shape (n_mels, time_frames, 1) — ready for CNN input.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)

    # Normalize
    mu, sigma = log_mel.mean(), log_mel.std()
    log_mel = (log_mel - mu) / (sigma + 1e-9)

    # Pad or truncate time axis
    T = log_mel.shape[1]
    if T < time_frames:
        pad_width = time_frames - T
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :time_frames]

    return log_mel[..., np.newaxis].astype(np.float32)  # (128, 130, 1)


def build_spectrogram_dataset(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Walk data_dir/GENRE/*.wav and build the full spectrogram dataset.

    Args:
        data_dir: Root directory containing genre sub-folders.

    Returns:
        X: float32 array of shape (N, 128, 130, 1).
        y: int32  array of shape (N,).
    """
    X_list, y_list = [], []
    data_path = Path(data_dir)

    for genre in GENRES:
        genre_dir = data_path / genre
        if not genre_dir.exists():
            print(f"[WARNING] Genre folder not found: {genre_dir}")
            continue
        wav_files = sorted(genre_dir.glob('*.wav'))
        for wav in tqdm(wav_files, desc=f'Spectrogram {genre}', leave=False):
            try:
                y_audio = load_audio(str(wav))
                spec = compute_melspectrogram(y_audio)
                X_list.append(spec)
                y_list.append(GENRE_TO_IDX[genre])
            except Exception as e:
                print(f"[SKIP] {wav.name}: {e}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ── Stratified Split ──────────────────────────────────────────────────────────

def stratified_split(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2,
                     random_state: int = 42):
    """Stratified 80/20 train-test split at the track level.

    Args:
        X:            Feature matrix.
        y:            Integer label array.
        test_size:    Fraction for test split (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)
