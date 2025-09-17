# -*- coding: utf-8 -*-
"""
Preprocessing numerik: konversi tipe, standarisasi (z-score) dengan scikit-learn.
PERBAIKAN:
- Tidak reset_index → index baris asli tetap sehingga mapping aman saat re-attach label
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def as_float32(arr: np.ndarray) -> np.ndarray:
    """
    Konversi array ke float32 (hemat memori).
    """
    if arr.dtype != np.float32:
        return arr.astype(np.float32, copy=False)
    return arr


def standardize(
    df_features: pd.DataFrame,
    return_scaler: bool = True,
    as_float32_output: bool = True,
) -> Tuple[np.ndarray, Optional[StandardScaler], pd.DataFrame]:
    """
    Standarisasi fitur (z-score).
    - df_features: DataFrame hanya berisi kolom fitur; fungsi ini TIDAK reset_index
    - NaN akan dibuang pada df_features (dropna)
    Return
    ------
    X : ndarray (n_samples_used, n_features)
    scaler : StandardScaler atau None
    X_used_df : DataFrame fitur terpakai (index = index asli yang valid)
    """
    if not isinstance(df_features, pd.DataFrame):
        raise TypeError("df_features harus pandas.DataFrame")

    fdf = df_features.copy()
    for c in fdf.columns:
        fdf[c] = pd.to_numeric(fdf[c], errors="coerce")
    # buang baris yang mengandung NaN
    fdf = fdf.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(fdf.values)

    if as_float32_output:
        X = as_float32(X)

    return X, scaler if return_scaler else None, fdf
