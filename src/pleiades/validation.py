# -*- coding: utf-8 -*-
"""
Utilitas validasi: overview klaster, ringkasan numerik per-klaster,
silhouette score (jika memungkinkan), komparasi global ke dataset valid,
serta centroid fitur per-klaster.

Fungsi penting (dipakai di notebook & skrip):
- cluster_overview(df, label_col="cluster", prob_col="prob")
- clustering_overview(labels, probs=None)
- numeric_summary(df, label_col="cluster", features=[...])
- silhouette_if_possible(X, labels, metric="euclidean")
- silhouette_safely(df, features, label_col="cluster", sample=20000)
- cluster_centroids(df, label_col="cluster", features=[...])
- compare_to_valid(df_members, valid_df)
- compare_to_valid_global(df_members, valid_df)  # alias
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "cluster_overview",
    "clustering_overview",
    "numeric_summary",
    "silhouette_if_possible",
    "silhouette_safely",
    "cluster_centroids",
    "compare_to_valid",
    "compare_to_valid_global",
]


# ---------------------------------------------------------------------------
# Ikhtisar klaster
# ---------------------------------------------------------------------------

def clustering_overview(labels: Iterable[int], probs: Optional[Iterable[float]] = None) -> pd.Series:
    """
    Ikhtisar cepat hasil klaster berdasarkan array label (dan opsional probabilitas).
    Return: pd.Series agar mudah .to_dict() / .to_json()
    """
    lab = np.asarray(list(labels))
    n = int(lab.size)
    non_noise = lab[lab >= 0]
    n_clusters = int(len(np.unique(non_noise))) if non_noise.size > 0 else 0
    frac_noise = float(np.mean(lab < 0)) if n > 0 else float("nan")
    max_cluster_size = int(np.max([np.sum(non_noise == c) for c in np.unique(non_noise)]) if non_noise.size else 0)

    ser = pd.Series({
        "n_samples": float(n),
        "n_clusters": float(n_clusters),
        "frac_noise": float(frac_noise),
        "max_cluster_size": float(max_cluster_size),
    })

    if probs is not None:
        p = np.asarray(list(probs), dtype=float)
        if p.size:
            ser["prob_mean"]   = float(np.nanmean(p))
            ser["prob_median"] = float(np.nanmedian(p))
    return ser


def cluster_overview(
    df: pd.DataFrame,
    label_col: str = "cluster",
    prob_col: Optional[str] = "prob",
) -> pd.Series:
    """
    Convenience wrapper: ambil label (dan probabilitas) langsung dari DataFrame.
    """
    if label_col not in df.columns:
        raise KeyError(f"Kolom label '{label_col}' tidak ditemukan.")
    labels = df[label_col].to_numpy()
    probs = df[prob_col].to_numpy() if (prob_col and prob_col in df.columns) else None
    return clustering_overview(labels, probs)


# ---------------------------------------------------------------------------
# Ringkasan numerik per-klaster
# ---------------------------------------------------------------------------

def numeric_summary(
    df: pd.DataFrame,
    label_col: str = "cluster",
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Statistik (count/mean/std/min/median/max) per klaster untuk fitur yang diminta.
    Abaikan noise (cluster = -1). Kompatibel dengan Pandas terbaru.
    """
    if label_col not in df.columns:
        raise KeyError(f"Kolom label '{label_col}' tidak ditemukan.")

    feats = [c for c in (features or []) if c in df.columns]
    if not feats:
        return pd.DataFrame()

    g = df[df[label_col] >= 0].copy()
    if g.empty:
        return pd.DataFrame()

    # pastikan numeric, non-numeric -> NaN biar agregasi aman
    gg = g.copy()
    gg[feats] = gg[feats].apply(pd.to_numeric, errors="coerce")

    out = gg.groupby(label_col)[feats].agg(["count", "mean", "std", "min", "median", "max"])
    # flatten kolom MultiIndex -> "fitur_stat"
    out.columns = [f"{feat}_{stat}" for feat, stat in out.columns]
    return out


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------

def silhouette_if_possible(
    X: np.ndarray,
    labels: Iterable[int],
    metric: str = "euclidean",
) -> Optional[float]:
    """
    Hitung silhouette score untuk non-noise jika jumlah klaster >= 2.
    Return None bila tak memenuhi syarat atau scikit-learn belum terpasang.
    """
    lab = np.asarray(list(labels))
    mask = lab >= 0
    if mask.sum() < 2:
        return None

    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return None

    try:
        score = silhouette_score(X[mask], lab[mask], metric=metric)
        return float(score)
    except Exception:
        return None


def silhouette_safely(
    df: pd.DataFrame,
    features: Sequence[str],
    label_col: str = "cluster",
    sample: Optional[int] = 20000,
    random_state: int = 0,
) -> Optional[float]:
    """
    Versi aman: ambil fitur langsung dari DataFrame, gunakan hanya label non-noise,
    drop NaN, sampling opsional, standardisasi, lalu hitung silhouette.
    Return None jika tidak memenuhi syarat atau scikit-learn tidak tersedia.
    """
    # validasi kolom
    feats = [c for c in features if c in df.columns]
    if not feats or label_col not in df.columns:
        return None

    # subset non-noise
    mask_non_noise = (df[label_col] >= 0)
    X = df.loc[mask_non_noise, feats].apply(pd.to_numeric, errors="coerce").dropna()
    if X.empty:
        return None
    y = df.loc[X.index, label_col]
    if y.nunique() < 2:
        return None

    # sampling opsional
    if sample is not None and len(X) > sample:
        X = X.sample(sample, random_state=random_state)
        y = y.loc[X.index]

    # standardisasi & silhouette
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
    except Exception:
        return None

    try:
        Xs = StandardScaler().fit_transform(X.values)
        return float(silhouette_score(Xs, y.values, metric="euclidean"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Centroid per-klaster
# ---------------------------------------------------------------------------

def cluster_centroids(
    df: pd.DataFrame,
    label_col: str = "cluster",
    features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Hitung centroid (mean) untuk setiap fitur pada tiap klaster non-noise.
    Return DataFrame indeks = label klaster, kolom = fitur.
    """
    if label_col not in df.columns:
        raise KeyError(f"Kolom label '{label_col}' tidak ditemukan.")

    feats = [c for c in (features or []) if c in df.columns]
    if not feats:
        return pd.DataFrame()

    sub = df[df[label_col] >= 0].copy()
    if sub.empty:
        return pd.DataFrame()

    sub[feats] = sub[feats].apply(pd.to_numeric, errors="coerce")
    out = sub.groupby(label_col)[feats].mean()
    out.columns = [f"{c}_mean" for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Komparasi global dengan dataset valid
# ---------------------------------------------------------------------------

def _pm_angle_deg(pmra: np.ndarray, pmdec: np.ndarray) -> float:
    """
    Hitung sudut vektor proper-motion rata-rata (derajat, -180..+180) dengan konvensi atan2(pmdec, pmra).
    """
    mean_pmra  = float(np.nanmean(pmra))
    mean_pmdec = float(np.nanmean(pmdec))
    ang = np.degrees(np.arctan2(mean_pmdec, mean_pmra))
    return ang


def compare_to_valid(df: pd.DataFrame, valid_df: pd.DataFrame) -> pd.Series:
    """
    Komparasi global (mean) antara anggota hasil klasterisasi (df) dan dataset valid (valid_df).
    Menghasilkan pd.Series (siap .to_json()).
    df diharapkan sudah terfilter ke anggota (atau memiliki kolom is_member=True).
    """
    ser = pd.Series(dtype=float)

    # pilih anggota (non-noise + is_member True jika ada)
    if "cluster" in df.columns:
        base = df[df["cluster"] >= 0].copy()
    else:
        base = df.copy()

    if "is_member" in base.columns:
        members = base[base["is_member"] == True].copy()
    else:
        members = base

    # Parallax mean
    if "parallax" in members.columns and "parallax" in valid_df.columns:
        plx_c = float(pd.to_numeric(members["parallax"], errors="coerce").mean())
        plx_v = float(pd.to_numeric(valid_df["parallax"], errors="coerce").mean())
        ser["parallax_mean_cluster"] = plx_c
        ser["parallax_mean_valid"]   = plx_v
        ser["parallax_mean_diff"]    = plx_c - plx_v

    # PM magnitude mean & angle diff
    if {"pmra","pmdec"}.issubset(members.columns) and {"pmra","pmdec"}.issubset(valid_df.columns):
        pmc = np.hypot(pd.to_numeric(members["pmra"], errors="coerce"),
                       pd.to_numeric(members["pmdec"], errors="coerce")).mean()
        pmv = np.hypot(pd.to_numeric(valid_df["pmra"], errors="coerce"),
                       pd.to_numeric(valid_df["pmdec"], errors="coerce")).mean()
        ser["pm_mag_mean_cluster"] = float(pmc)
        ser["pm_mag_mean_valid"]   = float(pmv)
        ser["pm_mag_mean_diff"]    = float(pmc - pmv)

        ang_c = _pm_angle_deg(pd.to_numeric(members["pmra"], errors="coerce").to_numpy(),
                              pd.to_numeric(members["pmdec"], errors="coerce").to_numpy())
        ang_v = _pm_angle_deg(pd.to_numeric(valid_df["pmra"], errors="coerce").to_numpy(),
                              pd.to_numeric(valid_df["pmdec"], errors="coerce").to_numpy())
        # beda sudut minimum (0..180)
        d = abs(ang_c - ang_v) % 360.0
        if d > 180.0:
            d = 360.0 - d
        ser["pm_angle_diff_deg"] = float(d)

    # Color & M_G mean (opsional)
    if "color" in members.columns and "color" in valid_df.columns:
        cm = float(pd.to_numeric(members["color"], errors="coerce").mean())
        cv = float(pd.to_numeric(valid_df["color"], errors="coerce").mean())
        ser["color_mean_cluster"] = cm
        ser["color_mean_valid"]   = cv
        ser["color_mean_diff"]    = cm - cv

    if "M_G" in members.columns and "M_G" in valid_df.columns:
        mm = float(pd.to_numeric(members["M_G"], errors="coerce").mean())
        mv = float(pd.to_numeric(valid_df["M_G"], errors="coerce").mean())
        ser["M_G_mean_cluster"] = mm
        ser["M_G_mean_valid"]   = mv
        ser["M_G_mean_diff"]    = mm - mv

    return ser


def compare_to_valid_global(df_members: pd.DataFrame, valid_df: pd.DataFrame) -> pd.Series:
    """
    Alias untuk kompatibilitas ke belakang (dipakai di beberapa template).
    Perilaku sama dengan compare_to_valid(...).
    """
    return compare_to_valid(df_members, valid_df)
