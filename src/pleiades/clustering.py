# -*- coding: utf-8 -*-
"""
Fungsi-fungsi untuk menjalankan HDBSCAN, memilih anggota, menempel label/prob,
dan menghitung centroid klaster.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 30,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = "eom",
    gen_min_span_tree: bool = True,
    allow_single_cluster: bool = False,
) -> Tuple[np.ndarray, np.ndarray, "object"]:
    """
    Jalankan HDBSCAN dan kembalikan (labels, probabilities, clusterer).
    Probabilitas akan difallback ke 1.0 untuk non-noise jika atribut probabilities_ tak tersedia.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    """
    try:
        import hdbscan
    except Exception as e:
        raise ImportError("Paket 'hdbscan' tidak tersedia. Install terlebih dahulu.") from e

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        gen_min_span_tree=gen_min_span_tree,
        allow_single_cluster=allow_single_cluster,
    ).fit(X)

    labels = getattr(clusterer, "labels_", None)
    if labels is None:
        raise RuntimeError("HDBSCAN tidak menghasilkan labels_.")

    if hasattr(clusterer, "probabilities_"):
        probs = clusterer.probabilities_
    else:
        # fallback: 1.0 untuk non-noise, 0.0 untuk noise
        probs = np.where(labels >= 0, 1.0, 0.0).astype(np.float32)

    return labels.astype(int, copy=False), probs.astype(np.float32, copy=False), clusterer


def choose_members(labels: np.ndarray, probs: np.ndarray, prob_threshold: float = 0.5) -> np.ndarray:
    """
    Mask anggota: label >= 0 dan probabilitas >= ambang.
    """
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    return (labels >= 0) & (probs >= float(prob_threshold))


def attach_cluster(
    df: pd.DataFrame,
    labels: Iterable[int],
    probs: Iterable[float],
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Tempelkan kolom 'cluster', 'prob', dan 'is_member' ke DataFrame yang INDEKS-nya
    sesuai dengan data fitur yang dipakai HDBSCAN (mapping harus sudah dijaga di hulu).
    """
    labels = np.asarray(labels)
    probs  = np.asarray(probs)

    if len(df) != len(labels) or len(df) != len(probs):
        raise ValueError("Panjang df, labels, dan probs harus sama untuk re-attach yang aman.")

    out = df.copy()
    out["cluster"]    = labels.astype(int)
    out["prob"]       = probs.astype(np.float32)
    out["is_member"]  = choose_members(labels, probs, prob_threshold=prob_threshold)
    return out


def cluster_centroids(
    df_labeled: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "cluster",
) -> pd.DataFrame:
    """
    Hitung centroid klaster: mean/median untuk fitur, plus ringkasan RA/Dec/PM.
    Hanya untuk cluster >= 0 (non-noise).
    """
    if label_col not in df_labeled.columns:
        raise KeyError(f"Kolom label '{label_col}' tidak ditemukan.")

    g = df_labeled[df_labeled[label_col] >= 0].copy()
    if g.empty:
        return pd.DataFrame(columns=[label_col, "n"])

    cols_feat = [c for c in (feature_cols or []) if c in g.columns]
    cols_extra = [c for c in ["ra", "dec", "pmra", "pmdec"] if c in g.columns]

    records = []
    for lab, sub in g.groupby(label_col, sort=False):
        rec = {label_col: int(lab), "n": int(len(sub))}
        # fitur: mean/std/median
        for c in cols_feat:
            arr = pd.to_numeric(sub[c], errors="coerce")
            rec[f"{c}_mean"]   = float(arr.mean())
            rec[f"{c}_std"]    = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
            rec[f"{c}_median"] = float(arr.median())
        # ekstra: posisi & pm rata-rata + |pm| mean
        if "ra" in cols_extra:
            rec["ra_mean"] = float(pd.to_numeric(sub["ra"], errors="coerce").mean())
        if "dec" in cols_extra:
            rec["dec_mean"] = float(pd.to_numeric(sub["dec"], errors="coerce").mean())
        if "pmra" in cols_extra and "pmdec" in cols_extra:
            pmra = pd.to_numeric(sub["pmra"], errors="coerce").to_numpy()
            pmdec= pd.to_numeric(sub["pmdec"], errors="coerce").to_numpy()
            rec["pmra_mean"] = float(np.nanmean(pmra))
            rec["pmdec_mean"] = float(np.nanmean(pmdec))
            rec["pm_mag_mean"] = float(np.nanmean(np.hypot(pmra, pmdec)))
        records.append(rec)

    return pd.DataFrame.from_records(records)
