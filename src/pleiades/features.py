# -*- coding: utf-8 -*-
"""
Harmonisasi kolom, rekayasa fitur (color, M_G), filter kualitas (RUWE), dan seleksi fitur.
PERBAIKAN PENTING:
- Validasi kolom wajib (parallax/pmra/pmdec) → error jika hilang
- Dukungan reskalasi unit (parallax & PM) dari argumen atau dari schema/params
- Sanity warning bila median parallax & |PM| "tidak masuk akal" untuk Pleiades
- Tidak reset_index agar pemetaan baris tetap aman
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------- helpers ---------------------------- #
def _build_rename_map(mapping: Dict[str, str], df_cols: Iterable[str]) -> Dict[str, str]:
    """
    mapping: dict target_internal → source_name
    return:  dict source_name → target_internal, hanya untuk kolom yang ada
    """
    inv = {}
    for tgt, src in (mapping or {}).items():
        if src in df_cols:
            inv[src] = tgt
    return inv


def add_absolute_magnitude(
    df: pd.DataFrame,
    parallax_col: str = "parallax",
    g_candidates: Tuple[str, ...] = ("g", "gmag"),
    out_col: str = "M_G",
) -> pd.DataFrame:
    """
    Hitung M_G memakai parallax dalam milliarcsecond (mas):
    M_G = G + 5*log10(parallax_mas) - 10
    (identik dengan M_G = G - 5*log10(1000/pi_mas) + 5)
    """
    if parallax_col not in df.columns:
        return df

    g_col = next((c for c in g_candidates if c in df.columns), None)
    if g_col is None:
        return df

    out = df.copy()
    plx = out[parallax_col].astype("float64")
    g = out[g_col].astype("float64")

    # hanya untuk parallax>0
    m = plx > 0
    M_G = pd.Series(np.nan, index=out.index, dtype="float64")
    M_G.loc[m] = g.loc[m] + 5.0 * np.log10(plx.loc[m]) - 10.0
    out[out_col] = M_G
    return out


# ---------------------------- main API ---------------------------- #
def harmonize_gaia(
    df: pd.DataFrame,
    schema: Optional[Dict] = None,
    ruwe_max: Optional[float] = 1.4,
    dropna_features: bool = False,
    unit_rescale: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    - Rename ke skema internal: ['ra','dec','parallax','pmra','pmdec','g','bp','rp','ruwe'] (jika ada)
    - Filter basic: parallax>0, ruwe<ruwe_max (jika kolom ada)
    - (Opsional) rescale unit: dict {'parallax': factor, 'pm': factor}
    - Tambah fitur: color = bp - rp, M_G
    - dropna_features: jika True → buang baris NaN minimal di (parallax, pmra, pmdec)
    """
    out = df.copy()

    # 1) Rename
    gmap: Dict[str, str] = (schema.get("gaia") or {}) if isinstance(schema, dict) else {}
    rename_map = _build_rename_map(gmap, out.columns)
    if rename_map:
        out = out.rename(columns=rename_map)

    # 2) Validasi kolom wajib
    core_need = ["parallax", "pmra", "pmdec"]
    missing = [c for c in core_need if c not in out.columns]
    if missing:
        raise KeyError(f"Kolom Gaia wajib hilang: {missing}. Cek configs.schema.gaia dan header CSV.")

    # 3) Filter basic
    out = out[out["parallax"].astype("float64") > 0].copy()
    if ruwe_max is not None and "ruwe" in out.columns:
        out = out[out["ruwe"].astype("float64") < float(ruwe_max)].copy()

    # 4) (Opsional) rescale unit
    if unit_rescale is None:
        # dukungan fallback: bisa tersimpan di schema['unit_rescale'] atau schema['params']['unit_rescale']
        if isinstance(schema, dict):
            unit_rescale = schema.get("unit_rescale") or (schema.get("params", {}) or {}).get("unit_rescale")
    if unit_rescale:
        if "parallax" in unit_rescale and "parallax" in out.columns:
            out["parallax"] = out["parallax"].astype("float64") * float(unit_rescale["parallax"])
        if "pm" in unit_rescale and {"pmra", "pmdec"}.issubset(out.columns):
            out["pmra"] = out["pmra"].astype("float64") * float(unit_rescale["pm"])
            out["pmdec"] = out["pmdec"].astype("float64") * float(unit_rescale["pm"])

    # 5) Fitur turunan: color & M_G
    if {"bp", "rp"}.issubset(out.columns):
        out["color"] = out["bp"].astype("float64") - out["rp"].astype("float64")
    out = add_absolute_magnitude(out, parallax_col="parallax", g_candidates=("g", "gmag"), out_col="M_G")

    # 6) Drop NA pada fitur inti (opsional, tanpa reset index)
    if dropna_features:
        base_feats = ["parallax", "pmra", "pmdec"]
        out = out.dropna(subset=[c for c in base_feats if c in out.columns])

    # 7) Sanity warning untuk unit
    try:
        med_plx = float(out["parallax"].median())
        med_pm = float(np.hypot(out["pmra"], out["pmdec"]).median())
        if med_plx < 2.0 and med_pm < 10.0:
            print("[WARN] Median parallax & |PM| tampak terlalu kecil untuk Pleiades. "
                  "Cek unit atau gunakan params.unit_rescale (parallax & pm).")
    except Exception:
        pass

    return out


def harmonize_valid(
    df: pd.DataFrame,
    schema: Optional[Dict] = None,
    ruwe_max: Optional[float] = None,
    dropna_features: bool = False,
) -> pd.DataFrame:
    """
    Rename dataset valid ke skema internal (jika mapping tersedia),
    tambah `M_G` bila memungkinkan, dan (opsional) filter RUWE.
    """
    out = df.copy()

    vmap: Dict[str, str] = (schema.get("valid") or {}) if isinstance(schema, dict) else {}
    rename_map = _build_rename_map(vmap, out.columns)
    if rename_map:
        out = out.rename(columns=rename_map)

    # filter dasar (opsional)
    if ruwe_max is not None and "ruwe" in out.columns:
        out = out[out["ruwe"].astype("float64") < float(ruwe_max)].copy()
    if "parallax" in out.columns:
        out = out[out["parallax"].astype("float64") > 0].copy()

    # fitur turunan untuk valid (jika ada g/gmag)
    out = add_absolute_magnitude(out, parallax_col="parallax", g_candidates=("g", "gmag"), out_col="M_G")

    if dropna_features:
        base_feats = ["parallax", "pmra", "pmdec"]
        out = out.dropna(subset=[c for c in base_feats if c in out.columns])

    return out


def select_feature_columns(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """
    Ambil subset fitur yang tersedia, buang baris yang mengandung NaN PADA FITUR TERSEBUT.
    *Tidak* reset index → penting agar pemetaan baris tetap aman untuk re-attach label.
    """
    feature_names = list(feature_names)
    valid_cols = [c for c in feature_names if c in df.columns]
    if not valid_cols:
        raise ValueError(f"Tidak ada fitur yang cocok di DataFrame. Diminta: {feature_names}")
    fdf = df.loc[:, valid_cols].apply(pd.to_numeric, errors="coerce")
    fdf = fdf.dropna()  # tetap pertahankan index asli
    return fdf
