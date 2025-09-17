# -*- coding: utf-8 -*-
"""
I/O utilities untuk membaca/menulis tabel Gaia dan dataset valid.
- Mendukung CSV/Parquet
- load_valid() langsung merapikan nama kolom ke skema internal
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


# ------------------------- low-level readers ------------------------- #
def _read_anytable(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    suf = path.suffix.lower()
    if suf in (".csv", ".tsv"):
        # engine=python + sep=None → autodetect delimiter
        return pd.read_csv(path, sep=None, engine="python")
    elif suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Ekstensi file tidak didukung: {path.suffix}")


# ------------------------- public API ------------------------- #
def load_gaia(
    path: Path,
    schema: Optional[Dict] = None,
    read_only_mapped_cols: bool = True,
) -> pd.DataFrame:
    """
    Membaca tabel Gaia mentah sesuai mapping di `schema['gaia']`.
    Fungsi ini TIDAK merename ke kolom internal—itu dilakukan di features.harmonize_gaia().
    Param
    -----
    read_only_mapped_cols: jika True, hanya kolom yang dipetakan diambil (hemat memori).
    """
    df = _read_anytable(Path(path))
    if schema is None:
        return df

    gmap: Dict[str, str] = (schema.get("gaia") or {})
    if not gmap:
        return df

    if read_only_mapped_cols:
        cols = [src for src in gmap.values() if src in df.columns]
        # sertakan RA/Dec jika ada konvensi nama umum
        for c in ("ra", "RA", "ra_icrs", "Ra", "dec", "DE", "Dec", "dec_icrs"):
            if c in df.columns and c not in cols:
                cols.append(c)
        if cols:
            df = df.loc[:, cols].copy()

    return df


def load_valid(path: Path, schema: Optional[Dict] = None) -> pd.DataFrame:
    """
    Membaca dataset valid (anggota referensi) dan MERENAME ke kolom internal standar:
    ['ra','dec','parallax','pmra','pmdec','g','gmag','bp','rp','color','ruwe'] (yang ada saja).
    """
    df = _read_anytable(Path(path))
    if not schema:
        return df

    vmap: Dict[str, str] = (schema.get("valid") or {})
    if not vmap:
        return df

    # buat rename map dari sumber → target internal
    rename_map = {}
    # target internal yang kita dukung
    targets = ["ra", "dec", "parallax", "pmra", "pmdec", "g", "gmag", "bp", "rp", "color", "ruwe"]
    for tgt in targets:
        src = vmap.get(tgt)
        if src and src in df.columns:
            rename_map[src] = tgt

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    """
    Simpan DataFrame ke CSV (ekstensi lain bisa ditambah bila perlu).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in ("", ".csv"):
        if suf == "":
            path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    elif suf in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        # default ke CSV
        df.to_csv(path, index=False)
