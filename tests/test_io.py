# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd

from pleiades.io import load_gaia, load_valid, save_df

def test_load_gaia_and_save(tmp_path, schema, gaia_df_raw):
    p = tmp_path / "gaia.csv"
    gaia_df_raw.to_csv(p, index=False)
    df = load_gaia(p, schema, read_only_mapped_cols=True)
    # Kolom yang diminta di schema harus ada
    assert set(df.columns) <= set(gaia_df_raw.columns)
    assert len(df) == len(gaia_df_raw)

    # Tes save_df
    outp = tmp_path / "out" / "saved.csv"
    save_df(df, outp)
    assert outp.exists()
    r = pd.read_csv(outp)
    assert len(r) == len(df)

def test_load_valid_and_rename(tmp_path, schema, valid_df_raw):
    p = tmp_path / "valid.csv"
    valid_df_raw.to_csv(p, index=False)
    df = load_valid(p, schema)
    # Sudah di-rename ke nama internal
    for c in ["ra", "dec", "parallax", "pmra", "pmdec"]:
        assert c in df.columns
    # Opsional kolom
    assert "G" in df.columns
    assert "color" in df.columns
