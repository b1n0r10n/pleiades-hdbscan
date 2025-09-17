# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from pleiades.features import harmonize_gaia, harmonize_valid, select_feature_columns, add_absolute_magnitude

def test_harmonize_gaia(schema, gaia_df_raw):
    df = harmonize_gaia(gaia_df_raw, schema, ruwe_max=1.4, dropna_features=True)
    # Harus terfilter RUWE > 1.4 (2 baris dibuang)
    assert len(df) <= len(gaia_df_raw) - 2
    # Kolom turunan
    for c in ["color", "M_G"]:
        assert c in df.columns
    # Rumus M_G konsisten pada satu baris
    i = 0
    G = df.loc[i, "G"]
    plx = df.loc[i, "parallax"]
    expected = G + 5*np.log10(plx) - 10.0
    assert np.isfinite(df.loc[i, "M_G"])
    assert np.isclose(df.loc[i, "M_G"], expected, atol=1e-6)

def test_harmonize_valid(schema, valid_df_raw):
    # io.load_valid biasanya sudah rename; di sini uji langsung jalur harmonize_valid setelah rename manual:
    df0 = valid_df_raw.rename(columns={
        schema["valid"]["ra"]: "ra",
        schema["valid"]["dec"]: "dec",
        schema["valid"]["parallax"]: "parallax",
        schema["valid"]["pmra"]: "pmra",
        schema["valid"]["pmdec"]: "pmdec",
        schema["valid"]["gmag"]: "G",
        schema["valid"]["color"]: "color",
    })
    df = harmonize_valid(df0, schema, ruwe_max=None, dropna_features=True)
    assert set(["ra","dec","parallax","pmra","pmdec"]).issubset(df.columns)
    # M_G harus ada karena G & parallax tersedia
    assert "M_G" in df.columns
    # Semua parallax > 0
    assert (df["parallax"] > 0).all()

def test_select_feature_columns(schema, gaia_df_raw):
    df = harmonize_gaia(gaia_df_raw, schema, ruwe_max=2.5, dropna_features=True)
    feats = ["parallax","pmra","pmdec","color","M_G"]
    X = select_feature_columns(df, features=feats)
    assert list(X.columns) == feats
    # Jika fitur tidak ada → error
    with pytest.raises(KeyError):
        select_feature_columns(df, features=["not_exist"])
