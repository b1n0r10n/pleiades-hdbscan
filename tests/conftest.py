# -*- coding: utf-8 -*-
# Fixture & konfigurasi umum untuk semua test.

import os, sys
import numpy as np
import pandas as pd
import pytest

# Pastikan package 'pleiades' di bawah src/ bisa diimport
sys.path.insert(0, os.path.abspath("src"))

def pytest_configure(config):
    # Gunakan backend non-GUI untuk plotting
    import matplotlib
    matplotlib.use("Agg")

@pytest.fixture
def schema():
    return {
        "gaia": {
            "ra": "ra",
            "dec": "dec",
            "parallax": "parallax",
            "pmra": "pmra",
            "pmdec": "pmdec",
            "g": "phot_g_mean_mag",
            "bp": "phot_bp_mean_mag",
            "rp": "phot_rp_mean_mag",
            "ruwe": "ruwe",
        },
        "valid": {
            "ra": "RA_ICRS",
            "dec": "DE_ICRS",
            "parallax": "plx",
            "pmra": "pmRA",
            "pmdec": "pmDE",
            "gmag": "Gmag",      # opsional → akan di-rename jadi 'G'
            "color": "Bp-Rp",    # opsional → akan di-rename jadi 'color'
        },
    }

@pytest.fixture
def gaia_df_raw():
    # Data sintetis kecil mirip Gaia
    n = 10
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "ra": rng.uniform(50, 60, n),
        "dec": rng.uniform(20, 25, n),
        "parallax": rng.uniform(5.0, 8.0, n),  # mas > 0
        "pmra": rng.normal(20, 2, n),
        "pmdec": rng.normal(-45, 2, n),
        "phot_g_mean_mag": rng.uniform(9.5, 12.0, n),
        "phot_bp_mean_mag": rng.uniform(12.0, 14.0, n),
        "phot_rp_mean_mag": rng.uniform(10.0, 12.5, n),
        "ruwe": np.concatenate([np.ones(n-2)*1.1, np.array([1.8, 2.2])]),  # 2 baris buruk
    })
    return df

@pytest.fixture
def valid_df_raw():
    # Data 'valid' dari sumber berbeda (skema lain)
    n = 8
    rng = np.random.default_rng(123)
    df = pd.DataFrame({
        "RA_ICRS": rng.uniform(55, 57, n),
        "DE_ICRS": rng.uniform(22, 23, n),
        "plx": rng.uniform(5.2, 7.8, n),
        "pmRA": rng.normal(19.5, 1.2, n),
        "pmDE": rng.normal(-44.5, 1.2, n),
        "Gmag": rng.uniform(9.8, 11.8, n),
        "Bp-Rp": rng.uniform(1.0, 1.8, n),
    })
    return df
