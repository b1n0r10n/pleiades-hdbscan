#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- membuat layout src/ bisa diimport saat menjalankan dari scripts/ ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathlib import Path
import argparse
import yaml
import pandas as pd
import numpy as np

from pleiades.io import load_gaia, load_valid, save_df
from pleiades.features import harmonize_gaia, harmonize_valid

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sanity_report(df: pd.DataFrame, title: str):
    print(f"[INFO] {title}: {df.shape}")
    cols = ["parallax", "pmra", "pmdec", "color", "M_G", "ruwe"]
    present = [c for c in cols if c in df.columns]
    desc = df[present].describe().T
    with pd.option_context("display.max_columns", 999, "display.width", 140):
        print(desc)

def main():
    ap = argparse.ArgumentParser(description="Preprocess (harmonisasi & feature engineering).")
    ap.add_argument("--config", default="configs/default.yaml", help="Path YAML konfigurasi.")
    ap.add_argument("--ruwe-max", type=float, default=None, help="Override RUWE max (opsional).")
    ap.add_argument("--dropna-features", action="store_true", help="Drop baris NaN pada fitur kunci.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths  = cfg.get("paths", {})
    schema = cfg.get("schema", {})
    params = cfg.get("params", {})

    if args.ruwe_max is not None:
        params["ruwe_max"] = args.ruwe_max

    raw_gaia = Path(paths.get("raw_gaia", "data/raw/pleiadesdataset.csv"))
    raw_valid = Path(paths.get("raw_valid", "data/raw/pleiadesvalid.csv"))
    out_interim = Path(paths.get("interim", "data/interim"))
    out_interim.mkdir(parents=True, exist_ok=True)

    print("[INFO] Memuat Gaia raw…")
    gdf_raw = load_gaia(raw_gaia, schema=schema, read_only_mapped_cols=True)
    print("[INFO] Memuat valid raw…")
    vdf_raw = load_valid(raw_valid, schema=schema)

    print("[INFO] Harmonisasi Gaia…")
    gdf = harmonize_gaia(
        gdf_raw, schema=schema,
        ruwe_max=params.get("ruwe_max", 1.4),
        dropna_features=args.dropna_features
    )
    sanity_report(gdf, "Gaia clean")

    print("[INFO] Harmonisasi valid…")
    vdf = harmonize_valid(
        vdf_raw, schema=schema,
        ruwe_max=None,  # biasanya dataset valid tidak punya RUWE; abaikan filter ini
        dropna_features=args.dropna_features
    )
    sanity_report(vdf, "Valid clean")

    # Simpan
    save_df(gdf, out_interim / "gaia_clean.csv")
    save_df(vdf, out_interim / "valid_clean.csv")
    print(f"[DONE] Preprocess selesai → {out_interim}")

    # Sanity: cek kemungkinan mismatch unit (warning saja)
    if {"pmra", "pmdec", "parallax"}.issubset(gdf.columns) and {"pmra", "pmdec", "parallax"}.issubset(vdf.columns):
        pm_g = np.hypot(gdf["pmra"], gdf["pmdec"]).median()
        pm_v = np.hypot(vdf["pmra"], vdf["pmdec"]).median()
        plx_g = gdf["parallax"].median()
        plx_v = vdf["parallax"].median()
        if pm_g > 0 and plx_g > 0:
            r_pm = pm_v / pm_g
            r_plx = plx_v / plx_g
            if 2 < r_pm < 100 and 2 < r_plx < 100:
                print(f"[WARN] Indikasi beda skala unit. Rasio median |PM| valid/gaia ≈ {r_pm:.2f}, parallax ≈ {r_plx:.2f}.")
                print("[WARN] Pertimbangkan set params.unit_rescale di configs/default.yaml (parallax & pm).")

if __name__ == "__main__":
    main()
