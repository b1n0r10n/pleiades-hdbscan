#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- membuat layout src/ bisa diimport saat menjalankan dari scripts/ ---
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import pandas as pd

from pleiades.preprocessing import standardize
from pleiades.clustering import run_hdbscan, attach_cluster
from pleiades.validation import clustering_overview

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Standarisasi fitur dan HDBSCAN clustering.")
    ap.add_argument("--config", default="configs/default.yaml", help="Path YAML konfigurasi.")
    ap.add_argument("--min-cluster-size", type=int, default=None, help="Override min_cluster_size HDBSCAN.")
    ap.add_argument("--prob-threshold", type=float, default=None, help="Override ambang probabilitas anggota.")
    ap.add_argument("--save-model", action="store_true", help="Simpan model HDBSCAN & scaler.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths  = cfg.get("paths", {})
    params = cfg.get("params", {})
    feats  = params.get("features", ["parallax", "pmra", "pmdec", "color", "M_G"])

    if args.min_cluster_size is not None:
        params.setdefault("hdbscan", {})["min_cluster_size"] = args.min_cluster_size
    if args.prob_threshold is not None:
        params["membership_prob_threshold"] = args.prob_threshold

    interim   = Path(paths.get("interim", "data/interim"))
    processed = Path(paths.get("processed", "data/processed"))
    models_dir = Path("models")
    processed.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data clean
    print("[INFO] Memuat Gaia clean…")
    data = pd.read_csv(interim / "gaia_clean.csv")
    print(f"[INFO] Gaia clean: {data.shape}")

    # --- Seleksi fitur & jaga pemetaan indeks (tanpa reset_index!) ---
    print(f"[INFO] Seleksi fitur: {feats}")
    fdf = data[feats].apply(pd.to_numeric, errors="coerce")
    mask = fdf.notnull().all(axis=1)
    fdf = fdf[mask]
    print(f"[INFO] Fitur matrix: {fdf.shape}")

    # --- Standarisasi ---
    print("[INFO] Standarisasi (z-score)…")
    X, scaler, X_used = standardize(fdf, return_scaler=True, as_float32_output=True)
    # X_used.index = indeks asli yang valid
    used_index = X_used.index

    # --- HDBSCAN ---
    print("[INFO] HDBSCAN…")
    hdb = params.get("hdbscan", {}) or {}
    labels, probs, clusterer = run_hdbscan(
        X,
        min_cluster_size=hdb.get("min_cluster_size", 30),
        min_samples=hdb.get("min_samples", None),
        metric=hdb.get("metric", "euclidean"),
        cluster_selection_epsilon=hdb.get("cluster_selection_epsilon", 0.0),
        cluster_selection_method=hdb.get("cluster_selection_method", "eom"),
        gen_min_span_tree=hdb.get("gen_min_span_tree", True),
        allow_single_cluster=hdb.get("allow_single_cluster", False),
    )

    # --- Rekatkan hasil ke baris asal yang digunakan (mapping aman) ---
    print("[INFO] Menempelkan label/probabilitas ke data…")
    data_used = data.loc[used_index].copy()
    out_used  = attach_cluster(
        data_used, labels=labels, probs=probs,
        prob_threshold=params.get("membership_prob_threshold", 0.5)
    )

    # Siapkan DataFrame akhir dengan default untuk baris yang tidak dipakai (NaN fitur)
    result = data.copy()
    for c, default in [("cluster", -1), ("prob", 0.0), ("is_member", False)]:
        if c not in result.columns:
            result[c] = default
    result.loc[used_index, ["cluster","prob","is_member"]] = out_used[["cluster","prob","is_member"]].to_numpy()

    # --- Simpan hasil & artefak ---
    print("[INFO] Menyimpan hasil berlabel → processed…")
    result.to_csv(processed / "gaia_clustered.csv", index=False)
    np.save(processed / "X.npy", X)

    # Overview ringkas
    meta = clustering_overview(labels, probs=probs).to_dict()
    meta.update({
        "features": feats,
        "min_cluster_size": hdb.get("min_cluster_size", 30),
        "prob_threshold": params.get("membership_prob_threshold", 0.5),
        "n_used": int(len(used_index)),
        "n_total": int(len(data)),
    })
    (processed / "cluster_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Simpan model (opsional)
    if args.save_model:
        try:
            import joblib
            joblib.dump(clusterer, models_dir / "hdbscan.joblib")
            joblib.dump(scaler, models_dir / "scaler.joblib")
            print("[INFO] Model & scaler disimpan di models/")
        except Exception as e:
            print(f"[WARN] Gagal menyimpan model/scaler: {e}")

    print("[DONE] Clustering selesai.")

if __name__ == "__main__":
    main()
