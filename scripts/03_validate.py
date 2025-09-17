#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Pastikan paket di src/ bisa diimport
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import pandas as pd

from pleiades.validation import (
    clustering_overview,
    numeric_summary,
    silhouette_if_possible,
    compare_to_valid,
)
from pleiades.clustering import cluster_centroids


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pm_magnitude(df: pd.DataFrame) -> np.ndarray:
    return np.hypot(df["pmra"].to_numpy(), df["pmdec"].to_numpy())


def choose_pleiades_cluster(df: pd.DataFrame, valid_df: pd.DataFrame | None = None) -> dict:
    """
    Pilih label klaster Pleiades:
      - jika valid_df ada: pilih label non-noise dengan median parallax & |PM| paling dekat
      - jika tidak: pilih klaster non-noise terbesar
    """
    out = {"label": None, "method": None, "score": None, "details": {}}
    if "cluster" not in df.columns:
        return out

    cand = df[df["cluster"] >= 0].copy()
    if cand.empty:
        return out

    groups = cand.groupby("cluster", sort=False)
    if valid_df is not None and {"pmra", "pmdec", "parallax"}.issubset(valid_df.columns):
        plx_v = float(valid_df["parallax"].median()) or 1.0
        pm_v = float(np.hypot(valid_df["pmra"], valid_df["pmdec"]).median()) or 1.0

        best_score = float("inf")
        best_label = None
        best_meta = None
        for lab, g in groups:
            if not {"parallax", "pmra", "pmdec"}.issubset(g.columns):
                continue
            plx_c = float(g["parallax"].median())
            pm_c = float(np.hypot(g["pmra"], g["pmdec"]).median())
            score = abs(plx_c - plx_v) / plx_v + abs(pm_c - pm_v) / pm_v
            if score < best_score:
                best_score = score
                best_label = int(lab)
                best_meta = (plx_c, pm_c, len(g))
        if best_label is not None:
            return {
                "label": best_label,
                "method": "closest_to_valid_median",
                "score": float(best_score),
                "details": {"plx_c": best_meta[0], "pm_c": best_meta[1], "n": best_meta[2], "plx_v": plx_v, "pm_v": pm_v},
            }

    # fallback: klaster non-noise terbesar
    sizes = groups.size().sort_values(ascending=False)
    return {"label": int(sizes.index[0]), "method": "largest_non_noise", "score": None, "details": {"n": int(sizes.iloc[0])}}


def main():
    ap = argparse.ArgumentParser(description="Validasi + tandai Pleiades-only + ekspor daftar anggota.")
    ap.add_argument("--config", default="configs/default.yaml", help="Path YAML konfigurasi.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    prob_thr = float(params.get("membership_prob_threshold", 0.5))

    processed = Path(paths.get("processed", "data/processed"))
    interim = Path(paths.get("interim", "data/interim"))
    processed.mkdir(parents=True, exist_ok=True)

    print("[INFO] Memuat hasil klaster & valid…")
    df = pd.read_csv(processed / "gaia_clustered.csv")
    valid = pd.read_csv(interim / "valid_clean.csv")

    # Overview
    print("[INFO] Ikhtisar klaster…")
    labels = df["cluster"].to_numpy()
    probs = df["prob"].to_numpy() if "prob" in df.columns else None
    ov = clustering_overview(labels, probs=probs)
    print(ov)
    (processed / "validation_overview.json").write_text(ov.to_json(), encoding="utf-8")

    # Silhouette (None bila <2 klaster non-noise — ini normal)
    sil_obj = {"silhouette": float("nan")}
    X_path = processed / "X.npy"
    if X_path.exists():
        X = np.load(X_path)
        sil = silhouette_if_possible(X, labels, metric="euclidean")
        sil_obj["silhouette"] = sil if sil is None or np.isnan(sil) else float(sil)
        print(f"[INFO] Silhouette: {sil_obj['silhouette']}")
    (processed / "validation_silhouette.json").write_text(json.dumps(sil_obj, indent=2), encoding="utf-8")

    # Ringkasan numerik per-klaster (tanpa noise agar ringkas)
    print("[INFO] Ringkasan numerik per-klaster…")
    feats_for_summary = [c for c in ["parallax", "pmra", "pmdec", "color", "M_G", "ruwe"] if c in df.columns]
    ns = numeric_summary(df, label_col="cluster", features=feats_for_summary)
    ns.to_csv(processed / "numeric_summary.csv", index=True)

    # Centroid
    print("[INFO] Centroid klaster…")
    cents = cluster_centroids(df, feature_cols=[c for c in ["parallax", "pmra", "pmdec", "color", "M_G"] if c in df.columns])
    cents.to_csv(processed / "cluster_centroids.csv", index=False)

    # Komparasi global ke valid
    print("[INFO] Perbandingan global dengan dataset valid…")
    comp = compare_to_valid(df, valid)
    (processed / "compare_to_valid.json").write_text(comp.to_json(), encoding="utf-8")
    print(comp)

    # ==== Tandai Pleiades-only dan tulis balik ====
    print("[INFO] Seleksi klaster Pleiades dan penandaan kolom…")
    chosen = choose_pleiades_cluster(df, valid_df=valid)
    (processed / "pleiades_cluster_selection.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    # is_member = anggota klaster mana pun (non-noise) yang lolos threshold
    df["is_member"] = (df["cluster"] >= 0) & (df["prob"] >= prob_thr)

    # is_pleiades = hanya klaster terpilih + lolos threshold
    if chosen["label"] is not None:
        df["is_pleiades"] = (df["cluster"] == int(chosen["label"])) & (df["prob"] >= prob_thr)
        print(f"[INFO] Klaster Pleiades terpilih: {chosen['label']} (is_pleiades sum = {int(df['is_pleiades'].sum())})")
    else:
        df["is_pleiades"] = False
        print("[WARN] Tidak ada klaster terpilih. is_pleiades semuanya False.")

    # Tulis balik agar skrip gambar konsumsi info yang sama
    df.to_csv(processed / "gaia_clustered.csv", index=False)

    # Ekspor daftar anggota
    df[df["is_member"]].to_csv(processed / "pleiades_members.csv", index=False)
    df[df["is_pleiades"]].to_csv(processed / "pleiades_main_cluster_members.csv", index=False)

    print("[DONE] Validasi + penandaan + ekspor selesai.")


if __name__ == "__main__":
    main()
