# -*- coding: utf-8 -*-
"""
scripts/04_make_figures.py
Membuat seluruh visualisasi hasil klasterisasi Pleiades–HDBSCAN.

Contoh:
    python scripts/04_make_figures.py \
        --config configs/default.yaml \
        --prob-thresh 0.5 \
        --panels-for all

Pencarian hasil otomatis di data/processed/:
  - *cluster*.parquet|csv|csv.gz atau *label*.*
  - fallback: parquet/csv terbaru apa pun
Atau tentukan langsung dengan:
  --input data/processed/namafile.parquet

Secara opsional memuat model HDBSCAN di models/hdbscan.(pkl|joblib)
untuk membuat condensed tree (jika tersedia).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

# backend non-interaktif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import joblib
import numpy as np
import pandas as pd
import yaml

# ---- pastikan modul lokal bisa diimport ----
try:
    import pleiades
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    import pleiades  # type: ignore

from pleiades import plotting as P  # semua helper plot ada di sini


# ---------------------- util I/O & config ---------------------- #

def _p(msg: str) -> None:
    print(msg, flush=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def find_processed_file(proc_dir: Path, explicit: Path | None = None) -> Path:
    """Cari file hasil klaster. Prioritas:
    1) --input (explicit)
    2) file paling baru di proc_dir dengan pola cluster*/label* (parquet/csv)
    3) file paling baru parquet/csv apa pun di proc_dir
    """
    if explicit is not None:
        p = explicit if explicit.is_absolute() else (proc_dir / explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"--input menunjuk ke file yang tidak ada: {p}")

    # kandidat dengan pola yang umum
    patterns = [
        "*cluster*.parquet", "*label*.parquet",
        "*cluster*.csv*", "*label*.csv*",
        "*.parquet", "*.csv*",   # fallback apa pun
    ]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(sorted(proc_dir.glob(pat)))

    if not candidates:
        raise FileNotFoundError(
            f"Tidak ada file parquet/csv di {proc_dir}. "
            f"Jalankan 01_preprocess.py & 02_cluster.py, atau berikan --input."
        )

    # pilih yang terbaru
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_valid_file(base_dir: Path) -> Path | None:
    cand = [
        base_dir / "valid.parquet",
        base_dir / "valid.csv.gz",
        base_dir / "valid.csv",
        (base_dir.parent / "interim" / "valid_clean.parquet"),
        (base_dir.parent / "interim" / "valid_clean.csv.gz"),
        (base_dir.parent / "interim" / "valid_clean.csv"),
    ]
    for c in cand:
        if c.exists():
            return c
    return None

def load_config(path: Path | None) -> dict:
    cfg = {}
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    # defaults
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("processed", "data/processed")
    cfg["paths"].setdefault("figures", "data/processed/figures")
    cfg["paths"].setdefault("models", "models")
    cfg.setdefault("cluster", {})
    cfg["cluster"].setdefault("prob_min", 0.5)
    cfg.setdefault("plot", {})
    cfg["plot"].setdefault("max_arrows", 800)
    return cfg


# ---------------------- pemrosesan data ------------------------ #

def prepare_dataframe(df: pd.DataFrame, prob_thresh: float) -> pd.DataFrame:
    """Harmonisasi nama kolom umum + bikin kolom turunan & flag keanggotaan."""
    # harmonisasi nama kolom yang sering berbeda
    rename_map = {}
    if "probability" in df.columns and "prob" not in df.columns:
        rename_map["probability"] = "prob"
    if "cluster_label" in df.columns and "cluster" not in df.columns:
        rename_map["cluster_label"] = "cluster"
    if "BP_RP" in df.columns and "color" not in df.columns:
        rename_map["BP_RP"] = "color"
    if "bp_rp" in df.columns and "color" not in df.columns:
        rename_map["bp_rp"] = "color"
    if "abs_g" in df.columns and "M_G" not in df.columns:
        rename_map["abs_g"] = "M_G"
    if "RA" in df.columns and "ra" not in df.columns:
        rename_map["RA"] = "ra"
    if "DEC" in df.columns and "dec" not in df.columns:
        rename_map["DEC"] = "dec"
    if rename_map:
        df = df.rename(columns=rename_map)

    # fitur turunan
    if "_pm_mag" not in df.columns and {"pmra", "pmdec"} <= set(df.columns):
        df["_pm_mag"] = np.sqrt(df["pmra"]**2 + df["pmdec"]**2)

    # flag anggota
    if "is_member" not in df.columns:
        if {"cluster", "prob"} <= set(df.columns):
            df["is_member"] = (df["cluster"] >= 0) & (df["prob"] >= float(prob_thresh))
        elif "prob" in df.columns:
            df["is_member"] = (df["prob"] >= float(prob_thresh))
        elif "cluster" in df.columns:
            df["is_member"] = (df["cluster"] >= 0)
        else:
            df["is_member"] = False

    return df


# ---------------------- membuat semua gambar ------------------- #

def make_all_figures(
    df: pd.DataFrame,
    out_dir: Path,
    prob_thresh: float,
    valid: pd.DataFrame | None,
    model_dir: Path,
    max_arrows: int,
    panels_for: list[int] | str = "all",
    skip_condensed: bool = False,
) -> list[str]:
    ensure_dir(out_dir)
    saved: list[str] = []

    # Info jika file "members-only"
    if "is_member" in df.columns and bool(df["is_member"].all()):
        _p("[NOTE] File hasil tampaknya hanya berisi anggota (members-only). "
           "Plot 'all' dan 'members' akan identik. "
           "Jika ingin bandingkan dengan field/noise, jalankan dengan --input ke file hasil lengkap.")

    # Sky (dengan overlay valid jika tersedia)
    _p("[PLOT] sky_scatter_all.png")
    P.plot_sky_scatter(df, members_only=False, color_by=None, valid=valid, savepath=str(out_dir / "sky_scatter_all.png"))
    saved.append("sky_scatter_all.png")

    _p("[PLOT] sky_scatter_members.png")
    P.plot_sky_scatter(df, members_only=True, color_by=None, valid=valid, savepath=str(out_dir / "sky_scatter_members.png"))
    saved.append("sky_scatter_members.png")

    _p("[PLOT] sky_scatter_prob_all.png")
    P.plot_sky_scatter(df, members_only=False, color_by="prob", valid=None, savepath=str(out_dir / "sky_scatter_prob_all.png"))
    saved.append("sky_scatter_prob_all.png")

    _p("[PLOT] sky_scatter_prob_members.png")
    P.plot_sky_scatter(df, members_only=True, color_by="prob", valid=None, savepath=str(out_dir / "sky_scatter_prob_members.png"))
    saved.append("sky_scatter_prob_members.png")

    _p("[PLOT] sky_hexbin_all.png")
    P.plot_sky_hexbin(df, members_only=False, savepath=str(out_dir / "sky_hexbin_all.png"))
    saved.append("sky_hexbin_all.png")

    _p("[PLOT] sky_quiver_members.png")
    P.plot_sky_quiver(df, members_only=True, sample=max_arrows, savepath=str(out_dir / "sky_quiver_members.png"))
    saved.append("sky_quiver_members.png")

    # PM plane
    _p("[PLOT] pm_plane_all.png")
    P.plot_pm_plane(df, members_only=False, color_by=None, savepath=str(out_dir / "pm_plane_all.png"))
    saved.append("pm_plane_all.png")

    _p("[PLOT] pm_plane_prob_all.png")
    P.plot_pm_plane(df, members_only=False, color_by="prob", savepath=str(out_dir / "pm_plane_prob_all.png"))
    saved.append("pm_plane_prob_all.png")

    _p("[PLOT] pm_plane_members.png")
    P.plot_pm_plane(df, members_only=True, color_by=None, savepath=str(out_dir / "pm_plane_members.png"))
    saved.append("pm_plane_members.png")

    _p("[PLOT] pm_plane_hexbin_members.png")
    P.plot_pm_plane_hexbin(df, members_only=True, savepath=str(out_dir / "pm_plane_hexbin_members.png"))
    saved.append("pm_plane_hexbin_members.png")

    # CMD
    _p("[PLOT] cmd_members.png")
    P.plot_cmd(df, members_only=True, color_by=None, savepath=str(out_dir / "cmd_members.png"))
    saved.append("cmd_members.png")

    _p("[PLOT] cmd_hexbin_members.png")
    P.plot_cmd_hexbin(df, members_only=True, savepath=str(out_dir / "cmd_hexbin_members.png"))
    saved.append("cmd_hexbin_members.png")

    # Overlay dengan valid (jika ada)
    if valid is not None and not valid.empty:
        _p("[PLOT] cmd_members_vs_valid.png")
        P.plot_overlay_cmd_with_valid(df[df["is_member"]], valid, savepath=str(out_dir / "cmd_members_vs_valid.png"))
        saved.append("cmd_members_vs_valid.png")

        if {"pmra", "pmdec"} <= set(valid.columns):
            _p("[PLOT] pm_plane_members_vs_valid.png")
            P.plot_overlay_pm_with_valid(df[df["is_member"]], valid, savepath=str(out_dir / "pm_plane_members_vs_valid.png"))
            saved.append("pm_plane_members_vs_valid.png")

    # M_G vs parallax (anggota)
    if {"parallax", "M_G"} <= set(df.columns):
        _p("[PLOT] mg_vs_parallax_members.png")
        fig, ax = plt.subplots(figsize=(7, 7))
        sub = df[df["is_member"]]
        ax.scatter(sub["parallax"], sub["M_G"], s=8, alpha=0.6)
        ax.set_xlabel("parallax [mas]"); ax.set_ylabel("M_G"); ax.invert_yaxis()
        ax.set_title("M_G vs parallax — members"); ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / "mg_vs_parallax_members.png", dpi=150, bbox_inches="tight"); plt.close(fig)
        saved.append("mg_vs_parallax_members.png")

    # Histogram fitur (all)
    for feat in ["pmra", "pmdec", "parallax", "color", "M_G", "ruwe", "_pm_mag"]:
        if feat in df.columns:
            fname = f"hist_{feat.replace('-', '_')}.png"
            _p(f"[PLOT] {fname}")
            P.plot_hist(df, feature=feat, members_only=False, savepath=str(out_dir / fname))
            saved.append(fname)

    # Box & Violin per cluster (anggota)
    if "cluster" in df.columns and df["is_member"].any():
        for feat in ["_pm_mag", "color", "M_G", "parallax", "ruwe"]:
            if feat in df.columns:
                bname = f"box_{feat.replace('-', '_')}_by_cluster_members.png"
                vname = f"violin_{feat.replace('-', '_')}_by_cluster_members.png"
                _p(f"[PLOT] {bname}")
                P.plot_box(df, feature=feat, savepath=str(out_dir / bname))
                saved.append(bname)
                _p(f"[PLOT] {vname}")
                P.plot_violin(df, feature=feat, savepath=str(out_dir / vname))
                saved.append(vname)

    # Probability hist & CDF + kurva anggota vs threshold
    if "prob" in df.columns:
        _p("[PLOT] prob_hist.png, prob_cdf_all.png, prob_cdf_members.png")
        P.plot_probability_hist_cdf(
            df, save_hist=str(out_dir / "prob_hist.png"),
            save_cdf=str(out_dir / "prob_cdf_all.png"), members_only=False
        )
        P.plot_probability_hist_cdf(
            df, save_hist=None, save_cdf=str(out_dir / "prob_cdf_members.png"),
            members_only=True
        )
        saved += ["prob_hist.png", "prob_cdf_all.png", "prob_cdf_members.png"]

        _p("[PLOT] members_vs_threshold.png")
        P.plot_members_vs_threshold(df, savepath=str(out_dir / "members_vs_threshold.png"))
        saved.append("members_vs_threshold.png")

    # Profil radial
    if {"ra", "dec"} <= set(df.columns):
        _p("[PLOT] radial_profile_members.png")
        P.plot_radial_profile(df, members_only=True, n_bins=25, savepath=str(out_dir / "radial_profile_members.png"))
        saved.append("radial_profile_members.png")

    # RUWE vs G (jika kolom ada)
    ax = P.plot_ruwe_vs_g(df, members_only=True, savepath=str(out_dir / "ruwe_vs_g_members.png"))
    if ax is not None:
        saved.append("ruwe_vs_g_members.png")

    # Condensed tree (jika model ada dan tidak di-skip)
    if not skip_condensed:
        model_path = None
        for cand in [model_dir / "hdbscan.pkl", model_dir / "hdbscan.joblib"]:
            if cand.exists():
                model_path = cand
                break
        if model_path:
            try:
                _p("[PLOT] condensed_tree.png")
                clusterer = joblib.load(model_path)
                ax = P.condensed_tree(clusterer, select_clusters=True)  # type: ignore
                ax.figure.savefig(out_dir / "condensed_tree.png", dpi=150, bbox_inches="tight")
                plt.close(ax.figure)
                saved.append("condensed_tree.png")
            except Exception as e:
                warnings.warn(f"Gagal membuat condensed tree: {e!r}")

    # Panel per-cluster (default: semua label non-noise)
    if "cluster" in df.columns:
        labels = sorted(df["cluster"].unique().tolist())
        if isinstance(panels_for, list):
            labels = [l for l in labels if l in panels_for]
        for lbl in labels:
            _p(f"[PLOT] panel_cluster_{lbl}.png")
            fig, _ = P.plot_panel_per_cluster(
                df, label=int(lbl), savepath=str(out_dir / f"panel_cluster_{lbl}.png"),
                max_arrows=max_arrows
            )
            saved.append(f"panel_cluster_{lbl}.png")

    # Manifest
    manifest = out_dir / "figures_manifest.txt"
    manifest.write_text("\n".join(saved), encoding="utf-8")
    _p(f"[DONE] {len(saved)} figur tersimpan di: {out_dir}")
    return saved


# ------------------------------ main --------------------------------------- #

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Buat seluruh figur hasil klasterisasi.")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"),
                    help="Path konfigurasi YAML.")
    ap.add_argument("--prob-thresh", type=float, default=None,
                    help="Ambang probabilitas anggota. Override config.")
    ap.add_argument("--panels-for", type=str, default="all",
                    help="'all' atau daftar label dipisah koma, mis: '0,1'")
    ap.add_argument("--input", type=Path, default=None,
                    help="File hasil klaster (parquet/csv). "
                         "Relatif ke paths.processed di config kalau bukan path absolut.")
    ap.add_argument("--skip-condensed", action="store_true",
                    help="Lewati pembuatan HDBSCAN condensed tree.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    paths = cfg["paths"]
    proc_dir = Path(paths["processed"])
    fig_dir = Path(paths["figures"])
    model_dir = Path(paths["models"])
    ensure_dir(fig_dir)

    prob_thresh = float(args.prob_thresh if args.prob_thresh is not None else cfg["cluster"]["prob_min"])
    max_arrows = int(cfg["plot"]["max_arrows"])

    # muat data hasil
    processed_path = find_processed_file(proc_dir, explicit=args.input)
    _p(f"[INFO] Memuat hasil klaster: {processed_path.name}")
    if "".join(processed_path.suffixes).endswith(".parquet"):
        df = pd.read_parquet(processed_path)
    else:
        df = pd.read_csv(processed_path)

    df = prepare_dataframe(df, prob_thresh=prob_thresh)
    _p(f"[INFO] n={len(df)}; members (prob>={prob_thresh} & cluster>=0): {int(df['is_member'].sum())}")

    # muat valid (jika ada)
    valid_path = find_valid_file(proc_dir)
    valid_df = None
    if valid_path:
        _p(f"[INFO] Memuat valid: {valid_path.name}")
        if "".join(valid_path.suffixes).endswith(".parquet"):
            valid_df = pd.read_parquet(valid_path)
        else:
            valid_df = pd.read_csv(valid_path)

        # harmonisasi nama kolom warna & M_G jika perlu
        ren = {}
        if "BP_RP" in valid_df.columns and "color" not in valid_df.columns:
            ren["BP_RP"] = "color"
        if "M_G" not in valid_df.columns and "abs_g" in valid_df.columns:
            ren["abs_g"] = "M_G"
        if ren:
            valid_df = valid_df.rename(columns=ren)

    # label panel cluster
    if args.panels_for != "all":
        try:
            panels_for: list[int] | str = [int(x.strip()) for x in args.panels_for.split(",") if x.strip() != ""]
        except Exception:
            panels_for = "all"
    else:
        panels_for = "all"

    # buat semua figur
    make_all_figures(
        df=df,
        out_dir=fig_dir,
        prob_thresh=prob_thresh,
        valid=valid_df,
        model_dir=model_dir,
        max_arrows=max_arrows,
        panels_for=panels_for,
        skip_condensed=args.skip_condensed,
    )


if __name__ == "__main__":
    main()
