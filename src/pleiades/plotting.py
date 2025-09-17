# -*- coding: utf-8 -*-
"""
pleiades.plotting — utilitas visualisasi untuk proyek Pleiades–HDBSCAN.

Semua fungsi
- tidak memanggil plt.show()
- mengembalikan Axes atau (Figure, Axes)
- aman jika sebagian kolom tidak ada (fungsi akan melewatkan layer opsional)

Konvensi kolom yang didukung:
    ra, dec, pmra, pmdec, parallax, color, M_G, ruwe, prob, cluster, is_member

Catatan:
- RA dibalik (invert_xaxis) untuk konvensi astronomi.
- Nilai ekstrem dipangkas dengan persentil agar skala sumbu stabil.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# --------------------------- util umum ------------------------------------- #

def _has_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return set(cols).issubset(df.columns)


def robust_limits(a: Union[np.ndarray, pd.Series], lo: float = 0.5, hi: float = 99.5) -> Tuple[float, float]:
    """Batas sumbu berbasis persentil agar outlier tidak mendominasi."""
    a = np.asarray(pd.Series(a).dropna())
    if a.size == 0:
        return 0.0, 1.0
    lo_v, hi_v = np.percentile(a, [lo, hi])
    if lo_v == hi_v:
        lo_v -= 1.0
        hi_v += 1.0
    return float(lo_v), float(hi_v)


def mad(a: Union[np.ndarray, pd.Series]) -> float:
    """Median Absolute Deviation."""
    a = np.asarray(pd.Series(a).dropna())
    if a.size == 0:
        return float("nan")
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def _annotate_stats(ax: plt.Axes, data: Union[np.ndarray, pd.Series], loc: str = "top-right",
                    fmt: str = r"$\tilde{{x}}$={med:.2f}\nMAD={mad:.2f}") -> None:
    med = np.nanmedian(data)
    md = mad(data)
    txt = fmt.format(med=med, mad=md)
    xy = {
        "top-right": (0.98, 0.97, "right", "top"),
        "top-left": (0.02, 0.97, "left", "top"),
        "bottom-left": (0.02, 0.03, "left", "bottom"),
        "bottom-right": (0.98, 0.03, "right", "bottom"),
    }[loc]
    ax.text(xy[0], xy[1], txt, ha=xy[2], va=xy[3], transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8), fontsize=9)


def _cov_ellipse(x: np.ndarray, y: np.ndarray, n_sigma: float = 2.0, **kwargs) -> Ellipse:
    """Ellipse kovarians 2D untuk scatter plane."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 5:
        return Ellipse((np.nan, np.nan), 0, 0)  # no-op
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_sigma * np.sqrt(vals)
    return Ellipse((np.nanmean(x), np.nanmean(y)), width, height, angle, fill=False, lw=1.5, **kwargs)


def _subset_members(df: pd.DataFrame, members_only: bool) -> pd.DataFrame:
    """Kembalikan subset anggota bila members_only=True & kolom is_member ada; selain itu kembalikan df apa adanya."""
    if members_only and ("is_member" in df.columns):
        return df[df["is_member"]]
    return df


# --------------------------- layer dasar ----------------------------------- #

def sky_scatter(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
                members_only: bool = False, color_by: Optional[str] = None,
                s: float = 6, alpha: float = 0.5, valid: Optional[pd.DataFrame] = None) -> plt.Axes:
    """Sebaran RA–Dec. Jika color_by='prob' akan diwarnai oleh probabilitas."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    data = _subset_members(df, members_only)
    if not _has_cols(data, ["ra", "dec"]):
        raise KeyError("Kolom 'ra' dan 'dec' diperlukan")
    c = data[color_by] if (color_by and color_by in data.columns) else None
    ax.scatter(data["ra"], data["dec"], s=s, alpha=alpha, c=c, cmap="viridis", edgecolors="none")
    if valid is not None and _has_cols(valid, ["ra", "dec"]):
        ax.scatter(valid["ra"], valid["dec"], s=8, facecolors="none", edgecolors="red", alpha=0.5, label="valid")
        ax.legend(loc="best", frameon=True)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]"); ax.set_title("Sky scatter")
    ax.grid(True, alpha=0.3); ax.invert_xaxis()
    return ax


def sky_hexbin(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
               members_only: bool = False, gridsize: int = 80) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    data = _subset_members(df, members_only)
    if not _has_cols(data, ["ra", "dec"]):
        raise KeyError("Kolom 'ra' dan 'dec' diperlukan")
    hb = ax.hexbin(data["ra"], data["dec"], gridsize=gridsize, cmap="magma", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title(f"Sky density (hexbin){' — members' if members_only else ' — all'}")
    ax.grid(True, alpha=0.3); ax.invert_xaxis()
    return ax


def sky_quiver(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
               members_only: bool = True, sample: int = 800, scale: float = 400.0) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    data = _subset_members(df, members_only)
    if sample and len(data) > sample:
        data = data.sample(sample, random_state=0)
    if not _has_cols(data, ["ra", "dec", "pmra", "pmdec"]):
        raise KeyError("Diperlukan kolom ra, dec, pmra, pmdec")
    ax.quiver(data["ra"], data["dec"], data["pmra"], data["pmdec"],
              angles="xy", scale_units="xy", scale=scale, width=0.002, color="k", alpha=0.7)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]"); ax.set_title(f"Sky quiver ({'members' if members_only else 'all'})")
    ax.grid(True, alpha=0.3); ax.invert_xaxis()
    return ax


def pm_plane(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
             members_only: bool = False, color_by: Optional[str] = None,
             valid: Optional[pd.DataFrame] = None, show_ellipse: bool = True) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    data = _subset_members(df, members_only)
    if not _has_cols(data, ["pmra", "pmdec"]):
        raise KeyError("Diperlukan kolom pmra & pmdec")
    c = data[color_by] if (color_by and color_by in data.columns) else None
    ax.scatter(data["pmra"], data["pmdec"], s=6, alpha=0.6, c=c, cmap="plasma", edgecolors="none")
    if valid is not None and _has_cols(valid, ["pmra", "pmdec"]):
        ax.scatter(valid["pmra"], valid["pmdec"], s=16, fc="none", ec="red", alpha=0.6, label="valid")
        ax.legend(loc="best", frameon=True)
    if show_ellipse:
        try:
            ell = _cov_ellipse(data["pmra"].values, data["pmdec"].values, n_sigma=2.0, ec="k")
            ax.add_patch(ell)
        except Exception:
            pass
    ax.set_xlabel("pmRA [mas/yr]"); ax.set_ylabel("pmDec [mas/yr]")
    ax.set_title(f"PM plane ({'members' if members_only else 'all'})"); ax.grid(True, alpha=0.3)
    return ax


def pm_plane_hexbin(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
                    members_only: bool = True, gridsize: int = 70) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    data = _subset_members(df, members_only)
    if not _has_cols(data, ["pmra", "pmdec"]):
        raise KeyError("Diperlukan kolom pmra & pmdec")
    hb = ax.hexbin(data["pmra"], data["pmdec"], gridsize=gridsize, cmap="magma", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("pmRA [mas/yr]"); ax.set_ylabel("pmDec [mas/yr]")
    ax.set_title(f"PM plane density (hexbin){' — members' if members_only else ' — all'}")
    ax.grid(True, alpha=0.3)
    return ax


def cmd(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None, members_only: bool = True,
        color_by: Optional[str] = None, valid: Optional[pd.DataFrame] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    data = _subset_members(df, members_only)
    if not _has_cols(data, ["color", "M_G"]):
        raise KeyError("Diperlukan kolom color & M_G")
    c = data[color_by] if (color_by and color_by in data.columns) else None
    ax.scatter(data["color"], data["M_G"], s=8, alpha=0.6, c=c, cmap="viridis", edgecolors="k", linewidths=0.2)
    if valid is not None and _has_cols(valid, ["color", "M_G"]):
        ax.scatter(valid["color"], valid["M_G"], s=20, fc="none", ec="red", alpha=0.6, label="valid")
        ax.legend(loc="best", frameon=True)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("M_G"); ax.invert_yaxis()
    ax.set_title(f"CMD ({'members' if members_only else 'all'})"); ax.grid(True, alpha=0.3)
    return ax


def cmd_hexbin(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
               members_only: bool = True, gridsize: int = 60) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    data = _subset_members(df, members_only)
    if not _has_cols(data, ["color", "M_G"]):
        raise KeyError("Diperlukan kolom color & M_G")
    hb = ax.hexbin(data["color"], data["M_G"], gridsize=gridsize, cmap="magma", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel("BP − RP"); ax.set_ylabel("M_G"); ax.invert_yaxis()
    ax.set_title(f"CMD density (hexbin){' — members' if members_only else ' — all'}")
    ax.grid(True, alpha=0.3)
    return ax


# ------------------------------ histogram & CDF ---------------------------- #

def prob_hist(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None, bins: int = 30,
              members_only: bool = False) -> plt.Axes:
    if "prob" not in df.columns:
        raise KeyError("Kolom 'prob' tidak ditemukan")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    data = _subset_members(df, members_only)
    ax.hist(data["prob"].dropna(), bins=bins, alpha=0.8)
    ax.set_xlabel("membership probability"); ax.set_ylabel("count")
    ax.set_title("Histogram membership probability")
    ax.grid(True, alpha=0.3)
    return ax


def prob_cdf(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None, members_only: bool = False) -> plt.Axes:
    if "prob" not in df.columns:
        raise KeyError("Kolom 'prob' tidak ditemukan")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    data = _subset_members(df, members_only)
    x = np.sort(data["prob"].dropna().values)
    y = np.linspace(0, 1, x.size, endpoint=True)
    ax.step(x, y, where="post")
    ax.set_xlabel("membership probability"); ax.set_ylabel("CDF")
    ax.set_title(f"Membership probability CDF — {'members' if members_only else 'all'}")
    ax.grid(True, alpha=0.3)
    return ax


def feature_hist_by_cluster(df: pd.DataFrame, feature: str, *, bins: int = 30,
                            ax: Optional[plt.Axes] = None, members_only: bool = False) -> plt.Axes:
    if feature not in df.columns:
        raise KeyError(f"Kolom '{feature}' tidak ditemukan")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    data = _subset_members(df, members_only)
    if "cluster" in data.columns:
        for lbl, sub in data.groupby("cluster"):
            ax.hist(sub[feature].dropna(), bins=bins, alpha=0.6, label=f"cluster {lbl}")
        ax.legend(frameon=True)
    else:
        ax.hist(data[feature].dropna(), bins=bins, alpha=0.8, label="all")
        ax.legend(frameon=True)
    ax.set_xlabel(feature); ax.set_ylabel("count")
    ax.set_title(f"Histogram {feature} ({'members' if members_only else 'all'})")
    ax.grid(True, alpha=0.3)
    return ax


def box_by_membership(df: pd.DataFrame, feature: str, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if feature not in df.columns or "cluster" not in df.columns:
        raise KeyError("Perlu kolom feature & 'cluster'")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    sub = df[df["is_member"]]
    ax.boxplot([sub[sub["cluster"] == k][feature].dropna().values for k in sorted(sub["cluster"].unique())],
               labels=[str(k) for k in sorted(sub["cluster"].unique())])
    ax.set_xlabel("cluster"); ax.set_ylabel(feature); ax.set_title(f"{feature} by cluster (boxplot, members)")
    return ax


def violin_by_membership(df: pd.DataFrame, feature: str, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if feature not in df.columns or "cluster" not in df.columns:
        raise KeyError("Perlu kolom feature & 'cluster'")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    sub = df[df["is_member"]]
    parts = [sub[sub["cluster"] == k][feature].dropna().values for k in sorted(sub["cluster"].unique())]
    ax.violinplot(parts, showmedians=True)
    ax.set_xticks(range(1, len(parts) + 1), [str(k) for k in sorted(sub["cluster"].unique())])
    ax.set_xlabel("cluster"); ax.set_ylabel(feature); ax.set_title(f"{feature} by cluster (violin, members)")
    return ax


# ------------------------------ evaluasi tambahan -------------------------- #

def members_vs_threshold(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
                         thresholds: Optional[Sequence[float]] = None) -> plt.Axes:
    """Kurva N anggota vs ambang probabilitas (cluster>=0 dan prob>=thr)."""
    if "prob" not in df.columns or "cluster" not in df.columns:
        raise KeyError("Diperlukan kolom 'prob' & 'cluster'")
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 41)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    n = []
    for t in thresholds:
        mask = (df["cluster"] >= 0) & (df["prob"] >= t)
        n.append(int(mask.sum()))
    ax.plot(list(thresholds), n, marker="o")
    ax.set_xlabel("probability threshold"); ax.set_ylabel("N selected")
    ax.set_title("Anggota terpilih vs ambang probabilitas")
    ax.grid(True, alpha=0.3)
    return ax


def radial_profile(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None,
                   members_only: bool = True, center: Optional[Tuple[float, float]] = None,
                   n_bins: int = 25) -> plt.Axes:
    """Profil radial kepadatan anggota terhadap pusat (RA,Dec)."""
    if not _has_cols(df, ["ra", "dec"]):
        raise KeyError("Kolom ra, dec diperlukan")
    data = _subset_members(df, members_only).copy()
    if center is None:
        center = (np.nanmedian(data["ra"]), np.nanmedian(data["dec"]))
    ra0, dec0 = np.radians(center[0]), np.radians(center[1])
    ra = np.radians(data["ra"].values); dec = np.radians(data["dec"].values)
    # jarak sudut great-circle (haversine)
    d_ra = ra - ra0
    d = 2 * np.arcsin(np.sqrt(np.sin((dec - dec0)/2)**2 + np.cos(dec)*np.cos(dec0)*np.sin(d_ra/2)**2))
    r_deg = np.degrees(d)
    # binning
    r = r_deg
    bins = np.linspace(0, np.nanpercentile(r, 99.5), n_bins+1)
    counts, edges = np.histogram(r, bins=bins)
    # luas gelang (aproks deg^2)
    area = np.pi*(edges[1:]**2 - edges[:-1]**2)
    dens = counts / area
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(edges[1:], dens, where="post")
    ax.set_xlabel("radius [deg]"); ax.set_ylabel("surface density [1/deg²]")
    ax.set_title("Profil radial (anggota)")
    ax.grid(True, alpha=0.3)
    return ax


def ruwe_vs_g(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None, members_only: bool = True) -> Optional[plt.Axes]:
    if not _has_cols(df, ["ruwe", "phot_g_mean_mag"]):
        return None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    data = _subset_members(df, members_only)
    ax.scatter(data["phot_g_mean_mag"], data["ruwe"], s=6, alpha=0.6)
    ax.set_xlabel("G [mag]"); ax.set_ylabel("RUWE"); ax.set_title("RUWE vs G")
    ax.grid(True, alpha=0.3); ax.invert_xaxis()
    return ax


def condensed_tree(clusterer, *, select_clusters: bool = True, label: str = "lambda value") -> plt.Axes:
    """Wrapper aman untuk hdbscan.plots.plot_condensed_tree."""
    try:
        from hdbscan import plots as hdbscan_plots  # type: ignore
    except Exception as e:
        raise ImportError("Visualisasi condensed tree memerlukan 'hdbscan.plots'.") from e
    if not hasattr(clusterer, "condensed_tree_"):
        raise ValueError("Objek clusterer tidak memiliki atribut 'condensed_tree_'")
    ax = hdbscan_plots.plot_condensed_tree(clusterer, select_clusters=select_clusters)
    ax.set_xlabel(label); ax.set_title("HDBSCAN Condensed Tree")
    return ax


# ------------------------------ overlay valid ------------------------------ #

def overlay_cmd_with_valid(df_members: pd.DataFrame, valid: pd.DataFrame, *,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    if not _has_cols(df_members, ["color", "M_G"]) or not _has_cols(valid, ["color", "M_G"]):
        raise KeyError("Butuh kolom color & M_G di keduanya")
    ax.scatter(df_members["color"], df_members["M_G"], s=8, alpha=0.35, label="members", edgecolors="none")
    ax.scatter(valid["color"], valid["M_G"], s=22, ec="red", fc="none", label="valid")
    ax.invert_yaxis(); ax.grid(True, alpha=0.3)
    ax.set_xlabel("BP − RP"); ax.set_ylabel("M_G"); ax.legend(frameon=True)
    ax.set_title("CMD: members vs valid overlay")
    return ax


def overlay_pm_with_valid(df_members: pd.DataFrame, valid: pd.DataFrame, *,
                          ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    if not _has_cols(df_members, ["pmra", "pmdec"]) or not _has_cols(valid, ["pmra", "pmdec"]):
        raise KeyError("Butuh kolom pmra & pmdec di keduanya")
    ax.scatter(df_members["pmra"], df_members["pmdec"], s=8, alpha=0.35, label="members", edgecolors="none")
    ax.scatter(valid["pmra"], valid["pmdec"], s=22, ec="red", fc="none", label="valid")
    ax.grid(True, alpha=0.3); ax.set_xlabel("pmRA [mas/yr]"); ax.set_ylabel("pmDec [mas/yr]")
    ax.set_title("PM plane: members vs valid overlay"); ax.legend(frameon=True)
    return ax


# ------------------------ panel lengkap per-klaster ------------------------ #

def panel_per_cluster(df: pd.DataFrame, label: int, *, features: Optional[Sequence[str]] = None,
                      prob_col: str = "prob", max_arrows: int = 500,
                      axarr: Optional[np.ndarray] = None) -> Tuple[plt.Figure, np.ndarray]:
    """Panel 3x3 untuk sebuah cluster spesifik (meniru versi notebook)."""
    subset = df[df["cluster"] == label].copy()
    if subset.empty:
        raise ValueError(f"Tidak ada data untuk cluster {label}")
    feats = features or ["parallax", "pmra", "pmdec", "color", "M_G", "ruwe"]
    # layout
    if axarr is None:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    else:
        axes = axarr
        fig = axes[0, 0].figure
    fig.suptitle(f"Cluster {label} – Analisis Lengkap", fontsize=16)

    # (0,0) RA vs Dec (warna prob)
    ax = axes[0, 0]
    if _has_cols(subset, ["ra", "dec"]):
        c = subset[prob_col] if prob_col in subset.columns else None
        sc = ax.scatter(subset["ra"], subset["dec"], s=10, alpha=0.7, c=c, cmap="viridis")
        ax.set_title("RA vs Dec"); ax.set_xlabel("RA (deg)"); ax.set_ylabel("Dec (deg)"); ax.grid(True)
        if c is not None:
            plt.colorbar(sc, ax=ax, label="Probability")
        ra_c, dec_c = subset["ra"].mean(), subset["dec"].mean()
        ax.scatter(ra_c, dec_c, color="red", marker="x")
        ax.text(0.05, 0.95, f"Centroid:\n({ra_c:.2f},{dec_c:.2f})", transform=ax.transAxes,
                ha="left", va="top", bbox=dict(boxstyle="round", fc="w"))
        ax.invert_xaxis()
    else:
        ax.axis("off")

    # (0,1) Parallax hist
    ax = axes[0, 1]
    if "parallax" in subset.columns:
        ax.hist(subset["parallax"], bins=20, alpha=0.7)
        ax.set_title("Parallax"); ax.set_xlabel("mas"); ax.set_ylabel("Count"); ax.grid(True)
        _annotate_stats(ax, subset["parallax"], loc="top-right")
    else:
        ax.axis("off")

    # (0,2) HR diagram
    ax = axes[0, 2]
    if _has_cols(subset, ["color", "M_G"]):
        c = subset[prob_col] if prob_col in subset.columns else None
        sc = ax.scatter(subset["color"], subset["M_G"], s=10, alpha=0.7, c=c, cmap="viridis")
        ax.invert_yaxis()
        ax.set_title("HR Diagram"); ax.set_xlabel("BP−RP"); ax.set_ylabel("M_G"); ax.grid(True)
        if c is not None:
            plt.colorbar(sc, ax=ax, label="Probability")
        _annotate_stats(ax, subset["M_G"], loc="top-right")
    else:
        ax.axis("off")

    # (1,0) Proper motion vectors
    ax = axes[1, 0]
    if _has_cols(subset, ["ra", "dec", "pmra", "pmdec"]):
        arrows = subset.sample(min(len(subset), max_arrows), random_state=0)
        ax.quiver(arrows["ra"], arrows["dec"], arrows["pmra"], arrows["pmdec"],
                  angles="xy", scale_units="xy", scale=500, width=0.002, alpha=0.7, color="tab:brown")
        pmra_m, pmdec_m = subset["pmra"].mean(), subset["pmdec"].mean()
        ra_c, dec_c = subset["ra"].mean(), subset["dec"].mean()
        ax.quiver(ra_c, dec_c, pmra_m, pmdec_m, angles="xy", scale_units="xy", scale=500, width=0.005, color="black")
        ax.set_title("Proper Motion Vectors"); ax.set_xlabel("RA"); ax.set_ylabel("Dec")
        ax.grid(True); ax.invert_xaxis()
        ax.text(0.05, 0.05, f"C{label}: n={len(arrows)}\nμ=({pmra_m:.2f},{pmdec_m:.2f})",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.7))
    else:
        ax.axis("off")

    # (1,1) PM magnitude
    ax = axes[1, 1]
    if _has_cols(subset, ["pmra", "pmdec"]):
        pm_vals = np.sqrt(subset["pmra"]**2 + subset["pmdec"]**2)
        ax.hist(pm_vals, bins=20, alpha=0.7)
        ax.set_title("PM Magnitude"); ax.set_xlabel("mas/yr"); ax.set_ylabel("Count"); ax.grid(True)
        _annotate_stats(ax, pm_vals, loc="top-right")
    else:
        ax.axis("off")

    # (1,2) RUWE
    ax = axes[1, 2]
    if "ruwe" in subset.columns:
        ax.hist(subset["ruwe"], bins=20, alpha=0.7)
        ax.set_title("RUWE"); ax.set_xlabel("RUWE"); ax.set_ylabel("Count"); ax.grid(True)
        _annotate_stats(ax, subset["ruwe"], loc="top-right")
    else:
        ax.axis("off")

    # (2,0) Color
    ax = axes[2, 0]
    if "color" in subset.columns:
        ax.hist(subset["color"], bins=20, alpha=0.7)
        ax.set_title("Color Distribution"); ax.set_xlabel("BP−RP"); ax.set_ylabel("Count"); ax.grid(True)
        _annotate_stats(ax, subset["color"], loc="top-right")
    else:
        ax.axis("off")

    # (2,1) M_G
    ax = axes[2, 1]
    if "M_G" in subset.columns:
        ax.hist(subset["M_G"], bins=20, alpha=0.7)
        ax.set_title("Absolute G Magnitude"); ax.set_xlabel("M_G"); ax.set_ylabel("Count"); ax.grid(True)
        _annotate_stats(ax, subset["M_G"], loc="top-right")
    else:
        ax.axis("off")

    # (2,2) kosong
    axes[2, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes


# ------------------------------ wrapper penyimpan -------------------------- #

def _save_fig(fig: plt.Figure, savepath: Optional[str] = None) -> None:
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_sky_scatter(df: pd.DataFrame, *, savepath: Optional[str] = None,
                     members_only: bool = False, color_by: Optional[str] = None,
                     valid: Optional[pd.DataFrame] = None) -> plt.Axes:
    ax = sky_scatter(df, members_only=members_only, color_by=color_by, valid=valid)
    _save_fig(ax.figure, savepath)
    return ax


def plot_sky_hexbin(df: pd.DataFrame, *, savepath: Optional[str] = None,
                    members_only: bool = False) -> plt.Axes:
    ax = sky_hexbin(df, members_only=members_only)
    _save_fig(ax.figure, savepath)
    return ax


def plot_sky_quiver(df: pd.DataFrame, *, savepath: Optional[str] = None,
                    members_only: bool = True, sample: int = 800, scale: float = 400.0) -> plt.Axes:
    ax = sky_quiver(df, members_only=members_only, sample=sample, scale=scale)
    _save_fig(ax.figure, savepath)
    return ax


def plot_pm_plane(df: pd.DataFrame, *, savepath: Optional[str] = None,
                  members_only: bool = False, color_by: Optional[str] = None,
                  valid: Optional[pd.DataFrame] = None, show_ellipse: bool = True) -> plt.Axes:
    ax = pm_plane(df, members_only=members_only, color_by=color_by, valid=valid, show_ellipse=show_ellipse)
    _save_fig(ax.figure, savepath)
    return ax


def plot_pm_plane_hexbin(df: pd.DataFrame, *, savepath: Optional[str] = None,
                         members_only: bool = True, gridsize: int = 70) -> plt.Axes:
    ax = pm_plane_hexbin(df, members_only=members_only, gridsize=gridsize)
    _save_fig(ax.figure, savepath)
    return ax


def plot_cmd(df: pd.DataFrame, *, savepath: Optional[str] = None,
             members_only: bool = True, color_by: Optional[str] = None,
             valid: Optional[pd.DataFrame] = None) -> plt.Axes:
    ax = cmd(df, members_only=members_only, color_by=color_by, valid=valid)
    _save_fig(ax.figure, savepath)
    return ax


def plot_cmd_hexbin(df: pd.DataFrame, *, savepath: Optional[str] = None,
                    members_only: bool = True, gridsize: int = 60) -> plt.Axes:
    ax = cmd_hexbin(df, members_only=members_only, gridsize=gridsize)
    _save_fig(ax.figure, savepath)
    return ax


def plot_hist(df: pd.DataFrame, feature: str, *, savepath: Optional[str] = None,
              bins: int = 30, members_only: bool = False) -> plt.Axes:
    ax = feature_hist_by_cluster(df, feature, bins=bins, members_only=members_only)
    _save_fig(ax.figure, savepath)
    return ax


def plot_box(df: pd.DataFrame, feature: str, *, savepath: Optional[str] = None) -> plt.Axes:
    ax = box_by_membership(df, feature)
    _save_fig(ax.figure, savepath)
    return ax


def plot_violin(df: pd.DataFrame, feature: str, *, savepath: Optional[str] = None) -> plt.Axes:
    ax = violin_by_membership(df, feature)
    _save_fig(ax.figure, savepath)
    return ax


def plot_probability_hist_cdf(df: pd.DataFrame, *, save_hist: Optional[str] = None,
                              save_cdf: Optional[str] = None, members_only: bool = False) -> Tuple[plt.Axes, plt.Axes]:
    ax1 = prob_hist(df, members_only=members_only)
    _save_fig(ax1.figure, save_hist)
    ax2 = prob_cdf(df, members_only=members_only)
    _save_fig(ax2.figure, save_cdf)
    return ax1, ax2


def plot_members_vs_threshold(df: pd.DataFrame, *, savepath: Optional[str] = None,
                              thresholds: Optional[Sequence[float]] = None) -> plt.Axes:
    ax = members_vs_threshold(df, thresholds=thresholds)
    _save_fig(ax.figure, savepath)
    return ax


def plot_radial_profile(df: pd.DataFrame, *, savepath: Optional[str] = None,
                        members_only: bool = True, center: Optional[Tuple[float, float]] = None,
                        n_bins: int = 25) -> plt.Axes:
    ax = radial_profile(df, members_only=members_only, center=center, n_bins=n_bins)
    _save_fig(ax.figure, savepath)
    return ax


def plot_ruwe_vs_g(df: pd.DataFrame, *, savepath: Optional[str] = None,
                   members_only: bool = True) -> Optional[plt.Axes]:
    ax = ruwe_vs_g(df, members_only=members_only)
    if ax is not None:
        _save_fig(ax.figure, savepath)
    return ax


def plot_overlay_cmd_with_valid(df_members: pd.DataFrame, valid: pd.DataFrame, *,
                                savepath: Optional[str] = None) -> plt.Axes:
    ax = overlay_cmd_with_valid(df_members, valid)
    _save_fig(ax.figure, savepath)
    return ax


def plot_overlay_pm_with_valid(df_members: pd.DataFrame, valid: pd.DataFrame, *,
                               savepath: Optional[str] = None) -> plt.Axes:
    ax = overlay_pm_with_valid(df_members, valid)
    _save_fig(ax.figure, savepath)
    return ax


def plot_panel_per_cluster(df: pd.DataFrame, label: int, *, savepath: Optional[str] = None,
                           features: Optional[Sequence[str]] = None, prob_col: str = "prob",
                           max_arrows: int = 500) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = panel_per_cluster(df, label, features=features, prob_col=prob_col, max_arrows=max_arrows)
    _save_fig(fig, savepath)
    return fig, axes
