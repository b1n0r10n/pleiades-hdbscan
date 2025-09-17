# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from pleiades.plotting import sky_scatter, sky_quiver, pm_plane, cmd, prob_hist, feature_hist_by_cluster, condensed_tree

def _dummy_df():
    n = 60
    df = pd.DataFrame({
        "ra": np.linspace(55, 56, n),
        "dec": np.linspace(22, 23, n),
        "pmra": np.r_[np.full(n//2, 20.0), np.full(n - n//2, 21.0)],
        "pmdec": np.r_[np.full(n//2, -45.0), np.full(n - n//2, -44.0)],
        "color": np.r_[np.full(n//2, 1.3), np.full(n - n//2, 1.5)],
        "M_G": np.r_[np.full(n//2, 4.3), np.full(n - n//2, 4.5)],
        "prob": np.linspace(0.2, 0.95, n),
        "cluster": np.r_[np.zeros(n//2, dtype=int), np.ones(n - n//2, dtype=int)],
        "is_member": np.r_[np.zeros(n//2, dtype=bool), np.ones(n - n//2, dtype=bool)],
    })
    return df

def test_plotting_smoke_all():
    df = _dummy_df()
    ax = sky_scatter(df, members_only=False); assert ax is not None
    ax = sky_scatter(df, members_only=True);  assert ax is not None
    ax = sky_quiver(df, sample=50, members_only=True); assert ax is not None
    ax = pm_plane(df, members_only=False); assert ax is not None
    ax = pm_plane(df, members_only=True);  assert ax is not None
    ax = cmd(df, members_only=True);       assert ax is not None
    ax = prob_hist(df, bins=10);           assert ax is not None
    ax = feature_hist_by_cluster(df, "parallax", bins=10, members_only=True); assert ax is not None

def test_condensed_tree_optional():
    hdbscan = pytest.importorskip("hdbscan")
    try:
        from hdbscan import plots as _plots  # noqa: F401
    except Exception:
        pytest.skip("hdbscan.plots tidak tersedia")

    # Buat clusterer kecil untuk condensed tree
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0,0], scale=0.3, size=(30,2)),
        rng.normal(loc=[3,3], scale=0.3, size=(30,2)),
    ]).astype("float32")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=8, gen_min_span_tree=True).fit(X)
    ax = condensed_tree(clusterer, select_clusters=True)
    assert ax is not None
