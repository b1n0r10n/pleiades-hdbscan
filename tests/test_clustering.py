# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from pleiades.clustering import run_hdbscan, attach_cluster, choose_members, cluster_sizes, cluster_centroids

def test_run_hdbscan_and_attach():
    # Dua klaster sintetis
    rng = np.random.default_rng(0)
    A = rng.normal(loc=[0,0,0,0], scale=0.3, size=(40,4))
    B = rng.normal(loc=[3,3,3,3], scale=0.3, size=(40,4))
    X = np.vstack([A,B]).astype("float32")

    labels, probs, clusterer = run_hdbscan(X, min_cluster_size=10, gen_min_span_tree=True)
    n_clusters = len(set(labels.tolist()) - {-1})
    assert n_clusters >= 2
    assert (labels >= -1).all()
    assert (probs >= 0).all() and (probs <= 1).all()

    # Attach ke DataFrame asal (dengan kolom langit/fisik minimal)
    df = pd.DataFrame({
        "parallax": np.r_[np.full(40, 6.0), np.full(40, 6.5)],
        "pmra": np.r_[np.full(40, 20.0), np.full(40, 21.0)],
        "pmdec": np.r_[np.full(40, -45.0), np.full(40, -44.0)],
        "ra": np.linspace(55, 56, len(X)),
        "dec": np.linspace(22, 23, len(X)),
    })
    out = attach_cluster(df, labels, probs, prob_threshold=0.5)
    assert {"cluster","prob","is_member"}.issubset(out.columns)
    # choose_members konsisten
    mask = choose_members(labels, probs, prob_threshold=0.5)
    assert (out["is_member"].to_numpy() == mask).all()

    # Ukuran klaster
    sizes = cluster_sizes(labels)
    assert sum(sizes.values()) == len(labels)

    # Centroid
    cents = cluster_centroids(out, feature_cols=["parallax","pmra","pmdec"])
    assert {"cluster","n","parallax_mean","pmra_mean","pmdec_mean"}.issubset(cents.columns)
    assert len(cents) == n_clusters
