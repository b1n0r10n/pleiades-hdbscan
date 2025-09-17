# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pleiades.validation import clustering_overview, numeric_summary, silhouette_if_possible, compare_to_valid

def test_overview_numeric_silhouette_compare():
    # Data sintetis hasil klaster
    n = 50
    df = pd.DataFrame({
        "parallax": np.r_[np.full(25, 6.2), np.full(25, 6.6)],
        "pmra": np.r_[np.full(25, 20.0), np.full(25, 21.0)],
        "pmdec": np.r_[np.full(25, -45.0), np.full(25, -44.0)],
        "color": np.r_[np.full(25, 1.3), np.full(25, 1.4)],
        "M_G": np.r_[np.full(25, 4.3), np.full(25, 4.4)],
        "cluster": np.r_[np.zeros(25, dtype=int), np.ones(25, dtype=int)],
    })
    labels = df["cluster"].to_numpy()
    probs = np.ones(n, dtype="float32")

    # Ikhtisar
    ov = clustering_overview(labels, probs=probs)
    assert ov["n_samples"] == n
    assert ov["n_clusters"] == 2
    assert ov["frac_noise"] == 0.0
    assert 0.9 <= ov["prob_mean"] <= 1.0

    # Ringkasan numerik
    ns = numeric_summary(df, label_col="cluster", features=["parallax","pmra","pmdec","color","M_G"])
    assert "parallax" in ns.columns.get_level_values(0)

    # Silhouette (pakai X sederhana)
    X = df[["parallax","pmra","pmdec","color","M_G"]].to_numpy(dtype="float32")
    sil = silhouette_if_possible(X, labels, metric="euclidean")
    assert np.isfinite(sil) or np.isnan(sil)

    # Compare ke valid
    valid = pd.DataFrame({
        "parallax": np.full(20, 6.4),
        "pmra": np.full(20, 20.5),
        "pmdec": np.full(20, -44.5),
        "color": np.full(20, 1.35),
        "M_G": np.full(20, 4.35),
    })
    comp = compare_to_valid(df, valid)
    for key in ["parallax_mean_cluster","parallax_mean_valid","pm_mag_mean_cluster","pm_mag_mean_valid","pm_angle_diff_deg"]:
        assert key in comp.index
