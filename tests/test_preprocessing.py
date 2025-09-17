# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pleiades.preprocessing import as_float32, standardize

def test_as_float32_and_standardize():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [10, 12, 14, 16],
    })
    df32 = as_float32(df)
    assert str(df32["a"].dtype) == "float32"
    assert str(df32["b"].dtype) == "float32"

    X, scaler, Xdf = standardize(df32, return_scaler=True, as_float32_output=True)
    # Mean ~ 0, Std ~ 1
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X.std(axis=0, ddof=0), 1.0, atol=1e-6)
