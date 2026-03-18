"""Tests for features."""
import pandas as pd, numpy as np
from src.data.features import compute_sma, compute_rsi, compute_features, compute_labels

def _df():
    np.random.seed(42)
    c = 1000+np.cumsum(np.random.randn(300)*10)
    return pd.DataFrame({"open":c,"high":c+5,"low":c-5,"close":c,"volume":np.full(300,5e6)},
                        index=pd.date_range("2023-01-01",periods=300,freq="B"))

def test_sma():
    assert compute_sma(pd.Series(range(1,31),dtype=float),20).isna().sum()==19

def test_rsi():
    assert (compute_rsi(_df()["close"]).dropna()>=0).all()

def test_features():
    for c in ["sma_20","rsi_14","macd"]:
        assert c in compute_features(_df()).columns

def test_labels():
    assert set(compute_labels(_df()).dropna().unique()).issubset({0,1})
