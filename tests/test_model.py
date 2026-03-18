"""Tests for model module."""
from src.models.trainer import DB_FEATURE_COLS

def test_feature_cols_defined():
    assert len(DB_FEATURE_COLS) > 0
    assert "rsi_14" in DB_FEATURE_COLS

def test_feature_cols_no_duplicates():
    assert len(DB_FEATURE_COLS) == len(set(DB_FEATURE_COLS))
