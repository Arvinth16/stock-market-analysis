"""Tests for data fetcher module."""

from src.data.universe import get_symbols, get_symbol_name, get_symbol_sector, get_universe


def test_universe_not_empty():
    """Universe should contain stocks."""
    assert len(get_universe()) > 0


def test_symbols_are_nse():
    """All symbols should end with .NS for Yahoo Finance."""
    for symbol in get_symbols():
        assert symbol.endswith(".NS"), f"{symbol} doesn't end with .NS"


def test_symbol_name():
    """Should return a human-readable name."""
    name = get_symbol_name("RELIANCE.NS")
    assert name == "Reliance Industries"


def test_symbol_sector():
    """Should return a sector."""
    sector = get_symbol_sector("TCS.NS")
    assert sector == "IT"


def test_unknown_symbol_name():
    """Unknown symbol should return the symbol itself."""
    name = get_symbol_name("UNKNOWN.NS")
    assert name == "UNKNOWN.NS"
