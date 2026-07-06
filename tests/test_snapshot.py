"""Snapshot round-trip: a saved bundle reloads to the same data (offline, no network)."""
from __future__ import annotations

import numpy as np

from taa.data.snapshot import save_snapshot, load_snapshot
from taa.data.synthetic import make_synthetic_bundle


def test_snapshot_roundtrip(tmp_path):
    b = make_synthetic_bundle()
    save_snapshot(b, str(tmp_path), vintage="2026-07-05")
    r = load_snapshot(str(tmp_path))

    assert list(r.assets) == list(b.assets)
    assert np.allclose(r.prices.raw_copy().values, b.prices.raw_copy().values, atol=1e-8)
    assert np.allclose(r.carry.raw_copy().values, b.carry.raw_copy().values, atol=1e-8)
    assert np.allclose(r.macro.raw_copy().values, b.macro.raw_copy().values, atol=1e-8)
    assert r.macro.release_lag == b.macro.release_lag

    # point-in-time behaviour survives the round-trip
    as_of = r.dates[100]
    assert r.prices.history(as_of).index.max() <= as_of
    assert r.macro.history(as_of).index.max() < as_of
