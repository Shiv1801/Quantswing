"""Purged walk-forward splits (05 §3): train N years / test M years, rolling.

For samples with a label span [start, end]: a training sample is dropped if its span
overlaps the test window extended by ``embargo`` sessions on both sides (purging +
embargo). Returns integer positions into the sample table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold."""

    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walk_forward(
    starts: pd.Series,
    ends: pd.Series,
    sessions: pd.DatetimeIndex,
    train_years: int,
    test_years: int,
    embargo: int,
) -> list[Fold]:
    """Rolling folds over calendar years present in ``sessions``."""
    s = pd.to_datetime(starts).to_numpy()
    e = pd.to_datetime(ends).to_numpy()
    years = sorted(set(sessions.year))
    folds: list[Fold] = []
    for k in range(train_years, len(years) - test_years + 1):
        tr_y0, te_y0, te_y1 = years[k - train_years], years[k], years[k + test_years - 1]
        test_mask_sess = (sessions.year >= te_y0) & (sessions.year <= te_y1)
        te_sess = sessions[test_mask_sess]
        if len(te_sess) == 0:
            continue
        t0, t1 = te_sess[0], te_sess[-1]
        p0 = max(int(sessions.searchsorted(t0)) - embargo, 0)
        p1 = min(int(sessions.searchsorted(t1)) + embargo, len(sessions) - 1)
        lo, hi = sessions[p0].to_datetime64(), sessions[p1].to_datetime64()
        tr0 = pd.Timestamp(f"{tr_y0}-01-01").to_datetime64()
        in_test = (s >= t0.to_datetime64()) & (s <= t1.to_datetime64())
        overlaps = (s <= hi) & (e >= lo)
        in_train = (s >= tr0) & (s < t0.to_datetime64()) & ~overlaps
        folds.append(
            Fold(pd.Timestamp(tr0), t0, t1, np.flatnonzero(in_train), np.flatnonzero(in_test))
        )
    return folds
