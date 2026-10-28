"""Why did the portfolio change? Switch each D31 (NSE) feature off one at a time and measure.

This is a diagnosis, not a search for better settings: nothing is logged to the run
tracker (so the Deflated-Sharpe trial count is unchanged) and no setting is changed.

Usage::

    python scripts/diagnose_portfolio.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from quantswing.alpha.events import event_block
from quantswing.alpha.research import AUX, COMPOSITE
from quantswing.backtest import metrics
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.factors.common import Panels
from quantswing.portfolio.construction import PortfolioResult, run_portfolio
from quantswing.portfolio.p5 import load_regime
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths


def without(p: Panels, drop: set[str]) -> Panels:
    """Panels minus some columns (e.g. every delisted stock)."""
    keep = [c for c in p.close.columns if c not in drop]

    def f(x: pd.DataFrame | None) -> pd.DataFrame | None:
        return None if x is None else x[[c for c in keep if c in x.columns]]

    return replace(
        p,
        open=p.open[keep],
        high=p.high[keep],
        low=p.low[keep],
        close=p.close[keep],
        volume=p.volume[keep],
        turnover=p.turnover[keep],
        deliv=f(p.deliv),
        restricted=f(p.restricted),
        pledge=f(p.pledge),
        div_yield=f(p.div_yield),
        sector=None if p.sector is None else p.sector.reindex(keep),
        delisted_last=None,
        delisted_haircut=None,
    )


def summary(
    res: PortfolioResult, bench: pd.Series, oos: pd.Timestamp, rf: float
) -> dict[str, float]:
    """OOS numbers + where the money went."""
    d = res.daily[res.daily.index >= oos]
    m = metrics.portfolio_metrics(d, bench, rf)
    t = res.trades
    t = t[pd.to_datetime(t["date"]) >= oos] if not t.empty else t
    yrs = max(len(d) / 252, 1e-9)
    dl = t[t.get("reason", pd.Series(dtype=str)) == "delisted"] if not t.empty else t
    return {
        "cagr": m.get("cagr", np.nan),
        "sharpe": m.get("sharpe", np.nan),
        "max_dd": m.get("max_dd", np.nan),
        "invested": float((d["invested"] / d["equity"]).mean()),
        "target": float(d["gross_target"].mean()),
        "turnover": float(t["value"].sum() / d["equity"].mean() / yrs) if not t.empty else 0.0,
        "costs_pa": float(t["cost"].sum() / d["equity"].mean() / yrs) if not t.empty else 0.0,
        "delist_n": float(len(dl)),
    }


def main() -> int:
    """Print the ablation table."""
    warnings.simplefilter("ignore", RuntimeWarning)
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    store = FactorStore(paths.factors)
    print("Loading data...", flush=True)
    p = load_panels(paths, cfg)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    scores = store.read("composite", COMPOSITE).reindex_like(p.close)
    reg = load_regime(paths)
    mult = reg["mult"] if reg is not None else None
    block = event_block(p, cfg.risk.event_window)
    bt = cfg.backtest
    first = scores.dropna(how="all").index.min()
    oos = max(
        pd.Timestamp(bt.start), pd.Timestamp(f"{first.year + bt.validation.train_years}-01-01")
    )
    start = pd.Timestamp(bt.start)
    dead = set(p.delisted_last.index) if p.delisted_last is not None else set()
    zero_h = None if p.delisted_haircut is None else p.delisted_haircut * 0.0

    cases = {
        "as now": (p, block),
        "no delisted stocks": (without(p, dead), block),
        "delisted, no haircut": (replace(p, delisted_haircut=zero_h), block),
        "no results-window block": (p, None),
        "no dividends": (replace(p, div_yield=None), block),
    }
    rows = []
    for name, (q, blk) in cases.items():
        runs = (
            (("no regime", None), ("regime", mult))
            if cfg.risk.portfolio.use_regime
            else (("-", None),)
        )
        for rname, m in runs:
            print(f"  {name} / {rname}", flush=True)
            res = run_portfolio(
                q,
                scores.reindex_like(q.close),
                gate.reindex_like(q.close).fillna(False),
                cfg,
                regime_mult=m,
                start=start,
                entry_block=blk,
            )
            rows.append(
                {"case": name, "regime": rname, **summary(res, p.bench, oos, bt.risk_free_annual)}
            )
    df = pd.DataFrame(rows)
    print(f"\nOut-of-sample from {oos.date()} (nothing logged; settings unchanged)\n")
    print(
        f"{'case':<26}{'regime':<11}{'CAGR':>7}{'Sharpe':>8}{'MaxDD':>8}{'Invest':>8}"
        f"{'Target':>8}{'Turn/yr':>8}{'Cost/yr':>8}{'Delist':>7}"
    )
    for r in df.itertuples():
        print(
            f"{r.case:<26}{r.regime:<11}{r.cagr:>7.1%}{r.sharpe:>8.2f}{r.max_dd:>8.1%}"
            f"{r.invested:>8.0%}{r.target:>8.0%}{r.turnover:>8.1f}{r.costs_pa:>8.2%}{r.delist_n:>7.0f}"
        )
    base = run_portfolio(p, scores, gate, cfg, regime_mult=mult, start=start, entry_block=block)
    eq = base.daily["equity"]
    yr = eq.groupby(pd.DatetimeIndex(eq.index).year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    bq = p.bench.reindex(eq.index)
    by = bq.groupby(pd.DatetimeIndex(bq.index).year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    inv = (
        (base.daily["invested"] / base.daily["equity"])
        .groupby(pd.DatetimeIndex(eq.index).year)
        .mean()
    )
    print("\nYear   portfolio          Nifty500   avg invested")
    for y in yr.index:
        print(f"{y}   {yr[y]:>+15.1%}  {by.get(y, np.nan):>+8.1%}   {inv[y]:>10.0%}")
    if reg is not None:
        share = reg["state"].dropna()
        share = share[share.index >= oos].value_counts(normalize=True)
        print("\nRegime time share (OOS): " + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
