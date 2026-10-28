"""Run every enabled sleeve through walk-forward validation; persist results.

Outputs: ``data/processed/factors/../sleeves_fitted.json`` (live parameters + verdicts),
one tracker run per sleeve (artifacts: OOS trades + daily curve) and the HTML report.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from quantswing.alpha.research import AUX, COMPOSITE
from quantswing.backtest import metrics
from quantswing.backtest.baseline import market_data
from quantswing.backtest.tracking import RunTracker
from quantswing.core.clock import now_ist
from quantswing.core.config import AppConfig, get_config
from quantswing.sleeves.base import REQUIRED, FactorView, MissingFactorError
from quantswing.sleeves.registry import SLEEVES
from quantswing.sleeves.validate import SleeveResult, run_sleeve
from quantswing.store.panels import FactorStore, load_panels
from quantswing.store.repositories import DataPaths


def fitted_path(paths: DataPaths) -> Path:
    """Where live sleeve parameters are stored."""
    return paths.factors.parent / "sleeves_fitted.json"


def run_all(
    paths: DataPaths,
    cfg: AppConfig | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[SleeveResult]:
    """Validate the requested (default: every enabled) sleeve. S5 needs NSE results data."""
    cfg = cfg or get_config()
    store = FactorStore(paths.factors)
    p = load_panels(paths, cfg, limit=limit)
    md = market_data(p, cfg)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    comp_path = store.path("composite", COMPOSITE)
    scores = (
        store.read("composite", COMPOSITE).reindex_like(p.close)
        if comp_path.exists()
        else pd.DataFrame(0.0, index=p.close.index, columns=p.close.columns)
    )
    fv = FactorView(p, store)
    todo = names or [s for s in SLEEVES if cfg.sleeves.enabled.get(s, False)]
    if "S5" in todo and (p.events is None or p.events.empty):
        todo = [s for s in todo if s != "S5"]
        if progress:
            progress("S5: skipped (no NSE results calendar - run nse_backfill.py)")
    missing = sorted(
        {f for s in todo for f in REQUIRED.get(s, ()) if not store.path("raw", f).exists()}
    )
    if missing:
        raise MissingFactorError(", ".join(missing))
    tracker = RunTracker(paths.runs, cfg.backtest.tracking.use_mlflow)
    results: list[SleeveResult] = []
    from quantswing.portfolio.p5 import load_regime

    regime = load_regime(paths)
    for name in todo:
        r = run_sleeve(name, p, md, fv, gate, scores, cfg, regime)
        sr, _, _, _ = (
            metrics.returns_moments(r.daily) if len(r.daily) > 2 else (float("nan"), 0, 3, 0)
        )
        run_metrics: dict[str, float | None] = {
            **r.port,
            **r.trade,
            "sharpe_pp": sr,
            "stress_expectancy_r": r.stress.get("expectancy_r"),
        }
        r.extra["run_id"] = tracker.log_run(
            f"sleeve_{name}",
            {
                "sleeve": name,
                "rules": getattr(cfg.sleeves.rules, name).model_dump(),
                "fit": cfg.sleeves.fit.model_dump(),
            },
            metrics.nan_to_none(run_metrics),
            tags={"verdict": r.verdict, "regime_filter": regime is not None},
            artifacts={"trades": r.trades, "daily": r.daily, "folds": r.folds},
        )
        results.append(r)
        if progress:
            progress(f"{name}: {r.verdict}")
    save_fitted(paths, results)
    return results


def save_fitted(paths: DataPaths, results: list[SleeveResult]) -> None:
    """Persist live parameters + verdicts (read by the dashboard and, from P6, the live engine)."""
    out: dict[str, Any] = {"updated": now_ist().isoformat(timespec="seconds"), "sleeves": {}}
    for r in results:
        out["sleeves"][r.name] = {
            "verdict": r.verdict,
            "reasons": r.reasons,
            "signals": r.n_signals,
            "oos": metrics.nan_to_none(
                {
                    **{k: r.port.get(k) for k in ("cagr", "sharpe", "max_dd")},
                    **{
                        k: r.trade.get(k)
                        for k in ("trades", "win_rate", "expectancy_r", "profit_factor")
                    },
                }
            ),
            "stress_expectancy_r": r.stress.get("expectancy_r"),
            "exits": r.live_exits,
            "ev": r.live_ev,
            "fit_confident": r.live_confident,
            "run_id": r.extra.get("run_id"),
        }
    f = fitted_path(paths)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
