"""End-to-end P2 on a small synthetic market: DB → factors → IC report (HTML)."""

from __future__ import annotations

from datetime import date

from quantswing.alpha.ic_report import verdicts, write_report
from quantswing.alpha.research import COMPOSITE, build_factors, ic_analysis
from quantswing.core.config import get_config
from quantswing.ingest import pipeline
from quantswing.store.database import build_database
from quantswing.store.panels import FactorStore
from quantswing.store.repositories import DataPaths
from tests.conftest import FakeFD, make_candles, symbol_master_csv, weekdays
from tests.store.test_pipeline import FakeHTTP

DATES = weekdays(date(2023, 1, 2), date(2025, 6, 30))


def test_p2_end_to_end(tmp_paths: DataPaths) -> None:
    cfg = get_config()
    syms = [f"NSE:S{i:02d}-EQ" for i in range(40)]
    rows = [
        {"name": f"CO{i}", "ticker": s, "isin": f"INE{i:03d}A01011"} for i, s in enumerate(syms)
    ]
    pipeline.refresh_universe(tmp_paths, cfg, http=FakeHTTP(symbol_master_csv(rows)))
    series = {s: make_candles(DATES, seed=i) for i, s in enumerate(cfg.settings.data.indices)}
    series |= {s: make_candles(DATES, seed=100 + i) for i, s in enumerate(syms)}
    fd = FakeFD(series)
    end = DATES[-1]
    pipeline.load_indices(fd, tmp_paths, date(2015, 1, 1), end, cfg)  # type: ignore[arg-type]
    pipeline.build_calendar(tmp_paths, cfg)
    pipeline.load_equities(fd, tmp_paths, syms, date(2015, 1, 1), end, None, cfg)  # type: ignore[arg-type]
    pipeline.validate_all(tmp_paths, syms, end, cfg)
    build_database(tmp_paths)

    s = build_factors(tmp_paths, cfg)
    assert s.symbols == 40 and s.sessions == len(DATES) and s.gated_avg > 0
    names = FactorStore(tmp_paths.factors).names("neutral")
    assert {"TQ60", "MOM126", "CMP", "KAL_T", "RMOM", "H52"} <= set(names)

    rep = ic_analysis(tmp_paths, cfg)
    hs = set(cfg.factors.ic.horizons)
    assert set(rep.stats["h"]) == hs
    assert set(rep.stats["factor"]) == set(names) | {COMPOSITE}
    v = verdicts(rep, cfg.factors.ic.t_stat_pass)
    assert set(v["verdict"]) <= {"PASS", "PASS(-)", "FAIL"}
    out = write_report(rep, tmp_paths.reports, cfg.factors.ic.t_stat_pass, ["note"])
    html = out.read_text(encoding="utf-8")
    assert "Mean IC by horizon" in html and "TQ60" in html and COMPOSITE in html
    assert (tmp_paths.reports / "p2_ic_stats.csv").exists()


def test_p3_baseline_end_to_end(tmp_paths: DataPaths) -> None:
    """P2 outputs → P3 baseline: reproducible, logged, report written."""
    from quantswing.backtest.baseline import run_baseline
    from quantswing.backtest.report import write_report

    test_p2_end_to_end(tmp_paths)
    cfg = get_config()
    cfg2 = cfg.model_copy(
        update={
            "backtest": cfg.backtest.model_copy(
                update={
                    "start": date(2024, 3, 1),
                    "baseline": cfg.backtest.baseline.model_copy(update={"min_score": 0.5}),
                    "validation": cfg.backtest.validation.model_copy(
                        update={"montecarlo_sims": 200}
                    ),
                }
            )
        }
    )
    rep = run_baseline(tmp_paths, cfg2)
    assert rep.reproducible and rep.run_id and rep.trade["trades"] > 0
    assert all(rep.checks.values())
    assert rep.labels["events"] > 0 and 0 <= rep.labels["p_upper"] <= 1
    out = write_report(rep, None, tmp_paths.reports, ["n"])
    assert "baseline backtest" in out.read_text(encoding="utf-8")
    assert (tmp_paths.runs / rep.run_id / "metrics.json").exists()
