"""End-to-end P1 pipeline on fakes: universe → indices → calendar → equities → validate → DB → exit."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from quantswing.core.config import get_config
from quantswing.ingest import pipeline
from quantswing.store.database import build_database
from quantswing.store.repositories import DataPaths, PriceStore, merge_candles
from tests.conftest import FakeFD, make_candles, symbol_master_csv, weekdays

DATES = weekdays(date(2024, 1, 1), date(2026, 9, 25))
HOLIDAY = date(2025, 1, 27)  # remove from index -> calendar must treat it as holiday
IDX_DATES = [d for d in DATES if d != HOLIDAY]


class FakeHTTP:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self, url: str, timeout: float) -> FakeHTTP:
        return self

    def raise_for_status(self) -> None:
        return None


def test_pipeline_end_to_end(tmp_paths: DataPaths) -> None:
    cfg = get_config()
    rows = [
        {"name": "ALPHA LTD", "ticker": "NSE:ALPHA-EQ", "isin": "INE100A01011"},
        {"name": "BETA LTD", "ticker": "NSE:BETA-EQ", "isin": "INE200A01011"},
        {"name": "GAMMA LTD", "ticker": "NSE:GAMMA-BE", "isin": "INE300A01011"},
    ]
    uni = pipeline.refresh_universe(tmp_paths, cfg, http=FakeHTTP(symbol_master_csv(rows)))
    assert len(uni) == 3

    bad = make_candles(IDX_DATES, seed=2)
    bad.loc[5, "low"] = bad.loc[5, "high"] * 2  # one reject row
    series = {s: make_candles(IDX_DATES, seed=i) for i, s in enumerate(cfg.settings.data.indices)}
    alpha = make_candles(IDX_DATES, seed=1)
    for col in ("open", "high", "low", "close"):
        alpha.loc[300:, col] = alpha.loc[300:, col] / 2  # unadjusted 1:1 bonus
    series |= {"NSE:ALPHA-EQ": alpha, "NSE:BETA-EQ": bad}
    fd = FakeFD(series)  # GAMMA has no data

    end = date(2026, 9, 25)
    s = pipeline.load_indices(fd, tmp_paths, date(2015, 1, 1), end, cfg)  # type: ignore[arg-type]
    assert s.fetched == len(cfg.settings.data.indices)
    n = pipeline.build_calendar(tmp_paths, cfg)
    assert n == len(IDX_DATES)

    es = pipeline.load_equities(
        fd, tmp_paths, list(uni["symbol"]), date(2015, 1, 1), end, None, cfg
    )  # type: ignore[arg-type]
    assert es.fetched == 2 and es.empty == 1

    iss = pipeline.validate_all(tmp_paths, list(uni["symbol"]), end, cfg)
    assert (iss["check"] == "ohlc").sum() == 1

    counts = build_database(tmp_paths)
    assert counts["prices_daily"] == 2 * len(IDX_DATES) - 1  # reject row excluded
    con = duckdb.connect(str(tmp_paths.duckdb), read_only=True)
    isin = con.execute(
        "SELECT DISTINCT isin FROM prices_daily WHERE symbol='NSE:ALPHA-EQ'"
    ).fetchone()
    px = con.execute(
        "SELECT date, close, adj_close, adj_factor FROM prices_daily "
        "WHERE symbol='NSE:ALPHA-EQ' ORDER BY date"
    ).fetchdf()
    con.close()
    assert isin == ("INE100A01011",)
    jump_raw = px["close"].iloc[300] / px["close"].iloc[299]
    jump_adj = px["adj_close"].iloc[300] / px["adj_close"].iloc[299]
    assert jump_raw < 0.6 and 0.9 < jump_adj < 1.1  # continuity restored
    assert px["adj_factor"].iloc[0] == 0.5 and px["adj_factor"].iloc[-1] == 1.0

    rep = pipeline.exit_report(tmp_paths, end, cfg)
    assert rep.with_data == 2 and rep.rejected_rows == 1 and rep.sessions == len(IDX_DATES)
    assert rep.adjustments == 1
    assert rep.earliest == date(2024, 1, 1)
    assert not rep.checks[f"universe >= {cfg.settings.validation.min_universe_symbols}"]


def test_merge_candles_new_wins() -> None:
    a = pd.DataFrame({"date": [date(2026, 1, 1), date(2026, 1, 2)], "close": [1.0, 2.0]})
    b = pd.DataFrame({"date": [date(2026, 1, 2), date(2026, 1, 5)], "close": [9.0, 3.0]})
    m = merge_candles(a, b)
    assert list(m["close"]) == [1.0, 9.0, 3.0]


def test_price_store_roundtrip(tmp_paths: DataPaths) -> None:
    from quantswing.store.repositories import candles_for_store

    st = PriceStore(tmp_paths.eq_dir)
    df = candles_for_store(
        make_candles(DATES[:5]), "NSE:M&M-EQ", pd.Timestamp.now(tz="Asia/Kolkata")
    )
    st.write("NSE:M&M-EQ", df)
    assert st.path("NSE:M&M-EQ").name == "NSE_MandM-EQ.parquet"
    assert st.last_date("NSE:M&M-EQ") == DATES[4]
    back = st.read("NSE:M&M-EQ")
    assert back is not None and str(back["ts"].dt.tz) == "Asia/Kolkata"
