"""Universe parsing from the Fyers symbol master."""

from __future__ import annotations

from datetime import date

from quantswing.core.config import get_config
from quantswing.ingest.fyers_universe import parse_symbol_master
from tests.conftest import symbol_master_csv

ROWS = [
    {"name": "RELIANCE INDUSTRIES LTD", "ticker": "NSE:RELIANCE-EQ", "isin": "INE002A01018"},
    {"name": "TATA CONSULTANCY SERV LT", "ticker": "NSE:TCS-EQ", "isin": "INE467B01029"},
    {"name": "SOME CO LTD", "ticker": "NSE:SOMECO-BE", "isin": "INE111A01011"},
    {"name": "SME CO", "ticker": "NSE:SMECO-SM", "isin": "INE222A01011"},
    {"name": "NIPPON INDIA ETF NIFTY BEES", "ticker": "NSE:NIFTYBEES-EQ", "isin": "INF204KB14I2"},
    {"name": "RELIANCE NCD", "ticker": "NSE:REL-N1", "isin": "INE002A08011", "itype": 2},
    {"name": "WARRANT X", "ticker": "NSE:WARX-EQ", "isin": "INE333A01011", "itype": 3},
    {"name": "NIFTY 50", "ticker": "NSE:NIFTY50-INDEX", "isin": "", "itype": 10},
]


def test_parse_filters_series_types_and_etfs() -> None:
    uc = get_config().settings.data.universe
    df = parse_symbol_master(symbol_master_csv(ROWS), uc, as_of=date(2026, 9, 25))
    assert list(df["symbol"]) == ["NSE:RELIANCE-EQ", "NSE:SOMECO-BE", "NSE:TCS-EQ"]
    assert set(df["series"]) == {"EQ", "BE"}
    assert df.loc[df.symbol == "NSE:RELIANCE-EQ", "isin"].item() == "INE002A01018"
    assert (df["as_of_date"] == date(2026, 9, 25)).all()
    assert df.attrs["type_counts"]["3"] == 1  # warrant seen, then dropped


def test_parse_autodetects_moved_columns() -> None:
    uc = get_config().settings.data.universe
    text = symbol_master_csv(ROWS)
    # shift every row by one leading column -> configured positions are now wrong
    shifted = "\n".join("X," + ln for ln in text.strip().splitlines()) + "\n"
    cols = uc.columns.model_copy(update={"ticker": 9, "isin": 5})
    df = parse_symbol_master(shifted, uc.model_copy(update={"columns": cols}))
    assert "NSE:RELIANCE-EQ" in set(df["symbol"])
