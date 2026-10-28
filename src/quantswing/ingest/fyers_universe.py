"""Universe from the public Fyers symbol master (``NSE_CM.csv``, no login needed).

Filter (config ``settings.data.universe``):
* ticker ``NSE:<SYM>-<SERIES>`` with SERIES in ``series_allowed`` (EQ, BE; SME excluded);
* ``instrument_type`` in ``instrument_types_allowed``;
* name does not contain an ``exclude_name_patterns`` entry (ETFs share the -EQ suffix).

The CSV has no header; column positions come from config and are sanity-checked (the
ticker column must look like ``NSE:...``; ISIN like ``INE...``). If the check fails the
columns are auto-detected by pattern, and a warning is logged.

Known limitation (D22): the master lists only CURRENTLY listed securities → the
research universe is not survivorship-free.
"""

from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd
import requests

from quantswing.core.clock import today_ist
from quantswing.core.config import UniverseCfg
from quantswing.core.logging import get_logger
from quantswing.store.schema import UNIVERSE_COLUMNS

log = get_logger(__name__)

TICKER_RE = re.compile(r"^NSE:.+-[A-Z0-9]{1,3}$")
ISIN_RE = re.compile(r"^IN[A-Z0-9]{10}$")


def download_symbol_master(cfg: UniverseCfg, http: object = requests) -> str:
    """Download ``NSE_CM.csv`` text."""
    resp = http.get(cfg.symbol_master_url, timeout=cfg.request_timeout_s)  # type: ignore[attr-defined]
    resp.raise_for_status()
    return str(resp.text)


def _share(series: pd.Series, rx: re.Pattern[str]) -> float:
    s = series.dropna().astype(str).str.strip()
    return float(s.str.match(rx).mean()) if len(s) else 0.0


def _detect(raw: pd.DataFrame, rx: re.Pattern[str]) -> int:
    scores = {c: _share(raw[c], rx) for c in raw.columns}
    best = max(scores, key=lambda c: scores[c])
    if scores[best] < 0.5:
        raise ValueError(f"cannot find a column matching {rx.pattern}")
    return int(best)


def parse_symbol_master(text: str, cfg: UniverseCfg, as_of: date | None = None) -> pd.DataFrame:
    """Parse and filter the symbol master into the universe frame (``UNIVERSE_COLUMNS``).

    Also returns rejected-by-type counts in ``df.attrs["type_counts"]`` for diagnostics.
    """
    raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)
    raw.columns = pd.Index(range(raw.shape[1]))
    c = cfg.columns.model_dump()
    if c["ticker"] >= raw.shape[1] or _share(raw[c["ticker"]], TICKER_RE) < 0.5:
        # Layout moved: assume a uniform column offset, anchored on the ticker column.
        offset = _detect(raw, TICKER_RE) - c["ticker"]
        log.warning("symbol master columns moved; applying offset", extra={"offset": offset})
        c = {k: v + offset for k, v in c.items()}
    if c["isin"] >= raw.shape[1] or _share(raw[c["isin"]], ISIN_RE) < 0.5:
        log.warning("symbol master ISIN column moved; auto-detecting")
        c["isin"] = _detect(raw, ISIN_RE)

    df = pd.DataFrame(
        {
            "symbol": raw[c["ticker"]].str.strip(),
            "name": raw[c["name"]].str.strip(),
            "isin": raw[c["isin"]].str.strip(),
            "instrument_type": pd.to_numeric(raw[c["instrument_type"]], errors="coerce"),
            "lot_size": pd.to_numeric(raw[c["lot_size"]], errors="coerce"),
            "tick_size": pd.to_numeric(raw[c["tick_size"]], errors="coerce"),
            "fytoken": raw[c["fytoken"]].str.strip(),
        }
    )
    df = df[df["symbol"].str.match(TICKER_RE)]
    df["series"] = df["symbol"].str.rsplit("-", n=1).str[-1]
    df = df[df["series"].isin(cfg.series_allowed)]
    type_counts = df["instrument_type"].value_counts(dropna=False).to_dict()
    df = df[df["instrument_type"].isin(cfg.instrument_types_allowed)]
    if cfg.exclude_name_patterns:
        pat = "|".join(re.escape(p.upper()) for p in cfg.exclude_name_patterns)
        df = df[~df["name"].str.upper().str.contains(pat, regex=True)]
    df["as_of_date"] = as_of or today_ist()
    out = (
        df.drop_duplicates(subset="symbol")
        .sort_values("symbol")
        .reset_index(drop=True)[list(UNIVERSE_COLUMNS)]
    )
    out.attrs["type_counts"] = {str(k): int(v) for k, v in type_counts.items()}
    return out
