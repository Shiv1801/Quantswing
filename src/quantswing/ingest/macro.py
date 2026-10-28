"""D39: find the cross-asset drivers in the Fyers symbol master (sector indices, G-Sec index,
gold ETF). Only what Fyers lists is used; nothing is guessed silently — every pick is printed.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pandas as pd

# driver name → exact Fyers tickers to try first, then keywords for a fallback search
SECTORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "IT": (("NSE:NIFTYIT-INDEX",), ()),
    "PHARMA": (("NSE:NIFTYPHARMA-INDEX",), ("PHARMA",)),
    "METAL": (("NSE:NIFTYMETAL-INDEX",), ("METAL",)),
    "AUTO": (("NSE:NIFTYAUTO-INDEX",), ("AUTO",)),
    "FMCG": (("NSE:NIFTYFMCG-INDEX",), ("FMCG",)),
    "ENERGY": (("NSE:NIFTYENERGY-INDEX",), ("ENERGY",)),
    "REALTY": (("NSE:NIFTYREALTY-INDEX",), ("REALTY",)),
    "PSUBANK": (("NSE:NIFTYPSUBANK-INDEX",), ("PSUBANK",)),
    "PVTBANK": (("NSE:NIFTYPVTBANK-INDEX",), ("PVTBANK",)),
    "MEDIA": (("NSE:NIFTYMEDIA-INDEX",), ("MEDIA",)),
    "FINSERV": (("NSE:FINNIFTY-INDEX", "NSE:NIFTYFINSERVICE-INDEX"), ("FINSERV", "FINNIFTY")),
    "INFRA": (("NSE:NIFTYINFRA-INDEX",), ("INFRA",)),
    "COMMODITIES": (("NSE:NIFTYCOMMODITIES-INDEX",), ("COMMODIT",)),
    "CONSUMPTION": (("NSE:NIFTYCONSUMPTION-INDEX",), ("CONSUMPTION",)),
    "HEALTHCARE": (("NSE:NIFTYHEALTHCARE-INDEX",), ("HEALTHCARE",)),
    "OILGAS": (("NSE:NIFTYOILANDGAS-INDEX",), ("OILGAS", "OILANDGAS", "OIL&GAS")),
    "CONSDUR": (("NSE:NIFTYCONSRDURBL-INDEX",), ("CONSRDURBL", "CONSUMERDUR", "CONSDUR")),
    "CHEMICALS": (("NSE:NIFTYCHEMICALS-INDEX",), ("CHEMICAL",)),
    "CPSE": (("NSE:NIFTYCPSE-INDEX",), ("CPSE",)),
}
GSEC = (("NSE:NIFTYGS10YR-INDEX",), ("GS10YR", "GSEC10", "10YR"))
GOLD_ETFS = (
    "NSE:GOLDBEES-EQ",
    "NSE:HDFCGOLD-EQ",
    "NSE:SETFGOLD-EQ",
    "NSE:ICICIGOLD-EQ",
    "NSE:AXISGOLD-EQ",
    "NSE:GOLD1-EQ",
)
TICKER_COL_RE = re.compile(r"^NSE:[A-Z0-9&_\-]+-(INDEX|EQ)$")


def master_tickers(text: str) -> list[str]:
    """Every NSE ticker in the symbol master (column auto-detected)."""
    raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)
    best, share = 0, -1.0
    for c in raw.columns:
        s = raw[c].astype(str).str.strip()
        sh = float(s.str.match(TICKER_COL_RE).mean())
        if sh > share:
            best, share = int(c), sh
    return sorted(set(raw[best].astype(str).str.strip()))


def pick(tickers: list[str], exact: tuple[str, ...], keys: tuple[str, ...]) -> str | None:
    """First exact ticker present, else the shortest ``-INDEX`` ticker containing a keyword."""
    have = set(tickers)
    for t in exact:
        if t in have:
            return t
    idx = [t for t in tickers if t.endswith("-INDEX")]
    for k in keys:
        hits = sorted((t for t in idx if k in t.upper()), key=len)
        if hits:
            return hits[0]
    return None


def find_drivers(tickers: list[str]) -> dict[str, str]:
    """Driver name → Fyers ticker, for what the symbol master lists."""
    out: dict[str, str] = {}
    for name, (exact, keys) in SECTORS.items():
        t = pick(tickers, exact, keys)
        if t:
            out[name] = t
    g = pick(tickers, *GSEC)
    if g:
        out["GSEC"] = g
    have = set(tickers)
    for t in GOLD_ETFS:
        if t in have:
            out["GOLD"] = t
            break
    return out


def save_symbols(path: Path, drivers: dict[str, str]) -> None:
    """Write the driver map read by ``load_panels`` / ``load_indices``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(drivers, indent=2, sort_keys=True), encoding="utf-8")


def load_symbols(path: Path) -> dict[str, str]:
    """The saved driver map (empty if ``macro_backfill.py`` has not run)."""
    if not path.exists():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
