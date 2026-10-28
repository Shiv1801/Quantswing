"""NSE file/API URLs and parsers (pure functions: bytes/JSON in → DataFrame out).

Formats (verified 2026-09-26 against NSE samples and the NseIndiaApi reference):

=================  =====================================================================
Bhavcopy (old)     ``content/historical/EQUITIES/{Y}/{MON}/cm{DD}{MON}{Y}bhav.csv.zip`` —
                   SYMBOL, SERIES, OPEN, HIGH, LOW, CLOSE, LAST, PREVCLOSE, TOTTRDQTY,
                   TOTTRDVAL, TIMESTAMP, TOTALTRADES, ISIN (until 2024-07-05)
Bhavcopy (UDiFF)   ``content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip`` — TradDt,
                   FinInstrmTp, ISIN, TckrSymb, SctySrs, OpnPric, HghPric, LwPric, ClsPric,
                   PrvsClsgPric, TtlTradgVol, TtlTrfVal, TtlNbOfTxsExctd (from 2024-07-08)
MTO delivery       ``archives/equities/mto/MTO_{DDMMYYYY}.DAT`` — header "Trade Date <DD-MON-
                   YYYY>", records ``20,<n>,SYMBOL,SERIES,traded,deliverable,pct``
Surveillance       ``content/cm/REG1_IND{DDMMYY}.csv`` (or ``REG_IND``) — Symbol, Series, GSM,
                   Long/Short_Term_Additional_Surveillance_Measure, ESM, IRP, Default,
                   Pledge, … ; ``100`` = not under the measure
Symbol changes     ``content/misc/symbol_change.csv`` (old symbol → new symbol, date)
API                corporates-corporateActions, corporates-financial-results,
                   corporate-board-meetings, holiday-master, NextApi GetQuoteApi (sector)
=================  =====================================================================
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

MON = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
BHAV_SERIES = ("EQ", "BE", "BZ")  # BZ kept: stocks often move there before delisting
BHAV_COLUMNS = (
    "date",
    "symbol",
    "series",
    "isin",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    "volume",
    "turnover",
    "trades",
)
MTO_COLUMNS = ("date", "symbol", "series", "traded_qty", "deliv_qty", "deliv_pct")
SURV_FLAGS = (
    "gsm",
    "asm_long",
    "asm_short",
    "esm",
    "irp",
    "default",
    "pledge",
    "total_pledge",
    "encumbered",
    "loss_making",
    "bz",
)
SURV_COLUMNS = ("date", "symbol", "series", *SURV_FLAGS)


# --------------------------------------------------------------------------- URLs
def bhav_urls(d: date, hosts: Sequence[str], udiff_switch: date) -> list[str]:
    """Bhavcopy URLs for session ``d`` (format by date)."""
    if d < udiff_switch:
        m = MON[d.month - 1]
        rel = f"/content/historical/EQUITIES/{d.year}/{m}/cm{d.day:02d}{m}{d.year}bhav.csv.zip"
    else:
        rel = f"/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return [h + rel for h in hosts]


def mto_urls(d: date, hosts: Sequence[str]) -> list[str]:
    """Security-wise delivery (MTO) URLs."""
    return [f"{h}/archives/equities/mto/MTO_{d:%d%m%Y}.DAT" for h in hosts]


def surveillance_urls(d: date, hosts: Sequence[str]) -> list[str]:
    """Daily surveillance-indicator URLs (two historical names)."""
    return [f"{h}/content/cm/{n}{d:%d%m%y}.csv" for h in hosts for n in ("REG1_IND", "REG_IND")]


def symbol_change_urls(hosts: Sequence[str]) -> list[str]:
    """Symbol-change list URLs."""
    rel = ("/content/misc/symbol_change.csv", "/content/equities/symbolchange.csv")
    return [h + r for h in hosts for r in rel]


# --------------------------------------------------------------------------- helpers
def _unzip(content: bytes) -> bytes:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            return z.read(names[0])
    return content


def _num(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(t.where(~t.isin(["-", "", "nan", "NaN", "None"])), errors="coerce")


def _norm_cols(cols: Iterable[str]) -> list[str]:
    return [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in cols]


DATE_FORMATS = (
    "%d-%b-%Y",
    "%Y-%m-%d",
    "%d-%b-%y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d%b%Y",
    "%Y%m%d",
)


def parse_date(x: Any) -> date | None:
    """Tolerant date parse (NSE uses several formats)."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    s = str(x).strip()
    if not s or s in ("-", "nan", "None"):
        return None
    s = s.split(" ")[0] if re.match(r"^\d{1,2}-[A-Za-z]{3}-\d{4} ", s) else s
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_datetime(x: Any) -> datetime | None:
    """``16-Jan-2025 20:20:21`` / ``16-Jan-2025 20:20`` / date-only → naive IST datetime."""
    if x is None:
        return None
    s = str(x).strip()
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    d = parse_date(s)
    return datetime(d.year, d.month, d.day) if d else None


def _empty(cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([], dtype="object") for c in cols})


# --------------------------------------------------------------------------- bhavcopy
def parse_bhav(content: bytes) -> pd.DataFrame:
    """Old or UDiFF bhavcopy → ``BHAV_COLUMNS`` (EQ/BE/BZ stock rows only)."""
    raw = pd.read_csv(io.BytesIO(_unzip(content)), dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    if "TckrSymb" in raw.columns:
        if "FinInstrmTp" in raw.columns:
            raw = raw[raw["FinInstrmTp"].str.strip() == "STK"]
        m = {
            "date": "TradDt",
            "symbol": "TckrSymb",
            "series": "SctySrs",
            "isin": "ISIN",
            "open": "OpnPric",
            "high": "HghPric",
            "low": "LwPric",
            "close": "ClsPric",
            "prev_close": "PrvsClsgPric",
            "volume": "TtlTradgVol",
            "turnover": "TtlTrfVal",
            "trades": "TtlNbOfTxsExctd",
        }
    elif "SYMBOL" in raw.columns and "TOTTRDQTY" in raw.columns:
        m = {
            "date": "TIMESTAMP",
            "symbol": "SYMBOL",
            "series": "SERIES",
            "isin": "ISIN",
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
            "prev_close": "PREVCLOSE",
            "volume": "TOTTRDQTY",
            "turnover": "TOTTRDVAL",
            "trades": "TOTALTRADES",
        }
    else:
        raise ValueError(f"unknown bhavcopy layout: {list(raw.columns)[:8]}")
    raw = raw[raw[m["series"]].str.strip().isin(BHAV_SERIES)]
    out = pd.DataFrame(index=raw.index)
    for k, c in m.items():
        if k in ("symbol", "series", "isin"):
            out[k] = raw[c].str.strip() if c in raw else ""
        elif k == "date":
            out[k] = [parse_date(v) for v in raw[c]] if c in raw else None
        else:
            out[k] = _num(raw[c]) if c in raw else np.nan
    return out.reset_index(drop=True)[list(BHAV_COLUMNS)]


# --------------------------------------------------------------------------- MTO
def parse_mto(content: bytes) -> pd.DataFrame:
    """MTO security-wise delivery file → ``MTO_COLUMNS``."""
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    d: date | None = None
    for ln in lines[:6]:
        if ln.lower().startswith("trade date"):
            m = re.search(r"<([^>]+)>", ln)
            d = parse_date(m.group(1)) if m else parse_date(ln.split(",")[-1])
    body = [ln for ln in lines if ln.startswith("20,")]
    if d is None or not body:
        return _empty(MTO_COLUMNS)
    df = pd.read_csv(
        io.StringIO("\n".join(body)),
        header=None,
        usecols=range(7),
        names=["_t", "_n", "symbol", "series", "traded_qty", "deliv_qty", "deliv_pct"],
        dtype=str,
    )
    out = pd.DataFrame(
        {
            "date": d,
            "symbol": df["symbol"].str.strip(),
            "series": df["series"].str.strip(),
            "traded_qty": _num(df["traded_qty"]),
            "deliv_qty": _num(df["deliv_qty"]),
            "deliv_pct": _num(df["deliv_pct"]),
        }
    )
    return out[out["series"].isin(BHAV_SERIES)].reset_index(drop=True)


# --------------------------------------------------------------------------- surveillance
# field order of REG_IND (NSE file-structure note) for header-less files
SURV_POSITIONAL = {
    1: "symbol",
    4: "series",
    5: "gsm",
    6: "asm_long",
    8: "irp",
    9: "asm_short",
    10: "default",
    14: "pledge",
    16: "total_pledge",
    18: "esm",
    19: "loss_making",
    20: "encumbered",
    21: "bz",
}


def _surv_key(c: str) -> str | None:
    if c == "symbol":
        return "symbol"
    if c == "series":
        return "series"
    if c == "gsm":
        return "gsm"
    if c.startswith("long_term_additional"):
        return "asm_long"
    if c.startswith("short_term_additional"):
        return "asm_short"
    if c == "esm":
        return "esm"
    if c.startswith("insolvency"):
        return "irp"
    if c == "default":
        return "default"
    if c == "pledge":
        return "pledge"
    if c == "total_pledge":
        return "total_pledge"
    if "encumbered" in c:
        return "encumbered"
    if "loss_making" in c:
        return "loss_making"
    if "bz" in c and "series" in c:
        return "bz"
    return None


def parse_surveillance(content: bytes, d: date) -> pd.DataFrame:
    """Daily surveillance indicators → ``SURV_COLUMNS`` (flags: True = under the measure)."""
    raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str, keep_default_na=False)
    if raw.empty:
        return _empty(SURV_COLUMNS)
    first = _norm_cols(raw.iloc[0].tolist())
    if "symbol" in first:
        mapping = {i: k for i, c in enumerate(first) if (k := _surv_key(c))}
        raw = raw.iloc[1:]
    else:
        mapping = {i: k for i, k in SURV_POSITIONAL.items() if i < raw.shape[1]}
    out = pd.DataFrame({"date": [d] * len(raw)})
    for i, k in mapping.items():
        col = raw.iloc[:, i].astype(str).str.strip()
        if k in ("symbol", "series"):
            out[k] = col.to_numpy()
        else:
            v = pd.to_numeric(col, errors="coerce").to_numpy()
            out[k] = np.isfinite(v) & (v != 100)
    for k in SURV_FLAGS:
        if k not in out:
            out[k] = False
    if "series" not in out:
        out["series"] = ""
    return out[list(SURV_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------- symbol changes
def parse_symbol_change(content: bytes) -> pd.DataFrame:
    """Symbol changes → ``old``, ``new``, ``date`` (tolerant to header / no header)."""
    raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str, keep_default_na=False)
    if raw.empty:
        return _empty(("old", "new", "date"))
    head = _norm_cols(raw.iloc[0].tolist())
    idx: dict[str, int] = {}
    for i, c in enumerate(head):
        if "old" in c or "key_symbol" in c or c.endswith("prev_symbol"):
            idx.setdefault("old", i)
        elif ("new" in c and "symbol" in c) or c == "new":
            idx.setdefault("new", i)
        elif "date" in c or "applicable" in c or "effective" in c:
            idx.setdefault("date", i)
    if len(idx) == 3:
        raw = raw.iloc[1:]
    else:
        n = raw.shape[1]
        idx = {"old": 1, "new": 2, "date": 3} if n >= 4 else {"old": 0, "new": 1, "date": 2}
    out = pd.DataFrame(
        {
            "old": raw.iloc[:, idx["old"]].str.strip().str.upper(),
            "new": raw.iloc[:, idx["new"]].str.strip().str.upper(),
            "date": [parse_date(x) for x in raw.iloc[:, idx["date"]]],
        }
    )
    out = out[(out["old"] != "") & (out["new"] != "") & out["date"].notna()]
    return out[out["old"] != out["new"]].drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------- JSON APIs
def as_rows(payload: Any) -> list[dict[str, Any]]:
    """API payloads come as a list or as ``{"data": [...]}``."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for k in ("data", "rows", "results"):
            if isinstance(payload.get(k), list):
                return [r for r in payload[k] if isinstance(r, dict)]
    return []


CA_COLUMNS = (
    "symbol",
    "series",
    "isin",
    "ex_date",
    "subject",
    "kind",
    "ratio",
    "dividend",
)


def _norm_subject(t: str) -> str:
    t = t.lower().replace("–", "-").replace("—", "-").replace("/-", " ")
    t = re.sub(r"(?<![a-z])(?:rs|re|inr)\.?(?![a-z])", " ", t)
    return re.sub(r"\s+", " ", t).strip()


REORG_WORDS = re.compile(
    r"amalgamat|merger|scheme of arrangement|arrangement|demerger|de-merger|delisting|"
    r"reverse book|buy ?back|open offer|takeover"
)


def parse_ca_subject(subject: str) -> list[tuple[str, float]]:
    """``[(kind, value)]``: split/consolidation/bonus → price ratio (new/old), dividend → ₹.

    ``Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share`` →
    ``("split", 0.2)``; ``Bonus 1:1`` → ``("bonus", 0.5)``;
    ``Interim Dividend - Rs 2.50 Per Share`` → ``("dividend", 2.5)``.
    Parts combined with ``/`` or ``AND`` are parsed independently.
    """
    out: list[tuple[str, float]] = []
    t = _norm_subject(subject or "")
    if REORG_WORDS.search(t):
        out.append(("reorg", float("nan")))  # merger/scheme: holders are paid, not wiped out
    parts = re.split(r"\s+and\s+|\s*/\s*(?=[a-z])|;", t)
    for part in parts:
        if re.search(r"split|sub[- ]?divi", part) or "consolidat" in part:
            m = re.search(r"from\s*(\d+(?:\.\d+)?)\s*(?:per share\s*)?to\s*(\d+(?:\.\d+)?)", part)
            if m and float(m.group(1)) > 0:
                kind = "consolidation" if "consolidat" in part else "split"
                out.append((kind, float(m.group(2)) / float(m.group(1))))
        elif "bonus" in part:
            m = re.search(r"(\d+)\s*:\s*(\d+)", part)
            if m and "preference" not in part and float(m.group(2)) > 0:
                a, b = float(m.group(1)), float(m.group(2))
                out.append(("bonus", b / (a + b)))
        elif "dividend" in part:
            if "%" in part:
                continue  # percent-of-face-value dividends: face value needed; skipped
            m = re.search(r"dividend[^0-9]{0,40}?(\d+(?:\.\d+)?)", part)
            if m:
                out.append(("dividend", float(m.group(1))))
    return out


def parse_corp_actions(payload: Any) -> pd.DataFrame:
    """Corporate-actions API → one row per parsed action (``CA_COLUMNS``)."""
    rows: list[dict[str, Any]] = []
    for r in as_rows(payload):
        ex = parse_date(r.get("exDate"))
        if ex is None:
            continue
        for kind, v in parse_ca_subject(str(r.get("subject") or "")):
            rows.append(
                {
                    "symbol": str(r.get("symbol") or "").strip().upper(),
                    "series": str(r.get("series") or "").strip(),
                    "isin": str(r.get("isin") or "").strip(),
                    "ex_date": ex,
                    "subject": str(r.get("subject") or ""),
                    "kind": kind,
                    "ratio": v if kind != "dividend" else np.nan,
                    "dividend": v if kind == "dividend" else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=list(CA_COLUMNS)) if rows else _empty(CA_COLUMNS)


RESULT_COLUMNS = ("symbol", "isin", "broadcast", "period_from", "period_to", "consolidated")


def parse_results(payload: Any) -> pd.DataFrame:
    """Financial-results filings → broadcast time per filing (``RESULT_COLUMNS``)."""
    rows = []
    for r in as_rows(payload):
        ts = (
            parse_datetime(r.get("broadCastDate"))
            or parse_datetime(r.get("exchdisstime"))
            or parse_datetime(r.get("filingDate"))
        )
        sym = str(r.get("symbol") or "").strip().upper()
        if ts is None or not sym:
            continue
        rows.append(
            {
                "symbol": sym,
                "isin": str(r.get("isin") or "").strip(),
                "broadcast": ts,
                "period_from": parse_date(r.get("fromDate")),
                "period_to": parse_date(r.get("toDate")),
                "consolidated": str(r.get("consolidated") or ""),
            }
        )
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS)) if rows else _empty(RESULT_COLUMNS)


MEETING_COLUMNS = ("symbol", "isin", "meeting_date", "intimation", "is_results", "industry")
RESULTS_WORDS = re.compile(r"financial result|quarterly result|audited result|results", re.I)


def parse_board_meetings(payload: Any) -> pd.DataFrame:
    """Board-meeting intimations → ``MEETING_COLUMNS``."""
    rows = []
    for r in as_rows(payload):
        md = parse_date(r.get("bm_date"))
        sym = str(r.get("bm_symbol") or r.get("symbol") or "").strip().upper()
        if md is None or not sym:
            continue
        text = f"{r.get('bm_purpose') or ''} {r.get('bm_desc') or ''}"
        rows.append(
            {
                "symbol": sym,
                "isin": str(r.get("sm_isin") or "").strip(),
                "meeting_date": md,
                "intimation": parse_datetime(r.get("bm_timestamp")),
                "is_results": bool(RESULTS_WORDS.search(text)),
                "industry": str(r.get("sm_indusrty") or r.get("sm_industry") or "").strip(),
            }
        )
    return pd.DataFrame(rows, columns=list(MEETING_COLUMNS)) if rows else _empty(MEETING_COLUMNS)


SECTOR_KEYS = {
    "macro": ("macro",),
    "sector": ("sector",),
    "industry": ("industryInfo", "industry"),
    "basic_industry": ("basicIndustry",),
}


def _find(obj: Any, keys: tuple[str, ...]) -> str | None:
    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip() and v.strip() != "-":
                return v.strip()
        for v in obj.values():
            hit = _find(v, keys)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find(v, keys)
            if hit:
                return hit
    return None


def parse_quote_sector(payload: Any) -> dict[str, str | None]:
    """Quote payload (NextApi or quote-equity) → macro/sector/industry/basic_industry."""
    return {k: _find(payload, keys) for k, keys in SECTOR_KEYS.items()}


def parse_holidays(payload: Any, segment: str = "CM") -> pd.DataFrame:
    """Holiday master → ``date``, ``description`` for the cash market."""
    rows = payload.get(segment, []) if isinstance(payload, dict) else []
    out = [
        {"date": parse_date(r.get("tradingDate")), "description": str(r.get("description", ""))}
        for r in rows
        if isinstance(r, dict)
    ]
    df = pd.DataFrame([o for o in out if o["date"] is not None], columns=["date", "description"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
