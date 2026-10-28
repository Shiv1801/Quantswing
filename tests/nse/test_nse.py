"""D31 NSE data: client etiquette, parsers (real formats), loader → build → DB end-to-end."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest

from quantswing.core.config import get_config
from quantswing.nse import formats as nf
from quantswing.nse.build import build_all, resolve, to_session
from quantswing.nse.client import NseClient, NseError
from quantswing.nse.loader import NseLoader, base_symbol, month_chunks
from quantswing.store.repositories import DataPaths, connect
from tests.conftest import FakeFD, make_candles, symbol_master_csv, weekdays

CFG = get_config()
HTTP = CFG.nse.http.model_copy(update={"min_interval_s": 0.0, "backoff_s": 0.0})


# --------------------------------------------------------------------------- fakes
class Resp:
    def __init__(self, code: int, content: bytes = b"", payload: Any = None) -> None:
        self.status_code = code
        self.content = content if payload is None else json.dumps(payload).encode()
        self.headers: dict[str, str] = {}
        self._payload = payload

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class Scripted:
    """Session returning scripted responses per URL substring (default 404)."""

    def __init__(self, script: dict[str, list[Resp]]) -> None:
        self.script = script
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url: str, **kw: Any) -> Resp:
        self.calls.append(url)
        for k, v in self.script.items():
            if k in url and v:
                return v.pop(0) if len(v) > 1 else v[0]
        return Resp(404)


def zipped(name: str, text: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, text)
    return b.getvalue()


# --------------------------------------------------------------------------- client
def test_client_statuses_and_rewarm() -> None:
    s = Scripted(
        {
            "www.nseindia.com/": [Resp(200, b"<html>home</html>")],
            "a.zip": [Resp(200, b"PK" + b"x" * 300)],
            "html.csv": [Resp(200, b"<!DOCTYPE html><html>not found</html>" + b" " * 300)],
            "blocked": [Resp(403)],
            "flaky": [Resp(403), Resp(200, b"data" * 100)],
        }
    )
    c = NseClient(HTTP, s, sleep=lambda _x: None)
    assert c.download(["https://h/a.zip"], b"PK").status == "ok"
    assert c.download(["https://h/missing.csv"]).status == "miss"
    assert c.download(["https://h/html.csv"]).status == "miss"
    assert c.download(["https://h/a.zip"], b"ZZ").status == "miss"  # wrong file type
    assert c.download(["https://h/blocked"]).status == "fail"  # never a "miss"
    warm_before = sum("www.nseindia.com/" in u for u in s.calls)
    assert c.download(["https://h/flaky"]).status == "ok"
    assert sum("www.nseindia.com/" in u for u in s.calls) > warm_before  # 403 → re-warm
    # second host used when the first misses
    assert c.download(["https://h/missing", "https://h/a.zip"], b"PK").status == "ok"


def test_client_api_and_throttle() -> None:
    s = Scripted(
        {
            "www.nseindia.com/api/ok": [Resp(200, payload=[{"a": 1}])],
            "www.nseindia.com/api/html": [Resp(200, b"<html>blocked</html>")],
        }
    )
    waits: list[float] = []
    t = [0.0]
    c = NseClient(
        HTTP.model_copy(update={"min_interval_s": 1.0}),
        s,
        sleep=lambda x: (waits.append(x), t.__setitem__(0, t[0] + x)),
        clock=lambda: t[0],
    )
    assert c.api("ok") == [{"a": 1}]
    with pytest.raises(NseError):
        c.api("html")
    assert waits and all(w <= 1.0 for w in waits)  # ≥ 1 s between requests (no burst)


# --------------------------------------------------------------------------- formats
OLD_HEAD = "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,"
UDIFF_HEAD = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,"
    "StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,"
    "SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,"
    "Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)
SURV_HEAD = (
    "ScripCode,Symbol,NSE Exclusive,Status,Series,GSM,"
    "Long_Term_Additional_Surveillance_Measure (Long Term ASM),Unsolicited_SMS,"
    "Insolvency_Resolution_Process(IRP),Short_Term_Additional_Surveillance_Measure (Short Term ASM),"
    "Default,ICA,Filler4,Filler5,Pledge,Add-on_PB,Total Pledge,Social Media Platforms,ESM,"
    "Loss making,Overall encumbered share in the scrip is more than 50 percent,Under BZ/SZ Series"
)


def old_line(sym: str, d: date, c: float, isin: str, series: str = "EQ") -> str:
    ts = f"{d:%d}-{nf.MON[d.month - 1]}-{d.year}"
    return f"{sym},{series},{c:.2f},{c * 1.01:.2f},{c * 0.99:.2f},{c:.2f},{c:.2f},{c:.2f},1000,{c * 1000:.2f},{ts},10,{isin},"


def udiff_line(sym: str, d: date, c: float, isin: str, series: str = "EQ") -> str:
    f = [""] * 34
    f[0], f[1], f[2], f[3], f[4] = d.isoformat(), d.isoformat(), "CM", "NSE", "STK"
    f[6], f[7], f[8] = isin, sym, series
    f[14], f[15], f[16], f[17], f[18], f[19] = (
        f"{c:.2f}",
        f"{c * 1.01:.2f}",
        f"{c * 0.99:.2f}",
        f"{c:.2f}",
        f"{c:.2f}",
        f"{c:.2f}",
    )
    f[24], f[25], f[26] = "1000", f"{c * 1000:.2f}", "10"
    return ",".join(f)


def mto_text(d: date, rows: list[tuple[str, str, int, int]]) -> str:
    lines = [
        "Security Wise Delivery Position - Compulsory Rolling Settlement",
        f"10,MTO,{d:%d%m%Y},1234567,0001234",
        f"Trade Date <{d:%d}-{nf.MON[d.month - 1]}-{d.year}>,Settlement Type <N>,Settlement No <1>,Settlement Date <x>",
        "Record Type,Sr No,Name of Security,Quantity Traded,Deliverable Quantity(gross across client level),% of Deliverable Quantity to Traded Quantity",
    ]
    for n, (sym, ser, q, dq) in enumerate(rows, 1):
        lines.append(f"20,{n},{sym},{ser},{q},{dq},{100 * dq / q:.2f}")
    return "\n".join(lines) + "\n"


def test_parse_bhav_both_formats_and_mto() -> None:
    d = date(2023, 6, 1)
    old = zipped(
        "x.csv",
        OLD_HEAD
        + "\n"
        + old_line("ALPHA", d, 100, "INE1")
        + "\n"
        + old_line("SMEX", d, 5, "INE9", "SM"),
    )
    df = nf.parse_bhav(old)
    assert (
        list(df["symbol"]) == ["ALPHA"] and df["date"].iloc[0] == d and df["close"].iloc[0] == 100
    )
    new = zipped("y.csv", UDIFF_HEAD + "\n" + udiff_line("BETA", date(2024, 8, 1), 50, "INE2"))
    df2 = nf.parse_bhav(new)
    assert df2.iloc[0][["symbol", "isin", "close", "volume"]].tolist() == [
        "BETA",
        "INE2",
        50.0,
        1000.0,
    ]
    m = nf.parse_mto(mto_text(d, [("ALPHA", "EQ", 1000, 400), ("SMEX", "SM", 10, 1)]).encode())
    assert m[["symbol", "deliv_qty", "deliv_pct"]].values.tolist() == [["ALPHA", 400.0, 40.0]]
    assert m["date"].iloc[0] == d


def test_parse_surveillance_header_and_flags() -> None:
    row = [
        "1",
        "BETA",
        "N",
        "A",
        "EQ",
        "100",
        "2",
        "100",
        "100",
        "100",
        "100",
        "100",
        "",
        "",
        "1",
        "100",
        "100",
        "100",
        "100",
        "0",
        "100",
        "100",
    ]
    ok = ["2", "ALPHA", "N", "A", "EQ"] + ["100"] * 17
    text = SURV_HEAD + "\n" + ",".join(row) + "\n" + ",".join(ok) + "\n"
    df = nf.parse_surveillance(text.encode(), date(2024, 3, 1)).set_index("symbol")
    assert (
        bool(df.loc["BETA", "asm_long"])
        and bool(df.loc["BETA", "pledge"])
        and bool(df.loc["BETA", "loss_making"])
    )
    assert not df.loc["BETA", "gsm"] and not df.loc["ALPHA", ["asm_long", "pledge", "bz"]].any()


def test_parse_ca_results_meetings_symbols() -> None:
    assert nf.parse_ca_subject(
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    ) == [("split", 0.2)]
    assert nf.parse_ca_subject("Bonus 1:1") == [("bonus", 0.5)]
    assert nf.parse_ca_subject("Final Dividend - Rs 2.50 Per Share") == [("dividend", 2.5)]
    res = nf.parse_results(
        [{"symbol": "X", "broadCastDate": "16-Jan-2025 20:20:21", "toDate": "31-Dec-2024"}]
    )
    assert res["broadcast"].iloc[0] == datetime(2025, 1, 16, 20, 20, 21)
    mt = nf.parse_board_meetings(
        {
            "data": [
                {
                    "bm_symbol": "X",
                    "bm_date": "30-Oct-2023",
                    "bm_purpose": "to consider Quarterly Financial results",
                    "bm_timestamp": "18-Oct-2023 22:42:01",
                }
            ]
        }
    )
    assert bool(mt["is_results"].iloc[0]) and mt["meeting_date"].iloc[0] == date(2023, 10, 30)
    sc = nf.parse_symbol_change(
        b"SM_NAME,SM_KEY_SYMBOL,SM_NEW_SYMBOL,SM_APPLICABLE_FROM\nNew Co Ltd,OLDCO,NEWCO,01-Jun-2023\n"
    )
    assert sc.values.tolist() == [["OLDCO", "NEWCO", date(2023, 6, 1)]]
    assert resolve("OLDCO", date(2023, 5, 31), {"OLDCO": [(date(2023, 6, 1), "NEWCO")]}) == "NEWCO"
    assert resolve("OLDCO", date(2023, 7, 1), {"OLDCO": [(date(2023, 6, 1), "NEWCO")]}) == "OLDCO"
    assert base_symbol("NSE:BAJAJ-AUTO-EQ") == ("BAJAJ-AUTO", "EQ")


def test_to_session_cutoff() -> None:
    s = pd.DatetimeIndex(pd.to_datetime(["2023-08-10", "2023-08-11", "2023-08-14"]))
    cut = CFG.nse.results_cutoff
    assert to_session(datetime(2023, 8, 10, 11, 0), s, cut) == pd.Timestamp("2023-08-10")
    assert to_session(datetime(2023, 8, 10, 17, 5), s, cut) == pd.Timestamp("2023-08-11")
    assert to_session(datetime(2023, 8, 11, 19, 0), s, cut) == pd.Timestamp("2023-08-14")


# --------------------------------------------------------------------------- fake NSE market
START, END = date(2023, 1, 2), date(2024, 12, 31)
CHANGE = date(2023, 6, 1)  # OLDCO → NEWCO
BONUS = date(2023, 6, 1)  # DEADCO 1:1 bonus
DEAD_LAST = date(2023, 12, 29)
SURV_FROM = date(2024, 1, 1)


class FakeNse:
    """Serves bhavcopy / MTO / surveillance / symbol change files and the JSON API."""

    def __init__(self, sessions: list[date]) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []
        self.sessions = sessions
        self.px = {
            s: make_candles(sessions, 100 + 20 * k, seed=900 + k).set_index(pd.Index(sessions))[
                "close"
            ]
            for k, s in enumerate(["ALPHA", "BETA", "NEWCO", "DEADCO"])
        }

    def rows(self, d: date) -> list[tuple[str, float, str]]:
        out = [("ALPHA", self.px["ALPHA"][d], "INEA"), ("BETA", self.px["BETA"][d], "INEB")]
        out.append(("OLDCO" if d < CHANGE else "NEWCO", self.px["NEWCO"][d], "INEN"))
        if d <= DEAD_LAST:
            c = self.px["DEADCO"][d] * (2.0 if d < BONUS else 1.0)  # raw, unadjusted
            out.append(("DEADCO", c, "INED"))
        return out

    def get(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> Resp:
        self.calls.append(url)
        path = urlparse(url).path
        name = path.rsplit("/", 1)[-1]
        if url.endswith("nseindia.com/"):
            return Resp(200, b"<html>home</html>")
        if name.startswith("cm") and name.endswith("bhav.csv.zip"):
            d = datetime.strptime(name[2:11], "%d%b%Y").date()
            if d not in self.sessions:
                return Resp(404)
            body = "\n".join(old_line(s, d, c, i) for s, c, i in self.rows(d))
            return Resp(200, zipped(name[:-4], OLD_HEAD + "\n" + body + "\n"))
        if name.startswith("BhavCopy_NSE_CM"):
            d = datetime.strptime(name.split("_")[6], "%Y%m%d").date()
            if d not in self.sessions:
                return Resp(404)
            body = "\n".join(udiff_line(s, d, c, i) for s, c, i in self.rows(d))
            return Resp(200, zipped(name[:-4], UDIFF_HEAD + "\n" + body + "\n"))
        if name.startswith("MTO_"):
            d = datetime.strptime(name[4:12], "%d%m%Y").date()
            if d not in self.sessions:
                return Resp(404)
            rows = [(s, "EQ", 1000, 300 + 10 * k) for k, (s, _, _) in enumerate(self.rows(d))]
            return Resp(200, mto_text(d, rows).encode())
        if name.startswith("REG1_IND"):
            d = datetime.strptime(name[8:14], "%d%m%y").date()
            if d < SURV_FROM or d not in self.sessions:
                return Resp(404)
            lines = [SURV_HEAD]
            for s, _, _ in self.rows(d):
                f = ["1", s, "N", "A", "EQ"] + ["100"] * 17
                if s == "BETA" and d.month == 3:
                    f[6] = "1"  # long-term ASM stage 1
                lines.append(",".join(f))
            return Resp(200, ("\n".join(lines) + "\n").encode())
        if name == "symbol_change.csv":
            return Resp(
                200,
                b"SM_NAME,SM_KEY_SYMBOL,SM_NEW_SYMBOL,SM_APPLICABLE_FROM\nNew Co Ltd,OLDCO,NEWCO,01-Jun-2023\n"
                + b" " * 300,
            )
        if "/api/" in path:
            return Resp(200, payload=self.api(path.split("/api/", 1)[1], params or {}))
        return Resp(404)

    def api(self, ep: str, p: dict[str, Any]) -> Any:
        rng = None
        if "from_date" in p:
            rng = (
                datetime.strptime(p["from_date"], "%d-%m-%Y").date(),
                datetime.strptime(p["to_date"], "%d-%m-%Y").date(),
            )
        inr = lambda d: rng is None or rng[0] <= d <= rng[1]  # noqa: E731
        if ep == "corporates-corporateActions":
            acts = [
                {
                    "symbol": "DEADCO",
                    "series": "EQ",
                    "subject": "Bonus 1:1",
                    "exDate": "01-Jun-2023",
                    "isin": "INED",
                },
                {
                    "symbol": "ALPHA",
                    "series": "EQ",
                    "subject": "Dividend - Rs 5 Per Share",
                    "exDate": "15-Sep-2023",
                    "isin": "INEA",
                },
            ]
            return [a for a in acts if inr(datetime.strptime(a["exDate"], "%d-%b-%Y").date())]
        if ep == "corporates-financial-results":
            rs = [
                {
                    "symbol": "ALPHA",
                    "broadCastDate": "10-Aug-2023 17:05:00",
                    "toDate": "30-Jun-2023",
                    "isin": "INEA",
                },
                {
                    "symbol": "ALPHA",
                    "broadCastDate": "10-Aug-2023 17:45:00",
                    "toDate": "30-Jun-2023",
                    "isin": "INEA",
                },
                {
                    "symbol": "BETA",
                    "broadCastDate": "10-Aug-2023 11:00:00",
                    "toDate": "30-Jun-2023",
                    "isin": "INEB",
                },
                {
                    "symbol": "OLDCO",
                    "broadCastDate": "12-May-2023 12:00:00",
                    "toDate": "31-Mar-2023",
                    "isin": "INEN",
                },
            ]
            return [
                r for r in rs if inr(datetime.strptime(r["broadCastDate"][:11], "%d-%b-%Y").date())
            ]
        if ep == "corporate-board-meetings":
            ms = [
                {
                    "bm_symbol": "ALPHA",
                    "bm_date": "10-Aug-2023",
                    "bm_purpose": "Financial Results",
                    "bm_timestamp": "01-Aug-2023 18:00:00",
                    "sm_indusrty": "x",
                }
            ]
            return [m for m in ms if inr(datetime.strptime(m["bm_date"], "%d-%b-%Y").date())]
        if ep == "holiday-master":
            return {"CM": [{"tradingDate": "26-Jan-2025", "description": "Republic Day"}]}
        if ep.startswith("NextApi"):
            sec = {
                "ALPHA": "Financial Services",
                "BETA": "Financial Services",
                "NEWCO": "Information Technology",
            }
            s = p.get("symbol", "")
            return {
                "equityResponse": [
                    {
                        "secInfo": {
                            "macro": "M",
                            "sector": sec.get(s, ""),
                            "industryInfo": "I",
                            "basicIndustry": "B",
                        }
                    }
                ]
            }
        return []


def fyers_root(root: Path) -> tuple[DataPaths, list[date]]:
    from quantswing.ingest import pipeline
    from quantswing.store.database import build_database
    from tests.store.test_pipeline import FakeHTTP

    paths = DataPaths.relocated(root)
    dates = weekdays(START, END)
    rows = [
        {"name": n, "ticker": f"NSE:{n}-EQ", "isin": f"INE{k}00A01011"}
        for k, n in enumerate(["ALPHA", "BETA", "NEWCO"])
    ]
    pipeline.refresh_universe(paths, CFG, http=FakeHTTP(symbol_master_csv(rows)))
    series = {s: make_candles(dates, seed=k) for k, s in enumerate(CFG.settings.data.indices)}
    series |= {
        f"NSE:{n}-EQ": make_candles(dates, seed=50 + k)
        for k, n in enumerate(["ALPHA", "BETA", "NEWCO"])
    }
    fd = FakeFD(series)
    pipeline.load_indices(fd, paths, START, dates[-1], CFG)  # type: ignore[arg-type]
    pipeline.build_calendar(paths, CFG)
    syms = [r["ticker"] for r in rows]
    pipeline.load_equities(fd, paths, syms, START, dates[-1], None, CFG)  # type: ignore[arg-type]
    pipeline.validate_all(paths, syms, dates[-1], CFG)
    build_database(paths)
    return paths, dates


@pytest.fixture(scope="module")
def nse_root(tmp_path_factory: pytest.TempPathFactory) -> tuple[DataPaths, FakeNse, NseLoader]:
    from quantswing.store.database import build_database

    paths, dates = fyers_root(tmp_path_factory.mktemp("nse"))
    cfg = CFG.model_copy(
        update={
            "nse": CFG.nse.model_copy(
                update={
                    "start": START,
                    "http": HTTP,
                    "files": CFG.nse.files.model_copy(update={"surveillance_probe_from": START}),
                    "survivorship": CFG.nse.survivorship.model_copy(
                        update={"delist_confirm_sessions": 5}
                    ),
                }
            )
        }
    )
    fake = FakeNse(dates)
    client = NseClient(cfg.nse.http, fake, sleep=lambda _x: None)
    loader = NseLoader(paths, cfg, client, today=END)
    uni = pd.read_parquet(paths.universe)["symbol"].tolist()
    summ = loader.run(tickers=uni)
    assert summ.aborted is None
    build_all(paths, cfg)
    build_database(paths, cfg)
    return paths, fake, loader


def test_loader_fetches_everything_once(nse_root: tuple[DataPaths, FakeNse, NseLoader]) -> None:
    paths, fake, loader = nse_root
    n = len(loader.sessions())
    assert len(list((paths.nse / "bhav").glob("*/*.parquet"))) == n
    assert len(list((paths.nse / "mto").glob("*/*.parquet"))) == n
    surv = sorted((paths.nse / "surv").glob("*/*.parquet"))
    assert surv and surv[0].stem >= "20240101"
    assert loader.state.get("surv_first") == "2024-01-01"
    before = len(fake.calls)
    loader2 = NseLoader(paths, loader.cfg, loader.client, today=END)
    loader2.run_files()
    assert len(fake.calls) == before  # all cached → no requests


def test_db_tables(nse_root: tuple[DataPaths, FakeNse, NseLoader]) -> None:
    paths, fake, _ = nse_root
    con = connect(paths.duckdb, read_only=True)
    try:
        q = lambda s: con.execute(s).fetchdf()  # noqa: E731
        # OLDCO delivery rows land on today's NEWCO ticker (symbol change)
        dl = q("SELECT min(date) AS d FROM nse_delivery WHERE symbol = 'NSE:NEWCO-EQ'")
        assert pd.Timestamp(dl["d"].iloc[0]).date() == START
        pd_ = q(
            "SELECT count(*) AS n, count(delivery_pct) AS k FROM prices_daily WHERE NOT delisted"
        )
        assert pd_["k"].iloc[0] == pd_["n"].iloc[0]
        # delisted stock rebuilt, bonus-adjusted (one adjustment: NSE feed wins over the detector)
        d = q("SELECT * FROM delisted")
        assert d["symbol"].tolist() == ["NSE:DEADCO-EQ"]
        assert pd.Timestamp(d["last_date"].iloc[0]).date() == DEAD_LAST
        px = q(
            "SELECT date, adj_close, close FROM prices_daily WHERE symbol = 'NSE:DEADCO-EQ' ORDER BY date"
        )
        r = np.log(px["adj_close"]).diff().abs().max()
        assert r < 0.1  # no 1:1 bonus gap left
        assert np.allclose(
            px["adj_close"].to_numpy(), fake.px["DEADCO"].loc[:DEAD_LAST].to_numpy(), rtol=1e-3
        )
        adj = q("SELECT * FROM adj_events WHERE symbol = 'NSE:DEADCO-EQ'")
        assert len(adj) == 1 and abs(adj["ratio"].iloc[0] - 0.5) < 1e-9
        # results: after 15:30 → next session; first broadcast per quarter; old symbol mapped
        ev = q("SELECT symbol, event_date FROM results_events ORDER BY symbol, event_date")
        got = {(s, pd.Timestamp(e).date()) for s, e in ev.values}
        assert ("NSE:ALPHA-EQ", date(2023, 8, 11)) in got and (
            "NSE:BETA-EQ",
            date(2023, 8, 10),
        ) in got
        assert ("NSE:NEWCO-EQ", date(2023, 5, 12)) in got and len(got) == 3
        mt = q("SELECT * FROM results_meetings")
        assert pd.Timestamp(mt["known_session"].iloc[0]).date() == date(2023, 8, 2)  # 18:00 → next
        dv = q("SELECT * FROM dividends")
        prev = fake.px["ALPHA"][date(2023, 9, 15)]  # fake files put the close in PREVCLOSE
        assert len(dv) == 1 and abs(dv["yield"].iloc[0] - 5 / round(prev, 2)) < 1e-6
        sv = q(
            "SELECT symbol, min(date) AS a, max(date) AS b FROM surveillance WHERE restricted GROUP BY 1"
        )
        assert sv["symbol"].tolist() == ["NSE:BETA-EQ"]
        assert pd.Timestamp(sv["a"].iloc[0]).month == 3
        sec = q("SELECT symbol, sector FROM sectors ORDER BY symbol")
        assert sec["sector"].tolist() == [
            "Financial Services",
            "Financial Services",
            "Information Technology",
        ]
        assert len(q("SELECT * FROM holidays")) == 1
    finally:
        con.close()


def test_panels_gate_and_factors(nse_root: tuple[DataPaths, FakeNse, NseLoader]) -> None:
    from quantswing.factors import nse_factors
    from quantswing.factors.gate import universe_gate
    from quantswing.store.panels import load_panels

    paths, _, _ = nse_root
    p = load_panels(paths, CFG)
    assert "NSE:DEADCO-EQ" in p.close.columns and p.delisted_last is not None
    assert p.deliv is not None and p.deliv["NSE:ALPHA-EQ"].notna().all()
    assert p.sector is not None and p.sector["NSE:NEWCO-EQ"] == "Information Technology"
    assert p.restricted is not None and bool(p.restricted.loc["2024-03-15", "NSE:BETA-EQ"])
    assert p.pledge is not None and p.pledge.loc["2023-06-01"].isna().all()
    assert (p.pledge.loc["2024-06-03"].dropna() == 0).all()
    assert p.div_yield is not None and p.div_yield.loc["2023-09-15", "NSE:ALPHA-EQ"] > 0
    g = universe_gate(
        p, CFG.factors.gate.model_copy(update={"min_history": 20, "continuity_window": 20})
    )
    assert not g.loc["2024-03-15", "NSE:BETA-EQ"] and g.loc["2024-04-15", "NSE:BETA-EQ"]
    f = nse_factors.compute(p, CFG.factors)
    assert set(f) >= {"DELZ", "PLEDGE", "EARN"}


# --------------------------------------------------------------------------- factor maths
def test_sector_return_leave_one_out_and_labels() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.DataFrame([[0.01, 0.03, 0.05, 0.02]] * 3, index=idx, columns=list("abcd"))
    rb = pd.Series(0.0, index=idx)
    labels = np.array(["X", "X", "X", "Y"])
    rs = nse_factors_mod().sector_return(r, labels, rb, 2)
    assert np.isclose(rs["a"].iloc[0], 0.04) and np.isclose(rs["c"].iloc[0], 0.02)
    assert rs["d"].iloc[0] == 0.0  # singleton sector → benchmark
    lab = nse_factors_mod().sector_labels(
        pd.Series({"a": "IT", "b": "IT", "c": None}), pd.Index(list("abc")), 2
    )
    assert lab.tolist() == ["IT", "IT", "Unknown"]


def nse_factors_mod():
    from quantswing.factors import nse_factors

    return nse_factors


def test_sector_neutralisation_removes_sector_effect() -> None:
    from quantswing.factors import xsec

    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = [f"s{i}" for i in range(40)]
    labels = np.array(["A"] * 20 + ["B"] * 20)
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (5, 40))
    raw = pd.DataFrame(noise + np.where(labels == "A", 5.0, 0.0), index=idx, columns=cols)
    gate = pd.DataFrame(True, index=idx, columns=cols)
    _, nz = xsec.process(raw, gate, None, CFG.factors.xsec, labels)
    means = nz.T.groupby(labels).mean().T
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-9)
    _, nz0 = xsec.process(raw, gate, None, CFG.factors.xsec, None)
    assert nz0.T.groupby(labels).mean().T["A"].min() > 0.5  # without sectors the effect stays


def test_delz_and_earn_no_lookahead() -> None:
    from quantswing.factors import nse_factors
    from tests.factors.test_factors import synth_panels

    p = synth_panels(400, 6)
    deliv = pd.DataFrame(
        np.random.default_rng(3).uniform(20, 60, p.close.shape),
        index=p.close.index,
        columns=p.close.columns,
    )
    z1 = nse_factors.delivery_surprise(deliv, 60)
    d2 = deliv.copy()
    d2.iloc[300:] *= 3
    z2 = nse_factors.delivery_surprise(d2, 60)
    assert np.allclose(z1.iloc[:300].fillna(0), z2.iloc[:300].fillna(0))
    e = p.close.index[300]
    ev = pd.DataFrame({"symbol": [p.close.columns[0]], "event_date": [e]})
    earn = nse_factors.earnings_reaction(p, ev, 252, 10)
    col = earn[p.close.columns[0]]
    assert col.loc[:e].isna().all()  # known only at the close of e+1
    assert col.iloc[301:312].notna().all() and col.iloc[312:].isna().all()


def test_event_block_windows() -> None:
    from quantswing.alpha.events import event_block
    from tests.factors.test_factors import synth_panels

    p = synth_panels(60, 3)
    idx, c0 = p.close.index, p.close.columns[0]
    ev = pd.DataFrame({"symbol": [c0], "event_date": [idx[30]]})
    mt = pd.DataFrame({"symbol": [c0], "meeting_session": [idx[30]], "known_session": [idx[20]]})
    from dataclasses import replace

    q = replace(p, events=ev, meetings=mt)
    b = event_block(q, CFG.risk.event_window)
    assert b is not None
    blocked = np.flatnonzero(b[c0].to_numpy())
    # before: entry sessions 27..30 → decisions 26..29; after: decisions 30..32
    assert blocked.tolist() == list(range(26, 33))
    assert not b[p.close.columns[1]].any()
    late = replace(p, events=None, meetings=mt.assign(known_session=[idx[28]]))
    b2 = event_block(late, CFG.risk.event_window)
    assert b2 is not None and np.flatnonzero(b2[c0].to_numpy()).tolist() == [28, 29]


def test_s5_signal_on_planted_event() -> None:
    from dataclasses import replace

    from quantswing.sleeves import s5_pead
    from quantswing.sleeves.base import FactorView
    from tests.factors.test_factors import synth_panels

    p = synth_panels(300, 4)
    c0 = p.close.columns[0]
    close = p.close.copy()
    close.iloc[200:, 0] *= 1.10  # +10% results jump that holds
    q = replace(
        p,
        close=close,
        events=pd.DataFrame(
            {
                "symbol": [c0, p.close.columns[1]],
                "event_date": [p.close.index[200], p.close.index[150]],
            }
        ),
    )
    one = pd.DataFrame(1.0, index=p.close.index, columns=p.close.columns)
    fv = FactorView(q, preset={"BETA": one, "IVOL": one * 0.02, "VOLZ": one * 5})
    gate = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    sig = s5_pead.build(q, fv, gate, CFG.sleeves.rules.S5).signal
    hits = sig.stack()
    hits = hits[hits]
    assert list(hits.index) == [(p.close.index[201], c0)]


def test_portfolio_delist_exit_dividends_and_block() -> None:
    from dataclasses import replace

    from quantswing.portfolio.construction import run_portfolio
    from tests.factors.test_factors import synth_panels

    p = synth_panels(200, 3)
    cols, idx = p.close.columns, p.close.index
    k, s0 = 2, idx[100]
    close, opn = p.close.copy(), p.open.copy()
    close.iloc[151:, k] = np.nan  # stock k delisted after session 150
    opn.iloc[151:, k] = np.nan
    q = replace(p, close=close, open=opn, delisted_last=pd.Series([idx[150]], index=[cols[k]]))
    score = pd.DataFrame(0.0, index=idx, columns=cols)
    score[cols[k]] = 2.0
    gate = pd.DataFrame(True, index=idx, columns=cols)
    pc = CFG.risk.portfolio.model_copy(update={"top_n": 1, "rebalance_every": 1000})
    cfg = CFG.model_copy(update={"risk": CFG.risk.model_copy(update={"portfolio": pc})})
    t = run_portfolio(q, score, gate, cfg, start=s0).trades
    dl = t[t["side"] == "delist"]
    assert len(dl) == 1 and pd.Timestamp(dl["date"].iloc[0]) == idx[151]
    haircut = cfg.backtest.execution.delist_exit_haircut
    assert np.isclose(dl["px"].iloc[0], close.iloc[150, k] * (1 - haircut))
    # dividends credited on the ex-date
    dy = pd.DataFrame(0.0, index=idx, columns=cols)
    dy.iloc[130, k] = 0.02
    q0 = replace(q, delisted_last=None)
    base = run_portfolio(q0, score, gate, cfg, start=s0)
    divd = run_portfolio(replace(q0, div_yield=dy), score, gate, cfg, start=s0)
    jump = divd.daily["equity"].iloc[40] - base.daily["equity"].iloc[40]
    assert jump > 0.015 * base.daily["invested"].iloc[29] > 0
    # entry block keeps a new name out
    blk = pd.DataFrame(False, index=idx, columns=cols)
    blk.iloc[100, k] = True
    b = run_portfolio(q0, score, gate, cfg, start=s0, entry_block=blk)
    assert b.trades.empty


def test_month_chunks() -> None:
    ch = month_chunks(date(2024, 1, 15), date(2024, 3, 3))
    assert ch == [
        (date(2024, 1, 15), date(2024, 1, 31)),
        (date(2024, 2, 1), date(2024, 2, 29)),
        (date(2024, 3, 1), date(2024, 3, 3)),
    ]
    assert (date(2024, 1, 1) + timedelta(days=1)).year == 2024


def test_client_falls_back_when_curl_ssl_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import quantswing.nse.client as nc

    class Broken:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def get(self, url: str, **kw: Any) -> Resp:
            raise RuntimeError(
                "curl: (60) SSL certificate problem: unable to get local issuer certificate"
            )

    good = Scripted({"a.zip": [Resp(200, b"PK" + b"x" * 300)]})
    monkeypatch.setattr(nc, "requests_session", lambda cfg: good)
    c = NseClient(HTTP, Broken(), sleep=lambda _x: None)
    c.backend, c._can_fallback = "curl_cffi", True
    c._answered = 5  # even after earlier successes, an SSL error switches backend
    assert c.download(["https://h/a.zip"], b"PK").status == "ok"
    assert c.backend == "requests (fallback)" and "SSL" in c.last_error


def test_meeting_events_fill_missing_quarters() -> None:
    from quantswing.nse.build import add_meeting_events

    s = pd.DatetimeIndex(
        pd.to_datetime(["2025-05-12", "2025-05-13", "2025-08-11", "2025-08-12", "2025-08-13"])
    )
    ev = pd.DataFrame(
        {
            "symbol": ["NSE:A-EQ"],
            "broadcast": [pd.Timestamp("2025-05-12 16:10")],
            "event_date": [s[1]],
            "period_to": [date(2025, 3, 31)],
        }
    )
    mt = pd.DataFrame(
        {
            "ticker": ["NSE:A-EQ", "NSE:A-EQ", "NSE:B-EQ", "NSE:B-EQ"],
            "meeting_date": [
                date(2025, 5, 12),
                date(2025, 8, 11),
                date(2025, 8, 12),
                date(2025, 11, 10),
            ],
            "is_results": [True, True, True, True],
        }
    )
    out = add_meeting_events(ev, mt, s)
    got = {(r.symbol, r.event_date, r.source) for r in out.itertuples()}
    assert got == {
        ("NSE:A-EQ", s[1], "filing"),  # feed had it: meeting not duplicated
        ("NSE:A-EQ", s[3], "meeting"),  # Aug quarter missing from the feed → next session
        ("NSE:B-EQ", s[4], "meeting"),
    }  # Nov meeting is in the future → not an event yet


def test_merger_delisting_exits_at_full_value() -> None:
    from quantswing.nse.build import delisting_reasons

    assert nf.parse_ca_subject("Scheme Of Amalgamation")[0][0] == "reorg"
    smap = pd.DataFrame(
        {
            "nse_symbol": ["HDFC", "BUST"],
            "d_from": [date(2015, 1, 1)] * 2,
            "d_to": [date(2023, 7, 12), date(2019, 5, 2)],
            "ticker": ["NSE:HDFC-EQ", "NSE:BUST-EQ"],
            "delisted": [True, True],
        }
    )
    ca = pd.DataFrame({"symbol": ["HDFC"], "ex_date": [date(2023, 7, 13)], "kind": ["reorg"]})
    r = delisting_reasons(smap, ca, 0.3).set_index("symbol")
    assert (
        r.loc["NSE:HDFC-EQ", "haircut"] == 0.0 and r.loc["NSE:HDFC-EQ", "reason"] == "merger/scheme"
    )
    assert r.loc["NSE:BUST-EQ", "haircut"] == 0.3
