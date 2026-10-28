"""D37 results: numbers vs Nifty 500, a self-contained HTML report and CSV logs."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantswing.backtest import metrics
from quantswing.portfolio.swing import SwingResult


@dataclass
class SwingSummary:
    """Everything the console and the report show."""

    start: pd.Timestamp
    end: pd.Timestamp
    capital: float
    sys: dict[str, float]
    nifty: dict[str, float]
    stress: dict[str, float]
    end_value: float
    nifty_end: float
    curve: pd.DataFrame  # date, system, nifty (₹), dd_system, dd_nifty
    yearly: pd.DataFrame
    monthly: pd.DataFrame  # year × month (system)
    exits: pd.DataFrame
    stats: dict[str, float]
    trades: pd.DataFrame


def _dd(s: pd.Series) -> pd.Series:
    out: pd.Series = s / s.cummax() - 1
    return out


def _year_ret(s: pd.Series) -> pd.Series:
    y = s.groupby(pd.DatetimeIndex(s.index).year)
    first = y.first()
    prev = s.shift(1).groupby(pd.DatetimeIndex(s.index).year).first().fillna(first)
    out: pd.Series = y.last() / prev - 1
    return out


def summarise(
    res: SwingResult, stress: SwingResult, bench: pd.Series, rf: float, capital: float
) -> SwingSummary:
    """Compare the system with buying and holding the Nifty 500 over the same sessions."""
    d = res.daily
    eq = d["equity"]
    b = bench.reindex(eq.index).ffill().bfill()
    nifty = b / b.iloc[0] * capital
    bdf = pd.DataFrame({"equity": nifty, "invested": nifty, "positions": 1})
    m = metrics.portfolio_metrics(d, bench, rf)
    mb = metrics.portfolio_metrics(bdf, None, rf)
    ms = metrics.portfolio_metrics(stress.daily, bench, rf)
    curve = pd.DataFrame(
        {"system": eq, "nifty": nifty, "dd_system": _dd(eq), "dd_nifty": _dd(nifty)}
    )
    t = res.trades.copy()
    done = t[t["exit_reason"] != "Open"] if not t.empty else t
    yr_s, yr_b = _year_ret(eq), _year_ret(nifty)
    dd_s = curve["dd_system"].groupby(pd.DatetimeIndex(curve.index).year).min()
    dd_b = curve["dd_nifty"].groupby(pd.DatetimeIndex(curve.index).year).min()
    ty = pd.DatetimeIndex(done["exit_date"]).year if not done.empty else pd.Index([])
    yearly = pd.DataFrame(
        {
            "system": yr_s,
            "nifty500": yr_b,
            "difference": yr_s - yr_b,
            "system_worst_fall": dd_s,
            "nifty_worst_fall": dd_b,
            "trades": done.groupby(ty).size() if not done.empty else 0,
            "win_rate": done.groupby(ty)["pnl"].apply(lambda x: (x > 0).mean())
            if not done.empty
            else np.nan,
            "pnl_inr": done.groupby(ty)["pnl"].sum() if not done.empty else 0.0,
        }
    )
    yearly["trades"] = yearly["trades"].fillna(0)
    yearly["pnl_inr"] = yearly["pnl_inr"].fillna(0.0)
    yearly.index.name = "year"
    me = eq.resample("ME").last()
    mret = me / me.shift(1).fillna(capital) - 1
    monthly = pd.DataFrame(
        {
            "year": pd.DatetimeIndex(mret.index).year,
            "month": pd.DatetimeIndex(mret.index).month,
            "ret": mret.to_numpy(),
        }
    ).pivot(index="year", columns="month", values="ret")
    if not done.empty:
        g = done.groupby("exit_reason")
        exits = pd.DataFrame(
            {
                "trades": g.size(),
                "share": g.size() / len(done),
                "win_rate": g["pnl"].apply(lambda x: (x > 0).mean()),
                "avg_return": g["return_pct"].mean(),
                "avg_R": g["r_multiple"].mean(),
                "avg_sessions": g["sessions"].mean(),
                "pnl_inr": g["pnl"].sum(),
            }
        ).sort_values("trades", ascending=False)
    else:
        exits = pd.DataFrame()
    wins, losses = (
        done[done["pnl"] > 0] if not done.empty else done,
        (done[done["pnl"] <= 0] if not done.empty else done),
    )
    yrs = max(len(eq) / 252, 1e-9)
    stats = {
        "trades": float(len(done)),
        "trades_per_year": len(done) / yrs,
        "open_at_end": float((t["exit_reason"] == "Open").sum()) if not t.empty else 0.0,
        "win_rate": float((done["pnl"] > 0).mean()) if len(done) else math.nan,
        "avg_win": float(wins["return_pct"].mean()) if len(wins) else math.nan,
        "avg_loss": float(losses["return_pct"].mean()) if len(losses) else math.nan,
        "profit_factor": float(wins["pnl"].sum() / -losses["pnl"].sum())
        if len(losses) and losses["pnl"].sum() < 0
        else math.nan,
        "expectancy_R": float(done["r_multiple"].mean()) if len(done) else math.nan,
        "avg_sessions": float(done["sessions"].mean()) if len(done) else math.nan,
        "best_trade": float(done["return_pct"].max()) if len(done) else math.nan,
        "worst_trade": float(done["return_pct"].min()) if len(done) else math.nan,
        "target_hit_rate": float(done["partial_date"].notna().mean()) if len(done) else math.nan,
        "avg_invested": float(done["invested"].mean()) if len(done) else math.nan,
        "costs_inr": float(t["costs"].sum()) if not t.empty else 0.0,
        "costs_pa": float(t["costs"].sum() / eq.mean() / yrs) if not t.empty else 0.0,
        "avg_positions": float(d["positions"].mean()),
        "invested_share": float((d["invested"] / eq).mean()),
        "beat_years": float((yearly["difference"] > 0).mean()),
    }
    return SwingSummary(
        start=eq.index[0],
        end=eq.index[-1],
        capital=capital,
        sys=m,
        nifty=mb,
        stress=ms,
        end_value=float(eq.iloc[-1]),
        nifty_end=float(nifty.iloc[-1]),
        curve=curve,
        yearly=yearly,
        monthly=monthly,
        exits=exits,
        stats=stats,
        trades=t,
    )


# ---------------------------------------------------------------------------- output
def lakh(v: float) -> str:
    """₹ in lakh (e.g. 32.6 L)."""
    return f"{v / 1e5:,.1f} L"


def write_csvs(res: SwingResult, s: SwingSummary, out: Path) -> dict[str, Path]:
    """Trade log, fills, daily curve, yearly and monthly tables as CSV (open in Excel)."""
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "trades": out / "swing_trades.csv",
        "fills": out / "swing_fills.csv",
        "daily": out / "swing_daily.csv",
        "yearly": out / "swing_yearly.csv",
        "monthly": out / "swing_monthly.csv",
    }
    t = s.trades.copy()
    for c in ("entry_date", "exit_date", "partial_date"):
        if c in t:
            t[c] = pd.to_datetime(t[c]).dt.date
    t.round(4).to_csv(files["trades"], index=False)
    res.fills.round({"px": 4, "value": 2, "cost": 2}).to_csv(files["fills"], index=False)
    pd.concat([res.daily, s.curve[["nifty", "dd_system", "dd_nifty"]]], axis=1).round(4).to_csv(
        files["daily"]
    )
    s.yearly.round(4).to_csv(files["yearly"])
    s.monthly.round(4).to_csv(files["monthly"])
    return files


def _pct(v: Any, digits: int = 1, sign: bool = False) -> str:
    if not isinstance(v, (int, float, np.floating)) or not math.isfinite(float(v)):
        return "—"
    return f"{float(v):+.{digits}%}" if sign else f"{float(v):.{digits}%}"


def _num(v: Any, digits: int = 2) -> str:
    if not isinstance(v, (int, float, np.floating)) or not math.isfinite(float(v)):
        return "—"
    return f"{float(v):,.{digits}f}"


def _cls(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "pos" if f > 0 else ("neg" if f < 0 else "")


def write_html(s: SwingSummary, out: Path, settings: dict[str, Any]) -> Path:
    """Self-contained report (charts drawn in the page; no internet needed)."""
    out.mkdir(parents=True, exist_ok=True)
    c = s.curve
    wk = c.resample("W-FRI").last().dropna()
    data = {
        "dates": [str(x.date()) for x in wk.index],
        "sys": [round(float(v), 0) for v in wk["system"]],
        "nifty": [round(float(v), 0) for v in wk["nifty"]],
        "ddS": [round(float(v), 4) for v in wk["dd_system"]],
        "ddN": [round(float(v), 4) for v in wk["dd_nifty"]],
        "years": [int(y) for y in s.yearly.index],
        "yS": [round(float(v), 4) for v in s.yearly["system"]],
        "yN": [round(float(v), 4) for v in s.yearly["nifty500"]],
    }
    st, m, mb = s.stats, s.sys, s.nifty

    def tile(label: str, a: str, b: str, note: str = "") -> str:
        return (
            f"<div class='tile'><div class='tl'>{label}</div><div class='tv'>{a}</div>"
            f"<div class='tn'>Nifty 500: {b}</div>{f'<div class=tn>{note}</div>' if note else ''}</div>"
        )

    tiles = "".join(
        [
            tile(
                f"₹{s.capital / 1e5:.0f} L grew to",
                "₹" + lakh(s.end_value),
                "₹" + lakh(s.nifty_end),
            ),
            tile("Return per year", _pct(m.get("cagr")), _pct(mb.get("cagr"))),
            tile("Worst fall", _pct(m.get("max_dd")), _pct(mb.get("max_dd"))),
            tile("Sharpe", _num(m.get("sharpe")), _num(mb.get("sharpe"))),
            tile(
                "Years beating Nifty",
                f"{round(st['beat_years'] * len(s.yearly))} of {len(s.yearly)}",
                "—",
            ),
            tile(
                "Trades · win rate",
                f"{st['trades']:.0f} · {_pct(st['win_rate'], 0)}",
                "—",
                f"{st['trades_per_year']:.0f} per year",
            ),
        ]
    )
    ytab = "".join(
        f"<tr><td>{y}</td><td class='{_cls(r.system)}'>{_pct(r.system, 1, True)}</td>"
        f"<td class='{_cls(r.nifty500)}'>{_pct(r.nifty500, 1, True)}</td>"
        f"<td class='{_cls(r.difference)}'><b>{_pct(r.difference, 1, True)}</b></td>"
        f"<td>{_pct(r.system_worst_fall)}</td><td>{_pct(r.nifty_worst_fall)}</td>"
        f"<td>{_num(r.trades, 0)}</td><td>{_pct(r.win_rate, 0)}</td>"
        f"<td class='{_cls(r.pnl_inr)}'>{_num(r.pnl_inr, 0)}</td></tr>"
        for y, r in s.yearly.iterrows()
    )
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mtab = "".join(
        f"<tr><td>{y}</td>"
        + "".join(
            (
                f"<td class='hm' style='--a:{min(abs(float(v)) / 0.12, 1):.2f}' data-s='{'p' if v > 0 else 'n'}'>"
                f"{float(v) * 100:+.1f}</td>"
                if np.isfinite(v)
                else "<td></td>"
            )
            for v in [row.get(mo, np.nan) for mo in range(1, 13)]
        )
        + "</tr>"
        for y, row in s.monthly.iterrows()
    )
    etab = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{int(r.trades)}</td><td>{_pct(r.share, 0)}</td>"
        f"<td>{_pct(r.win_rate, 0)}</td><td class='{_cls(r.avg_return)}'>{_pct(r.avg_return, 1, True)}</td>"
        f"<td>{_num(r.avg_R)}</td><td>{_num(r.avg_sessions, 1)}</td>"
        f"<td class='{_cls(r.pnl_inr)}'>{_num(r.pnl_inr, 0)}</td></tr>"
        for k, r in s.exits.iterrows()
    )
    stat_rows = [
        ("Win rate", _pct(st["win_rate"], 0)),
        ("Average winning trade", _pct(st["avg_win"], 1, True)),
        ("Average losing trade", _pct(st["avg_loss"], 1, True)),
        ("Profit factor (₹ won ÷ ₹ lost)", _num(st["profit_factor"])),
        ("Average result per trade (in R, 1R = planned risk)", _num(st["expectancy_R"])),
        ("Average holding (sessions)", _num(st["avg_sessions"], 1)),
        (
            "Best / worst trade",
            f"{_pct(st['best_trade'], 1, True)} / {_pct(st['worst_trade'], 1, True)}",
        ),
        ("Trades that reached the target", _pct(st["target_hit_rate"], 0)),
        ("Average money per trade", "₹" + _num(st["avg_invested"], 0)),
        (
            "Average stocks held · money invested",
            f"{st['avg_positions']:.1f} · {_pct(st['invested_share'], 0)}",
        ),
        ("Total costs (₹) · per year", f"₹{_num(st['costs_inr'], 0)} · {_pct(st['costs_pa'], 2)}"),
        ("Return per year at 2× costs", _pct(s.stress.get("cagr"))),
        (
            "Beta to Nifty 500 · alpha per year",
            f"{_num(m.get('beta'))} · {_pct(m.get('alpha_ann'))}",
        ),
        ("Trades still open at the end", _num(st["open_at_end"], 0)),
    ]
    stab = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in stat_rows)
    cols = [
        ("trade_id", "#", "n0"),
        ("symbol", "Stock", "s"),
        ("entry_date", "Bought", "d"),
        ("entry_px", "Buy ₹", "n2"),
        ("qty", "Qty", "n0"),
        ("invested", "Invested ₹", "n0"),
        ("score_z", "Score z", "n2"),
        ("sl", "SL ₹", "n2"),
        ("target", "Target ₹", "n2"),
        ("partial_date", "Half sold", "d"),
        ("partial_px", "at ₹", "n2"),
        ("exit_date", "Exit", "d"),
        ("exit_px", "Exit ₹", "n2"),
        ("exit_reason", "Why", "s"),
        ("sessions", "Days", "n0"),
        ("costs", "Costs ₹", "n0"),
        ("pnl", "P&L ₹", "p0"),
        ("return_pct", "Return", "pp"),
        ("r_multiple", "R", "p2"),
    ]

    def cell(v: Any, kind: str) -> str:
        if kind == "s":
            return f"<td>{html.escape(str(v))}</td>"
        if kind == "d":
            return f"<td>{'' if pd.isna(v) else pd.Timestamp(v).date()}</td>"
        if kind == "pp":
            return f"<td class='{_cls(v)}'>{_pct(v, 1, True)}</td>"
        if kind.startswith("p"):
            return f"<td class='{_cls(v)}'>{_num(v, int(kind[1]))}</td>"
        return f"<td>{_num(v, int(kind[1]))}</td>"

    trows = "".join(
        "<tr>" + "".join(cell(r[k], kind) for k, _, kind in cols) + "</tr>"
        for r in (
            s.trades.sort_values("entry_date", ascending=False).to_dict("records")
            if not s.trades.empty
            else []
        )
    )
    thead = "".join(f"<th>{h}</th>" for _, h, _ in cols)
    rules = "".join(f"<li>{html.escape(x)}</li>" for x in settings.get("rules", []))
    page = TEMPLATE.format(
        period=f"{s.start.date()} → {s.end.date()}",
        tiles=tiles,
        ytab=ytab,
        mtab=mtab,
        mhead="".join(f"<th>{x}</th>" for x in months),
        etab=etab,
        stab=stab,
        thead=thead,
        trows=trows,
        ntr=len(s.trades),
        rules=rules,
        data=json.dumps(data),
    )
    path = out / "swing_report.html"
    path.write_text(page, encoding="utf-8")
    return path


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantSwing Results</title>
<style>
:root{{color-scheme:light;--page:#f9f9f7;--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--pos:#006300;--neg:#d03b3b;
--hp:42,120,214;--hn:227,73,72}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--page:#0d0d0d;--surf:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--mute:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--pos:#0ca30c;--neg:#e66767;--hp:57,135,229;--hn:230,103,103}}}}
:root[data-theme="dark"]{{color-scheme:dark;--page:#0d0d0d;--surf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mute:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--pos:#0ca30c;--neg:#e66767;
--hp:57,135,229;--hn:230,103,103}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:32px 0 10px}}.sub{{color:var(--ink2);margin:0 0 18px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.tile,.card{{background:var(--surf);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}}
.tl{{color:var(--ink2);font-size:12px}}.tv{{font-size:24px;font-weight:600;margin:2px 0}}.tn{{color:var(--mute);font-size:12px}}
.card{{padding:14px 14px 8px}}.legend{{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:0 0 6px}}
.legend i{{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}}
svg{{display:block;width:100%;height:auto;overflow:visible}}svg text{{fill:var(--mute);font-size:11px}}
.tip{{position:fixed;pointer-events:none;background:var(--surf);border:1px solid var(--ring);border-radius:8px;
padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:9}}
.tip b{{font-weight:600}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media (max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.tw{{overflow-x:auto;background:var(--surf);border:1px solid var(--ring);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 9px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}}
th:first-child,td:first-child{{text-align:left}}th{{color:var(--ink2);font-weight:600;font-size:12px;
position:sticky;top:0;background:var(--surf)}}.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
td.hm[data-s=p]{{background:rgba(var(--hp),calc(var(--a)*.55))}}td.hm[data-s=n]{{background:rgba(var(--hn),calc(var(--a)*.55))}}
.scroll{{max-height:560px;overflow:auto}}input{{font:inherit;padding:6px 10px;border:1px solid var(--ring);
border-radius:8px;background:var(--surf);color:var(--ink);width:260px;max-width:100%;margin:0 0 8px}}
ul{{color:var(--ink2);padding-left:18px}}
</style></head><body><main>
<h1>QuantSwing: system vs Nifty 500</h1>
<p class="sub">Backtest {period} · all costs and slippage included · every decision uses only data known at the time</p>
<div class="tiles">{tiles}</div>

<h2>Growth of the money</h2>
<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>System</span>
<span><i style="background:var(--s2)"></i>Nifty 500 (buy &amp; hold)</span></div><svg id="eq" viewBox="0 0 1000 340"></svg></div>

<div class="grid2">
<div><h2>Fall from the last peak</h2><div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>System</span>
<span><i style="background:var(--s2)"></i>Nifty 500</span></div><svg id="dd" viewBox="0 0 500 240"></svg></div></div>
<div><h2>Return each year</h2><div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>System</span>
<span><i style="background:var(--s2)"></i>Nifty 500</span></div><svg id="yr" viewBox="0 0 500 240"></svg></div></div>
</div>

<h2>Year by year</h2>
<div class="tw"><table><tr><th>Year</th><th>System</th><th>Nifty 500</th><th>Difference</th><th>System worst fall</th>
<th>Nifty worst fall</th><th>Trades closed</th><th>Win rate</th><th>P&amp;L ₹</th></tr>{ytab}</table></div>

<h2>Month by month (system, %)</h2>
<div class="tw"><table><tr><th>Year</th>{mhead}</tr>{mtab}</table></div>

<div class="grid2">
<div><h2>How trades ended</h2><div class="tw"><table><tr><th>Exit</th><th>Trades</th><th>Share</th><th>Win rate</th>
<th>Avg return</th><th>Avg R</th><th>Avg days</th><th>P&amp;L ₹</th></tr>{etab}</table></div></div>
<div><h2>Trade statistics</h2><div class="tw"><table>{stab}</table></div></div>
</div>

<h2>Rules used</h2><ul>{rules}</ul>

<h2>Every trade ({ntr})</h2>
<input id="q" placeholder="Filter by stock or exit reason…">
<div class="tw scroll"><table id="tl"><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table></div>
<p class="sub" style="margin-top:8px">SL = stop-loss · Target: half sold, stop moved to the buy price (BE) · TSL = trailing stop ·
Score = model no longer likes it · Time = held 20 sessions · Replaced = swapped for a much stronger stock · Open = still held at the end (marked at last price).
The same log is in <code>swing_trades.csv</code>.</p>
</main><div class="tip" id="tip"></div>
<script>
const D={data};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const NS="http://www.w3.org/2000/svg",tip=document.getElementById("tip");
function el(t,a,p){{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);p&&p.appendChild(e);return e}}
const fmtL=v=>"₹"+(v/1e5).toFixed(1)+" L",fmtP=v=>(v*100).toFixed(1)+"%",fmtS=v=>(v>0?"+":"")+(v*100).toFixed(1)+"%";
function showTip(ev,h){{tip.innerHTML=h;tip.style.display="block";const x=Math.min(ev.clientX+14,innerWidth-tip.offsetWidth-8);
tip.style.left=x+"px";tip.style.top=(ev.clientY+14)+"px"}}
function hide(){{tip.style.display="none"}}
function ticks(lo,hi,n){{const s=Math.pow(10,Math.floor(Math.log10((hi-lo)/n||1)));const st=[1,2,2.5,5,10].map(m=>m*s)
.find(x=>(hi-lo)/x<=n)||s*10;const out=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)out.push(v);return out}}
function lines(id,W,H,series,fmt,zero){{const svg=document.getElementById(id);svg.innerHTML="";const L=58,R=W-12,T=10,B=H-24;
const n=D.dates.length;let lo=Infinity,hi=-Infinity;series.forEach(s=>s.v.forEach(v=>{{lo=Math.min(lo,v);hi=Math.max(hi,v)}}));
if(zero)hi=Math.max(hi,0);const pad=(hi-lo)*.05;lo-=zero?pad:pad;hi+=zero?0:pad;
const x=i=>L+(R-L)*i/(n-1),y=v=>B-(B-T)*(v-lo)/(hi-lo);
ticks(lo,hi,5).forEach(v=>{{el("line",{{x1:L,x2:R,y1:y(v),y2:y(v),stroke:css("--grid"),"stroke-width":1}},svg);
const t=el("text",{{x:L-8,y:y(v)+4,"text-anchor":"end"}},svg);t.textContent=fmt(v)}});
let py=null;D.dates.forEach((d,i)=>{{const yv=+d.slice(0,4);if(yv!==py){{py=yv;const t=el("text",{{x:x(i),y:H-6}},svg);t.textContent=yv;
el("line",{{x1:x(i),x2:x(i),y1:B,y2:B+4,stroke:css("--axis")}},svg)}}}});
el("line",{{x1:L,x2:R,y1:zero?y(0):B,y2:zero?y(0):B,stroke:css("--axis"),"stroke-width":1}},svg);
series.slice().reverse().forEach(s=>el("path",{{d:s.v.map((v,i)=>(i?"L":"M")+x(i).toFixed(1)+" "+y(v).toFixed(1)).join(""),
fill:"none",stroke:css(s.c),"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}},svg));
const cross=el("line",{{y1:T,y2:B,stroke:css("--axis"),"stroke-width":1,visibility:"hidden"}},svg);
const dots=series.map(s=>el("circle",{{r:4,fill:css(s.c),stroke:css("--surf"),"stroke-width":2,visibility:"hidden"}},svg));
const hit=el("rect",{{x:L,y:T,width:R-L,height:B-T,fill:"transparent"}},svg);
hit.addEventListener("mousemove",ev=>{{const r=svg.getBoundingClientRect();const px=(ev.clientX-r.left)*W/r.width;
const i=Math.max(0,Math.min(n-1,Math.round((px-L)/(R-L)*(n-1))));cross.setAttribute("x1",x(i));cross.setAttribute("x2",x(i));
cross.setAttribute("visibility","visible");series.forEach((s,k)=>{{dots[k].setAttribute("cx",x(i));dots[k].setAttribute("cy",y(s.v[i]));
dots[k].setAttribute("visibility","visible")}});showTip(ev,"<b>"+D.dates[i]+"</b><br>"+series.map(s=>
"<span style='color:"+css(s.c)+"'>■</span> "+s.n+": <b>"+fmt(s.v[i])+"</b>").join("<br>"))}});
hit.addEventListener("mouseleave",()=>{{hide();cross.setAttribute("visibility","hidden");dots.forEach(d=>d.setAttribute("visibility","hidden"))}});}}
function bars(){{const id="yr",W=500,H=240,svg=document.getElementById(id);svg.innerHTML="";const L=46,R=W-8,T=10,B=H-24;
const vals=D.yS.concat(D.yN);let lo=Math.min(0,...vals),hi=Math.max(0,...vals);const p=(hi-lo)*.08;lo-=lo<0?p:0;hi+=p;
const y=v=>B-(B-T)*(v-lo)/(hi-lo);ticks(lo,hi,5).forEach(v=>{{el("line",{{x1:L,x2:R,y1:y(v),y2:y(v),stroke:css("--grid")}},svg);
const t=el("text",{{x:L-6,y:y(v)+4,"text-anchor":"end"}},svg);t.textContent=Math.round(v*100)+"%"}});
const n=D.years.length,gw=(R-L)/n,bw=Math.min(22,(gw-10)/2);
D.years.forEach((yr,i)=>{{const cx=L+gw*i+gw/2;const t=el("text",{{x:cx,y:H-6,"text-anchor":"middle"}},svg);t.textContent=yr;
[["--s1",D.yS[i],"System",-1],["--s2",D.yN[i],"Nifty 500",1]].forEach(([c,v,nm,sd])=>{{const x0=cx+(sd<0?-bw-1:1);
const y0=y(Math.max(v,0)),h=Math.max(Math.abs(y(v)-y(0)),1);const g=el("g",{{}},svg);
el("rect",{{x:x0,y:y0,width:bw,height:h,rx:3,fill:css(c)}},g);
const hr=el("rect",{{x:x0-2,y:T,width:bw+4,height:B-T,fill:"transparent"}},g);
hr.addEventListener("mousemove",ev=>showTip(ev,"<b>"+yr+"</b><br>System: <b>"+fmtS(D.yS[i])+"</b><br>Nifty 500: <b>"+fmtS(D.yN[i])+"</b>"));
hr.addEventListener("mouseleave",hide)}})}});el("line",{{x1:L,x2:R,y1:y(0),y2:y(0),stroke:css("--axis")}},svg)}}
function draw(){{lines("eq",1000,340,[{{n:"System",v:D.sys,c:"--s1"}},{{n:"Nifty 500",v:D.nifty,c:"--s2"}}],fmtL,false);
lines("dd",500,240,[{{n:"System",v:D.ddS,c:"--s1"}},{{n:"Nifty 500",v:D.ddN,c:"--s2"}}],v=>Math.round(v*100)+"%",true);bars()}}
draw();matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
document.getElementById("q").addEventListener("input",e=>{{const q=e.target.value.toLowerCase();
document.querySelectorAll("#tl tbody tr").forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?"":"none")}});
</script></body></html>"""
