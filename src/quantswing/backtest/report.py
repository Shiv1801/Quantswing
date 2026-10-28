"""HTML report for the P3 baseline (no plotting dependency: inline SVG equity curve)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from quantswing.backtest.baseline import BaselineReport
from quantswing.backtest.metrics import drawdown


def _svg_line(series: pd.Series, bench: pd.Series | None, w: int = 900, h: int = 220) -> str:
    s = series.dropna()
    if len(s) < 2:
        return ""
    vals = [s / s.iloc[0]]
    if bench is not None:
        b = bench.reindex(s.index).ffill().dropna()
        if len(b) > 1:
            vals.append(b / b.iloc[0])
    lo = min(float(v.min()) for v in vals)
    hi = max(float(v.max()) for v in vals)
    span = hi - lo or 1.0

    def pts(v: pd.Series) -> str:
        n = len(v)
        return " ".join(
            f"{k / (n - 1) * w:.1f},{h - (float(x) - lo) / span * h:.1f}" for k, x in enumerate(v)
        )

    lines = [f'<polyline fill="none" stroke="#2b6cb0" stroke-width="1.6" points="{pts(vals[0])}"/>']
    if len(vals) > 1:
        lines.append(
            f'<polyline fill="none" stroke="#999" stroke-width="1" points="{pts(vals[1])}"/>'
        )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">{"".join(lines)}</svg>'


def _fmt(k: str, v: object) -> str:
    if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    pct = {
        "cagr",
        "vol",
        "max_dd",
        "exposure",
        "win_rate",
        "alpha_ann",
        "p_upper",
        "p_lower",
        "p_vertical",
        "capture",
    }
    if k in pct:
        return f"{v:.1%}"
    if k in {"net_pnl", "total_costs"}:
        return f"₹{v:,.0f}"
    return f"{v:,.2f}"


def _table(d: dict[str, object]) -> str:
    return (
        "<table>"
        + "".join(f"<tr><th>{html.escape(k)}</th><td>{_fmt(k, v)}</td></tr>" for k, v in d.items())
        + "</table>"
    )


def write_report(
    rep: BaselineReport, bench: pd.Series | None, out_dir: Path, notes: list[str]
) -> Path:
    """Write ``p3_backtest_report.html``; return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eq = rep.base.daily["equity"]
    dd = float(drawdown(eq).min()) if len(eq) else float("nan")
    checks = "".join(
        f'<tr><th>{html.escape(k)}</th><td class="{"ok" if v else "no"}">{"OK" if v else "NO"}</td></tr>'
        for k, v in {**rep.checks, **rep.quality}.items()
    )
    years = "".join(f"<tr><th>{y}</th><td>₹{v:,.0f}</td></tr>" for y, v in rep.by_year.items())
    reasons = (
        rep.base.trades["reason"].value_counts().to_dict() if not rep.base.trades.empty else {}
    )
    mc = rep.mc
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>QuantSwing P3 baseline</title>
<style>
:root{{--bg:#fff;--fg:#1b1d21;--mut:#667;--line:#e3e5ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15171b;--fg:#e8eaee;--mut:#99a;--line:#2a2e35}}}}
body{{background:var(--bg);color:var(--fg);font:13px/1.35 system-ui,Segoe UI,sans-serif;margin:16px}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:14px;margin:16px 0 6px}} .mut{{color:var(--mut)}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;margin-right:16px}}
th,td{{border:1px solid var(--line);padding:3px 8px;text-align:right}} th{{text-align:left;font-weight:600}}
.grid{{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}} .ok{{color:#1f9d55;font-weight:700}} .no{{color:#c53030;font-weight:700}}
</style></head><body>
<h1>QuantSwing P3 — baseline backtest</h1>
<div class="mut">Run {html.escape(rep.run_id)} · trials logged {rep.n_trials} · blue = strategy, grey = Nifty 500 (price) · max DD {dd:.1%}</div>
{_svg_line(eq, bench)}
<div class="grid">
<div><h2>Portfolio</h2>{_table(dict(rep.port))}</div>
<div><h2>Trades (1× costs)</h2>{_table(dict(rep.trade))}</div>
<div><h2>Trades (2× costs)</h2>{_table({k: rep.stress_trade.get(k) for k in ("expectancy_r", "profit_factor", "win_rate", "net_pnl")})}</div>
<div><h2>Checks</h2><table>{checks}</table>
<h2>Validation</h2>{_table({"DSR": rep.dsr, "MC DD p50": mc.dd_p50_reshuffle, "MC DD p95": mc.dd_p95_reshuffle, "Bootstrap DD p95": mc.dd_p95_bootstrap, "Bootstrap wealth p05": mc.final_p05_bootstrap, "Max year P&L share": rep.year_share})}</div>
<div><h2>P&amp;L by exit year</h2><table>{years}</table></div>
<div><h2>Exit reasons</h2>{_table({str(k): float(v) for k, v in reasons.items()})}</div>
<div><h2>Signal labels (triple barrier)</h2>{_table(dict(rep.labels))}</div>
</div>
<h2>Notes</h2><ul>{"".join(f"<li>{html.escape(n)}</li>" for n in notes)}</ul>
</body></html>"""
    path = out_dir / "p3_backtest_report.html"
    path.write_text(page, encoding="utf-8")
    rep.base.trades.to_csv(out_dir / "p3_trades.csv", index=False)
    rep.base.daily.to_csv(out_dir / "p3_equity.csv")
    return path
