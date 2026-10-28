"""HTML report for P4 sleeve validation."""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from quantswing.backtest.report import _svg_line
from quantswing.sleeves.validate import SleeveResult


def _f(v: object, pct: bool = False) -> str:
    if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.1%}" if pct else f"{v:,.2f}"


def write_report(results: list[SleeveResult], out_dir: Path, notes: list[str]) -> Path:
    """Write ``p4_sleeves_report.html``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"<tr><th>{r.name}</th><td class='{'ok' if r.verdict == 'ON' else 'no'}'>{r.verdict}</td>"
        f"<td>{r.n_signals:,}</td><td>{_f(r.trade.get('trades'))}</td><td>{_f(r.port.get('cagr'), True)}</td>"
        f"<td>{_f(r.port.get('sharpe'))}</td><td>{_f(r.port.get('max_dd'), True)}</td>"
        f"<td>{_f(r.trade.get('win_rate'), True)}</td><td>{_f(r.trade.get('expectancy_r'))}</td>"
        f"<td>{_f(r.trade.get('profit_factor'))}</td><td>{_f(r.stress.get('expectancy_r'))}</td>"
        f"<td>{html.escape('; '.join(r.reasons)) or '—'}</td></tr>"
        for r in results
    )
    blocks = []
    for r in results:
        ex = r.live_exits
        params = " · ".join(
            f"{k} {ex[k]:.2f}" if isinstance(ex[k], float) else f"{k} {ex[k]}"
            for k in (
                "k_sl_sigma",
                "partial_at_r",
                "trail_k_below_switch",
                "trail_k_above_switch",
                "time_stop_sessions",
                "max_hold_sessions",
            )
        )
        folds = (
            r.folds.to_html(index=False, float_format=lambda x: f"{x:,.2f}", border=0)
            if not r.folds.empty
            else "<p>No folds.</p>"
        )
        curve = _svg_line(r.daily["equity"], None, h=140) if len(r.daily) > 1 else ""
        blocks.append(
            f"<h2>{r.name} — {r.verdict}</h2><div class='mut'>Live exits ({'fitted' if r.live_confident else 'defaults: too few events'}): {html.escape(params)}</div>{curve}{folds}"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>QuantSwing P4 sleeves</title>
<style>
:root{{--bg:#fff;--fg:#1b1d21;--mut:#667;--line:#e3e5ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15171b;--fg:#e8eaee;--mut:#99a;--line:#2a2e35}}}}
body{{background:var(--bg);color:var(--fg);font:13px/1.35 system-ui,Segoe UI,sans-serif;margin:16px}}
h1{{font-size:18px}} h2{{font-size:14px;margin:18px 0 4px}} .mut{{color:var(--mut)}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{text-align:left}} .ok{{color:#1f9d55;font-weight:700}} .no{{color:#c53030;font-weight:700}}
</style></head><body><h1>QuantSwing P4 — sleeve walk-forward validation</h1>
<div class="mut">Out-of-sample only: each test year uses exits and p(win) fitted on the prior years (purged + embargo).</div>
<table><tr><th>Sleeve</th><th>Verdict</th><th>Signals</th><th>OOS trades</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Win</th><th>Exp R</th><th>PF</th><th>2x-cost Exp R</th><th>Why OFF</th></tr>{rows}</table>
{"".join(blocks)}
<h2>Notes</h2><ul>{"".join(f"<li>{html.escape(n)}</li>" for n in notes)}</ul></body></html>"""
    path = out_dir / "p4_sleeves_report.html"
    path.write_text(page, encoding="utf-8")
    summary = pd.DataFrame(
        [
            {
                "sleeve": r.name,
                "verdict": r.verdict,
                **r.port,
                **{f"t_{k}": v for k, v in r.trade.items()},
            }
            for r in results
        ]
    )
    summary.to_csv(out_dir / "p4_sleeves_summary.csv", index=False)
    return path
