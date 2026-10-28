"""HTML report for P5 (factor portfolio + regime)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

from quantswing.backtest.report import _svg_line
from quantswing.portfolio.p5 import Variant


def _f(v: object, pct: bool = False) -> str:
    if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.1%}" if pct else f"{v:,.2f}"


def write_report(
    variants: list[Variant],
    bench: dict[str, float],
    regime: pd.DataFrame,
    bench_px: pd.Series,
    out_dir: Path,
    notes: list[str],
) -> Path:
    """Write ``p5_portfolio_report.html``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        ("cagr", True),
        ("vol", True),
        ("sharpe", False),
        ("sortino", False),
        ("max_dd", True),
        ("calmar", False),
        ("exposure", True),
        ("alpha_ann", True),
        ("beta", False),
    ]
    head = "".join(f"<th>{k}</th>" for k, _ in keys)
    rows = "".join(
        f"<tr><th>{html.escape(v.name)} (OOS)</th>"
        + "".join(f"<td>{_f(v.oos.get(k), pct)}</td>" for k, pct in keys)
        + "</tr>"
        + f"<tr class='mut'><th>{html.escape(v.name)} 2x costs</th>"
        + "".join(f"<td>{_f(v.oos_stress.get(k), pct)}</td>" for k, pct in keys)
        + "</tr>"
        for v in variants
    )
    rows += (
        "<tr><th>Nifty 500 buy &amp; hold (OOS)</th>"
        + "".join(f"<td>{_f(bench.get(k), pct)}</td>" for k, pct in keys)
        + "</tr>"
    )
    checks = "".join(
        f"<h2>{html.escape(v.name)} — DSR {_f(v.dsr)} · turnover {_f(v.turnover)}x/yr</h2>"
        + _svg_line(v.res.daily["equity"], bench_px, h=160)
        + "<table>"
        + "".join(
            f"<tr><th>{html.escape(k)}</th><td class='{'ok' if ok else 'no'}'>{'OK' if ok else 'NO'}</td></tr>"
            for k, ok in v.checks.items()
        )
        + "</table>"
        + "<table><tr>"
        + "".join(f"<th>{y}</th>" for y in v.by_year.index)
        + "</tr><tr>"
        + "".join(f"<td>{r:+.1%}</td>" for r in v.by_year)
        + "</tr></table>"
        for v in variants
    )
    st = regime["state"].dropna()
    share = st.value_counts(normalize=True).to_dict() if len(st) else {}
    reg_tbl = (
        "<table>"
        + "".join(f"<tr><th>{k}</th><td>{val:.0%}</td></tr>" for k, val in share.items())
        + "</table>"
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>QuantSwing P5 portfolio</title>
<style>
:root{{--bg:#fff;--fg:#1b1d21;--mut:#667;--line:#e3e5ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15171b;--fg:#e8eaee;--mut:#99a;--line:#2a2e35}}}}
body{{background:var(--bg);color:var(--fg);font:13px/1.35 system-ui,Segoe UI,sans-serif;margin:16px}}
h1{{font-size:18px}} h2{{font-size:14px;margin:16px 0 4px}} .mut{{color:var(--mut)}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:6px 0}} th,td{{border:1px solid var(--line);padding:3px 7px;text-align:right}}
th{{text-align:left}} .ok{{color:#1f9d55;font-weight:700}} .no{{color:#c53030;font-weight:700}} tr.mut td,tr.mut th{{color:var(--mut)}}
</style></head><body><h1>QuantSwing P5 — factor portfolio + regime</h1>
<div class="mut">OOS from {int(bench.get("oos_start", 0))} · blue = strategy (full period), grey = Nifty 500</div>
<table><tr><th>Variant</th>{head}</tr>{rows}</table>{checks}
<h2>Regime share of sessions</h2>{reg_tbl}
<h2>Notes</h2><ul>{"".join(f"<li>{html.escape(n)}</li>" for n in notes)}</ul></body></html>"""
    path = out_dir / "p5_portfolio_report.html"
    path.write_text(page, encoding="utf-8")
    return path
