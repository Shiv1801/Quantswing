"""Self-contained HTML + CSV rendering of the P2 IC report (no plotting dependency).

Layout: factor × horizon heatmap of mean rank IC (cell text: IC and non-overlapping t),
a pass column at the primary horizon (05 §3: IC > 0 and t > threshold, sign-aware), the
IC-by-year table, and the latest IC-IR weights.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantswing.alpha.research import COMPOSITE, IcReport


def verdicts(rep: IcReport, t_pass: float) -> pd.DataFrame:
    """Per factor at the primary horizon: mean IC, t, and PASS / PASS(−) / FAIL.

    ``PASS(−)``: significant with a negative sign — usable if the factor is inverted.
    """
    s = rep.stats[rep.stats["h"] == rep.primary].set_index("factor")
    out = s[["mean", "t_nonoverlap", "hit", "n"]].copy()
    out["verdict"] = np.where(
        out["t_nonoverlap"] > t_pass,
        "PASS",
        np.where(out["t_nonoverlap"] < -t_pass, "PASS(-)", "FAIL"),
    )
    return out


def _f(x: Any) -> float:
    """Coerce a pandas scalar to float."""
    return float(x)


def _color(v: float, scale: float = 0.06) -> str:
    if not np.isfinite(v):
        return "transparent"
    a = min(abs(v) / scale, 1.0) * 0.85
    return f"rgba(34,160,90,{a:.2f})" if v > 0 else f"rgba(210,60,60,{a:.2f})"


def render_html(rep: IcReport, t_pass: float, title: str, notes: list[str]) -> str:
    """Build the HTML page."""
    piv_m = rep.stats.pivot(index="factor", columns="h", values="mean")
    piv_t = rep.stats.pivot(index="factor", columns="h", values="t_nonoverlap")
    ver = verdicts(rep, t_pass)
    order = [f for f in piv_m.index if f != COMPOSITE] + (
        [COMPOSITE] if COMPOSITE in piv_m.index else []
    )
    head = "".join(f"<th>{h}d</th>" for h in rep.horizons)
    body = []
    for f in order:
        cells = "".join(
            f'<td style="background:{_color(_f(piv_m.at[f, h]))}">{_f(piv_m.at[f, h]):+.3f}'
            f"<small> t={_f(piv_t.at[f, h]):.1f}</small></td>"
            for h in rep.horizons
        )
        v = str(ver.at[f, "verdict"])
        cls = "pass" if v.startswith("PASS") else "fail"
        w = rep.weights_last.get(f, np.nan)
        wtxt = f"{w:.2f}" if np.isfinite(w) else "—"
        body.append(
            f'<tr class="{"comp" if f == COMPOSITE else ""}"><th>{html.escape(f)}</th>{cells}'
            f'<td class="{cls}">{v}</td><td>{_f(ver.at[f, "hit"]):.0%}</td><td>{wtxt}</td></tr>'
        )
    yrs = rep.by_year
    ycols = list(yrs.columns)
    yhead = "".join(f"<th>{y}</th>" for y in ycols)
    ybody = "".join(
        f"<tr><th>{html.escape(str(f))}</th>"
        + "".join(
            f'<td style="background:{_color(_f(yrs.at[f, y]))}">{_f(yrs.at[f, y]):+.3f}</td>'
            for y in ycols
        )
        + "</tr>"
        for f in [x for x in order if x in yrs.index]
    )
    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
:root{{--bg:#fff;--fg:#1b1d21;--mut:#667;--line:#e3e5ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15171b;--fg:#e8eaee;--mut:#99a;--line:#2a2e35}}}}
body{{background:var(--bg);color:var(--fg);font:13px/1.35 system-ui,Segoe UI,sans-serif;margin:16px}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:14px;margin:18px 0 6px}}
.mut{{color:var(--mut)}} table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{border:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}}
th{{text-align:left;font-weight:600}} small{{color:var(--mut);margin-left:3px}}
.pass{{color:#1f9d55;font-weight:700}} .fail{{color:var(--mut)}} tr.comp th{{color:#2b6cb0}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="mut">Generated {datetime.now():%Y-%m-%d %H:%M} · rank IC of neutralised factor vs forward log return ·
t on non-overlapping samples · primary horizon {rep.primary}d · pass: |t| &gt; {t_pass}</div>
<h2>Mean IC by horizon</h2><div class="wrap"><table>
<tr><th>Factor</th>{head}<th>{rep.primary}d verdict</th><th>hit</th><th>IC-IR w</th></tr>
{"".join(body)}</table></div>
<h2>Mean IC by year ({rep.primary}d)</h2><div class="wrap"><table><tr><th>Factor</th>{yhead}</tr>{ybody}</table></div>
<h2>Notes</h2><ul>{note_html}</ul>
</body></html>"""


def write_report(rep: IcReport, out_dir: Path, t_pass: float, notes: list[str]) -> Path:
    """Write ``p2_ic_report.html`` + CSVs; return the HTML path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep.stats.to_csv(out_dir / "p2_ic_stats.csv", index=False)
    rep.by_year.to_csv(out_dir / "p2_ic_by_year.csv")
    verdicts(rep, t_pass).to_csv(out_dir / "p2_ic_verdicts.csv")
    path = out_dir / "p2_ic_report.html"
    path.write_text(
        render_html(rep, t_pass, "QuantSwing P2 — Factor IC report", notes), encoding="utf-8"
    )
    return path
