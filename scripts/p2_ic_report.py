"""P2 exit test: factor ICs reported by horizon (07 §1; 05 §3 "Factor IC").

Reads the factor panels from run_factors.py, computes rank IC per factor and horizon,
IC-IR weights and the composite, writes reports/p2_ic_report.html (+ CSVs), prints a
summary and ``P2 EXIT TEST: PASS`` when every factor has IC at every horizon.

Usage::

    python scripts/p2_ic_report.py
"""

from __future__ import annotations

import os
import sys

from quantswing.alpha.ic_report import verdicts, write_report
from quantswing.alpha.research import COMPOSITE, ic_analysis
from quantswing.core.config import get_config
from quantswing.core.logging import setup_logging
from quantswing.store.repositories import DataPaths

NOTES = [
    "Data: Fyers only (D22). Universe = currently listed NSE EQ/BE -> survivorship bias; ICs are optimistic.",
    "Neutralised for size (ln 60d median turnover) only; sector neutralisation unavailable (no sector data).",
    "Not built: DELZ (no delivery data), QUAL-lite (no filings). RSEC replaced by RMOM (residual vs Nifty 500).",
    "Gate: history >= 250 sessions, continuity >= 95%, no circuit-lock streak >= 3 in 20. No ASM/GSM/event filters.",
    "Composite uses IC-IR weights re-estimated monthly from ICs already known at the time (no look-ahead).",
    "Forward returns are close-to-close from the signal date; execution lag/costs are handled in the P3 backtest.",
]


def main() -> int:
    """Run the IC analysis. Returns 0 on PASS."""
    setup_logging()
    cfg = get_config()
    paths = DataPaths(cfg)
    ic_cfg = cfg.factors.ic
    print("Computing ICs (each line = one factor)...")
    rep = ic_analysis(paths, cfg, progress=lambda n: print(f"  done {n}", flush=True))
    out = write_report(rep, paths.reports, ic_cfg.t_stat_pass, NOTES)

    v = verdicts(rep, ic_cfg.t_stat_pass)
    print(f"\n{'factor':<11}{'IC ' + str(rep.primary) + 'd':>9}{'t':>7}{'hit':>6}   verdict")
    for f, row in v.iterrows():
        print(
            f"{f!s:<11}{row['mean']:>+9.3f}{row['t_nonoverlap']:>7.1f}{row['hit']:>6.0%}   {row['verdict']}"
        )
    n_pass = int(v.drop(index=COMPOSITE, errors="ignore")["verdict"].str.startswith("PASS").sum())
    total = len(v) - (1 if COMPOSITE in v.index else 0)
    complete = bool(rep.stats.groupby("factor")["n"].min().gt(0).all())
    print(f"\nSignificant factors at {rep.primary}d: {n_pass}/{total}")
    print(f"Report: {out}")
    print(f"\nP2 EXIT TEST: {'PASS' if complete and total > 0 else 'FAIL'}")
    if sys.platform == "win32" and "--no-open" not in sys.argv:
        os.startfile(out)  # type: ignore[attr-defined]
    return 0 if complete and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
