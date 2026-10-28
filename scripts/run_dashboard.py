"""Start the QuantSwing dashboard (local only) and open it in the browser.

Usage::

    python scripts/run_dashboard.py          # http://localhost:8501
Stop with Ctrl+C.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "quantswing" / "app" / "dashboard" / "app.py"

if __name__ == "__main__":
    os.chdir(ROOT)  # picks up .streamlit/config.toml (dark theme, localhost only)
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(APP), "--server.headless=false"]
    sys.exit(cli.main())
