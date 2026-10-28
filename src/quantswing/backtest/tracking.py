"""Run tracker (D25): every backtest/trial is logged under ``runs/<run_id>/``.

Files: ``params.json``, ``metrics.json``, ``tags.json`` + artifacts (trade log, equity).
The number of logged trials and the variance of their Sharpes feed the Deflated Sharpe.
If ``tracking.use_mlflow`` is true and mlflow is installed, runs are mirrored to MLflow.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantswing.core.clock import now_ist


def fingerprint(obj: Any) -> str:
    """Stable short hash of a JSON-serialisable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class RunTracker:
    """Filesystem run registry."""

    def __init__(self, root: Path, use_mlflow: bool = False) -> None:
        self.root = root
        self.use_mlflow = use_mlflow

    def log_run(
        self,
        name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        tags: dict[str, Any] | None = None,
        artifacts: dict[str, pd.DataFrame] | None = None,
    ) -> str:
        """Persist a run; returns its id."""
        ts: datetime = now_ist()
        run_id = f"{ts:%Y%m%d-%H%M%S}-{fingerprint(params)}"
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "params.json").write_text(json.dumps(params, indent=1, default=str), encoding="utf-8")
        (d / "metrics.json").write_text(
            json.dumps(metrics, indent=1, default=str), encoding="utf-8"
        )
        (d / "tags.json").write_text(
            json.dumps(
                {"name": name, "time": ts.isoformat(), **(tags or {})}, indent=1, default=str
            ),
            encoding="utf-8",
        )
        for fname, df in (artifacts or {}).items():
            df.to_parquet(d / f"{fname}.parquet")
        if self.use_mlflow:
            self._mirror_mlflow(name, params, metrics)
        return run_id

    def _mirror_mlflow(self, name: str, params: dict[str, Any], metrics: dict[str, Any]) -> None:
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError:
            return
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            )

    def runs(self) -> pd.DataFrame:
        """All logged runs: id, name, sharpe_pp (per-period Sharpe) and params fingerprint."""
        rows = []
        for d in sorted(self.root.glob("*/metrics.json")):
            m = json.loads(d.read_text(encoding="utf-8"))
            t = json.loads((d.parent / "tags.json").read_text(encoding="utf-8"))
            rows.append({"run_id": d.parent.name, "name": t.get("name"), **m})
        return pd.DataFrame(rows)

    def trial_stats(self, metric: str = "sharpe_pp") -> tuple[int, float]:
        """(number of trials, variance of their per-period Sharpes)."""
        df = self.runs()
        if df.empty or metric not in df:
            return 0, 0.0
        s = pd.to_numeric(df[metric], errors="coerce").dropna()
        return len(s), float(s.var()) if len(s) > 1 else 0.0
