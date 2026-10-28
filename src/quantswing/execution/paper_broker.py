"""Paper book (04 §7, 02 §4 "Operational: SQLite"): orders + state log, equity, positions,
controls (kill switch), incidents and job runs. One SQLite file; every write is a
transaction. Order fills come from the portfolio simulator (:mod:`portfolio.construction`)
so paper and backtest share one execution model.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantswing.core.clock import now_ist
from quantswing.execution.order_state import OPEN_STATES, OrderState, check

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL, decided_on TEXT NOT NULL, exec_session TEXT,
    symbol TEXT NOT NULL, side TEXT NOT NULL, target_weight REAL,
    qty REAL, fill_px REAL, value REAL, cost REAL,
    state TEXT NOT NULL, reason TEXT, strategy TEXT, model_version TEXT, manual INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_orders_state ON orders(state);
CREATE TABLE IF NOT EXISTS order_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, ts TEXT NOT NULL,
    from_state TEXT, to_state TEXT NOT NULL, px REAL, reason TEXT, model_version TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    date TEXT PRIMARY KEY, equity REAL, cash REAL, invested REAL, positions INTEGER,
    halted INTEGER, gross_target REAL, regime_mult REAL, regime TEXT, dd REAL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, qty REAL, last_px REAL, value REAL, weight REAL, since TEXT
);
CREATE TABLE IF NOT EXISTS controls (key TEXT PRIMARY KEY, value TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, kind TEXT, message TEXT, resolved INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started TEXT, finished TEXT, status TEXT,
    sessions INTEGER, message TEXT
);
"""


def _ts(t: datetime | None = None) -> str:
    return (t or now_ist()).strftime("%Y-%m-%d %H:%M:%S")


class PaperBook:
    """SQLite-backed paper account."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.tx() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Connection in a transaction (commit on success, rollback on error)."""
        con = sqlite3.connect(self.path, timeout=30)
        try:
            with con:
                yield con
        finally:
            con.close()

    # ------------------------------------------------------------------ meta / controls
    def get(self, key: str, default: Any = None) -> Any:
        """JSON value from ``meta``."""
        with self.tx() as c:
            row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def put(self, key: str, value: Any, con: sqlite3.Connection | None = None) -> None:
        """Store a JSON value in ``meta``."""
        sql = "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)"
        if con is not None:
            con.execute(sql, (key, json.dumps(value, default=str)))
            return
        with self.tx() as c:
            c.execute(sql, (key, json.dumps(value, default=str)))

    def control(self, key: str) -> bool:
        """Boolean control flag (pause_entries, flatten)."""
        with self.tx() as c:
            row = c.execute("SELECT value FROM controls WHERE key = ?", (key,)).fetchone()
        return row is not None and row[0] == "1"

    def set_control(self, key: str, on: bool) -> None:
        """Set a control flag."""
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO controls(key, value, ts) VALUES (?, ?, ?)",
                (key, "1" if on else "0", _ts()),
            )

    # ------------------------------------------------------------------ orders
    def _log(
        self,
        c: sqlite3.Connection,
        oid: int,
        src: str | None,
        dst: str,
        px: float | None,
        reason: str,
        mv: str,
    ) -> None:
        c.execute(
            "INSERT INTO order_log(order_id, ts, from_state, to_state, px, reason, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (oid, _ts(), src, dst, px, reason, mv),
        )

    def open_orders(self) -> pd.DataFrame:
        """Orders not yet final."""
        q = ",".join(f"'{s}'" for s in OPEN_STATES)
        with self.tx() as c:
            return pd.read_sql_query(f"SELECT * FROM orders WHERE state IN ({q}) ORDER BY id", c)

    def place(
        self,
        c: sqlite3.Connection,
        decided_on: str,
        symbol: str,
        side: str,
        target_weight: float,
        reason: str,
        strategy: str,
        mv: str,
    ) -> int:
        """New PENDING (buy) / EXIT_PENDING (sell) order for the next open."""
        st = OrderState.PENDING if side == "buy" else OrderState.EXIT_PENDING
        cur = c.execute(
            "INSERT INTO orders(created, decided_on, symbol, side, target_weight, state, reason, "
            "strategy, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_ts(), decided_on, symbol, side, target_weight, st.value, reason, strategy, mv),
        )
        oid = int(cur.lastrowid or 0)
        self._log(c, oid, None, st.value, None, reason, mv)
        return oid

    def move(
        self,
        c: sqlite3.Connection,
        oid: int,
        dst: OrderState,
        *,
        reason: str,
        mv: str,
        exec_session: str | None = None,
        qty: float | None = None,
        px: float | None = None,
        cost: float | None = None,
    ) -> None:
        """Validated state change (+ fill fields) with a log row."""
        row = c.execute("SELECT state FROM orders WHERE id = ?", (oid,)).fetchone()
        src = OrderState(row[0])
        check(src, dst)
        c.execute(
            "UPDATE orders SET state = ?, exec_session = COALESCE(?, exec_session), "
            "qty = COALESCE(?, qty), fill_px = COALESCE(?, fill_px), "
            "value = COALESCE(?, value), cost = COALESCE(?, cost) WHERE id = ?",
            (
                dst.value,
                exec_session,
                qty,
                px,
                None if qty is None or px is None else qty * px,
                cost,
                oid,
            ),
        )
        self._log(c, oid, src.value, dst.value, px, reason, mv)

    def orders(self) -> pd.DataFrame:
        """All orders."""
        with self.tx() as c:
            return pd.read_sql_query("SELECT * FROM orders ORDER BY id", c)

    def order_log(self) -> pd.DataFrame:
        """All state changes."""
        with self.tx() as c:
            return pd.read_sql_query("SELECT * FROM order_log ORDER BY id", c)

    # ------------------------------------------------------------------ equity / positions
    def equity(self) -> pd.DataFrame:
        """Daily equity curve."""
        with self.tx() as c:
            df = pd.read_sql_query("SELECT * FROM equity ORDER BY date", c)
        return df.set_index(pd.DatetimeIndex(pd.to_datetime(df.pop("date")), name="date"))

    def positions(self) -> pd.DataFrame:
        """Current holdings snapshot."""
        with self.tx() as c:
            return pd.read_sql_query("SELECT * FROM positions ORDER BY value DESC", c)

    # ------------------------------------------------------------------ incidents / runs
    def incident(self, kind: str, message: str) -> None:
        """Record an operational incident (G4)."""
        with self.tx() as c:
            c.execute(
                "INSERT INTO incidents(ts, kind, message) VALUES (?, ?, ?)",
                (_ts(), kind, message[:2000]),
            )

    def incidents(self, open_only: bool = False) -> pd.DataFrame:
        """Incidents (optionally unresolved only)."""
        where = "WHERE resolved = 0" if open_only else ""
        with self.tx() as c:
            return pd.read_sql_query(f"SELECT * FROM incidents {where} ORDER BY id", c)

    def resolve_incidents(self) -> int:
        """Mark every incident resolved (after a clean run). Returns how many."""
        with self.tx() as c:
            return c.execute("UPDATE incidents SET resolved = 1 WHERE resolved = 0").rowcount

    def run_start(self) -> int:
        """Log a job start."""
        with self.tx() as c:
            cur = c.execute("INSERT INTO runs(started, status) VALUES (?, 'running')", (_ts(),))
            return int(cur.lastrowid or 0)

    def run_end(self, rid: int, status: str, sessions: int, message: str = "") -> None:
        """Log a job end."""
        with self.tx() as c:
            c.execute(
                "UPDATE runs SET finished = ?, status = ?, sessions = ?, message = ? WHERE id = ?",
                (_ts(), status, sessions, message[:2000], rid),
            )

    def runs(self) -> pd.DataFrame:
        """Job history."""
        with self.tx() as c:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY id", c)
