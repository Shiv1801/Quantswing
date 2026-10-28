"""D40 live paper book (SQLite): engine state, action list, official fills / trades / equity,
live quotes and live events. Paper only — nothing here talks to a broker's order API.

Official numbers come from the end-of-day run, which replays the day through the *same*
trade engine as the backtest (daily bars). Live polls only add provisional events
("SL hit at ~₹x") so the screen and Telegram stay current during the day.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantswing.core.clock import now_ist
from quantswing.portfolio.swing import SwingState

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, session TEXT, at TEXT, symbol TEXT,
  action TEXT, qty INTEGER, order_type TEXT, price REAL, reason TEXT, status TEXT,
  detail TEXT, key TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS fills (
  date TEXT, symbol TEXT, side TEXT, qty INTEGER, px REAL, value REAL, cost REAL,
  reason TEXT, trade_id INTEGER);
CREATE TABLE IF NOT EXISTS trades (
  trade_id INTEGER PRIMARY KEY, symbol TEXT, entry_date TEXT, entry_px REAL, qty INTEGER,
  exit_date TEXT, exit_px REAL, exit_reason TEXT, sessions INTEGER, costs REAL, pnl REAL,
  return_pct REAL, r_multiple REAL, data TEXT);
CREATE TABLE IF NOT EXISTS equity (
  date TEXT PRIMARY KEY, equity REAL, cash REAL, invested REAL, positions INTEGER,
  open_risk REAL, nifty REAL);
CREATE TABLE IF NOT EXISTS quotes (
  symbol TEXT PRIMARY KEY, ts TEXT, ltp REAL, open REAL, high REAL, low REAL, prev_close REAL);
CREATE TABLE IF NOT EXISTS quote_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, ltp REAL, qty INTEGER,
  entry_px REAL, stop REAL, target REAL, pnl REAL, pnl_pct REAL);
CREATE INDEX IF NOT EXISTS idx_quote_log_symbol_ts ON quote_log (symbol, ts);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, session TEXT, symbol TEXT, kind TEXT,
  message TEXT, key TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS cash_moves (ts TEXT, amount REAL, note TEXT);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, started TEXT, ended TEXT, status TEXT,
  message TEXT);
"""


def _ts(t: datetime | None = None) -> str:
    return (t or now_ist()).strftime("%Y-%m-%d %H:%M:%S")


class LiveBook:
    """Thin SQLite wrapper; every write is one transaction."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.tx() as con:
            con.executescript(SCHEMA)
            cols = {r[1] for r in con.execute("PRAGMA table_info(runs)").fetchall()}
            if "pid" not in cols:  # books created before the Stop button
                con.execute("ALTER TABLE runs ADD COLUMN pid INTEGER")

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """Connection in a transaction (commit on success)."""
        con = sqlite3.connect(self.path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _df(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        with self.tx() as con:
            return pd.read_sql_query(sql, con, params=params)

    # ------------------------------------------------------------------ meta / state
    def get(self, key: str, default: Any = None) -> Any:
        """JSON value from ``meta``."""
        with self.tx() as con:
            row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def put(self, key: str, value: Any, con: sqlite3.Connection | None = None) -> None:
        """Store a JSON value in ``meta``."""
        sql = "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)"
        if con is not None:
            con.execute(sql, (key, json.dumps(value)))
            return
        with self.tx() as c:
            c.execute(sql, (key, json.dumps(value)))

    @property
    def started(self) -> bool:
        """True once the account has been opened (first end-of-day run)."""
        return self.get("state") is not None

    def state(self) -> SwingState | None:
        """Engine state after the last end-of-day run."""
        d = self.get("state")
        return None if d is None else SwingState.from_json(d)

    def capital(self, default: float) -> float:
        """Starting capital (set in the app; used when the account is opened)."""
        return float(self.get("capital", default))

    def save_live_positions(self, session: str, positions: list[dict[str, Any]]) -> None:
        """Today's provisional positions, as the live poll sees them intraday (today's live
        buys added, today's live sells/exits removed). Overwritten by the official book at
        the next end-of-day run; used only to show something live before then."""
        self.put("live_positions", {"session": session, "positions": positions})

    def live_positions(self) -> list[dict[str, Any]]:
        """Today's provisional positions from the live poll, if the official book (``state``)
        hasn't caught up to today yet; else [] (the official positions are already current)."""
        st = self.state()
        d = self.get("live_positions")
        if d is None or st is None:
            return []
        if str(st.last_session) == str(d.get("session")):
            return []  # EOD already ran for today's session - state() is authoritative
        return list(d.get("positions") or [])

    def reset(self, capital: float) -> None:
        """Delete everything and set a new starting capital (next EOD opens the account)."""
        with self.tx() as con:
            for t in (
                "meta",
                "actions",
                "fills",
                "trades",
                "equity",
                "quotes",
                "events",
                "cash_moves",
            ):
                con.execute(f"DELETE FROM {t}")
            self.put("capital", float(capital), con)

    def move_cash(self, amount: float, note: str) -> SwingState:
        """Add (+) or withdraw (−) paper cash; takes effect immediately."""
        st = self.state()
        if st is None:
            raise ValueError("The account is not open yet — run the end-of-day update first.")
        if amount < 0 and -amount > st.cash:
            raise ValueError(f"Only ₹{st.cash:,.0f} cash is free (the rest is in stocks).")
        st.cash += amount
        st.equity += amount
        for b in st.buys:  # planned buys are sized on the equity at decision time
            b["equity"] = float(b["equity"]) + amount
        with self.tx() as con:
            self.put("state", st.to_json(), con)
            con.execute("INSERT INTO cash_moves VALUES (?, ?, ?)", (_ts(), float(amount), note))
        return st

    # ------------------------------------------------------------------ end of day
    def save_eod(
        self,
        state: SwingState,
        daily: pd.DataFrame,
        fills: pd.DataFrame,
        trades: pd.DataFrame,
        actions: list[dict[str, Any]],
        nifty: pd.Series | None = None,
    ) -> None:
        """Persist one end-of-day run atomically."""
        with self.tx() as con:
            self.put("state", state.to_json(), con)
            self.put("last_eod", _ts(), con)
            for d, row in zip(daily.index, daily.to_dict("records"), strict=True):
                r = row
                con.execute(
                    "INSERT OR REPLACE INTO equity VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(pd.Timestamp(str(d)).date()),
                        float(r["equity"]),
                        float(r["cash"]),
                        float(r["invested"]),
                        int(r["positions"]),
                        float(r["open_risk"]),
                        None if nifty is None else float(nifty.get(d, float("nan"))),
                    ),
                )
            for fr in fills.to_dict("records"):
                r = fr
                con.execute(
                    "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(pd.Timestamp(r["date"]).date()),
                        r["symbol"],
                        r["side"],
                        int(r["qty"]),
                        float(r["px"]),
                        float(r["value"]),
                        float(r["cost"]),
                        r["reason"],
                        int(r["trade_id"]),
                    ),
                )
            for tr in trades.to_dict("records"):
                r = tr
                con.execute(
                    "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        int(r["trade_id"]),
                        r["symbol"],
                        str(pd.Timestamp(r["entry_date"]).date()),
                        float(r["entry_px"]),
                        int(r["qty"]),
                        str(pd.Timestamp(r["exit_date"]).date()),
                        float(r["exit_px"]),
                        r["exit_reason"],
                        int(r["sessions"]),
                        float(r["costs"]),
                        float(r["pnl"]),
                        float(r["return_pct"]),
                        float(r["r_multiple"]),
                        json.dumps({k: str(v) for k, v in r.items()}),
                    ),
                )
            for a in actions:
                self.add_action(a, con)

    def add_action(self, a: dict[str, Any], con: sqlite3.Connection | None = None) -> None:
        """Insert an action (ignored if its ``key`` exists — idempotent re-runs)."""
        row = (
            _ts(),
            a["session"],
            a.get("at", ""),
            a["symbol"],
            a["action"],
            int(a.get("qty") or 0),
            a.get("order_type", ""),
            None if a.get("price") is None else float(a["price"]),
            a.get("reason", ""),
            a.get("status", "Planned"),
            json.dumps(a.get("detail", {})),
            a["key"],
        )
        sql = (
            "INSERT OR IGNORE INTO actions (created, session, at, symbol, action, qty, order_type,"
            " price, reason, status, detail, key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        if con is not None:
            con.execute(sql, row)
            return
        with self.tx() as c:
            c.execute(sql, row)

    def set_action(self, key: str, status: str, **kw: Any) -> None:
        """Update an action's status (and optionally qty / price / reason)."""
        sets, vals = ["status = ?"], [status]
        for k in ("qty", "price", "reason", "order_type"):
            if k in kw and kw[k] is not None:
                sets.append(f"{k} = ?")
                vals.append(kw[k])
        with self.tx() as con:
            con.execute(f"UPDATE actions SET {', '.join(sets)} WHERE key = ?", (*vals, key))

    def close_sessions(self, sessions: list[str]) -> None:
        """Before the official fills are applied: default outcomes for replayed sessions."""
        with self.tx() as con:
            for d in sessions:
                con.execute(
                    "UPDATE actions SET status = 'Done (paper)' WHERE session = ?"
                    " AND action LIKE 'MODIFY%'",
                    (d,),
                )
                con.execute(
                    "UPDATE actions SET status = 'Not filled (gap-up above the limit, no cash"
                    " or upper circuit)' WHERE session = ? AND action = 'BUY'",
                    (d,),
                )
                con.execute(
                    "UPDATE actions SET status = 'Not filled (lower circuit - retried next"
                    " session)' WHERE session = ? AND action = 'SELL' AND key LIKE '%|SELL|%'",
                    (d,),
                )

    # ------------------------------------------------------------------ live
    def save_quotes(self, q: pd.DataFrame) -> None:
        """Latest quote per symbol (price only; see ``log_positions`` for the P&L history)."""
        ts = _ts()
        with self.tx() as con:
            for r in q.to_dict("records"):
                sym = r["symbol"]
                con.execute(
                    "INSERT OR REPLACE INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (sym, ts, r["ltp"], r["open"], r["high"], r["low"], r["prev_close"]),
                )

    def log_positions(self, quotes_by_symbol: dict[str, float], positions: list[dict[str, Any]]) -> None:
        """One dated ``quote_log`` row per currently-held symbol (official + today's live,
        provisional buys), with unrealised P&L at this poll's price, so the trade log shows
        exactly when each price came in and how P&L moved between polls."""
        ts = _ts()
        with self.tx() as con:
            for p in positions:
                sym = str(p["symbol"])
                ltp = quotes_by_symbol.get(sym)
                if ltp is None:
                    continue
                qty, entry = int(p["qty"]), float(p["entry_px"])
                stop = float(p["stop"])
                target = None if p.get("partial") else float(p.get("target"))
                pnl = (float(ltp) - entry) * qty
                pnl_pct = float(ltp) / entry - 1 if entry else None
                con.execute(
                    "INSERT INTO quote_log (ts, symbol, ltp, qty, entry_px, stop, target, pnl,"
                    " pnl_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, sym, ltp, qty, entry, stop, target, pnl, pnl_pct),
                )

    def event(self, session: str, symbol: str, kind: str, message: str, key: str) -> bool:
        """Record a live event once. True if it is new (→ send an alert)."""
        with self.tx() as con:
            cur = con.execute(
                "INSERT OR IGNORE INTO events (ts, session, symbol, kind, message, key)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (_ts(), session, symbol, kind, message, key),
            )
            return cur.rowcount > 0

    def run_start(self, kind: str, pid: int | None = None) -> int:
        """Log the start of an EOD / poll run (with the process id, for the Stop button)."""
        with self.tx() as con:
            cur = con.execute(
                "INSERT INTO runs (kind, started, status, pid) VALUES (?, ?, 'running', ?)",
                (kind, _ts(), pid if pid is not None else os.getpid()),
            )
            return int(cur.lastrowid or 0)

    def active_run(self, kind: str) -> dict[str, Any] | None:
        """The run of ``kind`` that is really still going (its process is alive), else None.
        Runs whose process has died (crash, PC restart, killed) are closed as 'stopped'."""
        from quantswing.live.proc import pid_alive

        r = self.runs(kind, 5)
        live: dict[str, Any] | None = None
        for rec in r.to_dict("records"):
            row = {str(k): v for k, v in rec.items()}
            if row["status"] != "running":
                continue
            pid = row.get("pid")
            if pid is not None and pid == pid and pid_alive(int(pid)):
                live = live or row
            else:
                self.run_end(int(row["id"]), "stopped", "process ended without finishing")
        return live

    def run_end(self, rid: int, status: str, message: str = "") -> None:
        """Log the end of a run."""
        with self.tx() as con:
            con.execute(
                "UPDATE runs SET ended = ?, status = ?, message = ? WHERE id = ?",
                (_ts(), status, message[:2000], rid),
            )

    # ------------------------------------------------------------------ reads (dashboard)
    def actions(self, session: str | None = None) -> pd.DataFrame:
        """Action list (newest session first)."""
        if session:
            return self._df("SELECT * FROM actions WHERE session = ? ORDER BY id", (session,))
        return self._df("SELECT * FROM actions ORDER BY session DESC, id")

    def fills(self) -> pd.DataFrame:
        """Official fills."""
        return self._df("SELECT * FROM fills ORDER BY date, rowid")

    def trades(self) -> pd.DataFrame:
        """Closed trades."""
        return self._df("SELECT * FROM trades ORDER BY exit_date DESC, trade_id DESC")

    def equity(self) -> pd.DataFrame:
        """Daily equity."""
        return self._df("SELECT * FROM equity ORDER BY date")

    def quotes(self) -> pd.DataFrame:
        """Latest live quotes."""
        return self._df("SELECT * FROM quotes")

    def quote_log(self, symbol: str | None = None, limit: int = 2000) -> pd.DataFrame:
        """Every poll's price (and, for held stocks, unrealised P&L at that moment),
        newest first — the detailed trade log across polls."""
        if symbol:
            return self._df(
                "SELECT * FROM quote_log WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                (symbol, limit),
            )
        return self._df("SELECT * FROM quote_log ORDER BY id DESC LIMIT ?", (limit,))

    def events(self, limit: int = 200) -> pd.DataFrame:
        """Latest live events."""
        return self._df("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    def runs(self, kind: str | None = None, limit: int = 20) -> pd.DataFrame:
        """Recent runs."""
        if kind:
            return self._df(
                "SELECT * FROM runs WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit)
            )
        return self._df("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))

    def cash_moves(self) -> pd.DataFrame:
        """Deposits / withdrawals."""
        return self._df("SELECT * FROM cash_moves ORDER BY ts")
