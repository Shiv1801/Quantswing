"""Event-driven daily backtester (05 §2) with the exit/sizing rules of 04.

Timeline for session t (all prices split/bonus-adjusted):
1. **Open** — positions flagged at the previous close (time stop / max hold) exit at the
   open. Stops: open ≤ SL → exit at the open (gap-through, 04 §2); else low ≤ SL → exit at
   SL. Partial: high ≥ entry + partial_at_r·R → sell ``partial_fraction`` at
   max(open, target), SL → breakeven; if the same bar's low also breaches the new SL the
   rest exits there (worst case, 05 §2). Pending entries (decided at t−1 close) fill at the
   open unless the gap exceeds ``max_entry_gap_sigma``·σ60.
2. **Close** — trailing SL from the highest close ``k·σ20·P`` (k tightens above
   ``trail_switch_r``), time-stop / max-hold flags, mark-to-market, then new signals for
   t+1 from the score panel.

Sizing (04 §5): ``qty = min(risk, vol, capacity, single-name, cash) × dd_mult`` (DD from the
rolling ``dd_peak_window_sessions`` equity peak — D26; entries pause at ``dd_stop_new_entries``), then
trimmed to fit the portfolio heat cap. Kelly and regime multipliers join in P4/P5.
Settlement T+1: sale proceeds become usable the next session. Idle cash accrues
``costs.idle_cash_annual_yield`` daily (04 §2), consistent with the Sharpe hurdle.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantswing.core.config import AppConfig, BaselineCfg, ExitsCfg
from quantswing.execution.costs import CostModel
from quantswing.execution.slippage import SlippageModel


@dataclass(frozen=True)
class MarketData:
    """Aligned panels (session index × symbol) the engine reads."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    sigma_sl: pd.DataFrame  # σ used for the initial stop and gap rule (60d)
    sigma_trail: pd.DataFrame  # σ used for the trail (20d)
    sigma_impact: pd.DataFrame  # σ used in the impact term (20d)
    adv_value: pd.DataFrame  # median traded value (₹)


@dataclass
class Position:
    """An open position."""

    symbol: str
    qty: int
    qty0: int
    entry_date: pd.Timestamp
    entry_px: float
    sl0: float
    sl: float
    unit: float  # R per share
    highest_close: float
    max_high: float
    min_low: float
    buy_cost: float
    sell_cost: float = 0.0
    gross: float = 0.0  # realised gross P&L of exited legs
    exit_value: float = 0.0
    exit_qty: int = 0
    partial_done: bool = False
    held: int = 0
    flag: str | None = None


@dataclass
class BacktestResult:
    """Engine outputs."""

    trades: pd.DataFrame
    daily: pd.DataFrame  # equity, cash, invested, positions, heat
    params: dict[str, object] = field(default_factory=dict)


class Backtester:
    """Runs one strategy over a score panel."""

    def __init__(
        self,
        md: MarketData,
        scores: pd.DataFrame,
        gate: pd.DataFrame,
        cfg: AppConfig,
        cost_multiplier: float = 1.0,
        *,
        exits: ExitsCfg | None = None,
        entry: BaselineCfg | None = None,
        entry_level: pd.DataFrame | None = None,
        confirm: pd.DataFrame | None = None,
        label: str = "",
        exposure: pd.Series | None = None,
        delisted_last: pd.Series | None = None,
        delisted_haircut: pd.Series | None = None,
    ) -> None:
        """``exits``/``entry`` override the config defaults (per-sleeve fitted params).

        ``entry_level`` (set at signal close t): stop-entry price for t+1 — fill only if the
        day's high reaches it, at max(open, level); NaN = market-on-open.
        ``confirm``: boolean panel that must be True on the entry day (e.g. volume surge).
        ``delisted_last``: symbol → last traded date (NSE, D31); a position still open after
        it is sold at last close × (1 − ``delist_exit_haircut``).
        """
        self.md = md
        self.scores = scores.reindex_like(md.close)
        self.gate = gate.reindex_like(md.close).fillna(0).astype(bool)
        self.cfg = cfg
        self.exits = exits or cfg.backtest.exits
        self.entry = entry or cfg.backtest.baseline
        self.label = label
        self._expo = (
            None if exposure is None else exposure.reindex(md.close.index).to_numpy(dtype="float64")
        )
        self._lvl = (
            None
            if entry_level is None
            else entry_level.reindex_like(md.close).to_numpy(dtype="float64")
        )
        self._conf = (
            None
            if confirm is None
            else confirm.reindex_like(md.close).fillna(False).to_numpy(dtype=bool)
        )
        self.costs = CostModel(cfg.costs, cost_multiplier)
        self.slip = SlippageModel(cfg.costs.slippage, cost_multiplier)
        f64 = "float64"
        self._adv: np.ndarray = md.adv_value.to_numpy(dtype=f64)
        self._simp: np.ndarray = md.sigma_impact.to_numpy(dtype=f64)
        self._ssl: np.ndarray = md.sigma_sl.to_numpy(dtype=f64)
        self._str: np.ndarray = md.sigma_trail.to_numpy(dtype=f64)
        idx = md.close.index
        self._last_i = np.full(md.close.shape[1], len(idx) + 1)
        if delisted_last is not None:
            dl = delisted_last.reindex(md.close.columns)
            ok = dl.notna().to_numpy()
            self._last_i[ok] = idx.searchsorted(pd.DatetimeIndex(dl[ok]), side="right") - 1
        self._haircut = np.full(md.close.shape[1], cfg.backtest.execution.delist_exit_haircut)
        if delisted_haircut is not None:
            h = delisted_haircut.reindex(md.close.columns).to_numpy(dtype="float64")
            self._haircut = np.where(np.isfinite(h), h, self._haircut)

    # ------------------------------------------------------------------ helpers
    def _sell(
        self, pos: Position, qty: int, px_raw: float, i: int, j: int, sold_today: set[str]
    ) -> float:
        """Sell ``qty`` at ``px_raw`` (pre-slippage). Returns net proceeds."""
        adv_v = float(self._adv[i, j])
        adv_sh = adv_v / px_raw if px_raw > 0 and np.isfinite(adv_v) else 0.0
        px = self.slip.price("sell", px_raw, qty, adv_sh, float(self._simp[i, j]))
        c = self.costs.fill("sell", qty, px, first_sell_today=pos.symbol not in sold_today).total
        sold_today.add(pos.symbol)
        pos.qty -= qty
        pos.exit_qty += qty
        pos.exit_value += qty * px
        pos.gross += qty * (px - pos.entry_px)
        pos.sell_cost += c
        return qty * px - c

    def _record(self, pos: Position, exit_date: pd.Timestamp, reason: str) -> dict[str, object]:
        net = pos.gross - pos.buy_cost - pos.sell_cost
        risk = pos.qty0 * pos.unit
        mfe_r = (pos.max_high - pos.entry_px) / pos.unit
        r_mult = net / risk if risk > 0 else float("nan")
        return {
            "symbol": pos.symbol,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "entry_px": pos.entry_px,
            "exit_px": pos.exit_value / pos.exit_qty if pos.exit_qty else float("nan"),
            "qty": pos.qty0,
            "sl0": pos.sl0,
            "unit": pos.unit,
            "gross_pnl": pos.gross,
            "costs": pos.buy_cost + pos.sell_cost,
            "net_pnl": net,
            "r_mult": r_mult,
            "mfe_r": mfe_r,
            "mae_r": (pos.min_low - pos.entry_px) / pos.unit,
            "capture": (pos.gross / risk) / mfe_r if mfe_r > 0 and risk > 0 else float("nan"),
            "holding": pos.held,
            "reason": reason,
            "partial": pos.partial_done,
            "sleeve": self.label,
        }

    # ------------------------------------------------------------------ main loop
    def run(
        self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> BacktestResult:
        """Simulate over [start, end] (session index)."""
        cfg, md = self.cfg, self.md
        ex, sz, bl, rk = self.exits, cfg.backtest.sizing, self.entry, cfg.risk
        idx = md.close.index
        cols = list(md.close.columns)
        col_of = {s: k for k, s in enumerate(cols)}
        i0 = 0 if start is None else int(idx.searchsorted(start))
        i1 = len(idx) - 1 if end is None else int(idx.searchsorted(end, side="right")) - 1
        op, hi, lw, cl = (x.to_numpy(dtype="float64") for x in (md.open, md.high, md.low, md.close))
        sc, gt = self.scores.to_numpy(dtype="float64"), self.gate.to_numpy(dtype=bool)
        cash = float(rk.paper_capital_inr)
        unsettled = 0.0
        idle_daily = (1 + cfg.costs.idle_cash_annual_yield) ** (1 / 252) - 1
        equity = cash
        eq_hist: deque[float] = deque([cash], maxlen=rk.dd_peak_window_sessions)
        last_close = np.full(len(cols), np.nan)
        positions: dict[str, Position] = {}
        pending: list[str] = []
        cooldown: dict[str, int] = {}
        trades: list[dict[str, object]] = []
        daily: list[dict[str, object]] = []

        for i in range(i0, i1 + 1):
            d = idx[i]
            cash += unsettled  # T+1: yesterday's sale proceeds usable today
            unsettled = 0.0
            if i > i0:  # idle cash earns the liquid-fund yield overnight (04 §2)
                cash *= 1 + idle_daily
            sold_today: set[str] = set()
            # ---------------- open: exits
            for sym in list(positions):
                pos, j = positions[sym], col_of[sym]
                o, h, lo, c = op[i, j], hi[i, j], lw[i, j], cl[i, j]
                if i > self._last_i[j]:  # delisted: exit at a haircut to the last close
                    px_d = last_close[j] if np.isfinite(last_close[j]) else pos.entry_px
                    px_d *= 1 - self._haircut[j]
                    unsettled += self._sell(pos, pos.qty, px_d, i, j, sold_today)
                    trades.append(self._record(pos, d, "delisted"))
                    del positions[sym]
                    continue
                if not np.isfinite(o):
                    continue
                reason = None
                if pos.flag:
                    unsettled += self._sell(pos, pos.qty, o, i, j, sold_today)
                    reason = pos.flag
                elif o <= pos.sl:
                    unsettled += self._sell(pos, pos.qty, o, i, j, sold_today)
                    reason = "sl_gap" if pos.sl < pos.entry_px else "trail_gap"
                elif lo <= pos.sl:
                    unsettled += self._sell(pos, pos.qty, pos.sl, i, j, sold_today)
                    reason = "sl" if pos.sl < pos.entry_px else "trail"
                else:
                    tgt = pos.entry_px + ex.partial_at_r * pos.unit
                    if not pos.partial_done and h >= tgt:
                        q = math.floor(pos.qty0 * ex.partial_fraction)
                        if 0 < q < pos.qty:
                            unsettled += self._sell(pos, q, max(o, tgt), i, j, sold_today)
                        pos.partial_done = True
                        if ex.move_sl_to_breakeven:
                            pos.sl = max(pos.sl, pos.entry_px)
                        if lo <= pos.sl:  # worst case: rest stopped on the same bar
                            unsettled += self._sell(pos, pos.qty, pos.sl, i, j, sold_today)
                            reason = "breakeven"
                pos.max_high = max(pos.max_high, h)
                pos.min_low = min(pos.min_low, lo)
                if reason or pos.qty <= 0:
                    trades.append(self._record(pos, d, reason or "partial_all"))
                    del positions[sym]
                    if reason in ("sl", "sl_gap"):
                        cooldown[sym] = i + ex.reentry_cooldown_sessions
                    continue
                pos.held += 1
            # ---------------- open: entries
            peak = max(eq_hist)
            dd = 1 - equity / peak if peak > 0 else 0.0
            halted = dd >= rk.dd_stop_new_entries
            dd_mult = min(max(1 - dd / rk.dd_mult_full_scale_dd, 0.0), 1.0)
            if self._expo is not None and i > 0:  # regime multiplier known at the prior close
                reg_m = self._expo[i - 1]
                if np.isfinite(reg_m):
                    dd_mult *= reg_m
                    halted = halted or reg_m <= 0
            for sym in pending:
                if sym in positions or len(positions) >= rk.max_open_positions or halted:
                    continue
                j = col_of[sym]
                o, prev_c = op[i, j], cl[i - 1, j] if i > 0 else np.nan
                sig = float(self._ssl[i - 1, j]) if i > 0 else float("nan")
                if not (np.isfinite(o) and np.isfinite(prev_c) and np.isfinite(sig) and sig > 0):
                    continue
                if self._conf is not None and not self._conf[i, j]:
                    continue
                lvl = self._lvl[i - 1, j] if self._lvl is not None and i > 0 else np.nan
                if np.isfinite(lvl):  # stop-entry: needs the high to reach the level
                    if not (np.isfinite(hi[i, j]) and hi[i, j] >= lvl):
                        continue
                    if o / lvl - 1 > bl.max_entry_gap_sigma * sig:
                        continue
                    o = max(o, lvl)
                elif o / prev_c - 1 > bl.max_entry_gap_sigma * sig:
                    continue
                adv_v = float(self._adv[i - 1, j])
                adv_sh = adv_v / o if np.isfinite(adv_v) else 0.0
                s_imp = float(self._simp[i, j])
                # size on a slippage-inclusive price estimate (impact at the risk-sized qty)
                q_guess = (
                    equity * rk.risk_per_trade / (min(ex.k_sl_sigma * sig, rk.sl_hard_cap) * o)
                )
                px_est = self.slip.price("buy", o, q_guess, adv_sh, s_imp)
                unit_est = min(ex.k_sl_sigma * sig, rk.sl_hard_cap) * px_est
                q_risk = equity * rk.risk_per_trade / unit_est
                q_vol = equity * sz.target_position_vol / (sig * math.sqrt(252) * px_est)
                q_cap = rk.capacity_adv_fraction * adv_v / px_est if np.isfinite(adv_v) else 0.0
                q_name = rk.single_name_max * equity / px_est
                q_cash = cash / (px_est * 1.01)
                qty = math.floor(min(q_risk, q_vol, q_cap, q_name, q_cash) * dd_mult)
                open_risk = sum(max(p.entry_px - p.sl, 0.0) * p.qty for p in positions.values())
                room = rk.max_heat * equity - open_risk
                qty = min(qty, math.floor(room / unit_est)) if room > 0 else 0
                if qty < 1:
                    continue
                px = self.slip.price("buy", o, qty, adv_sh, s_imp)
                cost = self.costs.fill("buy", qty, px).total
                if qty * px + cost > cash:
                    continue
                cash -= qty * px + cost
                sl = px * (1 - min(ex.k_sl_sigma * sig, rk.sl_hard_cap))
                positions[sym] = Position(
                    symbol=sym,
                    qty=qty,
                    qty0=qty,
                    entry_date=d,
                    entry_px=px,
                    sl0=sl,
                    sl=sl,
                    unit=px - sl,
                    highest_close=px,
                    max_high=px,
                    min_low=px,
                    buy_cost=cost,
                )
                p_new = positions[sym]
                if np.isfinite(lw[i, j]):
                    p_new.min_low = min(px, lw[i, j])
                if np.isfinite(hi[i, j]):
                    p_new.max_high = max(px, hi[i, j])
                if np.isfinite(lw[i, j]) and lw[i, j] <= sl:  # worst case: stopped the same day
                    unsettled += self._sell(p_new, p_new.qty, sl, i, j, sold_today)
                    trades.append(self._record(p_new, d, "sl_entry_day"))
                    del positions[sym]
                    cooldown[sym] = i + ex.reentry_cooldown_sessions
            pending = []
            # ---------------- close: trail, flags, mark-to-market
            last_close = np.where(np.isfinite(cl[i]), cl[i], last_close)
            invested = 0.0
            for sym, pos in positions.items():
                j = col_of[sym]
                c = cl[i, j]
                if np.isfinite(c):
                    pos.highest_close = max(pos.highest_close, c)
                    r_now = (c - pos.entry_px) / pos.unit
                    k = (
                        ex.trail_k_above_switch
                        if r_now >= ex.trail_switch_r
                        else ex.trail_k_below_switch
                    )
                    s20 = float(self._str[i, j])
                    if np.isfinite(s20):
                        pos.sl = max(pos.sl, pos.highest_close * (1 - k * s20))
                    if pos.held >= ex.max_hold_sessions:
                        pos.flag = "max_hold"
                    elif pos.held >= ex.time_stop_sessions and r_now < ex.time_stop_min_r:
                        pos.flag = "time"
                invested += pos.qty * (
                    last_close[j] if np.isfinite(last_close[j]) else pos.entry_px
                )
            equity = cash + unsettled + invested
            eq_hist.append(equity)
            heat = sum(max(p.entry_px - p.sl, 0.0) * p.qty for p in positions.values())
            daily.append(
                {
                    "date": d,
                    "equity": equity,
                    "cash": cash + unsettled,
                    "invested": invested,
                    "positions": len(positions),
                    "halted": halted,
                    "heat": heat / equity if equity > 0 else 0.0,
                }
            )
            # ---------------- close: signals for t+1
            if i < i1:
                row = sc[i]
                ok = gt[i] & np.isfinite(row) & (row >= bl.min_score)
                cand = np.flatnonzero(ok)
                if len(cand):
                    order = cand[np.argsort(-row[cand], kind="stable")]
                    free = rk.max_open_positions - len(positions)
                    for j in order:
                        if len(pending) >= min(bl.max_new_per_day, free):
                            break
                        sym = cols[j]
                        if sym in positions or cooldown.get(sym, -1) >= i:
                            continue
                        pending.append(sym)
        # ---------------- end of data: close out at last close
        d_end = idx[i1]
        for sym, pos in list(positions.items()):
            j = col_of[sym]
            px = last_close[j] if np.isfinite(last_close[j]) else pos.entry_px
            self._sell(pos, pos.qty, px, i1, j, set())
            trades.append(self._record(pos, d_end, "end_of_data"))
        tr = pd.DataFrame(trades)
        dl = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame()
        return BacktestResult(trades=tr, daily=dl)
