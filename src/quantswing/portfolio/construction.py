"""Ranked factor portfolio (D29 → D33): hold the top composite names, rebalance weekly.

D33 (default): only the ``universe_top_n`` most traded stocks; up to ``top_n`` names with
a positive score, a fixed slot each (``equal_slot``; unfilled slots stay in cash — the
score decides exposure); no volatility target, no regime multiplier; the D26 drawdown
brake stays. The D29 options (inverse-vol, vol target, HMM regime) remain switchable.

Timeline: scores/gate/regime known at the close of t → on rebalance sessions, trade at the
open of t+1 toward target weights (slippage + full Fyers costs), mark at each close. Idle
cash earns ``costs.idle_cash_annual_yield``. Membership uses a buffer: a holding stays while
it ranks within ``top_n × hold_buffer_mult`` and keeps a score > ``min_score``; new names
fill the book from the top. Target weights per :mod:`sizing` × regime × drawdown multiplier
(D26 rolling-peak DD). No leverage, long-only.

NSE data (D31): new names are not bought inside the results window (``entry_block``);
dividends are credited in cash on the ex-date (``credit_dividends``); a delisted holding is
sold at the open after its last traded session at last close × (1 − ``delist_exit_haircut``).

Circuits (04 §2): a sell on a lower-circuit-locked session (one-price day, down) is
blocked and retried next session; a buy on an upper-circuit-locked session is skipped and
retried next session.

:class:`PortfolioSim` is resumable (:class:`SimState`), so the paper engine (P6) steps the
*same* code one session at a time; :func:`run_portfolio` is the backtest wrapper.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantswing.core.config import AppConfig
from quantswing.execution.costs import CostModel
from quantswing.execution.slippage import SlippageModel
from quantswing.factors.common import Panels, rolling_sigma
from quantswing.factors.nse_factors import OTHER, UNKNOWN, sector_labels
from quantswing.portfolio.covariance import ledoit_wolf
from quantswing.portfolio.sizing import capped_normalise, sector_capped, vol_scale


@dataclass
class PortfolioResult:
    """Daily curve + trade log."""

    daily: pd.DataFrame  # equity, cash, invested, positions, halted, gross_target, regime_mult
    trades: pd.DataFrame  # date, symbol, side, qty, px, value, cost, reason


@dataclass
class Controls:
    """Operator controls for one session's close decision (P6 kill switch, 04 §8)."""

    pause_entries: bool = False  # no NEW names (holdings keep their targets)
    flatten: bool = False  # target zero everywhere at the next open


@dataclass
class SimState:
    """Everything needed to continue a simulation at the next session (JSON-serialisable)."""

    cash: float
    anchor: str  # rebalance schedule anchor (ISO date): rebalance every N sessions from here
    qty: dict[str, float] = field(default_factory=dict)
    last: dict[str, float] = field(default_factory=dict)  # last traded close
    eq_hist: list[float] = field(default_factory=list)
    pending: dict[str, float] | None = None  # target weights for the next open (NaN = no-op)
    target_gross: float = 0.0
    last_session: str | None = None
    peak: dict[str, float] = field(default_factory=dict)  # highest close since entry (trailing)
    cooldown: dict[str, str] = field(default_factory=dict)  # no re-entry until (ISO date)
    entry: dict[str, float] = field(default_factory=dict)  # entry fill price (take-profit)

    def to_json(self) -> dict[str, Any]:
        """Plain dict (NaN → None)."""
        d = asdict(self)
        if d["pending"] is not None:
            d["pending"] = {k: (None if not np.isfinite(v) else v) for k, v in d["pending"].items()}
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> SimState:
        """Inverse of :meth:`to_json`."""
        pend = d.get("pending")
        if pend is not None:
            pend = {k: (np.nan if v is None else float(v)) for k, v in pend.items()}
        return cls(
            cash=float(d["cash"]),
            anchor=str(d["anchor"]),
            qty={k: float(v) for k, v in d.get("qty", {}).items()},
            last={k: float(v) for k, v in d.get("last", {}).items()},
            eq_hist=[float(x) for x in d.get("eq_hist", [])],
            pending=pend,
            target_gross=float(d.get("target_gross", 0.0)),
            last_session=d.get("last_session"),
            peak={k: float(v) for k, v in d.get("peak", {}).items()},
            cooldown={k: str(v) for k, v in d.get("cooldown", {}).items()},
            entry={k: float(v) for k, v in d.get("entry", {}).items()},
        )


def locked(p: Panels) -> tuple[np.ndarray, np.ndarray]:
    """(lower-circuit lock, upper-circuit lock) per session: one-price day, down / up."""
    one = (p.high == p.low) & p.close.notna()
    prev = p.close.shift(1)
    return (one & (p.close < prev)).to_numpy(), (one & (p.close > prev)).to_numpy()


class PortfolioSim:
    """Precomputed inputs + a resumable session loop."""

    def __init__(
        self,
        p: Panels,
        scores: pd.DataFrame,
        gate: pd.DataFrame,
        cfg: AppConfig,
        *,
        regime_mult: pd.Series | None = None,
        cost_multiplier: float = 1.0,
        entry_block: pd.DataFrame | None = None,
        edge_ic: pd.Series | None = None,
        exposure: pd.Series | None = None,
    ) -> None:
        self.cfg = cfg
        pc, ex = cfg.risk.portfolio, cfg.backtest.execution
        self.costs = CostModel(cfg.costs, cost_multiplier)
        self.slip = SlippageModel(cfg.costs.slippage, cost_multiplier)
        self.idx = p.close.index
        self.cols = np.asarray(p.close.columns)
        self.col_of = {str(c): k for k, c in enumerate(self.cols)}
        self.op = p.open.to_numpy(dtype="float64")
        self.cl = p.close.to_numpy(dtype="float64")
        self.sc = scores.reindex_like(p.close).to_numpy(dtype="float64")
        self.gt = gate.reindex_like(p.close).fillna(0).astype(bool).to_numpy()
        self.r = p.r
        self.sig = rolling_sigma(self.r, pc.vol_window).to_numpy(dtype="float64")
        aw = cfg.backtest.sizing.adv_window
        self.adv = p.turnover.rolling(aw, min_periods=aw).median().to_numpy(dtype="float64")
        self.s_imp = rolling_sigma(self.r, ex.impact_vol_window).to_numpy(dtype="float64")
        self.mult = (
            regime_mult.reindex(self.idx).to_numpy(dtype="float64")
            if regime_mult is not None and pc.use_regime
            else np.ones(len(self.idx))
        )
        # B: invested share decided by the score's own recent accuracy (1 = off)
        ex_ = (
            exposure.reindex(self.idx).ffill().to_numpy(dtype="float64")
            if exposure is not None
            else np.ones(len(self.idx))
        )
        self.expo = np.clip(np.where(np.isfinite(ex_), ex_, 1.0), 0.0, 1.0)
        self.liquid = np.ones(p.close.shape, dtype=bool)
        if pc.universe_top_n > 0:  # D33: only the N most traded stocks (median ₹ value)
            w = pc.universe_window
            med = p.turnover.where(p.volume > 0).rolling(w, min_periods=w // 2).median()
            rank = med.rank(axis=1, ascending=False, method="first")
            self.liquid = (rank <= pc.universe_top_n).to_numpy()
        self.blk = (
            entry_block.reindex_like(p.close).fillna(False).astype(bool).to_numpy()
            if entry_block is not None
            else np.zeros(p.close.shape, dtype=bool)
        )
        self.dy = (
            p.div_yield.reindex_like(p.close).fillna(0.0).to_numpy(dtype="float64")
            if ex.credit_dividends and p.div_yield is not None
            else None
        )
        self.last_i = np.full(len(self.cols), len(self.idx) + 1)
        if p.delisted_last is not None:
            dls = p.delisted_last.reindex(self.cols)
            ok = dls.notna().to_numpy()
            self.last_i[ok] = self.idx.searchsorted(pd.DatetimeIndex(dls[ok]), side="right") - 1
        self.haircut = np.full(len(self.cols), ex.delist_exit_haircut)
        if p.delisted_haircut is not None:
            h = p.delisted_haircut.reindex(self.cols).to_numpy(dtype="float64")
            self.haircut = np.where(np.isfinite(h), h, self.haircut)
        self.lower_lock, self.upper_lock = locked(p)
        # v2: point-in-time IC of the score (edge per unit z) and the round-trip switch cost
        ic = (
            edge_ic.reindex(self.idx).ffill().to_numpy(dtype="float64")
            if edge_ic is not None
            else np.full(len(self.idx), pc.edge_ic)
        )
        self.edge_ic = np.where(np.isfinite(ic) & (ic > 0), ic, pc.edge_ic)
        ref_q, ref_px = 100.0, 1000.0
        fees = (
            self.costs.fill("buy", ref_q, ref_px).total
            + self.costs.fill("sell", ref_q, ref_px).total
        )
        self.cost_rt = (
            fees / (ref_q * ref_px)
            + 2 * cfg.costs.slippage.default_half_spread_pct * cost_multiplier
        )
        # final round: sector cap uses NSE sectors (pooled "Other" and missing are not capped)
        lab = sector_labels(p.sector, p.close.columns, cfg.factors.xsec.min_sector_names)
        self.sector = np.where(lab == OTHER, UNKNOWN, lab)

    # ------------------------------------------------------------------ state
    def init_state(self, start: pd.Timestamp | None = None) -> SimState:
        """Fresh account (``paper_capital_inr``) anchored at ``start`` (default: first session)."""
        cap = float(self.cfg.risk.paper_capital_inr)
        i0 = 0 if start is None else int(self.idx.searchsorted(start))
        return SimState(cash=cap, anchor=str(self.idx[i0].date()), eq_hist=[cap])

    def _arr(self, d: dict[str, float], fill: float) -> np.ndarray:
        a = np.full(len(self.cols), fill)
        for k, v in d.items():
            j = self.col_of.get(k)
            if j is not None:
                a[j] = v
        return a

    def _dict(self, a: np.ndarray, keep: np.ndarray) -> dict[str, float]:
        return {str(self.cols[j]): float(a[j]) for j in np.flatnonzero(keep)}

    # ------------------------------------------------------------------ decision
    def _select_buffer(
        self, i: int, held: np.ndarray, s: np.ndarray, blocked: np.ndarray
    ) -> list[int]:
        """D29/D33: keep holdings within top_n × buffer; fill from the top."""
        pc = self.cfg.risk.portfolio
        n = len(self.cols)
        ranked = np.argsort(-np.nan_to_num(s, nan=-np.inf), kind="stable")
        ranked = ranked[np.isfinite(s[ranked])]
        rank_of = np.full(n, np.inf)
        rank_of[ranked] = np.arange(len(ranked))
        book: list[int] = [
            int(j) for j in np.flatnonzero(held) if rank_of[j] < pc.top_n * pc.hold_buffer_mult
        ]
        for j in ranked:
            if len(book) >= pc.top_n:
                break
            if j not in book and not blocked[j]:  # results window / cooldown: no NEW names
                book.append(int(j))
        return book

    def _select_edge(
        self, i: int, held: np.ndarray, s: np.ndarray, blocked: np.ndarray
    ) -> list[int]:
        """v2: expected edge = IC · σ · √h · z (z > min_score). Free slots are filled from
        the top (stay invested); a new name replaces the weakest holding only if its edge
        beats that holding's by more than the round-trip cost."""
        pc = self.cfg.risk.portfolio
        edge = self.edge_ic[i] * self.sig[i] * np.sqrt(pc.edge_horizon) * s
        book = [int(j) for j in np.flatnonzero(held & np.isfinite(s))]
        book.sort(key=lambda j: -edge[j])
        book = book[: pc.max_names]
        cand = np.flatnonzero(np.isfinite(s) & ~held & ~blocked)
        cand = cand[np.argsort(-edge[cand], kind="stable")]
        k = 0
        while k < len(cand) and len(book) < pc.max_names:  # free slots: stay invested
            book.append(int(cand[k]))
            k += 1
        for j in cand[k:]:
            weakest = min(book, key=lambda b: edge[b]) if book else None
            if weakest is None or edge[j] <= edge[weakest] + pc.switch_cost_mult * self.cost_rt:
                break
            book.remove(weakest)
            book.append(int(j))
        return book

    def decide(
        self,
        i: int,
        held: np.ndarray,
        dd: float,
        halted: bool,
        ctl: Controls,
        cooled: np.ndarray | None = None,
    ) -> np.ndarray:
        """Target weights for the next open, decided at the close of session ``i``."""
        pc, rk = self.cfg.risk.portfolio, self.cfg.risk
        n = len(self.cols)
        w = np.zeros(n)
        if ctl.flatten:
            return w
        sc = self.sc[i]
        sg = self.sig[i]
        ok = (
            self.gt[i]
            & self.liquid[i]
            & np.isfinite(sc)
            & (sc > pc.min_score)
            & np.isfinite(sg)
            & (sg > 0)
        )
        s = np.where(ok, sc, np.nan)
        blocked = self.blk[i] | (cooled if cooled is not None else False)
        book = (
            self._select_edge(i, held, s, blocked)
            if pc.selection == "edge"
            else self._select_buffer(i, held, s, blocked)
        )
        if not book:
            return w
        b = np.asarray(book)
        sb = self.sig[i, b]
        good = np.isfinite(sb) & (sb > 0)  # a flat / untraded name gets no weight
        if pc.weighting == "equal_slot":  # D33: fixed slot per name; empty slots = cash
            wb = good * min(pc.max_gross / pc.top_n, rk.single_name_max)
        elif pc.weighting == "conviction":  # v2: w ∝ z/σ (mean-variance with E[r] ∝ IC·σ·z)
            raw = np.where(
                good, np.maximum(np.nan_to_num(sc[b]), 0.0) / np.where(good, sb, 1.0), 0.0
            )
            lab = self.sector[b]

            def alloc(x: np.ndarray) -> np.ndarray:
                if pc.sector_cap > 0:
                    return sector_capped(x, lab, pc.max_gross, pc.max_weight, pc.sector_cap)
                return capped_normalise(x, pc.max_gross, pc.max_weight)

            wb = alloc(raw)
            while (wb > 0).sum() > 1 and wb[wb > 0].min() < pc.min_weight:
                raw = np.where(wb > 0, raw, 0.0)
                raw[np.flatnonzero(wb > 0)[np.argmin(wb[wb > 0])]] = 0.0
                wb = alloc(raw)
        else:
            raw = (
                np.where(good, 1.0 / np.where(good, sb, 1.0), 0.0)
                if pc.weighting == "inverse_vol"
                else good.astype(float)
            )
            wb = capped_normalise(raw, pc.max_gross, rk.single_name_max)
        if pc.vol_target:
            lo = max(0, i - pc.cov_window + 1)
            cov = ledoit_wolf(self.r.iloc[lo : i + 1, b])
            wb = wb * vol_scale(wb, cov, pc.vol_target_level or rk.target_portfolio_vol)
        m_reg = self.mult[i] if np.isfinite(self.mult[i]) else 1.0
        dd_mult = min(max(1 - dd / rk.dd_mult_full_scale_dd, 0.0), 1.0) if pc.dd_brake else 1.0
        if halted or ctl.pause_entries:  # no new names; existing ones kept at reduced size
            wb = np.where(held[b], wb, 0.0)
        w[b] = wb * m_reg * dd_mult * self.expo[i]
        return w

    # ------------------------------------------------------------------ loop
    def run(
        self,
        state: SimState,
        i_from: int,
        i_to: int,
        controls: dict[pd.Timestamp, Controls] | None = None,
    ) -> tuple[SimState, list[dict[str, Any]], list[dict[str, Any]]]:
        """Step sessions ``[i_from, i_to]``. Returns (new state, daily rows, trades)."""
        cfg = self.cfg
        pc, rk = cfg.risk.portfolio, cfg.risk
        idle = (1 + cfg.costs.idle_cash_annual_yield) ** (1 / 252) - 1
        cash = state.cash
        qty = self._arr(state.qty, 0.0)
        last = self._arr(state.last, np.nan)
        eq_hist: deque[float] = deque(state.eq_hist or [cash], maxlen=rk.dd_peak_window_sessions)
        equity = eq_hist[-1]
        pending = None if state.pending is None else self._arr(state.pending, np.nan)
        peak_px = self._arr(state.peak, np.nan)
        entry_px = self._arr(state.entry, np.nan)
        cool_until = np.full(len(self.cols), -1)
        for k, v in state.cooldown.items():
            jc = self.col_of.get(k)
            if jc is not None:
                cool_until[jc] = int(self.idx.searchsorted(pd.Timestamp(v)))
        halt_level = rk.dd_stop_new_entries if pc.dd_brake else pc.halt_dd
        trail_exit = np.zeros(len(self.cols), dtype=bool)  # set at a close, executed next open
        exit_why = np.full(len(self.cols), "", dtype=object)  # "trail" / "target"
        target_gross = state.target_gross
        i_anchor = int(self.idx.searchsorted(pd.Timestamp(state.anchor)))
        started = state.last_session is not None
        daily: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        costs, slip = self.costs, self.slip
        op, cl, adv, s_imp = self.op, self.cl, self.adv, self.s_imp

        def trade(
            d: pd.Timestamp, j: Any, side: str, q: float, px: float, c: float, why: str
        ) -> None:
            trades.append(
                {
                    "date": d,
                    "symbol": self.cols[j],
                    "side": side,
                    "qty": q,
                    "px": px,
                    "value": q * px,
                    "cost": c,
                    "reason": why,
                }
            )

        for i in range(i_from, i_to + 1):
            d = self.idx[i]
            if started:
                cash *= 1 + idle
            started = True
            # ---------- open: delisted holdings leave the book
            for j in np.flatnonzero((qty > 0) & (i > self.last_i)):
                px = float(last[j]) * (1 - self.haircut[j]) if np.isfinite(last[j]) else 0.0
                c = costs.fill("sell", qty[j], px).total if px > 0 else 0.0
                cash += qty[j] * px - c
                trade(d, j, "delist", qty[j], px, c, "delisted")
                qty[j] = 0
                if pending is not None:
                    pending[j] = np.nan
            # ---------- open: execute pending targets (NaN = no instruction)
            if pending is not None:
                px_open = np.where(np.isfinite(op[i]), op[i], last)
                act = np.isfinite(pending) & ((pending > 0) | (qty > 0))
                touched = np.flatnonzero(act)
                tgt_all = np.array(
                    [
                        (
                            math.floor(pending[j] * equity / px_open[j])
                            if np.isfinite(px_open[j]) and px_open[j] > 0
                            else qty[j]
                        )
                        for j in touched
                    ]
                )
                carry = np.full(len(self.cols), np.nan)
                order = touched[np.argsort(tgt_all - qty[touched], kind="stable")]  # sells first
                for j in order:
                    px = px_open[j]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    tgt_q = math.floor(pending[j] * equity / px)
                    dq = tgt_q - qty[j]
                    if dq == 0 or (abs(dq) * px < pc.min_trade_value_frac * equity and tgt_q > 0):
                        continue
                    if (  # v2: a holding is not resized for small drifts
                        pc.rebalance_band > 0
                        and qty[j] > 0
                        and tgt_q > 0
                        and abs(dq) * px < pc.rebalance_band * pending[j] * equity
                    ):
                        continue
                    side = "buy" if dq > 0 else "sell"
                    if (side == "sell" and self.lower_lock[i, j]) or (
                        side == "buy" and self.upper_lock[i, j]
                    ):
                        carry[j] = pending[j]  # circuit-locked: retry next session
                        continue
                    adv_sh = adv[i - 1, j] / px if i > 0 and np.isfinite(adv[i - 1, j]) else 0.0
                    fill = slip.price(side, px, abs(dq), adv_sh, s_imp[i, j])
                    if side == "buy":
                        dq = min(dq, math.floor(cash / (fill * 1.003)))
                        if dq <= 0:
                            continue
                    c = costs.fill(side, abs(dq), fill).total
                    cash -= dq * fill + c
                    if side == "buy" and qty[j] <= 0:
                        peak_px[j] = fill  # trailing reference starts at entry
                        entry_px[j] = fill
                    qty[j] += dq
                    if qty[j] <= 0:
                        peak_px[j] = np.nan
                        entry_px[j] = np.nan
                    why = exit_why[j] if pending[j] == 0 and trail_exit[j] else "rebalance"
                    trade(d, j, side, abs(dq), fill, c, why)
                pending = carry if np.isfinite(carry).any() else None
            # ---------- close: dividends (ex-date), then mark to market
            if self.dy is not None:
                held = qty > 0
                if held.any():
                    cash += float(np.nansum(qty[held] * last[held] * self.dy[i, held]))
            last = np.where(np.isfinite(cl[i]), cl[i], last)
            invested = float(np.nansum(qty * np.nan_to_num(last)))
            equity = cash + invested
            eq_hist.append(equity)
            peak = max(eq_hist)
            dd = 1 - equity / peak if peak > 0 else 0.0
            halted = dd >= halt_level
            held_mask = qty > 0
            daily.append(
                {
                    "date": d,
                    "equity": equity,
                    "cash": cash,
                    "invested": invested,
                    "positions": int(held_mask.sum()),
                    "halted": halted,
                    "gross_target": target_gross,
                    "regime_mult": self.mult[i],
                }
            )
            # ---------- close: trailing stop (v2 option) → exit at the next open
            trail_exit = np.zeros(len(self.cols), dtype=bool)
            exit_why = np.full(len(self.cols), "", dtype=object)
            if pc.take_profit > 0:  # final round: sell once up take_profit from entry
                tp = (
                    held_mask
                    & np.isfinite(cl[i])
                    & np.isfinite(entry_px)
                    & (cl[i] >= entry_px * (1 + pc.take_profit))
                )
                if tp.any():
                    trail_exit = trail_exit | tp
                    exit_why[tp] = "target"
                    pending = pending if pending is not None else np.full(len(self.cols), np.nan)
                    pending[tp] = 0.0
                    cool_until[tp] = i + pc.reentry_cooldown
            if pc.trail_sigma > 0:
                peak_px = np.where(held_mask, np.fmax(peak_px, cl[i]), np.nan)
                band = pc.trail_sigma * self.sig[i] * np.sqrt(pc.trail_horizon)
                hit = (
                    held_mask
                    & np.isfinite(cl[i])
                    & np.isfinite(band)
                    & (cl[i] < peak_px * (1 - band))
                )
                hit = hit & ~trail_exit
                if hit.any():
                    trail_exit = trail_exit | hit
                    exit_why[hit] = "trail"
                    pending = pending if pending is not None else np.full(len(self.cols), np.nan)
                    pending[hit] = 0.0
                    cool_until[hit] = i + pc.reentry_cooldown
            # ---------- close: decide the next open
            ctl = (controls or {}).get(d, Controls())
            if (i - i_anchor) % pc.rebalance_every == 0 or ctl.flatten:
                w = self.decide(i, held_mask & ~trail_exit, dd, halted, ctl, cool_until > i)
                target_gross = float(w.sum())
                if pending is not None:  # keep circuit retries / trailing exits
                    keep = np.isfinite(pending) & (w == 0) & held_mask
                    w = np.where(keep, pending, w)
                    w = np.where(trail_exit, 0.0, w)
                pending = w
        new = SimState(
            cash=cash,
            anchor=state.anchor,
            qty=self._dict(qty, qty > 0),
            last=self._dict(last, np.isfinite(last)),
            eq_hist=list(eq_hist),
            pending=(None if pending is None else self._dict(pending, np.isfinite(pending))),
            target_gross=target_gross,
            last_session=str(self.idx[i_to].date()) if i_to >= i_from else state.last_session,
            peak=self._dict(peak_px, np.isfinite(peak_px) & (qty > 0)),
            entry=self._dict(entry_px, np.isfinite(entry_px) & (qty > 0)),
            cooldown={
                str(self.cols[j]): str(self.idx[min(cool_until[j], len(self.idx) - 1)].date())
                for j in np.flatnonzero(cool_until > (i_to if i_to >= i_from else -1))
            },
        )
        return new, daily, trades


def run_portfolio(
    p: Panels,
    scores: pd.DataFrame,
    gate: pd.DataFrame,
    cfg: AppConfig,
    *,
    regime_mult: pd.Series | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    cost_multiplier: float = 1.0,
    entry_block: pd.DataFrame | None = None,
    edge_ic: pd.Series | None = None,
    exposure: pd.Series | None = None,
) -> PortfolioResult:
    """Simulate the ranked portfolio over [start, end] (backtest)."""
    sim = PortfolioSim(
        p,
        scores,
        gate,
        cfg,
        regime_mult=regime_mult,
        cost_multiplier=cost_multiplier,
        entry_block=entry_block,
        edge_ic=edge_ic,
        exposure=exposure,
    )
    idx = sim.idx
    i0 = 0 if start is None else int(idx.searchsorted(start))
    i1 = len(idx) - 1 if end is None else int(idx.searchsorted(end, side="right")) - 1
    _, daily, trades = sim.run(sim.init_state(idx[i0] if len(idx) else None), i0, i1)
    dl = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame()
    return PortfolioResult(dl, pd.DataFrame(trades))
