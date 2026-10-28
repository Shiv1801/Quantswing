"""D37 trade engine: the score picks, maths sizes, and every trade has SL / target / TSL.

Daily timeline (all decisions at a close, all orders at or after the next open):

* **Close of t** — mark to market, raise trailing stops, flag score / time exits, pick new
  buys (z ≥ ``z_entry``, expected edge IC·σ₁₀·z > cost), and at most
  ``max_replacements`` swaps of the weakest holding (only if the edge gain > 2× cost).
* **Open of t+1** — (1) resting stops that the open gapped through fill at the open;
  (2) flagged exits sell at the open; (3) buys fill at the open unless it gapped up more
  than ``gap_skip_sigma`` × daily σ (no chasing) or the stock is upper-circuit locked.
  A buy sets SL = fill − 1.5σ₁₀, target = fill + 3σ₁₀; quantity = risk ₹ ÷ (fill − SL),
  capped at ``max_weight`` of capital and by cash.
* **During t+1** — a resting SL-M order fills if the low reaches the stop (at the stop,
  minus slippage). A resting target limit order sells ``target_sell_frac`` if the high
  reaches the target; the stop then moves to the entry price (break-even). If both
  levels are inside the same day's range the stop is assumed to fill first (pessimistic).

σ₁₀ = daily σ of log returns (``vol_window`` sessions) × √``horizon``, frozen at entry.
Lower-circuit-locked sessions block every sell (retried next session); delisted holdings
leave at last close × (1 − haircut); dividends are credited on the ex-date; idle cash earns
``costs.idle_cash_annual_yield``. Every fill pays slippage (market / stop orders) and the
full Fyers delivery cost model.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantswing.core.config import AppConfig
from quantswing.execution.costs import CostModel
from quantswing.execution.slippage import SlippageModel
from quantswing.factors.common import Panels, rolling_sigma
from quantswing.factors.macro import Exposures
from quantswing.portfolio.construction import locked


@dataclass
class Position:
    """One open trade (a round trip in the log once closed)."""

    j: int
    trade_id: int
    entry_date: pd.Timestamp
    entry_px: float
    qty0: int
    qty: int
    s10: float  # σ₁₀ at entry (fraction)
    z0: float
    risk_inr: float  # planned ₹ loss at the initial SL
    sl0: float
    target: float
    stop: float
    stop_kind: str = "SL"  # SL → BE (after target) → TSL
    peak: float = 0.0
    partial: bool = False
    held: int = 0  # sessions since entry
    buy_cost: float = 0.0
    sell_value: float = 0.0
    sell_cost: float = 0.0
    dividends: float = 0.0
    partial_date: pd.Timestamp | None = None
    partial_px: float = float("nan")
    partial_qty: int = 0
    exit_flag: str = ""  # set at a close → sold at the next open
    sl_grace_days: int = 0  # D41: captured from config AT ENTRY, so a settings change
    # (e.g. D37 -> T24) never retroactively alters an already-open position's grace period


POS_DATES = ("entry_date", "partial_date")


@dataclass
class SwingState:
    """Everything needed to continue the engine at the next session (JSON-safe).

    Positions are keyed by symbol (not column number), dates are ISO strings, and a
    cooldown is stored as the session the stock was sold (+ ``cooldown`` sessions).
    """

    cash: float
    positions: list[dict[str, Any]] = field(default_factory=list)
    last: dict[str, float] = field(default_factory=dict)  # last traded close per symbol
    cool: dict[str, str] = field(default_factory=dict)  # symbol → session it was sold
    buys: list[dict[str, Any]] = field(default_factory=list)  # planned for the next open
    next_id: int = 1
    equity: float = 0.0
    last_session: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Plain dict."""
        return {
            "cash": self.cash,
            "positions": self.positions,
            "last": self.last,
            "cool": self.cool,
            "buys": self.buys,
            "next_id": self.next_id,
            "equity": self.equity,
            "last_session": self.last_session,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> SwingState:
        """Inverse of :meth:`to_json`."""
        return cls(
            cash=float(d["cash"]),
            positions=list(d.get("positions", [])),
            last={k: float(v) for k, v in d.get("last", {}).items()},
            cool={k: str(v) for k, v in d.get("cool", {}).items()},
            buys=list(d.get("buys", [])),
            next_id=int(d.get("next_id", 1)),
            equity=float(d.get("equity", d["cash"])),
            last_session=d.get("last_session"),
        )


def _pos_to_dict(ps: Position, sym: str) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in ps.__dict__.items() if k != "j"}
    out["symbol"] = sym
    for k in POS_DATES:
        v = out.get(k)
        out[k] = None if v is None or pd.isna(v) else str(pd.Timestamp(v).date())
    for k, v in list(out.items()):
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
    return out


def _pos_from_dict(d: dict[str, Any], j: int) -> Position:
    kw = {k: v for k, v in d.items() if k != "symbol"}
    for k in POS_DATES:
        kw[k] = None if kw.get(k) is None else pd.Timestamp(kw[k])
    if kw.get("partial_px") is None:
        kw["partial_px"] = float("nan")
    return Position(j=j, **kw)


@dataclass
class SwingResult:
    """Daily curve, every fill, every round-trip trade."""

    daily: pd.DataFrame  # equity, cash, invested, positions, open_risk
    fills: pd.DataFrame  # date, symbol, side, qty, px, value, cost, reason, trade_id
    trades: pd.DataFrame  # one row per round trip
    meta: dict[str, Any] = field(default_factory=dict)


class SwingSim:
    """Precomputed arrays + the session loop."""

    def __init__(
        self,
        p: Panels,
        scores: pd.DataFrame,
        gate: pd.DataFrame,
        cfg: AppConfig,
        *,
        edge_ic: pd.Series | None = None,
        cost_multiplier: float = 1.0,
        macro: Exposures | None = None,
        regime_mult: pd.Series | None = None,
    ) -> None:
        self.cfg = cfg
        sw, pc, ex = cfg.risk.swing, cfg.risk.portfolio, cfg.backtest.execution
        self.sw = sw
        self.costs = CostModel(cfg.costs, cost_multiplier)
        self.slip = SlippageModel(cfg.costs.slippage, cost_multiplier)
        self.idx = p.close.index
        self.cols = np.asarray(p.close.columns)
        self.col_of = {str(c): k for k, c in enumerate(self.cols)}
        f = "float64"
        self.op = p.open.to_numpy(dtype=f)
        self.hi = p.high.to_numpy(dtype=f)
        self.lo = p.low.to_numpy(dtype=f)
        self.cl = p.close.to_numpy(dtype=f)
        sc = scores.reindex_like(p.close).to_numpy(dtype=f)
        gt = gate.reindex_like(p.close).fillna(0).astype(bool).to_numpy()
        self.sig = rolling_sigma(p.r, sw.vol_window).to_numpy(dtype=f)  # daily σ
        self.s10 = self.sig * math.sqrt(sw.horizon)
        liquid = np.ones(p.close.shape, dtype=bool)
        if pc.universe_top_n > 0:
            w = pc.universe_window
            med = p.turnover.where(p.volume > 0).rolling(w, min_periods=w // 2).median()
            liquid = (
                med.rank(axis=1, ascending=False, method="first") <= pc.universe_top_n
            ).to_numpy()
        ok_sig = np.isfinite(self.sig) & (self.sig > 0)
        self.elig = gt & liquid & np.isfinite(sc) & ok_sig  # may be BOUGHT
        # cross-sectional z of the score among eligible names (per session)
        e = np.where(self.elig, sc, np.nan)
        with warnings.catch_warnings():  # sessions with no eligible stock → NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(e, axis=1, keepdims=True)
            sd = np.nanstd(e, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (sc - mu) / np.where(sd > 0, sd, np.nan)
        self.z = np.where(gt & np.isfinite(sc), z, np.nan)  # held names keep a z while gated
        aw = cfg.backtest.sizing.adv_window
        self.adv = p.turnover.rolling(aw, min_periods=aw).median().to_numpy(dtype=f)
        self.s_imp = rolling_sigma(p.r, ex.impact_vol_window).to_numpy(dtype=f)
        self.lower_lock, self.upper_lock = locked(p)
        self.dy = (
            p.div_yield.reindex_like(p.close).fillna(0.0).to_numpy(dtype=f)
            if ex.credit_dividends and p.div_yield is not None
            else None
        )
        self.last_i = np.full(len(self.cols), len(self.idx) + 1)
        if p.delisted_last is not None:
            dls = p.delisted_last.reindex(self.cols)
            has = dls.notna().to_numpy()
            self.last_i[has] = self.idx.searchsorted(pd.DatetimeIndex(dls[has]), side="right") - 1
        self.haircut = np.full(len(self.cols), ex.delist_exit_haircut)
        if p.delisted_haircut is not None:
            h = p.delisted_haircut.reindex(self.cols).to_numpy(dtype=f)
            self.haircut = np.where(np.isfinite(h), h, self.haircut)
        ic = (
            edge_ic.reindex(self.idx).ffill().to_numpy(dtype=f)
            if edge_ic is not None
            else np.full(len(self.idx), pc.edge_ic)
        )
        self.ic = np.where(np.isfinite(ic) & (ic > 0), ic, pc.edge_ic)
        ref_q, ref_px = 100.0, 1000.0
        fees = (
            self.costs.fill("buy", ref_q, ref_px).total
            + self.costs.fill("sell", ref_q, ref_px).total
        )
        self.cost_rt = (
            fees / (ref_q * ref_px)
            + 2 * cfg.costs.slippage.default_half_spread_pct * cost_multiplier
        )
        self.tiers = sorted(sw.risk_tiers)
        # D39: correlation risk + macro protection inputs
        self.rv = p.r.to_numpy(dtype=f)
        self.spike = np.zeros(len(self.idx), dtype=bool)
        if sw.spike_risk_mult < 1.0:
            self.spike = avg_corr_spike(
                p.r.where(pd.DataFrame(self.elig, index=p.r.index, columns=p.r.columns)),
                sw.spike_window,
                sw.spike_pct,
                sw.spike_lookback,
            )
        self.rb = p.rb.to_numpy(dtype=f)
        # D41: HMM regime multiplier - below regime_gate_mult, no NEW buys (exits unaffected)
        self.regime_ok = np.ones(len(self.idx), dtype=bool)
        if sw.regime_gate and regime_mult is not None:
            rm = regime_mult.reindex(self.idx).ffill().to_numpy(dtype=f)
            self.regime_ok = np.isfinite(rm) & (rm >= sw.regime_gate_mult)
        self.bm = self.bs = self.dsec = None
        if macro is not None:
            self.bm = macro.beta_mkt.reindex_like(p.close).to_numpy(dtype=f)
            self.bs = macro.beta_sec.reindex_like(p.close).to_numpy(dtype=f)
            self.dsec = macro.dsec.reindex_like(p.close).to_numpy(dtype=f)

    # ------------------------------------------------------------------ helpers
    def risk_frac(self, z: float) -> float:
        """Risk per trade (fraction of capital) for a score z (conviction tiers)."""
        r = 0.0
        for lvl, frac in self.tiers:
            if z >= lvl:
                r = frac
        return r

    def book_corr(self, i: int, j: int, book: list[int]) -> float:
        """Average correlation of stock ``j`` with the ``book`` over the last ``corr_window``."""
        if not book:
            return 0.0
        w = self.sw.corr_window
        blk = self.rv[max(0, i - w + 1) : i + 1][:, [j, *book]]
        ok = np.isfinite(blk).all(axis=1)
        if ok.sum() < w // 2:
            return 0.0
        c = np.corrcoef(blk[ok].T)
        v = c[0, 1:]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else 0.0

    def edge(self, i: int, j: np.ndarray | int) -> Any:
        """Expected 10-day gain from the score: IC · σ₁₀ · z."""
        return self.ic[i] * self.s10[i, j] * self.z[i, j]

    # ------------------------------------------------------------------ run
    def init_state(self, capital: float | None = None) -> SwingState:
        """A fresh all-cash account (``paper_capital_inr`` unless given)."""
        cap = float(self.cfg.risk.paper_capital_inr if capital is None else capital)
        return SwingState(cash=cap, equity=cap)

    def run(self, i0: int, i1: int) -> SwingResult:
        """Backtest: sessions [i0, i1] from a fresh account; open trades marked at the end."""
        _, res = self.step(self.init_state(), i0, i1, plan_last=False, mark_open=True)
        return res

    def step(
        self,
        state: SwingState,
        i0: int,
        i1: int,
        *,
        plan_last: bool = True,
        mark_open: bool = False,
    ) -> tuple[SwingState, SwingResult]:
        """Continue ``state`` through sessions [i0, i1] (live paper: one or more new days).

        ``plan_last``: decide buys / exits at the close of ``i1`` for the next open.
        ``mark_open``: close the book at the end (backtest report); live keeps it open.
        """
        sw, cfg = self.sw, self.cfg
        cash = float(state.cash)
        idle = (1 + cfg.costs.idle_cash_annual_yield) ** (1 / 252) - 1
        pos: dict[int, Position] = {}
        for pd_ in state.positions:
            jj = self.col_of.get(str(pd_["symbol"]))
            if jj is not None:
                pos[jj] = _pos_from_dict(pd_, jj)
        last = np.full(len(self.cols), np.nan)
        for sk, sv in state.last.items():
            jj = self.col_of.get(sk)
            if jj is not None:
                last[jj] = sv
        cool = np.full(len(self.cols), -1)
        for sk, sd in state.cool.items():
            jj = self.col_of.get(sk)
            if jj is not None:
                cool[jj] = int(self.idx.searchsorted(pd.Timestamp(sd))) + sw.cooldown
        buys: list[tuple[int, float, float, float]] = []  # (j, z, risk_frac, equity at decision)
        for b in state.buys:
            jj = self.col_of.get(str(b["symbol"]))
            if jj is not None:
                buys.append((jj, float(b["z"]), float(b["risk"]), float(b["equity"])))
        daily: list[dict[str, Any]] = []
        fills: list[dict[str, Any]] = []
        closed: list[dict[str, Any]] = []
        next_id = int(state.next_id)
        equity = float(state.equity or cash)
        costs, slip = self.costs, self.slip

        def adv_sh(i: int, j: int, px: float) -> float:
            a = self.adv[i - 1, j] if i > 0 else np.nan
            return float(a / px) if np.isfinite(a) and px > 0 else 0.0

        def fill(
            d: pd.Timestamp, ps: Position, side: str, q: int, px: float, c: float, why: str
        ) -> None:
            fills.append(
                {
                    "date": d,
                    "symbol": str(self.cols[ps.j]),
                    "side": side,
                    "qty": q,
                    "px": px,
                    "value": q * px,
                    "cost": c,
                    "reason": why,
                    "trade_id": ps.trade_id,
                }
            )

        def sell(i: int, ps: Position, q: int, px: float, why: str, *, limit: bool = False) -> None:
            nonlocal cash
            fpx = (
                px if limit else slip.price("sell", px, q, adv_sh(i, ps.j, px), self.s_imp[i, ps.j])
            )
            c = costs.fill("sell", q, fpx).total
            cash += q * fpx - c
            ps.qty -= q
            ps.sell_value += q * fpx
            ps.sell_cost += c
            fill(self.idx[i], ps, "sell", q, fpx, c, why)

        def close_trade(i: int, ps: Position, why: str) -> None:
            pnl = ps.sell_value - ps.sell_cost - ps.qty0 * ps.entry_px - ps.buy_cost + ps.dividends
            cost_basis = ps.qty0 * ps.entry_px
            sold_q = ps.qty0
            closed.append(
                {
                    "trade_id": ps.trade_id,
                    "symbol": str(self.cols[ps.j]),
                    "entry_date": ps.entry_date,
                    "entry_px": ps.entry_px,
                    "qty": ps.qty0,
                    "invested": cost_basis,
                    "score_z": ps.z0,
                    "sigma10": ps.s10,
                    "sl": ps.sl0,
                    "target": ps.target,
                    "risk_inr": ps.risk_inr,
                    "partial_date": ps.partial_date,
                    "partial_px": ps.partial_px,
                    "partial_qty": ps.partial_qty,
                    "exit_date": self.idx[i],
                    "exit_px": (
                        (ps.sell_value - (ps.partial_px * ps.partial_qty if ps.partial else 0.0))
                        / max(sold_q - ps.partial_qty, 1)
                    ),
                    "exit_reason": why,
                    "sessions": ps.held,
                    "costs": ps.buy_cost + ps.sell_cost,
                    "dividends": ps.dividends,
                    "pnl": pnl,
                    "return_pct": pnl / cost_basis if cost_basis > 0 else np.nan,
                    "r_multiple": pnl / ps.risk_inr if ps.risk_inr > 0 else np.nan,
                }
            )
            cool[ps.j] = i + sw.cooldown
            del pos[ps.j]

        started = state.last_session is not None
        for i in range(i0, i1 + 1):
            d = self.idx[i]
            if started:
                cash *= 1 + idle
            started = True
            # ---------------- open: delisted holdings
            for j in [j for j in pos if i > self.last_i[j]]:
                ps = pos[j]
                px = float(last[j]) * (1 - self.haircut[j]) if np.isfinite(last[j]) else 0.0
                c = costs.fill("sell", ps.qty, px).total if px > 0 else 0.0
                cash += ps.qty * px - c
                ps.sell_value += ps.qty * px
                ps.sell_cost += c
                fill(d, ps, "sell", ps.qty, px, c, "Delisted")
                ps.qty = 0
                close_trade(i, ps, "Delisted")
            # ---------------- open: stops gapped through, then flagged exits (sells first)
            for j in list(pos):
                ps = pos[j]
                o = self.op[i, j]
                if not np.isfinite(o) or self.lower_lock[i, j]:
                    continue  # no trade / locked at lower circuit: retry next session
                if o <= ps.stop:
                    sell(i, ps, ps.qty, o, ps.stop_kind)
                    close_trade(i, ps, ps.stop_kind)
                elif ps.exit_flag:
                    why = ps.exit_flag
                    sell(i, ps, ps.qty, o, why)
                    close_trade(i, ps, why)
            # ---------------- open: buys decided at the last close
            for j, zv, rf, eq_dec in buys:
                o = self.op[i, j]
                if j in pos or not np.isfinite(o) or o <= 0 or self.upper_lock[i, j]:
                    continue
                pc_ = self.cl[i - 1, j] if i > 0 else np.nan
                if np.isfinite(pc_) and o > pc_ * (1 + sw.gap_skip_sigma * self.sig[i - 1, j]):
                    continue  # gapped up too much: no chasing
                s10 = float(self.s10[i - 1, j])
                px = slip.price("buy", o, 1.0, adv_sh(i, j, o), self.s_imp[i, j])
                sl = px * (1 - sw.sl_sigma * s10)
                per_sh = px - sl
                if per_sh <= 0:
                    continue
                q = math.floor(rf * eq_dec / per_sh)
                q = min(q, math.floor(sw.max_weight * eq_dec / px), math.floor(cash / (px * 1.003)))
                if q <= 0:
                    continue
                px = slip.price("buy", o, q, adv_sh(i, j, o), self.s_imp[i, j])  # size-aware
                sl = px * (1 - sw.sl_sigma * s10)
                c = costs.fill("buy", q, px).total
                if q * px + c > cash:
                    continue
                cash -= q * px + c
                ps = Position(
                    j=j,
                    trade_id=next_id,
                    entry_date=d,
                    entry_px=px,
                    qty0=q,
                    qty=q,
                    s10=s10,
                    z0=zv,
                    risk_inr=q * (px - sl),
                    sl0=sl,
                    target=px * (1 + sw.target_sigma * s10),
                    stop=sl,
                    peak=px,
                    buy_cost=c,
                    sl_grace_days=sw.sl_grace_days,
                )
                next_id += 1
                pos[j] = ps
                fill(d, ps, "buy", q, px, c, "Entry")
            buys = []
            # ---------------- during the session: resting SL-M and target orders
            for j in list(pos):
                ps = pos[j]
                lo, hi = self.lo[i, j], self.hi[i, j]
                if not (np.isfinite(lo) and np.isfinite(hi)) or self.lower_lock[i, j]:
                    continue
                hit_stop = lo <= ps.stop and ps.held >= ps.sl_grace_days  # D41: per-position grace
                hit_tgt = (not ps.partial) and hi >= ps.target
                if hit_stop and not (hit_tgt and self.op[i, j] >= ps.target):
                    sell(i, ps, ps.qty, ps.stop, ps.stop_kind)  # pessimistic: stop first
                    close_trade(i, ps, ps.stop_kind)
                    continue
                if hit_tgt:
                    q = ps.qty - math.floor(ps.qty * (1 - sw.target_sell_frac))
                    q = ps.qty if ps.qty - q <= 0 else q
                    tpx = max(ps.target, self.op[i, j])  # a gap above the target fills at the open
                    sell(i, ps, q, tpx, "Target", limit=True)
                    ps.partial, ps.partial_date, ps.partial_px, ps.partial_qty = True, d, tpx, q
                    if ps.qty <= 0:
                        close_trade(i, ps, "Target")
                        continue
                    if ps.entry_px > ps.stop:
                        ps.stop, ps.stop_kind = ps.entry_px, "BE"
                    if lo <= ps.stop and self.op[i, j] < ps.target:  # fell back to entry same day
                        sell(i, ps, ps.qty, ps.stop, ps.stop_kind)
                        close_trade(i, ps, ps.stop_kind)
            # ---------------- close: dividends, mark to market
            if self.dy is not None:
                for ps in pos.values():
                    dv = ps.qty * last[ps.j] * self.dy[i, ps.j]
                    if np.isfinite(dv) and dv > 0:
                        cash += dv
                        ps.dividends += dv
            last = np.where(np.isfinite(self.cl[i]), self.cl[i], last)
            invested = float(
                sum(ps.qty * last[ps.j] for ps in pos.values() if np.isfinite(last[ps.j]))
            )
            equity = cash + invested
            heat = sum(max(ps.entry_px - ps.stop, 0.0) * ps.qty for ps in pos.values())
            daily.append(
                {
                    "date": d,
                    "equity": equity,
                    "cash": cash,
                    "invested": invested,
                    "positions": len(pos),
                    "open_risk": heat / equity if equity > 0 else 0.0,
                }
            )
            # ---------------- close: trailing stops, score / time exits
            for ps in pos.values():
                ps.held += 1
                c_ = self.cl[i, ps.j]
                if np.isfinite(c_):
                    ps.peak = max(ps.peak, c_)
                if ps.peak >= ps.entry_px * (1 + sw.tsl_start_sigma * ps.s10):
                    t = ps.peak * (1 - sw.tsl_sigma * ps.s10)
                    if t > ps.stop:
                        ps.stop, ps.stop_kind = t, "TSL"
                if (  # D39 M3: today's market + sector moves imply a hit → protect the profit
                    sw.macro_protect
                    and self.bm is not None
                    and self.bs is not None
                    and self.dsec is not None
                    and i > 0
                    and np.isfinite(c_)
                    and c_ > ps.entry_px
                ):
                    j = ps.j
                    imp = np.nan_to_num(self.bm[i - 1, j]) * np.nan_to_num(self.rb[i]) + (
                        np.nan_to_num(self.bs[i - 1, j]) * np.nan_to_num(self.dsec[i, j])
                    )
                    sg = self.sig[i - 1, j]
                    if np.isfinite(sg) and imp < -sw.macro_protect_k * sg:
                        t = ps.peak * (1 - sw.macro_tsl_sigma * ps.s10)
                        if t > ps.stop:
                            ps.stop, ps.stop_kind = t, "MacroTSL"
                zc = self.z[i, ps.j]
                if not np.isfinite(zc) or zc < sw.z_exit:
                    ps.exit_flag = ps.exit_flag or "Score"
                elif ps.held >= sw.max_hold:
                    ps.exit_flag = ps.exit_flag or "Time"
            # ---------------- close: new buys and swaps
            if i >= i1 and not plan_last:
                continue
            staying = [ps for ps in pos.values() if not ps.exit_flag]
            heat_eq = heat / equity if equity > 0 else 1.0
            zi = self.z[i]
            if self.regime_ok[i]:  # D41: skip new entries while the regime gate is closed
                cand = np.flatnonzero(self.elig[i] & (zi >= sw.z_entry) & (cool <= i))
                cand = np.array([j for j in cand if j not in pos], dtype=int)
            else:
                cand = np.array([], dtype=int)
            if cand.size:
                ed = self.edge(i, cand)
                cand = cand[ed > sw.entry_hurdle_mult * self.cost_rt]
                cand = cand[np.argsort(-zi[cand], kind="stable")]
            free = sw.max_positions - len(staying)
            k = 0
            planned = 0.0
            book = [ps.j for ps in staying]
            spike_m = sw.spike_risk_mult if self.spike[i] else 1.0
            while k < cand.size and free > 0:
                j = int(cand[k])
                if sw.corr_book_max > 0 and self.book_corr(i, j, book) > sw.corr_book_max:
                    k += 1
                    continue  # D39: too similar to what we already hold
                rf = self.risk_frac(float(zi[j])) * spike_m
                if heat_eq + planned + rf > sw.max_open_risk:
                    break
                buys.append((j, float(zi[j]), rf, equity))
                book.append(j)
                planned += rf
                free -= 1
                k += 1
            swaps = 0
            while k < cand.size and swaps < sw.max_replacements and staying:
                weakest = min(
                    staying, key=lambda s: float(np.nan_to_num(self.edge(i, s.j), nan=-1.0))
                )
                j = int(cand[k])
                gain = float(self.edge(i, j)) - float(
                    np.nan_to_num(self.edge(i, weakest.j), nan=-1.0)
                )
                if gain <= sw.replace_hurdle_mult * self.cost_rt:
                    break
                if (
                    sw.corr_book_max > 0
                    and self.book_corr(i, j, [b for b in book if b != weakest.j]) > sw.corr_book_max
                ):
                    k += 1
                    continue
                rf = self.risk_frac(float(zi[j])) * spike_m
                freed = max(weakest.entry_px - weakest.stop, 0.0) * weakest.qty / max(equity, 1e-9)
                if heat_eq + planned + rf - freed > sw.max_open_risk:
                    break
                weakest.exit_flag = "Replaced"
                staying.remove(weakest)
                buys.append((j, float(zi[j]), rf, equity))
                book = [b for b in book if b != weakest.j] + [j]
                planned += rf - freed
                swaps += 1
                k += 1
        new = SwingState(
            cash=cash,
            positions=[_pos_to_dict(ps, str(self.cols[j])) for j, ps in pos.items()],
            last={str(self.cols[j]): float(last[j]) for j in np.flatnonzero(np.isfinite(last))},
            cool={
                str(self.cols[j]): str(self.idx[int(cool[j]) - sw.cooldown].date())
                for j in np.flatnonzero(cool > i1)
                if 0 <= int(cool[j]) - sw.cooldown < len(self.idx)
            },
            buys=[
                {"symbol": str(self.cols[j]), "z": zv, "risk": rf, "equity": eq}
                for j, zv, rf, eq in buys
            ],
            next_id=next_id,
            equity=equity,
            last_session=str(self.idx[i1].date()) if i1 >= i0 else state.last_session,
        )
        if mark_open:  # backtest: open trades marked at the last close (not sold)
            for ps in list(pos.values()):
                ps.sell_value += ps.qty * float(last[ps.j]) if np.isfinite(last[ps.j]) else 0.0
                close_trade(i1, ps, "Open")
        dl = pd.DataFrame(daily).set_index("date") if daily else pd.DataFrame()
        res = SwingResult(dl, pd.DataFrame(fills), pd.DataFrame(closed), {"cost_rt": self.cost_rt})
        return new, res


def avg_corr_spike(r: pd.DataFrame, window: int, pct: float, lookback: int) -> np.ndarray:
    """Market-wide correlation spike flag per session.

    Average pairwise correlation from standardised returns (no N×N matrix):
    ``ρ̄ = (N·Var(mean z) − 1) / (N − 1)`` over the last ``window`` sessions; a spike is ρ̄
    above its own ``pct`` percentile of the previous ``lookback`` sessions (point-in-time).
    """
    sd = r.rolling(window, min_periods=window // 2).std()
    z = r / sd.where(sd > 0)
    n = z.notna().sum(axis=1)
    zbar = z.mean(axis=1)
    v = zbar.rolling(window, min_periods=window // 2).var()
    nn = n.rolling(window, min_periods=window // 2).mean()
    rho = (nn * v - 1) / (nn - 1).where(nn > 1)
    thr = rho.shift(1).rolling(lookback, min_periods=lookback // 4).quantile(pct / 100)
    out: np.ndarray = (rho > thr).fillna(False).to_numpy(dtype=bool)
    return out


def run_swing(
    p: Panels,
    scores: pd.DataFrame,
    gate: pd.DataFrame,
    cfg: AppConfig,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    edge_ic: pd.Series | None = None,
    cost_multiplier: float = 1.0,
    macro: Exposures | None = None,
    regime_mult: pd.Series | None = None,
) -> SwingResult:
    """Backtest the D37 trade engine over [start, end] (D39/D41 options via ``risk.swing``)."""
    sim = SwingSim(
        p,
        scores,
        gate,
        cfg,
        edge_ic=edge_ic,
        cost_multiplier=cost_multiplier,
        macro=macro,
        regime_mult=regime_mult,
    )
    idx = sim.idx
    i0 = 0 if start is None else int(idx.searchsorted(start))
    i1 = len(idx) - 1 if end is None else int(idx.searchsorted(end, side="right")) - 1
    return sim.run(i0, i1)
