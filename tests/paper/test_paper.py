"""P6: resumable simulator, order state machine, paper engine (daily + catch-up), controls,
reconciliation, paper-vs-backtest replay, Telegram commands and login relay."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantswing.core.config import get_config
from quantswing.execution.order_state import IllegalTransition, OrderState, check
from quantswing.execution.paper_broker import PaperBook
from quantswing.factors.common import Panels
from quantswing.portfolio.construction import PortfolioSim, SimState, run_portfolio
from quantswing.store.repositories import DataPaths
from tests.factors.test_factors import synth_panels

CFG = get_config()


def _pc(**kw: Any) -> Any:
    pc = CFG.risk.portfolio.model_copy(update={"top_n": 3, "rebalance_every": 5, **kw})
    return CFG.model_copy(update={"risk": CFG.risk.model_copy(update={"portfolio": pc})})


def _inputs(t: int = 260, n: int = 8) -> tuple[Panels, pd.DataFrame, pd.DataFrame]:
    p = synth_panels(t, n)
    rng = np.random.default_rng(5)
    score = pd.DataFrame(
        rng.normal(0, 1, p.close.shape), index=p.close.index, columns=p.close.columns
    )
    gate = p.close.notna()
    return p, score, gate


def test_resumable_sim_equals_backtest() -> None:
    p, score, gate = _inputs()
    cfg = _pc()
    idx = p.close.index
    full = run_portfolio(p, score, gate, cfg, start=idx[80])
    sim = PortfolioSim(p, score, gate, cfg)
    st = sim.init_state(idx[80])
    daily: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for a, b in ((80, 80), (81, 120), (121, 121), (122, len(idx) - 1)):
        st, d, t = sim.run(st, a, b)
        st = SimState.from_json(json.loads(json.dumps(st.to_json())))  # through storage
        daily += d
        trades += t
    eq = pd.DataFrame(daily).set_index("date")["equity"]
    assert np.allclose(eq.to_numpy(), full.daily["equity"].to_numpy(), rtol=0, atol=1e-6)
    assert len(trades) == len(full.trades) > 0


def test_lower_circuit_defers_the_sell() -> None:
    p, _score, gate = _inputs()
    cfg = _pc(top_n=1)
    idx, cols = p.close.index, p.close.columns
    k = 3
    sc = pd.DataFrame(0.0, index=idx, columns=cols)
    sc.iloc[:, k] = 2.0
    sc.iloc[100:, k] = -1.0  # decided at close 100 → exit at open 101
    hi, lo, cl = p.high.copy(), p.low.copy(), p.close.copy()
    cl.iloc[101, k] = cl.iloc[100, k] * 0.9
    hi.iloc[101, k] = lo.iloc[101, k] = cl.iloc[101, k]  # locked at the lower circuit
    q = replace(p, high=hi, low=lo, close=cl)
    t = run_portfolio(q, sc, gate, cfg, start=idx[80]).trades
    sells = t[
        (t["symbol"] == cols[k]) & (t["side"] == "sell") & (pd.to_datetime(t["date"]) > idx[100])
    ]
    assert pd.Timestamp(sells["date"].iloc[0]) == idx[102]  # blocked on 101, retried on 102
    assert sells["qty"].sum() > 0 and len(t[pd.to_datetime(t["date"]) == idx[101]]) == 0


def test_order_state_machine() -> None:
    check(OrderState.PENDING, OrderState.FILLED)
    check(OrderState.EXIT_PENDING, OrderState.EXIT_BLOCKED)
    check(OrderState.EXIT_BLOCKED, OrderState.EXIT_PENDING)
    with pytest.raises(IllegalTransition):
        check(OrderState.FILLED, OrderState.CANCELLED)
    with pytest.raises(IllegalTransition):
        check(OrderState.PENDING, OrderState.CLOSED)


# --------------------------------------------------------------------------- engine e2e
@pytest.fixture(scope="module")
def market(tmp_path_factory: pytest.TempPathFactory) -> tuple[DataPaths, Any, Panels]:
    from quantswing.factors.gate import universe_gate
    from quantswing.portfolio.p5 import build_regime
    from quantswing.store.panels import load_panels
    from tests.sleeves.test_sleeves import build_long_root

    root = build_long_root(tmp_path_factory.mktemp("paper"))
    paths = DataPaths.relocated(root)
    rc = CFG.risk.model_copy(
        update={
            "regime": CFG.risk.regime.model_copy(
                update={"min_train_sessions": 300, "breadth_min_names": 20}
            ),
            "portfolio": CFG.risk.portfolio.model_copy(update={"top_n": 6}),
        }
    )
    cfg = CFG.model_copy(update={"risk": rc})
    p = load_panels(paths, cfg)
    build_regime(p, universe_gate(p, cfg.factors.gate), paths, cfg)
    return paths, cfg, p


def _upto(p: Panels, k: int) -> Panels:
    cut = lambda x: x.iloc[: k + 1]  # noqa: E731
    return replace(
        p,
        open=cut(p.open),
        high=cut(p.high),
        low=cut(p.low),
        close=cut(p.close),
        volume=cut(p.volume),
        turnover=cut(p.turnover),
        bench=p.bench.iloc[: k + 1],
    )


def _run_days(
    monkeypatch: pytest.MonkeyPatch, market: Any, book: PaperBook, days: list[int]
) -> list[Any]:
    import quantswing.paper.engine as eng

    paths, cfg, p = market
    reps = []
    for k in days:
        monkeypatch.setattr(eng, "load_panels", lambda *_a, _k=k, **_kw: _upto(p, _k))
        reps.append(eng.step(paths, cfg, book))
    return reps


def test_paper_daily_matches_backtest_and_catch_up(
    monkeypatch: pytest.MonkeyPatch, market: Any, tmp_path: Path
) -> None:
    from quantswing.alpha.research import AUX, COMPOSITE
    from quantswing.paper.engine import reconcile
    from quantswing.portfolio.p5 import load_regime
    from quantswing.store.panels import FactorStore

    paths, cfg, p = market
    n = len(p.close)
    k0 = n - 40
    daily = PaperBook(tmp_path / "daily.sqlite")
    reps = _run_days(monkeypatch, market, daily, list(range(k0, n)))
    assert reps[0].opened and not reps[0].plan.empty  # account opened, first buys planned
    catch = PaperBook(tmp_path / "catch.sqlite")
    _run_days(monkeypatch, market, catch, [k0, k0 + 7, k0 + 8, n - 1])  # PC off for days
    e1, e2 = daily.equity()["equity"], catch.equity()["equity"]
    assert len(e1) == len(e2) == 40 and np.allclose(e1.to_numpy(), e2.to_numpy(), atol=1e-6)
    # identical to a backtest started on the paper start date
    store = FactorStore(paths.factors)
    gate = store.read(AUX, "gate").reindex_like(p.close).fillna(0).astype(bool)
    scores = store.read("composite", COMPOSITE).reindex_like(p.close)
    reg = load_regime(paths)
    from quantswing.alpha.events import event_block

    bt = run_portfolio(
        p,
        scores,
        gate,
        cfg,
        regime_mult=reg["mult"],
        start=p.close.index[k0],
        entry_block=event_block(p, cfg.risk.event_window),
    )
    assert np.allclose(e1.to_numpy(), bt.daily["equity"].to_numpy(), atol=1e-6)
    # orders: every fill logged, positions reconcile, states legal
    o = daily.orders()
    assert len(o[o["state"].isin(["FILLED", "CLOSED"])]) == len(bt.trades)
    assert reconcile(daily) == {} and reconcile(catch) == {}
    lg = daily.order_log()
    assert lg["to_state"].isin([s.value for s in OrderState]).all()
    # nothing new → no-op
    again = _run_days(monkeypatch, market, daily, [n - 1])[0]
    assert again.sessions == [] and len(daily.equity()) == 40


def test_controls_pause_and_flatten(
    monkeypatch: pytest.MonkeyPatch, market: Any, tmp_path: Path
) -> None:
    _, _, p = market
    n = len(p.close)
    book = PaperBook(tmp_path / "ctl.sqlite")
    book.set_control("pause_entries", True)
    rep = _run_days(monkeypatch, market, book, [n - 20])[0]
    assert rep.plan.empty  # opened paused: nothing bought
    book.set_control("pause_entries", False)
    _run_days(monkeypatch, market, book, list(range(n - 19, n - 10)))
    assert not book.positions().empty
    book.set_control("flatten", True)
    _run_days(monkeypatch, market, book, [n - 9])
    assert not book.control("flatten")  # one-shot
    _run_days(monkeypatch, market, book, [n - 8])
    assert book.positions().empty


def test_paper_status_exit_test(
    monkeypatch: pytest.MonkeyPatch, market: Any, tmp_path: Path
) -> None:
    import quantswing.paper.status as ps

    paths, cfg, p = market
    n = len(p.close)
    book = PaperBook(tmp_path / "st.sqlite")
    _run_days(monkeypatch, market, book, list(range(n - 12, n)))
    rid = book.run_start()
    book.run_end(rid, "ok", 1)
    monkeypatch.setattr(ps, "load_panels", lambda *_a, **_kw: p)
    st = ps.paper_status(paths, cfg, book)
    assert st.sessions == 12 and st.replay_gap < 1e-9 and st.passed, st.checks
    book.incident("run_failed", "boom")
    assert not ps.paper_status(paths, cfg, book, replay=False).passed


# --------------------------------------------------------------------------- telegram
class FakeTg:
    def __init__(self, updates: list[list[dict[str, Any]]]) -> None:
        self.batches = updates
        self.sent: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> Any:
        method = url.rsplit("/", 1)[-1]

        class R:
            def __init__(self, body: dict[str, Any]) -> None:
                self.body = body

            def json(self) -> dict[str, Any]:
                return self.body

        if method == "sendMessage":
            self.sent.append(json)
            return R({"ok": True, "result": {}})
        if method == "getUpdates":
            return R({"ok": True, "result": self.batches.pop(0) if self.batches else []})
        return R({"ok": False})


def _msg(uid: int, text: str, chat: int = 42) -> dict[str, Any]:
    return {"update_id": uid, "message": {"chat": {"id": chat}, "text": text}}


def test_telegram_commands_and_owner_only(tmp_path: Path) -> None:
    from quantswing.alerts.telegram_bot import Telegram, process_commands

    book = PaperBook(tmp_path / "tg.sqlite")
    http = FakeTg(
        [[_msg(1, "/pause"), _msg(2, "/flatten"), _msg(3, "/flatten", chat=99), _msg(4, "/help")]]
    )
    tg = Telegram("TKN", "42", http=http)
    assert process_commands(tg, book) == 3  # chat 99 ignored
    assert book.control("pause_entries") and not book.control("flatten")  # needs "yes"
    assert "yes" in http.sent[1]["text"] and book.get("tg_offset") == 5
    http.batches = [[_msg(5, "/flatten yes"), _msg(6, "/resume")]]
    process_commands(tg, book)
    assert not book.control("pause_entries") and not book.control("flatten")


def test_telegram_login_relay(tmp_path: Path) -> None:
    from quantswing.alerts.telegram_bot import Telegram, wait_for_auth

    book = PaperBook(tmp_path / "login.sqlite")
    got: list[str] = []

    def login(text: str) -> None:
        if "bad" in text:
            raise ValueError("expired")
        got.append(text)

    url = "https://127.0.0.1/?s=ok&auth_code=GOOD&state=x"
    http = FakeTg([[], [_msg(1, "https://127.0.0.1/?auth_code=bad")], [_msg(2, url)]])
    tg = Telegram("TKN", "42", http=http)
    assert wait_for_auth(tg, book, "https://login", 1, 1, login, sleep=lambda _s: None)
    assert got == [url] and "Login failed" in http.sent[1]["text"]
    assert not wait_for_auth(Telegram(None, None), book, "u", 1, 1, login)


def test_flat_zombie_stock_is_never_picked() -> None:
    p, score, gate = _inputs()
    cfg = _pc(top_n=2)
    idx, cols = p.close.index, p.close.columns
    close, opn, hi, lo = p.close.copy(), p.open.copy(), p.high.copy(), p.low.copy()
    for x in (close, opn, hi, lo):
        x.iloc[:, 2] = 50.0  # no price movement for months (untraded prints)
    sc = score.copy()
    sc.iloc[:, 2] = 9.0  # best score (e.g. lowest volatility)
    t = run_portfolio(
        replace(p, close=close, open=opn, high=hi, low=lo), sc, gate, cfg, start=idx[80]
    ).trades
    assert not t.empty and cols[2] not in set(t["symbol"])
    vol = p.volume.copy()
    vol.iloc[:, 3] = 0.0
    zero = replace(p, volume=vol)
    from quantswing.factors.gate import universe_gate

    g = universe_gate(
        zero, CFG.factors.gate.model_copy(update={"min_history": 20, "continuity_window": 20})
    )
    assert not g[cols[3]].any()  # zero-volume prints are not trades
