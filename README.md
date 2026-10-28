# QuantSwing NSE

Systematic multi-sleeve swing platform for NSE equities. Spec: project docs 00–09.
**Status:** P0–P3, P5 ✅ passed on real Fyers data · P4 sleeves all OFF · P1b NSE data (D31) running on user PC · P6 paper trading (D32) built, pending start · Dashboard v1 (P7-early) built.

## P0 contents
| Area | File(s) |
|---|---|
| Config | `config/{settings,costs,risk,sleeves}.yaml` → `src/quantswing/core/config.py` (pydantic-settings) |
| Logging | `core/logging.py` — JSON lines, IST timestamps, secrets masked |
| Time | `core/clock.py` (IST), `core/calendar.py` (NSE calendar stub; holidays loaded in P1) |
| Auth | `broker/fyers_auth.py` — per doc 09 (direct `validate-authcode` POST), Fernet-encrypted token |
| Throttle | `broker/throttler.py` — sec/min/day token buckets + exponential backoff |
| Data | `broker/fyers_data.py` — chunked `history()`, batched `quotes()` |
| Scripts | `scripts/fyers_login.py`, `scripts/p0_smoke_test.py` |
| Tests | `tests/` — throttler, auth-code parsing/exchange, candle UTC→IST, calendar, masking |

## P1 contents (D22: Fyers is the only data source)
| Area | File(s) |
|---|---|
| Universe | `ingest/fyers_universe.py` — public Fyers symbol master → NSE EQ/BE, ETFs/SME excluded |
| History | `ingest/fyers_history.py` — resumable per-symbol daily backfill + incremental updates |
| Pipeline | `ingest/pipeline.py` — universe → indices → calendar → equities → validation → DB |
| Storage | `store/repositories.py` (Parquet per symbol), `store/database.py` (DuckDB) |
| Validation | `store/validation.py` — OHLC, volume, calendar, spikes, unadjusted split/bonus suspects |
| Calendar | `core/calendar.py` — sessions observed from NIFTY 50 history |
| Scripts | `scripts/backfill.py`, `scripts/run_eod.py`, `scripts/p1_exit_test.py` |

### Run P1
```powershell
python scripts\fyers_login.py              # token is valid for today only
python scripts\backfill.py --limit 20      # trial: 20 stocks, ~2 min
python scripts\backfill.py                 # full market, ~3 h first time; Ctrl+C safe, re-run resumes
python scripts\p1_exit_test.py             # look for: P1 EXIT TEST: PASS
python scripts\run_eod.py                  # daily after 15:30: fetches only the last few days
```
Outputs: `data\raw\fyers\eq_daily\*.parquet`, `data\db\quantswing.duckdb` (tables `prices_daily`, `indices_daily`, `universe`, `calendar`, `validation_issues`).

## P1+ — automatic split/bonus adjustment
If Fyers history has an unadjusted split/bonus (overnight gap matching 1:2, 1:5, 1:1 bonus, …),
validation records it in `data\processed\adjustments.parquet` and `prices_daily` gets
`adj_factor` + `adj_open/high/low/close/volume`. Toggle: `settings.validation.ca_auto_adjust`.

## P1b contents — NSE secondary data (D31)
Fyers stays the only source of listed-stock prices. NSE fills what Fyers does not have.
| Data | NSE source | Used for |
|---|---|---|
| Delivery % | MTO security-wise delivery file | DELZ factor |
| Results calendar | financial-results broadcast time + results board meetings | results-window entry block, S5, EARN |
| Corporate actions | corporate-actions API | dividends (credited on ex-date), delisted-stock adjustment |
| Sector | quote API (monthly) | sector-neutral factors, RSEC |
| Surveillance | daily REG1_IND file (ASM/GSM/ESM/IRP/default/BZ, pledge) | gate exclusion, PLEDGE factor |
| Delisted stocks | daily bhavcopies (+ symbol changes) | survivorship-free backtests |
| Holidays | holiday master | scheduler (P6) |

| Area | File(s) |
|---|---|
| HTTP | `nse/client.py` — ≤ 1 req/s, cookies, browser TLS (curl_cffi), retries, Windows cert store |
| Formats | `nse/formats.py` — URL builders + parsers (old/UDiFF bhavcopy, MTO, surveillance, APIs) |
| Loader | `nse/loader.py` — resumable backfill + daily update, Parquet cache under `data\processed\nse` |
| Build | `nse/build.py` (symbol map, delisted candles, events, dividends), `nse/db.py` (DuckDB tables) |
| Models | `factors/nse_factors.py` (DELZ, RSEC, PLEDGE, EARN), `factors/xsec.py` (sector neutral), `alpha/events.py` (entry block), `sleeves/s5_pead.py` |
| Config | `config/nse.yaml` |

### Run P1b (after P1)
```powershell
python scripts\nse_backfill.py --probe    # 2 min: NSE reachable + formats parse → NSE PROBE: PASS
python scripts\nse_backfill.py            # first time ~3 h (1 request/s), Ctrl+C safe → NSE DATA TEST: PASS
```
Daily: `run_eod.py` runs the Fyers update then the NSE update (`--no-nse` to skip).

## P2 contents — factor library + IC analysis
| Area | File(s) |
|---|---|
| Factors | `factors/{trend,momentum,volatility,liquidity,hurst,ou,kalman,drawdown,market}.py` — TQ60/120, MOM63/126/252, CMP, RNG, ILLIQ, VOLZ, HURST, DEV, OUHL, KAL_SLOPE/T, DD, RR, H52, BETA, IVOL, RMOM |
| Gate | `factors/gate.py` — history ≥ 250, continuity ≥ 95%, no circuit-lock streak ≥ 3 |
| Cross-section | `factors/xsec.py` — winsorise 1/99 → z → size-neutralise |
| IC | `alpha/ic.py` (rank IC, non-overlapping t), `alpha/combine.py` (IC-IR weights, composite) |
| Report | `alpha/ic_report.py` → `reports\p2_ic_report.html` + CSVs |
| Config | `config/factors.yaml` |

### Run P2 (after the P1 backfill)
```powershell
python scripts\run_factors.py              # ~1–3 min, no Fyers calls
python scripts\p2_ic_report.py             # opens the HTML report; look for: P2 EXIT TEST: PASS
```

## P3 contents — backtester + validation suite
| Area | File(s) |
|---|---|
| Costs | `execution/costs.py` — Fyers delivery schedule (brokerage, STT, txn, SEBI, stamp, GST, DP) |
| Slippage | `execution/slippage.py` — half-spread + η·σ·√(Q/ADV); gap-through stops fill at the open |
| Labels | `alpha/labels.py` — triple barrier (worst case on two-sided bars) + uniqueness weights |
| Engine | `backtest/engine.py` — next-open entries, SL / partial +1.5R → breakeven / σ-trail / time stop, risk-vol-capacity sizing, heat & position caps, rolling-peak DD pause (D26), T+1 cash |
| Metrics | `backtest/metrics.py` — CAGR, Sharpe, Sortino, Calmar, DD, win rate, expectancy R, PF, capture, alpha/beta |
| Validation | `backtest/dsr.py` (Deflated Sharpe), `montecarlo.py` (5,000 reshuffles/bootstrap), `walkforward.py` (purged + embargo) |
| Tracking | `backtest/tracking.py` — every run in `runs\<id>\` (D25; MLflow optional) |
| Baseline | `backtest/baseline.py` + `scripts\run_backtest.py` → `reports\p3_backtest_report.html` |
| Config | `config/backtest.yaml` |

### Run P3 (after P2)
```powershell
python scripts\run_backtest.py             # ~1 min; opens the report; look for: P3 EXIT TEST: PASS
```

## P4 contents — strategy sleeves
| Area | File(s) |
|---|---|
| Rules | `sleeves/s1_breakout.py` … `s4_high_momentum.py` (03 §4), `sleeves/base.py`, `registry.py` |
| Exit fitting | `sleeves/exit_fit.py` — SL from winners' MAE p90, partial from MFE reach, trail from giveback, time stop from half-life |
| p(win) / EV | `alpha/calibration.py` — logistic on composite score; EV = p·W̄ − (1−p)·L̄ − costs (R) |
| Validation | `sleeves/validate.py` — purged walk-forward per sleeve; ON iff OOS Sharpe ≥ 0.7, ≥ 30 trades, 2×-cost expectancy > 0 |
| Output | `scripts\run_sleeves.py` → `reports\p4_sleeves_report.html`, `data\processed\sleeves_fitted.json`, dashboard *Sleeves* page |
| Config | `config/sleeves.yaml` (`rules`, `fit`, `validation`) |

### Run P4 (after P2)
```powershell
python scripts\run_sleeves.py              # a few minutes; look for: P4 EXIT TEST: PASS
```

## P5 contents — regime model + factor portfolio (D28, D29)
| Area | File(s) |
|---|---|
| Composite | `alpha/combine.py` — D28 sign-aware IC-IR weights (negative-IC factors inverted, point-in-time) |
| Regime | `regime/breadth.py` (share of names in an uptrend), `regime/hmm.py` — 4-state Gaussian HMM on Nifty 500 return, 20d vol, breadth, India VIX change; monthly refit, filtered probabilities, 2-day switch rule |
| Portfolio | `portfolio/construction.py` — top 12 by composite (buffer: top 24), weekly rebalance at next open, `sizing.py` (inverse-vol, 15% cap, vol target), `covariance.py` (Ledoit-Wolf) |
| Run | `portfolio/p5.py` + `scripts\run_portfolio.py` → `reports\p5_portfolio_report.html`, `data\processed\regime.parquet` |
| Sleeves | P4 sleeves now use the regime: S1/S3/S4 off in bear/crisis, S2 scaled |
| Config | `config/risk.yaml` → `regime:` and `portfolio:` |

### Run P5 (after P2)
```powershell
python scripts\p2_ic_report.py            # rebuild the composite with D28
python scripts\run_portfolio.py           # ~1–2 min; look for: P5 EXIT TEST: PASS
python scripts\run_sleeves.py             # sleeves again, now with the regime filter
```

## D33 — the trading rule (simple, score-driven)
| Rule | Value |
|---|---|
| Universe | 1,000 most traded stocks (median ₹ value, 60 sessions), re-ranked every decision |
| Holdings | up to 10 stocks with a positive composite score, 10% each; keep while in the top 20 |
| Exposure | decided by the score: unfilled slots stay in cash (no market filter, no vol target) |
| Risk | drawdown brake (D26), results-window entry block (±3 sessions) |
Config: `config/risk.yaml` → `portfolio:`. The D29 options (HMM regime, vol target, inverse-vol) stay switchable.

## P6 contents — paper trading (D32)
The D29 ranked portfolio with the regime (D30) trades on paper once per evening, using the
**same simulator as the backtest** (`PortfolioSim`, resumable), so paper can be replayed
and compared exactly.
| Area | File(s) |
|---|---|
| Simulator | `portfolio/construction.py` — `PortfolioSim` + `SimState` (JSON), circuit-lock retries, controls |
| Book | `execution/paper_broker.py` (SQLite: orders, state log, equity, positions, controls, incidents, runs), `execution/order_state.py` (04 §7) |
| Engine | `paper/engine.py` — research refresh, session catch-up, fills at the open, next-open plan, reconciliation |
| Telegram | `alerts/telegram_bot.py` — daily report, alerts, `/status /positions /pause /resume /flatten yes /report`, Fyers login relay |
| Exit test | `paper/status.py` + `scripts\paper_status.py` — sessions ≥ 10, no failed runs / incidents, fills reconcile, paper = backtest replay (≤ 0.5%) |
| Scheduling | `scripts\install_schedule.py` — Windows Task Scheduler, Mon–Fri 19:30 (after NSE publishes), catches up if the PC was off |
| Dashboard | Watchlist (next-open plan), Positions (equity curve), Orders & Journal (state log, runs, incidents) |

### Run P6
```powershell
python scripts\telegram_setup.py          # once: bot token + chat id → .env
python scripts\paper_daily.py             # first run opens the account and plans the first buys
python scripts\install_schedule.py        # then it runs by itself every weekday evening
python scripts\paper_status.py            # any time; after 2 weeks: P6 EXIT TEST: PASS
```

## Dashboard (Streamlit v1, D13)
```powershell
python scripts\run_dashboard.py            # opens http://localhost:8501 (local only); Ctrl+C to stop
```
| Screen | Contents |
|---|---|
| Command Center | Status KPIs, pipeline checklist, top composite scores, latest backtest |
| Scanner | Whole universe ranked by composite, factor z-scores, filters, CSV export |
| Stock | Adjusted price + volume, quarantined rows, factor history, latest factor values |
| Factor IC | Mean IC by horizon, t-stats, verdicts, IC by year |
| Backtest Lab | Runs compared, equity vs Nifty 500, drawdown, R histogram, exit reasons, trades |
| Sleeves | P4 verdicts, OOS equity per sleeve, fitted live exits, per-year folds |
| Data Health | Live backfill progress (15 s refresh — safe while backfill runs), validation, adjustments, files |
| Settings | All parameters (read-only v1) |
| Watchlist / Positions / Orders & Journal | Paper trading (P6) — next-open plan, holdings + equity, order log |
| Risk | Placeholder until P7 |

The dashboard never locks the database (each query opens read-only and closes).

## Setup (Windows, PowerShell)
Requires **Python 3.11 or 3.12** (64-bit). Python 3.13 is not supported: `fyers-apiv3` pins `aiohttp==3.9.3`, which has no 3.13 wheels.

```powershell
cd C:\path\to\quantswing
python -m venv .venv                  # or: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt       # pinned deps + editable install of quantswing
copy .env.example .env
notepad .env                          # fill FYERS_CLIENT_ID / FYERS_SECRET_KEY / FYERS_REDIRECT_URI
python -m pytest                      # expect: 180 passed
```

### `.env`
| Key | Value |
|---|---|
| `FYERS_CLIENT_ID` | QuantSwing app ID, e.g. `XXXXXXXXXX-100` |
| `FYERS_SECRET_KEY` | App secret |
| `FYERS_REDIRECT_URI` | **Exactly** as registered on the app (incl. trailing `/`) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Used from P6 |

## Daily login
```powershell
python scripts\fyers_login.py         # opens browser; paste the full redirect URL (or the code)
python scripts\fyers_login.py --check # is today's token valid?
python scripts\fyers_login.py --force # re-login even if valid
```
The redirect page may show "can't connect". That's fine: copy the URL from the address bar. The token is saved encrypted in `.secrets\fyers_token.enc` (key: `.secrets\token.key`) and is treated as valid for the IST day on which it was obtained.

## P0 exit test
```powershell
python scripts\p0_smoke_test.py
```
It logs in if needed, runs a profile check, pulls 1 year of daily candles for `NSE:RELIANCE-EQ` and `NSE:NIFTY50-INDEX`, writes `data\raw\fyers\daily\*.parquet`, prints row counts and first/last dates, and ends with `P0 EXIT TEST: PASS`.

## Quality
```powershell
ruff check src tests scripts
black --check src tests scripts
mypy                                   # strict on src/
```

## Configuration
* All parameters live in YAML; nothing is hard-coded. Unknown keys cause an error, so typos are caught.
* To override a YAML value from the environment, use `QS_` + a `__` path, e.g. `QS_SETTINGS__LOGGING__LEVEL=DEBUG`.
* Logs go to `logs\quantswing.jsonl` (rotating).

## Security notes
* `.env`, `.secrets\`, `data\`, `logs\`, `*.db` and `*.parquet` are git-ignored.
* The Fernet key file is the only protection for the token at rest. Keep the repo in your user profile, and turn on BitLocker if you can (06 §5 "`.env` encrypted at rest" relies on OS disk encryption in P0).

## D34/D35 — variant tests (one command)
`python scripts\run_variants.py` runs the pre-registered final round (D35) and prints ₹10 lakh → value, return/yr,
worst fall, return ÷ fall, year-by-year vs Nifty, and the winner. `--all` also re-runs V0, V1, V3 (D34).

| Variant | What it adds |
|---|---|
| V2 (reference) | 10-day score from significant factors, conviction sizing (5–20%, ≤ 15 names), cost-aware switching |
| F1 | decide every 10 sessions; swap only if the gain beats 2× the round-trip cost |
| F2 | + up to 20 names, ≤ 30% per NSE sector, portfolio risk capped at 20%/yr |
| F3 | + trailing stop (2.5σ√10 below the peak, 20-session cooldown) |
| F4 | + sell once a stock is up 20% from entry |

Winner rule (fixed in advance): return ≥ Nifty + 5%/yr and beats cash at 2× costs → highest return ÷ worst fall.
Config keys: `risk.portfolio.switch_cost_mult`, `sector_cap`, `vol_target_level`, `take_profit`.

### D36 — score-set exposure (default run of `run_variants.py`)
B1 = F1 + invest `clip(t/2, 0, 1)` of capital, where t is the t-stat of the score's own recent (63-session, already-known) IC;
B2 = F1 + trailing stop; B3 = B1 + trailing stop. No winner is picked. The run also prints each factor's accuracy by year
(`reports/factor_ic_by_year.parquet`). Config: `risk.portfolio.ic_exposure`, `ic_exposure_window`, `ic_exposure_full_t`.

## D37 — trade engine (score entry, SL / half-target / TSL)
```powershell
python scripts\run_swing.py      # ~3–5 min → reports\swing\swing_report.html + swing_trades.csv
```
| Step | Rule (config `risk.swing`) |
|---|---|
| Buy | next open, score z ≥ 1, expected gain IC·σ₁₀·z > round-trip cost, not after a > 1σ gap-up |
| Size | quantity = risk ₹ ÷ (buy − SL); risk 0.75 / 1.0 / 1.25% of capital for z ≥ 1 / 1.5 / 2; ≤ 20% per stock; ≤ 12 stocks; open risk ≤ 10% |
| SL | buy − 1.5σ₁₀ (SL-M; a gap below fills at the open) |
| Target | buy + 3σ₁₀ → sell half, SL moves to the buy price |
| TSL | once up 1σ₁₀: highest close − 1.5σ₁₀ |
| Other exits | score z < 0 · 20 sessions · swapped for a stock whose gain beats it by 2× cost |
σ₁₀ = daily volatility (60 sessions) × √10, fixed at the buy.

## D39 — cross-asset correlation (Fyers-only drivers)
```powershell
python scripts\fyers_login.py        # today's token
python scripts\macro_backfill.py     # finds sector indices / G-Sec index / gold ETF in the Fyers master, downloads, builds MAC_* factors
python scripts\run_macro_test.py     # D37 vs M1/M2/M3/ALL + stability check (~20–30 min)
```
| Variant | Idea |
|---|---|
| M1 | Macro spill-over factors (size spread, VIX, gold, rates, best-matched sector × its 10-day move) may enter the score if significant |
| M2 | Skip buys highly correlated (> 0.6) with holdings; half risk during market-wide correlation spikes |
| M3 | Tighten the TSL of profitable holdings when today's market + sector moves imply a hit > 1 daily σ |
All options are OFF in `risk.swing` (D37 unchanged). Crude oil and USD/INR are not used: Fyers has no expired-futures history.

## D40 — live paper trading of D37 (paper only)
| Piece | What it does |
|---|---|
| **Login** page | One-click Fyers login: the app listens on `http://127.0.0.1:8765/callback` (set this as the redirect URL in the Fyers portal **and** as `FYERS_REDIRECT_URI` in `.env`). Paste login as a backup |
| **Actions** page | Account (₹5 lakh default, editable), **Run end-of-day update** button, the action list (date, when, stock, BUY/SELL/MODIFY SL-M/PLACE SL-M/PLACE TARGET, qty, price, status, order, reason), live holdings, live events, equity vs Nifty 500, closed trades |
| `scripts/live_eod.py` | Data → factors → D37 score → replay the new session(s) through the backtest engine → next session's actions (+ Telegram) |
| `scripts/live_poll.py` | Every 5 min in market hours: live prices → buys at the open (gap limit), SL / target / TSL events (+ Telegram). Provisional; the EOD run is official |
| `scripts/install_live_task.py` | Windows task: live updates Mon–Fri 09:10–15:35 every 5 min |
Settings: `config/paper.yaml` → `live:`; strategy settings are D37 (`config/risk.yaml` → `swing:`).
