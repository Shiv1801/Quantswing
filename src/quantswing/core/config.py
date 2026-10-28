"""Typed configuration: YAML files in ``config/`` + secrets from ``.env``.

Loading model
-------------
* ``config/settings.yaml``, ``costs.yaml``, ``risk.yaml``, ``sleeves.yaml`` each hold ONE
  top-level key (``settings`` / ``costs`` / ``risk`` / ``sleeves``). They are merged by
  pydantic-settings' ``YamlConfigSettingsSource`` into :class:`AppConfig`.
* Any YAML value can be overridden by an env var with prefix ``QS_`` and ``__`` as the
  nesting delimiter, e.g. ``QS_SETTINGS__LOGGING__LEVEL=DEBUG``.
* Secrets (Fyers / Telegram) live only in ``.env`` and are loaded into :class:`Secrets`
  as ``SecretStr`` so they never print in reprs or logs.
* Config directory: ``QS_CONFIG_DIR`` env var, else ``<repo>/config``.
"""

from __future__ import annotations

import os
from datetime import date, time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG_FILES: tuple[str, ...] = (
    "settings.yaml",
    "costs.yaml",
    "risk.yaml",
    "sleeves.yaml",
    "factors.yaml",
    "backtest.yaml",
    "nse.yaml",
    "paper.yaml",
)


def config_dir() -> Path:
    """Return the config directory (``QS_CONFIG_DIR`` or ``<repo>/config``)."""
    return Path(os.environ.get("QS_CONFIG_DIR", REPO_ROOT / "config")).resolve()


class _Strict(BaseModel):
    """Base model: unknown keys are errors (catches YAML typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- settings.yaml
class AppCfg(_Strict):
    name: str
    timezone: str
    model_version: str


class PathsCfg(_Strict):
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    db_dir: Path
    log_dir: Path
    secrets_dir: Path
    token_file: Path
    token_key_file: Path
    api_usage_file: Path

    def resolve(self, p: Path) -> Path:
        """Resolve a configured path against the repo root (absolute paths pass through)."""
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()


class LoggingCfg(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    file_name: str
    max_bytes: int = Field(gt=0)
    backup_count: int = Field(ge=0)
    console: bool
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    mask_keys: list[str]
    mask_env_vars: list[str]
    mask_visible_chars: int = Field(ge=0)


class MarketCfg(_Strict):
    exchange: str
    session_open: time
    session_close: time
    pre_open_start: time
    weekend_days: list[int]
    holidays: list[str]
    special_sessions: list[str]

    @field_validator("weekend_days")
    @classmethod
    def _weekday_range(cls, v: list[int]) -> list[int]:
        if any(d < 0 or d > 6 for d in v):
            raise ValueError("weekend_days must be 0..6 (Mon=0)")
        return v


class HistoryCfg(_Strict):
    date_format: str
    cont_flag: str
    max_days_per_request: dict[str, int]


class QuotesCfg(_Strict):
    max_symbols_per_request: int = Field(gt=0)


class FyersCfg(_Strict):
    login_url_state: str
    response_type: str
    grant_type: str
    validate_authcode_url: str
    request_timeout_s: float = Field(gt=0)
    token_valid_same_ist_day: bool
    history: HistoryCfg
    quotes: QuotesCfg
    success_status: str
    no_data_status: str


class BackoffCfg(_Strict):
    max_retries: int = Field(ge=0)
    base_delay_s: float = Field(gt=0)
    max_delay_s: float = Field(gt=0)
    jitter_s: float = Field(ge=0)
    retryable_http_codes: list[int]
    retryable_api_codes: list[int]
    fatal_api_codes: list[int]


class ThrottleCfg(_Strict):
    per_second: int = Field(gt=0)
    per_minute: int = Field(gt=0)
    per_day: int = Field(gt=0)
    safety_factor: float = Field(gt=0, le=1)
    day_reset_tz: str
    backoff: BackoffCfg


class PlatformCfg(_Strict):
    dashboard: str
    alerts: str
    host: str


class SmokeTestCfg(_Strict):
    symbols: list[str]
    lookback_days: int = Field(gt=0)
    resolution: str
    output_subdir: str


class UniverseColumnsCfg(_Strict):
    fytoken: int
    name: int
    instrument_type: int
    lot_size: int
    tick_size: int
    isin: int
    ticker: int


class UniverseCfg(_Strict):
    symbol_master_url: str
    request_timeout_s: float = Field(gt=0)
    columns: UniverseColumnsCfg
    series_allowed: list[str]
    instrument_types_allowed: list[int]
    exclude_name_patterns: list[str]


class DataCfg(_Strict):
    source: Literal["fyers"]
    history_start: date
    resolution: str
    equity_subdir: str
    index_subdir: str
    universe_file: Path
    calendar_file: Path
    issues_file: Path
    adjustments_file: Path
    factors_dir: Path
    reports_dir: Path
    backfill_state_file: Path
    duckdb_file: Path
    incremental_lookback_days: int = Field(gt=0)
    progress_every: int = Field(gt=0)
    abort_after_consecutive_failures: int = Field(gt=0)
    calendar_symbol: str
    universe: UniverseCfg
    indices: list[str]


class KnownEventCfg(_Strict):
    symbol: str
    ex_date: date
    ratio: float = Field(gt=0)
    note: str


class ValidationCfg(_Strict):
    spike_sigma: float = Field(gt=0)
    spike_vol_window: int = Field(gt=1)
    spike_min_obs: int = Field(gt=1)
    ca_gap_min_abs_log: float = Field(gt=0)
    ca_candidate_ratios: list[float]
    ca_ratio_tolerance: float = Field(gt=0)
    ca_auto_adjust: bool
    stale_max_sessions: int = Field(ge=0)
    min_universe_symbols: int = Field(gt=0)
    min_coverage: float = Field(gt=0, le=1)
    max_reject_row_frac: float = Field(ge=0, le=1)
    min_fresh_frac: float = Field(gt=0, le=1)
    known_events: list[KnownEventCfg]
    known_event_tolerance: float = Field(gt=0)


class SettingsCfg(_Strict):
    version: str
    app: AppCfg
    paths: PathsCfg
    logging: LoggingCfg
    market: MarketCfg
    fyers: FyersCfg
    throttle: ThrottleCfg
    platform: PlatformCfg
    smoke_test: SmokeTestCfg
    data: DataCfg
    validation: ValidationCfg


# --------------------------------------------------------------------------- costs.yaml
class BrokerCostCfg(_Strict):
    delivery_brokerage_flat_inr: float = Field(ge=0)
    delivery_brokerage_pct: float = Field(ge=0)


class DpChargeCfg(_Strict):
    per_scrip_per_sell_day_inr: float = Field(ge=0)
    gst_applies: bool


class StatutoryCfg(_Strict):
    stt_delivery_pct: float = Field(ge=0)
    nse_txn_pct: float = Field(ge=0)
    sebi_fee_per_crore_inr: float = Field(ge=0)
    stamp_duty_buy_pct: float = Field(ge=0)
    gst_pct: float = Field(ge=0)


class SlippageCfg(_Strict):
    default_half_spread_pct: float = Field(ge=0)
    impact_eta: float = Field(ge=0)


class TaxCfg(_Strict):
    stcg_pct: float = Field(ge=0)
    ltcg_pct: float = Field(ge=0)


class CostsCfg(_Strict):
    version: str
    source: str
    broker: BrokerCostCfg
    dp_charge: DpChargeCfg
    statutory: StatutoryCfg
    slippage: SlippageCfg
    tax_reporting: TaxCfg
    idle_cash_annual_yield: float = Field(ge=0)
    settlement: str


# --------------------------------------------------------------------------- risk.yaml
class EntryWindowCfg(_Strict):
    start: time
    end: time


class RegimeCfg(_Strict):
    n_states: int = Field(gt=1)
    features: list[str]
    rv_window: int = Field(gt=1)
    breadth_window: int = Field(gt=1)
    vix_change_window: int = Field(gt=0)
    min_train_sessions: int = Field(gt=0)
    refit: Literal["monthly"]
    n_iter: int = Field(gt=0)
    seed: int
    switch_prob: float = Field(gt=0, lt=1)
    switch_days: int = Field(gt=0)
    vix_symbol: str
    breadth_min_names: int = Field(gt=0)


class PortfolioCfg(_Strict):
    top_n: int = Field(gt=0)
    hold_buffer_mult: float = Field(ge=1)
    rebalance_every: int = Field(gt=0)
    min_score: float
    weighting: Literal["inverse_vol", "equal", "equal_slot", "conviction"]
    vol_window: int = Field(gt=1)
    cov_window: int = Field(gt=1)
    max_gross: float = Field(gt=0, le=1)
    min_trade_value_frac: float = Field(ge=0)
    universe_top_n: int = Field(default=0, ge=0)  # D33: most-traded N stocks only (0 = all)
    universe_window: int = Field(default=60, gt=1)  # sessions of median ₹ traded value
    vol_target: bool = True  # scale the book to target_portfolio_vol (D29); off in D33
    use_regime: bool = True  # multiply by the HMM regime exposure (D29); off in D33
    # --- v2 options (D34 candidates; defaults = D33 behaviour)
    selection: Literal["buffer", "edge"] = "buffer"  # edge: switch only if gain > cost
    max_names: int = Field(default=15, gt=0)  # conviction / edge book size cap
    min_weight: float = Field(default=0.05, ge=0, le=1)  # conviction: drop smaller weights
    max_weight: float = Field(default=0.20, gt=0, le=1)  # conviction: cap per name
    edge_horizon: int = Field(default=10, gt=0)  # sessions the score forecasts
    edge_ic: float = Field(default=0.05, gt=0)  # fallback IC when no measured series
    rebalance_band: float = Field(default=0.0, ge=0)  # skip resizing within ±band × target
    dd_brake: bool = True  # D26 brake; False → only halt new buys at halt_dd
    halt_dd: float = Field(default=0.30, gt=0, le=1)
    trail_sigma: float = Field(default=0.0, ge=0)  # 0 = off; exit below peak × (1 − kσ√h)
    trail_horizon: int = Field(default=10, gt=0)
    reentry_cooldown: int = Field(default=20, ge=0)  # sessions after a trailing exit
    switch_cost_mult: float = Field(default=1.0, gt=0)  # edge swap hurdle = mult × round trip
    sector_cap: float = Field(default=0.0, ge=0, le=1)  # max weight per NSE sector (0 = off)
    vol_target_level: float = Field(default=0.0, ge=0)  # 0 → risk.target_portfolio_vol
    ic_exposure: bool = False  # B: invested share = clip(t(IC of the score, recent) / full_t, 0, 1)
    ic_exposure_window: int = Field(default=63, gt=5)  # sessions of (already-known) daily IC
    ic_exposure_full_t: float = Field(default=2.0, gt=0)  # t-stat at which 100% is invested
    take_profit: float = Field(default=0.0, ge=0)  # sell once up this much from entry (0 = off)


class SwingCfg(_Strict):
    """D37 trade engine: score entry, σ-based SL / partial target / TSL, score & time exits."""

    z_entry: float = 1.0  # buy only if the cross-sectional score z ≥ this
    z_exit: float = 0.0  # sell (next open) once the score z falls below this
    horizon: int = Field(default=10, gt=0)  # σ₁₀ = daily σ × √horizon
    vol_window: int = Field(default=60, gt=5)
    entry_hurdle_mult: float = Field(default=1.0, ge=0)  # IC·σ₁₀·z > mult × round-trip cost
    replace_hurdle_mult: float = Field(default=2.0, ge=0)  # swap only if edge gain > mult × cost
    max_replacements: int = Field(default=2, ge=0)  # per session
    gap_skip_sigma: float = Field(default=1.0, gt=0)  # skip buy if open > prev close × (1 + kσ)
    risk_tiers: list[tuple[float, float]] = [(1.0, 0.0075), (1.5, 0.01), (2.0, 0.0125)]
    max_weight: float = Field(default=0.20, gt=0, le=1)
    max_positions: int = Field(default=12, gt=0)
    max_open_risk: float = Field(default=0.10, gt=0, le=1)  # Σ qty × (entry − stop)⁺ / equity
    sl_sigma: float = Field(default=1.5, gt=0)
    target_sigma: float = Field(default=3.0, gt=0)
    target_sell_frac: float = Field(default=0.5, gt=0, le=1)
    tsl_start_sigma: float = Field(default=1.0, ge=0)
    tsl_sigma: float = Field(default=1.5, gt=0)
    max_hold: int = Field(default=20, gt=0)  # sessions
    cooldown: int = Field(default=10, ge=0)  # sessions before a sold stock can be bought again
    # D39 correlation risk + macro protection (all OFF by default = D37)
    corr_book_max: float = Field(default=0.0, ge=0, le=1)  # skip a buy if avg ρ with book > x
    corr_window: int = Field(default=60, gt=5)
    spike_risk_mult: float = Field(default=1.0, gt=0, le=1)  # risk × this in a correlation spike
    spike_window: int = Field(default=20, gt=2)  # sessions for the market-wide average ρ
    spike_pct: float = Field(default=90.0, gt=0, lt=100)  # spike = above this percentile …
    spike_lookback: int = Field(default=500, gt=20)  # … of its own past (sessions)
    use_macro_factors: bool = False  # D39 M1: MAC_* factors may enter the score
    macro_protect: bool = False  # tighten the TSL when today's driver moves hurt a holding
    macro_protect_k: float = Field(default=1.0, gt=0)  # implied hit > k × daily σ
    macro_tsl_sigma: float = Field(default=1.0, gt=0)  # tightened TSL = peak − x·σ₁₀
    # D41: drawdown-reduction candidates (all OFF by default = D37)
    sl_grace_days: int = Field(default=0, ge=0)  # no SL exit for this many sessions after entry
    regime_gate: bool = False  # skip NEW buys when the HMM regime multiplier is below the cutoff
    regime_gate_mult: float = Field(default=0.5, ge=0, le=1)  # cutoff (below → no new buys)


class RiskCfg(_Strict):
    version: str
    paper_capital_inr: float = Field(gt=0)
    direction: Literal["long_only", "long_short"]
    beta_hedge_enabled: bool
    risk_per_trade: float = Field(gt=0, lt=1)
    max_heat: float = Field(gt=0, lt=1)
    target_portfolio_vol: float = Field(gt=0, lt=1)
    max_open_positions: int = Field(gt=0)
    sector_max_positions: int = Field(gt=0)
    sector_max_gross: float = Field(gt=0, le=1)
    single_name_max: float = Field(gt=0, le=1)
    max_avg_pairwise_corr: float = Field(ge=-1, le=1)
    daily_cvar95_max: float = Field(gt=0)
    overnight_gap_stress_sigma: float = Field(gt=0)
    overnight_gap_budget: float = Field(gt=0)
    dd_mult_full_scale_dd: float = Field(gt=0)
    dd_stop_new_entries: float = Field(gt=0)
    dd_peak_window_sessions: int = Field(gt=0)
    corr_spike_percentile: float = Field(gt=0, lt=100)
    corr_spike_gross_mult: float = Field(ge=0, le=1)
    kelly_fraction: float = Field(ge=0, le=1)
    capacity_adv_fraction: float = Field(gt=0, le=1)
    sl_hard_cap: float = Field(gt=0, lt=1)
    regime_exposure_mult: dict[str, float]
    entry_window: EntryWindowCfg
    websocket_outage_alert_s: int = Field(gt=0)
    regime: RegimeCfg
    portfolio: PortfolioCfg
    event_window: EventWindowCfg
    swing: SwingCfg = Field(default_factory=SwingCfg)


# --------------------------------------------------------------------------- sleeves.yaml
class AllocationCfg(_Strict):
    method: str
    tilt_bounds: tuple[float, float]
    sharpe_lookback_months: int = Field(gt=0)


class S1Cfg(_Strict):
    cmp_max: float
    rng_xs_quantile_max: float
    tq_min: float
    hurst_min: float
    range_window: int = Field(gt=1)
    breakout_k_sigma: float = Field(ge=0)
    volz_min: float


class S2Cfg(_Strict):
    tq_xs_quantile_min: float
    r2_frac_of_median: float
    r2_median_window: int = Field(gt=1)
    dev_min: float
    dev_max: float
    ouhl_max: float
    knife_volz_max: float
    knife_window: int = Field(gt=0)
    kal_t_trigger: float
    setup_memory: int = Field(gt=0)


class S3Cfg(_Strict):
    dd_min: float
    dd_max: float
    dd_lookback: int = Field(gt=1)
    rr_min: float
    hurst_min: float
    kal_t_min: float
    kal_slope_accel_window: int = Field(gt=0)


class S4Cfg(_Strict):
    h52_min: float
    mom_xs_quantile_min: float
    rmom_min: float
    high_window: int = Field(gt=1)
    volz_min: float


class S5Cfg(_Strict):
    ar_sigma_min: float = Field(gt=0)  # event abnormal return > k · idiosyncratic σ
    volz_min: float
    gap_hold_frac: float = Field(ge=0, le=1)  # day-1 close keeps ≥ this share of the event move


class SleeveRulesCfg(_Strict):
    S1: S1Cfg
    S2: S2Cfg
    S3: S3Cfg
    S4: S4Cfg
    S5: S5Cfg


class ExitFitCfg(_Strict):
    horizon: int = Field(gt=1)
    winner_horizon: int = Field(gt=0)
    winner_mae_pct: float = Field(gt=0, lt=100)
    sl_buffer: float = Field(ge=0)
    k_sl_bounds: tuple[float, float]
    partial_reach_frac: float = Field(gt=0, lt=1)
    partial_r_bounds: tuple[float, float]
    trail_giveback_pct: float = Field(gt=0, lt=100)
    trail_k_bounds: tuple[float, float]
    time_stop_bounds: tuple[int, int]
    min_events: int = Field(gt=0)


class SleeveValidationCfg(_Strict):
    sharpe_min_oos: float
    stress_expectancy_min: float
    min_oos_trades: int = Field(ge=0)
    calib_bins: int = Field(gt=1)


class SleevesCfg(_Strict):
    version: str
    enabled: dict[str, bool]
    allocation: AllocationCfg
    reentry_cooldown_sessions: int = Field(ge=0)
    rules: SleeveRulesCfg
    fit: ExitFitCfg
    validation: SleeveValidationCfg


# --------------------------------------------------------------------------- factors.yaml
class GateCfg(_Strict):
    min_history: int = Field(gt=0)
    continuity_window: int = Field(gt=0)
    min_continuity: float = Field(gt=0, le=1)
    lock_window: int = Field(gt=0)
    max_lock_streak: int = Field(ge=0)
    exclude_surveillance: bool = True  # D31: ASM/GSM/ESM/IRP/default/BZ (NSE, point-in-time)


class WindowsCfg(_Strict):
    windows: list[int]


class MomentumCfg(_Strict):
    lookbacks: list[int]
    skip: int = Field(ge=0)


class CompressionCfg(_Strict):
    short: int = Field(gt=1)
    long: int = Field(gt=1)
    rank_window: int = Field(gt=1)


class RangeCfg(_Strict):
    window: int = Field(gt=1)
    vol_window: int = Field(gt=1)


class WindowCfg(_Strict):
    window: int = Field(gt=1)


class HurstCfg(_Strict):
    window: int = Field(gt=1)
    q: int = Field(gt=1)


class KalmanCfg(_Strict):
    obs_var: float = Field(gt=0)
    level_var: float = Field(gt=0)
    slope_var: float = Field(gt=0)
    init_slope_var: float = Field(gt=0)


class BetaCfg(_Strict):
    window: int = Field(gt=1)
    benchmark: str


class ResidualMomentumCfg(_Strict):
    window: int = Field(gt=1)
    skip: int = Field(ge=0)


class XsecCfg(_Strict):
    winsor_lower: float = Field(ge=0, lt=0.5)
    winsor_upper: float = Field(gt=0.5, le=1)
    min_names_per_date: int = Field(gt=2)
    neutralise_size: bool
    neutralise_sector: bool = True  # D31: NSE sector dummies (current classification)
    min_sector_names: int = Field(default=10, gt=0)  # smaller sectors pooled into "Other"


class SectorMomentumCfg(_Strict):
    window: int = Field(gt=1)
    skip: int = Field(ge=0)
    beta_window: int = Field(gt=1)


class EarningsReactionCfg(_Strict):
    hold_sessions: int = Field(gt=0)  # EARN value valid this long after the event


class CombineCfg(_Strict):
    lookback_days: int = Field(gt=0)
    shrink_to_equal: float = Field(ge=0, le=1)
    min_history_days: int = Field(gt=0)
    sign_aware: bool = False
    min_t: float = Field(default=0.0, ge=0)  # v2: only factors with trailing |t| ≥ min_t


class IcCfg(_Strict):
    horizons: list[int]
    primary_horizon: int = Field(gt=0)
    min_names_per_date: int = Field(gt=2)
    t_stat_pass: float = Field(gt=0)
    combine: CombineCfg


class MacroCfg(_Strict):
    """D39 cross-asset (Fyers-only) drivers → per-stock sensitivities and spill-over factors."""

    enabled: bool = True
    beta_window: int = Field(default=250, gt=20)  # sessions for the rolling sensitivities
    min_obs: int = Field(default=120, gt=10)
    move_window: int = Field(default=10, gt=0)  # driver move that "spills over" (sessions)
    symbols_file: str = "data/processed/macro_symbols.json"  # written by macro_backfill.py
    # drivers from settings.data.indices (name → Fyers symbol)
    base: dict[str, str] = {
        "N50": "NSE:NIFTY50-INDEX",
        "MID": "NSE:NIFTYMIDCAP150-INDEX",
        "SMALL": "NSE:NIFTYSMLCAP250-INDEX",
        "BANK": "NSE:NIFTYBANK-INDEX",
        "VIX": "NSE:INDIAVIX-INDEX",
    }


class FactorsCfg(_Strict):
    version: str
    annualisation_days: int = Field(gt=0)
    crore: float = Field(gt=0)
    gate: GateCfg
    trend: WindowsCfg
    momentum: MomentumCfg
    compression: CompressionCfg
    range: RangeCfg
    volume_surprise: WindowCfg
    hurst: HurstCfg
    illiquidity: WindowCfg
    deviation: WindowCfg
    ou_half_life: WindowCfg
    kalman: KalmanCfg
    high52: WindowCfg
    beta: BetaCfg
    residual_momentum: ResidualMomentumCfg
    size_proxy: WindowCfg
    delivery_surprise: WindowCfg
    sector_momentum: SectorMomentumCfg
    earnings_reaction: EarningsReactionCfg
    macro: MacroCfg = Field(default_factory=MacroCfg)
    xsec: XsecCfg
    ic: IcCfg


# --------------------------------------------------------------------------- backtest.yaml
class BaselineCfg(_Strict):
    name: str
    signal: str
    min_score: float
    max_new_per_day: int = Field(gt=0)
    max_entry_gap_sigma: float = Field(gt=0)


class ExitsCfg(_Strict):
    sigma_window: int = Field(gt=1)
    k_sl_sigma: float = Field(gt=0)
    partial_at_r: float = Field(gt=0)
    partial_fraction: float = Field(gt=0, lt=1)
    move_sl_to_breakeven: bool
    trail_sigma_window: int = Field(gt=1)
    trail_k_below_switch: float = Field(gt=0)
    trail_k_above_switch: float = Field(gt=0)
    trail_switch_r: float = Field(gt=0)
    time_stop_sessions: int = Field(gt=0)
    time_stop_min_r: float
    max_hold_sessions: int = Field(gt=0)
    reentry_cooldown_sessions: int = Field(ge=0)


class SizingCfg(_Strict):
    target_position_vol: float = Field(gt=0)
    adv_window: int = Field(gt=0)


class BtExecutionCfg(_Strict):
    impact_vol_window: int = Field(gt=1)
    cost_stress_multiplier: float = Field(gt=0)
    delist_exit_haircut: float = Field(default=0.3, ge=0, le=1)  # D31: exit at last px × (1−h)
    credit_dividends: bool = True  # 04 §2: dividends credited on the ex-date (NSE, D31)


class EventWindowCfg(_Strict):
    enabled: bool
    before: int = Field(ge=0)  # sessions before a scheduled results meeting
    after: int = Field(ge=0)  # sessions after results were published


class LabelsCfg(_Strict):
    upper_r: float = Field(gt=0)
    k_sl_sigma: float = Field(gt=0)
    max_hold_sessions: int = Field(gt=0)


class BtValidationCfg(_Strict):
    train_years: int = Field(gt=0)
    test_years: int = Field(gt=0)
    embargo_sessions: int = Field(ge=0)
    montecarlo_sims: int = Field(gt=0)
    montecarlo_seed: int
    montecarlo_dd_mult_pass: float = Field(gt=0)
    dsr_pass: float = Field(gt=0, lt=1)
    max_year_pnl_share: float = Field(gt=0, le=1)


class TrackingCfg(_Strict):
    runs_dir: Path
    use_mlflow: bool


class BacktestCfg(_Strict):
    version: str
    start: date
    risk_free_annual: float = Field(ge=0)
    baseline: BaselineCfg
    exits: ExitsCfg
    sizing: SizingCfg
    execution: BtExecutionCfg
    labels: LabelsCfg
    validation: BtValidationCfg
    tracking: TrackingCfg


# --------------------------------------------------------------------------- nse.yaml (D31)
class NseHttpCfg(_Strict):
    site_url: str
    api_url: str
    archive_hosts: list[str]
    warmup_path: str
    impersonate: str
    use_system_certs: bool
    user_agent: str
    min_interval_s: float = Field(ge=0)
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    backoff_s: float = Field(ge=0)
    rewarm_every: int = Field(gt=0)


class NseStorageCfg(_Strict):
    raw_dir: Path
    processed_dir: Path
    state_file: Path
    keep_raw: bool


class NseFilesCfg(_Strict):
    bhav: bool
    delivery: bool
    surveillance: bool
    udiff_switch: date
    surveillance_probe_from: date
    retry_recent_misses_days: int = Field(ge=0)


class NseApiCfg(_Strict):
    chunk_days: int = Field(gt=0)
    refresh_recent_chunks: int = Field(ge=0)
    sector_refresh_days: int = Field(gt=0)


class SurvivorshipCfg(_Strict):
    enabled: bool
    delist_confirm_sessions: int = Field(gt=0)
    min_sessions: int = Field(gt=0)


class NseCfg(_Strict):
    version: str
    enabled: bool
    start: date
    http: NseHttpCfg
    storage: NseStorageCfg
    files: NseFilesCfg
    api: NseApiCfg
    series: list[str]
    results_cutoff: time
    survivorship: SurvivorshipCfg
    surveillance_exclude: list[
        Literal["gsm", "asm_long", "asm_short", "esm", "irp", "default", "bz", "pledge"]
    ]


# --------------------------------------------------------------------------- paper.yaml (D32)
class PaperScheduleCfg(_Strict):
    time: time
    task_name: str


class PaperLoginCfg(_Strict):
    via_telegram: bool
    wait_minutes: int = Field(gt=0)
    poll_seconds: int = Field(gt=0)


class PaperTelegramCfg(_Strict):
    enabled: bool
    timeout_s: float = Field(gt=0)


class PaperExitCfg(_Strict):
    min_sessions: int = Field(gt=0)
    max_replay_gap: float = Field(gt=0)


class LiveCfg(_Strict):
    """D40: live paper trading of the D37 trade engine."""

    db_file: Path = Path("data/db/live.sqlite")
    capital: float = Field(default=500000.0, gt=0)  # starting paper capital (editable in the app)
    poll_minutes: int = Field(default=5, ge=1, le=60)
    poll_start: time = time(9, 10)  # the Windows task runs between these times, Mon–Fri
    poll_end: time = time(15, 35)
    task_name: str = "QuantSwing Live Poll"
    eod_log: Path = Path("logs/live_eod.log")
    poll_log: Path = Path("logs/live_poll.log")
    callback_timeout_s: int = Field(default=300, gt=10)  # wait this long for the Fyers redirect


class PaperCfg(_Strict):
    version: str
    enabled: bool
    db_file: Path
    log_file: Path
    start_capital_from_risk: bool
    schedule: PaperScheduleCfg
    login: PaperLoginCfg
    telegram: PaperTelegramCfg
    exit_test: PaperExitCfg
    live: LiveCfg = Field(default_factory=LiveCfg)


# --------------------------------------------------------------------------- root
class AppConfig(BaseSettings):
    """Merged, validated configuration from the four YAML files (+ ``QS_`` env overrides)."""

    settings: SettingsCfg
    costs: CostsCfg
    risk: RiskCfg
    sleeves: SleevesCfg
    factors: FactorsCfg
    backtest: BacktestCfg
    nse: NseCfg
    paper: PaperCfg

    model_config = SettingsConfigDict(
        env_prefix="QS_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init kwargs > QS_ env vars > YAML files."""
        cdir = config_dir()
        files = [cdir / f for f in CONFIG_FILES]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing config file(s): {missing}")
        yaml_src = YamlConfigSettingsSource(
            settings_cls, yaml_file=files, yaml_file_encoding="utf-8"
        )
        return (init_settings, env_settings, yaml_src)


class Secrets(BaseSettings):
    """Secrets from ``.env`` (never committed, never logged)."""

    FYERS_CLIENT_ID: SecretStr
    FYERS_SECRET_KEY: SecretStr
    FYERS_REDIRECT_URI: str
    TELEGRAM_BOT_TOKEN: SecretStr | None = None
    TELEGRAM_CHAT_ID: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the application config."""
    return AppConfig()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Load and cache secrets from ``.env`` / environment."""
    return Secrets()
