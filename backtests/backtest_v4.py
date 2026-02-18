#!/usr/bin/env python3
"""
PTJ 매매법 v4 - 1분봉 백테스트 시뮬레이션 (선별 매매형)
=======================================================
v2 엔진을 기반으로 v4 규칙 적용:
- 쌍둥이 ENTRY 갭: 1.5% → 2.2%
- 물타기: 7회 → 4회, 종목당 최대 1,000만 → 700만
- COIN/CONL 트리거: 3.0% → 4.5%
- 중복 쿨타임: 5분 → 20분
- 종목당 일일 1회 거래
- 진입 시간 제한 (V4_ENTRY_CUTOFF 이후 매수 금지)
- 횡보장 감지 → 현금 100%

Data:
  - data/backtest_1min_v2.parquet  (동일 데이터 사용)
  - polymarket/history/*.json

Dependencies:
  - config.py          : v4 parameters
  - signals_v4.py      : v4 signal functions
  - backtest_common.py : shared utilities
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
import signals_v4
import backtest_common

# ============================================================
# Constants
# ============================================================
MARKET_OPEN = time(8, 0)
MARKET_CLOSE = time(16, 0)
ENTRY_CUTOFF = time(config.V4_ENTRY_CUTOFF_HOUR, config.V4_ENTRY_CUTOFF_MINUTE)
ENTRY_DEFAULT_START = time(config.V4_ENTRY_DEFAULT_START_HOUR, config.V4_ENTRY_DEFAULT_START_MINUTE)
ENTRY_EARLY_START = time(config.V4_ENTRY_EARLY_START_HOUR, config.V4_ENTRY_EARLY_START_MINUTE)
CRASH_BUY_TIME = time(config.V4_CRASH_BUY_TIME_HOUR, config.V4_CRASH_BUY_TIME_MINUTE)


# ============================================================
# Data Models (v2와 동일 구조)
# ============================================================
@dataclass
class Position:
    """개별 포지션 상태."""
    ticker: str
    entries: list
    total_qty: float
    avg_price: float
    total_invested_krw: float
    dca_count: int
    initial_entry_price: float
    first_entry_time: datetime
    staged_sell_active: bool = False
    staged_sell_start: datetime | None = None
    staged_sell_remaining: float = 0.0
    staged_sell_step: int = 0
    fixed_tp_active: bool = False
    fixed_tp_target_pct: float = 0.0
    entry_atr: float = 0.0
    overheat_origin: str = ""
    is_crash_buy: bool = False
    crash_entry_date: date | None = None
    crash_ref_close: float = 0.0
    crash_flat_monitor_end: datetime | None = None
    uses_daily_fallback: bool = False
    is_swing: bool = False
    swing_role: str = ""
    swing_peak_price: float = 0.0
    carry: bool = False
    signal_type: str = ""
    is_conl_conditional: bool = False
    is_coin_conditional: bool = False


@dataclass
class TradeV4:
    """매매 기록."""
    date: date
    ticker: str
    side: str
    price: float
    qty: float
    amount_krw: float
    pnl_krw: float
    pnl_pct: float
    signal_type: str
    exit_reason: str = ""
    dca_level: int = 0
    fees_krw: float = 0.0
    net_pnl_krw: float = 0.0
    entry_time: object = None
    exit_time: object = None


# ============================================================
# Backtest Engine v4
# ============================================================
class BacktestEngineV4:
    def __init__(
        self,
        initial_capital_krw: float = config.V4_TOTAL_CAPITAL,
        start_date: date = date(2025, 2, 18),
        end_date: date = date(2026, 2, 17),
        use_fees: bool = True,
    ):
        # NOTE:
        # - 필드명은 하위 호환을 위해 *_krw를 유지하지만
        # - 값 자체는 USD 단위로 계산/저장한다.
        self.initial_capital_krw = initial_capital_krw
        self.cash_krw = initial_capital_krw
        self.start_date = start_date
        self.end_date = end_date
        self.use_fees = use_fees
        self._fx_fallback = 1.0
        self._fx_series: pd.Series | None = None
        self._current_fx_rate: float = 1.0
        self._verbose = False

        # State
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeV4] = []
        self.equity_curve: list[tuple[date, float]] = []
        self._sold_today: set[str] = set()
        self._traded_today: dict[str, int] = {}  # v4: 종목별 당일 트레이드 수
        self._last_buy_time: dict[str, datetime] = {}

        # v4 횡보장 상태
        self._sideways_active: bool = False
        self._sideways_last_eval: datetime | None = None
        self._gap_fail_count: int = 0      # 당일 갭 수렴 실패 횟수
        self._trigger_fail_count: int = 0  # 당일 트리거 불발 횟수
        self._gap_entry_tickers: set[str] = set()  # 당일 ENTRY 후 수렴 대기 중
        self._trigger_entry_active: bool = False    # 당일 트리거 발동 여부
        self._high_vol_hits: dict[str, int] = {}
        self._high_vol_last_above: set[str] = set()

        # v4 통계
        self.sideways_days: int = 0
        self.sideways_blocks: int = 0      # 횡보장으로 차단된 매수 수
        self.entry_cutoff_blocks: int = 0  # 시간 제한으로 차단된 매수 수
        self.daily_limit_blocks: int = 0   # 일일 1회 제한으로 차단된 매수 수
        self.cb_buy_blocks: int = 0        # 서킷브레이커로 차단된 매수 수
        self.cb_sell_halt_bars: int = 0    # 서킷브레이커로 매도 로직 정지된 바 수

        # v4 서킷브레이커 상태
        self._cb_vix_cooldown_days: int = 0
        self._cb_gld_cooldown_days: int = 0
        self._cb_block_all: bool = False
        self._cb_block_new_buys: bool = False
        self._cb_overheated_tickers: set[str] = set()
        self._cb_last_flags: dict[str, bool] = {
            "CB-1_VIX_SPIKE": False,
            "CB-2_GLD_SPIKE": False,
            "CB-3_BTC_CRASH": False,
            "CB-4_BTC_SURGE": False,
            "CB-5_RATE_HIKE": False,
            "CB-6_OVERHEAT_ANY": False,
        }
        self._cb_overheat_state: dict[str, dict[str, float | bool]] = {}
        self._conl_indicators: dict[object, dict[str, float]] = {}
        self._atr_by_day: dict[tuple[str, date], float] = {}
        self._high_vol_by_day: dict[tuple[str, date], bool] = {}
        self._sideways_daily_metrics: dict[date, dict[str, bool]] = {}
        self._entry_early_ok: dict[tuple[str, object], bool] = {}
        self._replacement_daily_close: dict[tuple[str, date], float] = {}
        self._intraday_tickers: set[str] = set()
        self._spy_day_open: float | None = None
        self._spy_day_high: float | None = None
        self._spy_day_low: float | None = None
        self._crash_buy_active_today: bool = False
        self._swing_active: bool = False
        self._swing_variant: str = ""  # momentum | vix
        self._swing_stage: int = 0     # 0:none, 1:first stage, 2:second stage
        self._swing_stage_start_date: date | None = None
        self._swing_stage_elapsed_days: int = 0
        self._swing_last_counted_date: date | None = None
        self._swing_stage1_targets: set[str] = set()
        self._swing_stage2_ticker: str = ""
        self._swing_last_trigger_date: date | None = None

        # Fee accumulators
        self.total_buy_fees_krw: float = 0.0
        self.total_sell_fees_krw: float = 0.0

        # Statistics
        self.total_trading_days: int = 0
        self.skipped_gold_bars: int = 0
        self.conl_filter_blocks: int = 0
        self.cb_overheat_switches: int = 0
        self.cb_overheat_no_substitute_blocks: int = 0
        self.entry_start_blocks: int = 0
        self.crash_buy_count: int = 0
        self.cb_events: list[dict] = []
        self.swing_events: list[dict] = []
        self.swing_entry_count: int = 0
        self.swing_exit_count: int = 0

    # ----------------------------------------------------------
    # FX rate lookup
    # ----------------------------------------------------------
    def _get_fx_rate(self, ts) -> float:
        _ = ts
        return 1.0

    # ----------------------------------------------------------
    # Net profit calculation
    # ----------------------------------------------------------
    def _calc_net_profit_pct(self, pos: Position, cur_price: float) -> float:
        gross_value_usd = pos.total_qty * cur_price
        gross_value_krw = gross_value_usd
        if self.use_fees:
            sell_fee = backtest_common.calc_sell_fee(gross_value_krw)
        else:
            sell_fee = 0.0
        net_value_krw = gross_value_krw - sell_fee
        if pos.total_invested_krw <= 0:
            return 0.0
        return (net_value_krw - pos.total_invested_krw) / pos.total_invested_krw * 100

    # ----------------------------------------------------------
    # v4: Buy eligibility (시간/일일제한/횡보장/쿨타임)
    # ----------------------------------------------------------
    def _can_buy(self, ticker: str, ts: datetime, is_conditional: bool = False) -> bool:
        """v4 매수 가능 여부. 횡보장/시간/일일제한/쿨타임 검증.

        Parameters
        ----------
        is_conditional : bool
            True이면 조건부 매매(COIN/CONL) — 진입 마감 시간 면제.
            (KST 17:30 프리마켓부터 가능 → 정규장 전체 시간 허용)
        """
        # 1. v4 서킷브레이커 → 신규 매수 차단
        if self._cb_block_new_buys:
            self.cb_buy_blocks += 1
            return False

        # 1-1. 과열 종목 추가 매수 금지 (CB-6)
        if ticker in self._cb_overheated_tickers:
            self.cb_buy_blocks += 1
            return False

        # 2. 횡보장 모드 → 매수 전면 차단
        if self._sideways_active:
            self.sideways_blocks += 1
            return False

        # 3. 진입 시간 제한 (V4_ENTRY_CUTOFF 이후 매수 금지)
        #    조건부 매매는 면제 (KST 17:30 프리마켓부터 가능)
        if not (is_conditional and config.V4_CONDITIONAL_EXEMPT_CUTOFF):
            bar_time = ts.time() if hasattr(ts, 'time') else None
            start_time = self._get_entry_start_time(ticker, ts)
            if bar_time and bar_time < start_time:
                self.entry_start_blocks += 1
                return False
            if bar_time and bar_time >= ENTRY_CUTOFF:
                self.entry_cutoff_blocks += 1
                return False

        # 4. 이미 보유 중이면 신규 매수 불가 (DCA는 별도)
        if ticker in self.positions:
            return False

        # 5. 당일 재매수 금지 (v2 동일)
        if ticker in self._sold_today:
            return False

        # 6. 종목당 일일 1트레이드 제한 (v4 신규)
        if self._traded_today.get(ticker, 0) >= config.V4_MAX_DAILY_TRADES_PER_STOCK:
            self.daily_limit_blocks += 1
            return False

        # 7. 쿨타임 (v4: 20분)
        last = self._last_buy_time.get(ticker)
        if last and (ts - last).total_seconds() < config.V4_SPLIT_BUY_INTERVAL_MIN * 60:
            return False

        return True

    def _can_dca(self, ticker: str, ts: datetime, is_conditional: bool = False) -> bool:
        """v4 물타기 가능 여부."""
        # 서킷브레이커 → 물타기도 차단
        if self._cb_block_new_buys:
            self.cb_buy_blocks += 1
            return False

        # 과열 종목 추가 매수 금지 (CB-6)
        if ticker in self._cb_overheated_tickers:
            self.cb_buy_blocks += 1
            return False

        # 횡보장 → 물타기도 차단
        if self._sideways_active:
            self.sideways_blocks += 1
            return False

        # 진입 시간 제한 → 조건부 매매는 면제
        if not (is_conditional and config.V4_CONDITIONAL_EXEMPT_CUTOFF):
            bar_time = ts.time() if hasattr(ts, 'time') else None
            start_time = self._get_entry_start_time(ticker, ts)
            if bar_time and bar_time < start_time:
                self.entry_start_blocks += 1
                return False
            if bar_time and bar_time >= ENTRY_CUTOFF:
                self.entry_cutoff_blocks += 1
                return False

        # 쿨타임
        last = self._last_buy_time.get(ticker)
        if last and (ts - last).total_seconds() < config.V4_SPLIT_BUY_INTERVAL_MIN * 60:
            return False

        return True

    # ----------------------------------------------------------
    # Market time elapsed
    # ----------------------------------------------------------
    @staticmethod
    def _market_minutes_elapsed(entry_time: datetime, current_time: datetime) -> float:
        if current_time <= entry_time:
            return 0.0
        entry_date = entry_time.date()
        current_date = current_time.date()
        total_minutes = 0.0
        d = entry_date
        while d <= current_date:
            day_open = datetime.combine(d, MARKET_OPEN)
            day_close = datetime.combine(d, MARKET_CLOSE)
            if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is not None:
                day_open = day_open.replace(tzinfo=entry_time.tzinfo)
                day_close = day_close.replace(tzinfo=entry_time.tzinfo)
            if d == entry_date:
                effective_start = max(entry_time, day_open)
            else:
                effective_start = day_open
            if d == current_date:
                effective_end = min(current_time, day_close)
            else:
                effective_end = day_close
            if effective_end > effective_start:
                total_minutes += (effective_end - effective_start).total_seconds() / 60
            d += timedelta(days=1)
        return total_minutes

    # ----------------------------------------------------------
    # Coin follow selection
    # ----------------------------------------------------------
    def _select_coin_follow(self, sym_bars_day: dict[str, list[dict]], trading_date=None) -> str:
        candidates = ["MSTU", "IRE"]
        stats: dict[str, dict] = {}
        for ticker in candidates:
            bars = sym_bars_day.get(ticker, [])[:30]
            if not bars:
                continue
            open_price = bars[0]["open"]
            if open_price <= 0:
                continue
            high_max = max(b["high"] for b in bars)
            low_min = min(b["low"] for b in bars)
            vol = (high_max - low_min) / open_price * 100
            volume = sum(b["volume"] for b in bars)
            stats[ticker] = {"volatility": vol, "volume": volume}
        if len(stats) == 0:
            return "MSTU"
        if len(stats) == 1:
            return list(stats.keys())[0]
        mstu_vol = stats["MSTU"]["volatility"]
        ire_vol = stats["IRE"]["volatility"]
        diff = abs(mstu_vol - ire_vol)
        if diff >= config.COIN_FOLLOW_VOLATILITY_GAP:
            return "MSTU" if mstu_vol > ire_vol else "IRE"
        else:
            return "MSTU" if stats["MSTU"]["volume"] >= stats["IRE"]["volume"] else "IRE"

    def _prepare_conl_indicators(self, df: pd.DataFrame) -> None:
        """CONL ADX(14) / EMA slope 지표를 timestamp 단위로 사전 계산한다."""
        conl = df.loc[df["symbol"] == "CONL", ["timestamp", "high", "low", "close"]].copy()
        if conl.empty:
            self._conl_indicators = {}
            return

        conl.sort_values("timestamp", inplace=True)
        conl.reset_index(drop=True, inplace=True)

        high = conl["high"].astype(float)
        low = conl["low"].astype(float)
        close = conl["close"].astype(float)

        prev_close = close.shift(1)
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=conl.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=conl.index,
        )

        adx_period = 14
        atr = tr.ewm(alpha=1 / adx_period, adjust=False).mean()
        plus_di = 100.0 * (
            plus_dm.ewm(alpha=1 / adx_period, adjust=False).mean()
            / atr.replace(0.0, np.nan)
        )
        minus_di = 100.0 * (
            minus_dm.ewm(alpha=1 / adx_period, adjust=False).mean()
            / atr.replace(0.0, np.nan)
        )
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        adx = dx.ewm(alpha=1 / adx_period, adjust=False).mean()

        ema = close.ewm(span=config.V4_CONL_EMA_PERIOD, adjust=False).mean()
        ema_prev = ema.shift(max(config.V4_CONL_EMA_SLOPE_LOOKBACK, 1))
        ema_slope_pct = (ema - ema_prev) / ema_prev.replace(0.0, np.nan) * 100.0

        indicator_df = pd.DataFrame(
            {
                "timestamp": conl["timestamp"],
                "adx": adx,
                "ema_slope_pct": ema_slope_pct,
            }
        ).dropna(subset=["adx", "ema_slope_pct"])

        self._conl_indicators = {
            row.timestamp: {
                "adx": float(row.adx),
                "ema_slope_pct": float(row.ema_slope_pct),
            }
            for row in indicator_df.itertuples(index=False)
        }

    def _passes_conl_entry_filter(self, ts: datetime) -> bool:
        """v4 CONL 진입 필터: ADX(14) >= 18, EMA slope > 0."""
        metric = self._conl_indicators.get(ts)
        if metric is None:
            self.conl_filter_blocks += 1
            return False

        adx_ok = metric["adx"] >= config.V4_CONL_ADX_MIN
        ema_ok = metric["ema_slope_pct"] > config.V4_CONL_EMA_SLOPE_MIN_PCT

        if not (adx_ok and ema_ok):
            self.conl_filter_blocks += 1
            return False

        return True

    @staticmethod
    def _build_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
        """1분봉을 일봉(OHLCV)으로 집계한다."""
        daily = (
            df.sort_values(["symbol", "timestamp"])
            .groupby(["symbol", "date"], as_index=False)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
        )
        return daily

    def _prepare_daily_risk_metrics(self, daily: pd.DataFrame) -> None:
        """ATR14와 최근 5거래일 고변동성 플래그를 (symbol, date) 단위로 계산한다."""
        atr_by_day: dict[tuple[str, date], float] = {}
        high_vol_by_day: dict[tuple[str, date], bool] = {}

        for symbol, g in daily.groupby("symbol"):
            g = g.sort_values("date").copy()
            prev_close = g["close"].shift(1)
            tr = pd.concat(
                [
                    (g["high"] - g["low"]).abs(),
                    (g["high"] - prev_close).abs(),
                    (g["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
            atr14_known = atr14.shift(1)

            daily_change_pct = g["close"].pct_change() * 100.0
            high_vol_hit = daily_change_pct.abs() >= config.V4_HIGH_VOL_MOVE_PCT
            high_vol_count_5d = high_vol_hit.rolling(window=5, min_periods=1).sum()
            high_vol_known = (high_vol_count_5d >= config.V4_HIGH_VOL_HIT_COUNT).shift(1, fill_value=False)

            for idx, row in g.iterrows():
                d = row["date"]
                atr_val = atr14_known.loc[idx]
                if pd.notna(atr_val) and atr_val > 0:
                    atr_by_day[(symbol, d)] = float(atr_val)
                high_vol_by_day[(symbol, d)] = bool(high_vol_known.loc[idx])

        self._atr_by_day = atr_by_day
        self._high_vol_by_day = high_vol_by_day

    def _prepare_sideways_metrics(self, daily: pd.DataFrame) -> None:
        """v4 횡보장 6지표 중 1~5번(일봉 기반) 지표를 날짜별로 계산한다."""
        spy = daily.loc[daily["symbol"] == "SPY"].sort_values("date").copy()
        if spy.empty:
            self._sideways_daily_metrics = {}
            return

        prev_close = spy["close"].shift(1)
        tr = pd.concat(
            [
                (spy["high"] - spy["low"]).abs(),
                (spy["high"] - prev_close).abs(),
                (spy["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
        atr20 = atr14.rolling(window=20, min_periods=20).mean()
        atr_decline = atr14 <= atr20 * (1.0 - config.V4_SIDEWAYS_ATR_DECLINE_PCT / 100.0)
        atr_decline = atr_decline.shift(1, fill_value=False)

        vol20 = spy["volume"].rolling(window=20, min_periods=20).mean()
        volume_decline = spy["volume"] <= vol20 * (1.0 - config.V4_SIDEWAYS_VOLUME_DECLINE_PCT / 100.0)
        volume_decline = volume_decline.shift(1, fill_value=False)

        ema20 = spy["close"].ewm(span=20, adjust=False).mean()
        ema_prev = ema20.shift(max(config.V4_SIDEWAYS_EMA_SLOPE_LOOKBACK_DAYS, 1))
        ema_slope_pct = ((ema20 - ema_prev) / ema_prev.replace(0.0, np.nan) * 100.0).abs()
        ema_flat = (ema_slope_pct <= config.V4_SIDEWAYS_EMA_SLOPE_MAX).shift(1, fill_value=False)

        diff = spy["close"].diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_box = (
            (rsi >= config.V4_SIDEWAYS_RSI_LOW)
            & (rsi <= config.V4_SIDEWAYS_RSI_HIGH)
        ).shift(1, fill_value=False)

        sma20 = spy["close"].rolling(window=20, min_periods=20).mean()
        std20 = spy["close"].rolling(window=20, min_periods=20).std(ddof=0)
        bb_width = ((sma20 + 2 * std20) - (sma20 - 2 * std20)) / sma20.replace(0.0, np.nan) * 100.0
        bb_pct_window = max(config.V4_SIDEWAYS_BB_PERCENTILE_WINDOW_DAYS, 20)
        bb_width_percentile = bb_width.rolling(window=bb_pct_window, min_periods=20).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1] * 100.0),
            raw=False,
        )
        bb_narrow = (bb_width_percentile <= config.V4_SIDEWAYS_BB_WIDTH_PERCENTILE).shift(1, fill_value=False)

        metrics = {}
        for idx, row in spy.iterrows():
            d = row["date"]
            metrics[d] = {
                "atr_decline": bool(atr_decline.loc[idx]),
                "volume_decline": bool(volume_decline.loc[idx]),
                "ema_flat": bool(ema_flat.loc[idx]),
                "rsi_box": bool(rsi_box.loc[idx]),
                "bb_narrow": bool(bb_narrow.loc[idx]),
            }
        self._sideways_daily_metrics = metrics

    def _get_sideways_indicator_state(self, trading_date: date) -> dict[str, bool]:
        """현재 날짜의 횡보장 일봉 지표 상태를 반환한다."""
        base = self._sideways_daily_metrics.get(trading_date, {})
        return {
            "atr_decline": bool(base.get("atr_decline", False)),
            "volume_decline": bool(base.get("volume_decline", False)),
            "ema_flat": bool(base.get("ema_flat", False)),
            "rsi_box": bool(base.get("rsi_box", False)),
            "bb_narrow": bool(base.get("bb_narrow", False)),
            "range_narrow": False,
        }

    def _update_spy_intraday_range(self, cur_prices: dict[str, float]) -> float:
        """당일 SPY 장중 변동폭(%)을 업데이트해서 반환한다."""
        spy = cur_prices.get("SPY")
        if spy is None or spy <= 0:
            return float("inf")
        if self._spy_day_open is None:
            self._spy_day_open = spy
            self._spy_day_high = spy
            self._spy_day_low = spy
        else:
            self._spy_day_high = max(self._spy_day_high or spy, spy)
            self._spy_day_low = min(self._spy_day_low or spy, spy)
        if not self._spy_day_open:
            return float("inf")
        return (self._spy_day_high - self._spy_day_low) / self._spy_day_open * 100.0

    def _is_high_vol_active(self, ticker: str, trading_date: date) -> bool:
        return bool(self._high_vol_by_day.get((ticker, trading_date), False))

    def _get_entry_atr(self, ticker: str, trading_date: date, price: float) -> float:
        atr = self._atr_by_day.get((ticker, trading_date))
        if atr is None or atr <= 0:
            # 초반 구간 ATR 데이터가 부족할 수 있어 보수적 fallback 적용
            return max(price * 0.03, 0.01)
        return float(atr)

    def _prepare_entry_early_metrics(self, df: pd.DataFrame) -> None:
        """조기 진입(ADX + EMA + 거래량배수) 조건을 timestamp 단위로 계산한다."""
        early_ok: dict[tuple[str, object], bool] = {}

        raw = df.loc[:, ["symbol", "timestamp", "date", "high", "low", "close", "volume"]].copy()
        raw.sort_values(["symbol", "timestamp"], inplace=True)

        # 프리마켓 데이터가 없으면(최소 시각 >= 08:00 ET) 조기진입 로직은 의미가 없다.
        first_time = raw["timestamp"].dt.tz_convert("US/Eastern").dt.time.min()
        if first_time >= ENTRY_DEFAULT_START:
            self._entry_early_ok = {}
            return

        for symbol, g in raw.groupby("symbol"):
            g = g.copy()
            high = g["high"].astype(float)
            low = g["low"].astype(float)
            close = g["close"].astype(float)

            prev_close = close.shift(1)
            prev_high = high.shift(1)
            prev_low = low.shift(1)

            tr = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)

            adx_period = 14
            up_move = high - prev_high
            down_move = prev_low - low
            plus_dm = pd.Series(
                np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                index=g.index,
            )
            minus_dm = pd.Series(
                np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                index=g.index,
            )
            atr = tr.ewm(alpha=1 / adx_period, adjust=False).mean()
            plus_di = 100.0 * (
                plus_dm.ewm(alpha=1 / adx_period, adjust=False).mean()
                / atr.replace(0.0, np.nan)
            )
            minus_di = 100.0 * (
                minus_dm.ewm(alpha=1 / adx_period, adjust=False).mean()
                / atr.replace(0.0, np.nan)
            )
            dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
            adx = dx.ewm(alpha=1 / adx_period, adjust=False).mean()

            ema = close.ewm(span=config.V4_ENTRY_EMA_PERIOD, adjust=False).mean()
            ema_prev = ema.shift(max(config.V4_ENTRY_EMA_SLOPE_LOOKBACK, 1))
            ema_slope_pct = (ema - ema_prev) / ema_prev.replace(0.0, np.nan) * 100.0

            tg = g.loc[:, ["timestamp", "date", "volume"]].copy()
            tg["tod"] = tg["timestamp"].dt.tz_convert("US/Eastern").dt.strftime("%H:%M")
            tg.sort_values(["tod", "date"], inplace=True)
            tg["vol_avg20"] = tg.groupby("tod")["volume"].transform(
                lambda s: s.shift(1).rolling(window=20, min_periods=5).mean()
            )
            tg.sort_values("timestamp", inplace=True)
            vol_ratio = tg["volume"].astype(float) / tg["vol_avg20"].replace(0.0, np.nan)

            cond = (
                (adx >= config.V4_ENTRY_TREND_ADX_MIN)
                & (ema_slope_pct > 0.0)
                & (vol_ratio >= config.V4_ENTRY_VOLUME_RATIO_MIN)
            )

            for ts, ok in zip(g["timestamp"], cond):
                if bool(ok):
                    early_ok[(symbol, ts)] = True

        self._entry_early_ok = early_ok

    @staticmethod
    def _is_close_bar(ts: datetime) -> bool:
        bar_time = ts.time() if hasattr(ts, "time") else None
        return bool(bar_time and bar_time >= time(15, 59))

    def _can_use_intraday_price(self, pos: Position, ts: datetime) -> bool:
        """일봉 fallback 종목은 EOD 바에서만 가격기반 규칙을 적용한다."""
        if not pos.uses_daily_fallback:
            return True
        return self._is_close_bar(ts)

    def _load_replacement_daily_prices(self) -> None:
        """SOXX/IREN 일봉 fallback 가격을 로드한다."""
        path = config.PROJECT_ROOT / "stock_history" / "soxx_iren_daily.parquet"
        if not path.exists():
            self._replacement_daily_close = {}
            return

        try:
            df = pd.read_parquet(path)
        except Exception:
            self._replacement_daily_close = {}
            return

        required = {"symbol", "close"}
        if not required.issubset(df.columns):
            self._replacement_daily_close = {}
            return

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("US/Eastern")
            dates = ts.dt.date
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.date
        else:
            self._replacement_daily_close = {}
            return

        out: dict[tuple[str, date], float] = {}
        for symbol, d, c in zip(df["symbol"], dates, df["close"]):
            if pd.isna(c):
                continue
            out[(str(symbol), d)] = float(c)

        self._replacement_daily_close = out

    def _is_early_entry_allowed(self, ticker: str, ts: datetime) -> bool:
        return bool(self._entry_early_ok.get((ticker, ts), False))

    def _get_entry_start_time(self, ticker: str, ts: datetime) -> time:
        if self._is_early_entry_allowed(ticker, ts):
            return ENTRY_EARLY_START
        return ENTRY_DEFAULT_START

    def _process_crash_buy(self, ts: datetime, cur_prices: dict[str, float], changes: dict[str, dict]) -> bool:
        """v4 급락 역매수(-40%, 15:55 ET, 95%)를 처리한다."""
        if any(pos.is_crash_buy for pos in self.positions.values()):
            # 급락 역매수 포지션 보유 중에는 당일 신규매수 중단
            return True
        if self._crash_buy_active_today:
            return True
        if not (hasattr(ts, "time") and ts.time() >= CRASH_BUY_TIME):
            return False
        if self._cb_block_all:
            return False

        candidates: list[tuple[str, float]] = []
        for ticker in config.V4_CRASH_BUY_STOCKS:
            pct = changes.get(ticker, {}).get("change_pct")
            price = cur_prices.get(ticker)
            if pct is None or price is None:
                continue
            if pct <= config.V4_CRASH_BUY_THRESHOLD_PCT:
                if ticker in self.positions or ticker in self._sold_today:
                    continue
                candidates.append((ticker, float(pct)))

        if not candidates:
            return False

        target = min(candidates, key=lambda x: x[1])[0]
        amount = min(
            self.cash_krw,
            self.initial_capital_krw * config.V4_CRASH_BUY_WEIGHT_PCT / 100.0,
        )
        if amount < 1.0:
            return False

        price = cur_prices.get(target)
        if price is None or price <= 0:
            return False

        self._buy(
            target,
            price,
            ts,
            "crash_buy",
            amount_krw=amount,
            is_crash_buy=True,
        )
        self._crash_buy_active_today = True
        self.crash_buy_count += 1
        return True

    def _process_crash_gap_open(
        self,
        trading_date: date,
        first_ts: datetime | None,
        first_prices: dict[str, float],
    ) -> None:
        """급락 역매수 포지션의 다음날 갭 시가 처리를 수행한다."""
        if not any(pos.is_crash_buy for pos in self.positions.values()):
            return

        if first_ts is None:
            return

        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.is_crash_buy:
                continue
            # 다음 거래일 오픈에서 1회만 판정
            if pos.crash_entry_date is None or pos.crash_entry_date >= trading_date:
                continue

            open_price = first_prices.get(ticker)
            if open_price is None or open_price <= 0:
                # 가격을 못 받으면 crash 상태는 유지하되 다음 바에서 재시도
                continue

            ref_close = pos.crash_ref_close if pos.crash_ref_close > 0 else pos.initial_entry_price
            if ref_close <= 0:
                continue

            gap_pct = (open_price - ref_close) / ref_close * 100.0
            if open_price > ref_close:
                self._sell_all(ticker, open_price, first_ts, "crash_gap_up")
                continue

            if abs(gap_pct) <= config.V4_CRASH_BUY_FLAT_BAND_PCT:
                pos.crash_flat_monitor_end = first_ts + timedelta(minutes=config.V4_CRASH_BUY_FLAT_OBSERVE_MIN)
                # 30분 관찰 후 일반 포지션으로 전환
            else:
                pos.crash_flat_monitor_end = None
                # 갭하락/보합대역 외는 즉시 일반 규칙으로 전환
                pos.is_crash_buy = False
            pos.carry = False

    def _resolve_overheat_buy_ticker(
        self, ticker: str, cur_prices: dict[str, float]
    ) -> str | None:
        """CB-6 과열 시 대체 종목 분기. 대체가 없거나 가격이 없으면 None."""
        if ticker not in self._cb_overheated_tickers:
            return ticker

        replacement = config.V4_CB_OVERHEAT_SWITCH_MAP.get(ticker)
        if replacement is None:
            self.cb_overheat_no_substitute_blocks += 1
            return None

        if cur_prices.get(replacement) is None:
            self.cb_overheat_no_substitute_blocks += 1
            return None

        return replacement

    # ----------------------------------------------------------
    # Buy execution
    # ----------------------------------------------------------
    def _buy(
        self,
        ticker: str,
        price: float,
        ts: datetime,
        signal_type: str,
        amount_krw: float,
        is_conl_conditional: bool = False,
        is_coin_conditional: bool = False,
        is_crash_buy: bool = False,
        overheat_origin: str = "",
        is_swing: bool = False,
        swing_role: str = "",
    ) -> None:
        if price <= 0 or amount_krw <= 0:
            return
        if amount_krw > self.cash_krw:
            amount_krw = self.cash_krw
        if amount_krw < 1.0:
            return

        if self.use_fees:
            buy_fee = backtest_common.calc_buy_fee(amount_krw)
            self.total_buy_fees_krw += buy_fee
        else:
            buy_fee = 0.0

        net_amount_krw = amount_krw - buy_fee
        qty = net_amount_krw / price

        self.cash_krw -= amount_krw

        entry = {"time": ts, "price": price, "qty": qty, "krw": amount_krw}

        if overheat_origin:
            self.cb_overheat_switches += 1

        if ticker in self.positions:
            pos = self.positions[ticker]
            pos.entries.append(entry)
            pos.total_qty += qty
            pos.total_invested_krw += amount_krw
            pos.avg_price = pos.total_invested_krw / pos.total_qty
            pos.dca_count += 1
            if is_swing:
                pos.is_swing = True
                if swing_role:
                    pos.swing_role = swing_role
                pos.swing_peak_price = max(pos.swing_peak_price, price)
            dca_level = pos.dca_count
        else:
            entry_atr = self._get_entry_atr(ticker, ts.date(), price)
            pos = Position(
                ticker=ticker,
                entries=[entry],
                total_qty=qty,
                avg_price=price,
                total_invested_krw=amount_krw,
                dca_count=0,
                initial_entry_price=price,
                first_entry_time=ts,
                signal_type=signal_type,
                is_conl_conditional=is_conl_conditional,
                is_coin_conditional=is_coin_conditional,
                is_crash_buy=is_crash_buy,
                crash_entry_date=ts.date() if is_crash_buy else None,
                entry_atr=entry_atr,
                overheat_origin=overheat_origin,
                uses_daily_fallback=(ticker not in self._intraday_tickers),
                is_swing=is_swing,
                swing_role=swing_role,
                swing_peak_price=price if is_swing else 0.0,
            )
            self.positions[ticker] = pos
            dca_level = 0

        self._last_buy_time[ticker] = ts

        self.trades.append(TradeV4(
            date=ts.date() if isinstance(ts, (datetime, pd.Timestamp)) else ts,
            ticker=ticker,
            side="BUY",
            price=price,
            qty=qty,
            amount_krw=amount_krw,
            pnl_krw=0.0,
            pnl_pct=0.0,
            signal_type=signal_type,
            dca_level=dca_level,
            fees_krw=buy_fee,
            entry_time=ts,
        ))

    # ----------------------------------------------------------
    # Sell execution
    # ----------------------------------------------------------
    def _sell(
        self,
        ticker: str,
        qty: float,
        price: float,
        ts: datetime,
        exit_reason: str,
    ) -> None:
        if ticker not in self.positions:
            return
        pos = self.positions[ticker]
        sell_qty = min(qty, pos.total_qty)
        if sell_qty <= 0:
            return

        gross_usd = sell_qty * price
        gross_krw = gross_usd

        if self.use_fees:
            sell_fee = backtest_common.calc_sell_fee(gross_krw)
            self.total_sell_fees_krw += sell_fee
        else:
            sell_fee = 0.0

        net_proceeds_krw = gross_krw - sell_fee
        fraction = sell_qty / pos.total_qty
        cost_krw = pos.total_invested_krw * fraction
        pnl_krw = net_proceeds_krw - cost_krw
        pnl_pct = (pnl_krw / cost_krw * 100) if cost_krw > 0 else 0.0

        self.cash_krw += net_proceeds_krw
        pos.total_qty -= sell_qty
        pos.total_invested_krw -= cost_krw

        self.trades.append(TradeV4(
            date=ts.date() if isinstance(ts, (datetime, pd.Timestamp)) else ts,
            ticker=ticker,
            side="SELL",
            price=price,
            qty=sell_qty,
            amount_krw=net_proceeds_krw,
            pnl_krw=round(pnl_krw, 2),
            pnl_pct=round(pnl_pct, 2),
            signal_type=pos.signal_type,
            exit_reason=exit_reason,
            fees_krw=sell_fee,
            net_pnl_krw=round(pnl_krw, 2),
            entry_time=pos.first_entry_time,
            exit_time=ts,
        ))

        self._sold_today.add(ticker)

        # v4: 트레이드 카운트 (전체 청산 시에만 카운트)
        if pos.total_qty < 1e-9:
            self._traded_today[ticker] = self._traded_today.get(ticker, 0) + 1
            del self.positions[ticker]

    def _sell_all(self, ticker: str, price: float, ts: datetime, exit_reason: str) -> None:
        if ticker not in self.positions:
            return
        self._sell(ticker, self.positions[ticker].total_qty, price, ts, exit_reason)

    def _partial_sell(self, ticker: str, qty: float, price: float, ts: datetime, exit_reason: str) -> None:
        self._sell(ticker, qty, price, ts, exit_reason)

    # ----------------------------------------------------------
    # Staged sell
    # ----------------------------------------------------------
    def _process_staged_sell(self, pos: Position, price: float, ts: datetime) -> None:
        if not pos.staged_sell_active or pos.staged_sell_start is None:
            return
        elapsed = (ts - pos.staged_sell_start).total_seconds() / 60
        expected_steps = int(elapsed / config.PAIR_SELL_INTERVAL_MIN)
        while pos.staged_sell_step < expected_steps and pos.staged_sell_remaining > 0:
            sell_qty = pos.staged_sell_remaining * config.PAIR_SELL_REMAINING_PCT
            if sell_qty < 0.001:
                sell_qty = pos.staged_sell_remaining
            self._partial_sell(pos.ticker, sell_qty, price, ts, "staged_sell")
            pos.staged_sell_remaining -= sell_qty
            pos.staged_sell_step += 1
        if pos.staged_sell_remaining <= 0:
            pos.staged_sell_active = False

    def _record_swing_event(self, ts: datetime, event: str, detail: str) -> None:
        self.swing_events.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "event": event,
                "detail": detail,
                "variant": self._swing_variant,
                "stage": self._swing_stage,
            }
        )

    def _advance_swing_day_counter(self, trading_date: date) -> None:
        if not self._swing_active:
            return
        if self._swing_last_counted_date is None:
            self._swing_last_counted_date = trading_date
            return
        if trading_date > self._swing_last_counted_date:
            self._swing_stage_elapsed_days += 1
            self._swing_last_counted_date = trading_date

    def _liquidate_all_positions(
        self,
        ts: datetime,
        cur_prices: dict[str, float],
        exit_reason: str,
        only_swing: bool = False,
    ) -> None:
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            if only_swing and not pos.is_swing:
                continue
            price = cur_prices.get(ticker, pos.avg_price)
            if price is None or price <= 0:
                continue
            self._sell_all(ticker, price, ts, exit_reason)

    def _reset_swing_state(self) -> None:
        self._swing_active = False
        self._swing_variant = ""
        self._swing_stage = 0
        self._swing_stage_start_date = None
        self._swing_stage_elapsed_days = 0
        self._swing_last_counted_date = None
        self._swing_stage1_targets.clear()
        self._swing_stage2_ticker = ""

    def _exit_swing_mode(self, ts: datetime, reason: str) -> None:
        self.swing_exit_count += 1
        self._record_swing_event(ts, "SWING_EXIT", reason)
        self._reset_swing_state()

    def _enter_swing_stage2(self, ts: datetime, cur_prices: dict[str, float], variant: str) -> None:
        self._swing_stage = 2
        self._swing_stage_start_date = ts.date()
        self._swing_stage_elapsed_days = 0
        self._swing_last_counted_date = ts.date()
        self._swing_stage1_targets.clear()

        if variant == "momentum":
            ticker = config.V4_SWING_STAGE2_GLD_TICKER
            self._swing_stage2_ticker = ticker
            price = cur_prices.get(ticker)
            if price is not None and price > 0:
                amount = min(
                    self.cash_krw,
                    self.cash_krw * config.V4_SWING_STAGE2_WEIGHT_PCT / 100.0,
                )
                if amount >= 1.0:
                    self._buy(
                        ticker=ticker,
                        price=price,
                        ts=ts,
                        signal_type="swing_stage2",
                        amount_krw=amount,
                        is_swing=True,
                        swing_role="stage2_gld",
                    )
            self._record_swing_event(
                ts,
                "SWING_STAGE_CHANGE",
                f"{variant}:stage2 ticker={self._swing_stage2_ticker or '-'}",
            )
            return

        # vix variant: stage2는 현금 100% 쿨다운
        self._swing_stage2_ticker = ""
        self._record_swing_event(ts, "SWING_STAGE_CHANGE", f"{variant}:stage2 cooldown")

    def _enter_swing_mode(
        self,
        ts: datetime,
        variant: str,
        cur_prices: dict[str, float],
        trigger_tickers: list[str],
    ) -> bool:
        if self._swing_active:
            return False

        # 스윙 전환 시 일반 포지션은 전량 정리
        self._liquidate_all_positions(ts, cur_prices, f"{variant}_swing_enter")

        self._swing_active = True
        self._swing_variant = variant
        self._swing_stage = 1
        self._swing_stage_start_date = ts.date()
        self._swing_stage_elapsed_days = 0
        self._swing_last_counted_date = ts.date()
        self._swing_stage2_ticker = ""
        self._swing_stage1_targets = set()

        if variant == "vix":
            ticker = config.V4_SWING_STAGE2_GLD_TICKER
            price = cur_prices.get(ticker)
            if price is None or price <= 0:
                self._reset_swing_state()
                return False
            amount = min(
                self.cash_krw,
                self.cash_krw * config.V4_SWING_VIX_STAGE1_WEIGHT_PCT / 100.0,
            )
            if amount < 1.0:
                self._reset_swing_state()
                return False
            self._buy(
                ticker=ticker,
                price=price,
                ts=ts,
                signal_type="swing_vix_stage1",
                amount_krw=amount,
                is_swing=True,
                swing_role="vix_stage1_gld",
            )
            self._swing_stage1_targets = {ticker}
        else:
            targets = [t for t in trigger_tickers if (cur_prices.get(t) or 0) > 0]
            if not targets:
                self._reset_swing_state()
                return False
            alloc_total = min(
                self.cash_krw,
                self.cash_krw * config.V4_SWING_STAGE1_WEIGHT_PCT / 100.0,
            )
            if alloc_total < 1.0:
                self._reset_swing_state()
                return False
            per_amount = alloc_total / len(targets)
            for ticker in targets:
                if self.cash_krw < 1.0:
                    break
                amount = min(self.cash_krw, per_amount)
                price = cur_prices.get(ticker)
                if price is None or price <= 0 or amount < 1.0:
                    continue
                self._buy(
                    ticker=ticker,
                    price=price,
                    ts=ts,
                    signal_type="swing_stage1",
                    amount_krw=amount,
                    is_swing=True,
                    swing_role="stage1",
                )
                pos = self.positions.get(ticker)
                if pos is not None:
                    pos.swing_peak_price = price
                    self._swing_stage1_targets.add(ticker)
            if not self._swing_stage1_targets:
                self._reset_swing_state()
                return False

        self._swing_last_trigger_date = ts.date()
        self.swing_entry_count += 1
        self._record_swing_event(
            ts,
            "SWING_ENTER",
            f"{variant}:targets={','.join(sorted(self._swing_stage1_targets))}",
        )
        return True

    def _evaluate_swing_mode(
        self,
        ts: datetime,
        changes: dict[str, dict],
        cur_prices: dict[str, float],
    ) -> None:
        vix_pct = changes.get("VIX", {}).get("change_pct", 0.0)
        vix_trigger = vix_pct >= config.V4_CB_VIX_SPIKE_PCT

        # v4 문서 우선순위: 급등 스윙 중 VIX 급등 시 VIX 스윙으로 전환
        if self._swing_active and self._swing_variant != "vix" and vix_trigger:
            self._liquidate_all_positions(ts, cur_prices, "swing_switch_to_vix", only_swing=True)
            self._exit_swing_mode(ts, "switch_to_vix")
            self._enter_swing_mode(ts, "vix", cur_prices, [config.V4_SWING_STAGE2_GLD_TICKER])
            return

        if self._swing_active:
            return
        if self._swing_last_trigger_date == ts.date():
            return

        if vix_trigger:
            self._enter_swing_mode(ts, "vix", cur_prices, [config.V4_SWING_STAGE2_GLD_TICKER])
            return

        momentum_targets: list[str] = []
        for ticker in config.V4_SWING_ELIGIBLE_TICKERS:
            pct = changes.get(ticker, {}).get("change_pct")
            if pct is None:
                continue
            if pct >= config.V4_SWING_TRIGGER_PCT and (cur_prices.get(ticker) or 0) > 0:
                momentum_targets.append(ticker)
        if momentum_targets:
            self._enter_swing_mode(ts, "momentum", cur_prices, momentum_targets)

    def _process_swing_sells(self, cur_prices: dict[str, float], ts: datetime) -> None:
        if not self._swing_active:
            return

        if self._swing_stage == 1:
            if self._swing_variant == "momentum":
                trigger_reason = ""
                for ticker in list(self.positions.keys()):
                    pos = self.positions.get(ticker)
                    if pos is None or not pos.is_swing or pos.swing_role != "stage1":
                        continue
                    price = cur_prices.get(ticker)
                    if price is None or price <= 0:
                        continue
                    pos.swing_peak_price = max(pos.swing_peak_price, price)
                    atr_stop = pos.initial_entry_price - (
                        config.V4_SWING_STAGE1_ATR_MULT * max(pos.entry_atr, 0.01)
                    )
                    peak = max(pos.swing_peak_price, 1e-9)
                    drawdown_pct = (price - peak) / peak * 100.0
                    if price <= atr_stop:
                        trigger_reason = "swing_stage1_atr_stop"
                        break
                    if drawdown_pct <= config.V4_SWING_STAGE1_DRAWDOWN_PCT:
                        trigger_reason = "swing_stage1_drawdown_stop"
                        break

                if not trigger_reason and self._swing_stage_elapsed_days >= config.V4_SWING_STAGE1_HOLD_DAYS:
                    trigger_reason = "swing_stage1_maturity"

                if trigger_reason:
                    for ticker in list(self.positions.keys()):
                        pos = self.positions.get(ticker)
                        if pos is None or not pos.is_swing or pos.swing_role != "stage1":
                            continue
                        price = cur_prices.get(ticker, pos.avg_price)
                        if price is None or price <= 0:
                            continue
                        self._sell_all(ticker, price, ts, trigger_reason)
                    self._enter_swing_stage2(ts, cur_prices, "momentum")
                return

            # vix stage1: GLD 보유
            ticker = config.V4_SWING_STAGE2_GLD_TICKER
            pos = self.positions.get(ticker)
            price = cur_prices.get(ticker)
            trigger_reason = ""
            if pos is not None and price is not None and pos.initial_entry_price > 0:
                pnl_pct = (price - pos.initial_entry_price) / pos.initial_entry_price * 100.0
                if pnl_pct <= config.V4_SWING_STAGE2_STOP_PCT:
                    trigger_reason = "swing_vix_stage1_stop"
            if not trigger_reason and self._swing_stage_elapsed_days >= config.V4_SWING_VIX_STAGE1_HOLD_DAYS:
                trigger_reason = "swing_vix_stage1_maturity"
            if trigger_reason:
                if pos is not None and price is not None and price > 0:
                    self._sell_all(ticker, price, ts, trigger_reason)
                self._enter_swing_stage2(ts, cur_prices, "vix")
            return

        if self._swing_stage == 2 and self._swing_variant == "momentum":
            ticker = self._swing_stage2_ticker or config.V4_SWING_STAGE2_GLD_TICKER
            pos = self.positions.get(ticker)
            price = cur_prices.get(ticker)
            trigger_reason = ""
            if pos is not None and price is not None and pos.initial_entry_price > 0:
                pnl_pct = (price - pos.initial_entry_price) / pos.initial_entry_price * 100.0
                if pnl_pct <= config.V4_SWING_STAGE2_STOP_PCT:
                    trigger_reason = "swing_stage2_stop"
            if not trigger_reason and self._swing_stage_elapsed_days >= config.V4_SWING_STAGE2_HOLD_DAYS:
                trigger_reason = "swing_stage2_maturity"
            if trigger_reason:
                if pos is not None and price is not None and price > 0:
                    self._sell_all(ticker, price, ts, trigger_reason)
                self._exit_swing_mode(ts, trigger_reason)
            return

        if self._swing_stage == 2 and self._swing_variant == "vix":
            if self._swing_stage_elapsed_days >= config.V4_SWING_VIX_STAGE2_COOLDOWN_DAYS:
                self._exit_swing_mode(ts, "swing_vix_cooldown_done")

    # ----------------------------------------------------------
    # v4: 횡보장 평가
    # ----------------------------------------------------------
    def _evaluate_sideways(
        self,
        ts: datetime,
        cur_prices: dict[str, float],
    ) -> None:
        """30분 간격으로 횡보장 지표를 재평가한다."""
        if not config.V4_SIDEWAYS_ENABLED:
            return

        # 30분 간격 체크
        if self._sideways_last_eval is not None:
            elapsed = (ts - self._sideways_last_eval).total_seconds() / 60
            if elapsed < config.V4_SIDEWAYS_EVAL_INTERVAL_MIN:
                return

        indicators = self._get_sideways_indicator_state(ts.date())
        intraday_range_pct = self._update_spy_intraday_range(cur_prices)
        indicators["range_narrow"] = intraday_range_pct <= config.V4_SIDEWAYS_RANGE_MAX_PCT

        result = signals_v4.evaluate_sideways(
            indicators=indicators,
            min_signals=config.V4_SIDEWAYS_MIN_SIGNALS,
        )

        self._sideways_active = result["is_sideways"]
        self._sideways_last_eval = ts

    # ----------------------------------------------------------
    # v4: 갭/트리거 실패 추적
    # ----------------------------------------------------------
    def _track_gap_convergence(self, sigs: dict) -> None:
        """쌍둥이 ENTRY 후 수렴 여부를 추적한다."""
        for pair_sig in sigs.get("twin_pairs", []):
            follow = pair_sig["follow"]
            signal = pair_sig["signal"]
            if signal == "ENTRY":
                self._gap_entry_tickers.add(follow)
            elif signal == "SELL" and follow in self._gap_entry_tickers:
                self._gap_entry_tickers.discard(follow)
                # 수렴 성공 → 카운트 안 함
            elif signal == "HOLD" and follow in self._gap_entry_tickers:
                # 여전히 대기 중 — 나중에 시간 손절/EOD에서 실패 처리됨
                pass

    def _update_high_vol_state(self, changes: dict[str, dict]) -> None:
        """고변동성 손절용 변동성 발생 횟수를 업데이트한다."""
        next_above: set[str] = set()
        for ticker, info in changes.items():
            pct = abs(info.get("change_pct", 0.0))
            above = pct >= config.V4_HIGH_VOL_MOVE_PCT
            if above:
                next_above.add(ticker)
                if ticker not in self._high_vol_last_above:
                    self._high_vol_hits[ticker] = self._high_vol_hits.get(ticker, 0) + 1
        self._high_vol_last_above = next_above

    def _count_gap_fail_on_sell(self, ticker: str, exit_reason: str) -> None:
        """갭 ENTRY 후 수렴 없이 손절/시간손절/EOD 매도 → 실패 카운트."""
        if ticker in self._gap_entry_tickers and exit_reason in (
            "stop_loss", "time_stop", "eod_close",
        ):
            self._gap_fail_count += 1
            self._gap_entry_tickers.discard(ticker)

    def _flush_gap_entry_eod(self) -> None:
        """EOD: ENTRY 시그널 발생 후 미수렴 상태로 남은 종목 → 실패 카운트.

        매수되지 않았지만 ENTRY 시그널이 발생한 경우도 실패로 카운트한다.
        (규칙서 1-2 #3: '시그널 발생 후 수렴 실패')
        """
        self._gap_fail_count += len(self._gap_entry_tickers)
        self._gap_entry_tickers.clear()

    def _count_trigger_fail_on_sell(self, ticker: str, exit_reason: str) -> None:
        """조건부 매매 트리거 발동 후 목표 수익 미달 매도 → 실패 카운트.

        COIN/CONL이 손절/시간손절/EOD로 청산되면 트리거 불발로 카운트한다.
        """
        if exit_reason not in ("stop_loss", "time_stop", "eod_close", "conl_avg_drop"):
            return
        pos = self.positions.get(ticker)
        if pos and (pos.is_conl_conditional or pos.is_coin_conditional):
            self._trigger_fail_count += 1

    def _record_cb_event(self, ts: datetime, rule: str, active: bool, detail: str) -> None:
        self.cb_events.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "rule": rule,
                "state": "ON" if active else "OFF",
                "detail": detail,
            }
        )

    def _track_cb_transition(self, ts: datetime, rule: str, active: bool, detail: str) -> None:
        prev = self._cb_last_flags.get(rule, False)
        if prev != active:
            self._record_cb_event(ts, rule, active, detail)
            self._cb_last_flags[rule] = active

    # ----------------------------------------------------------
    # v4: Circuit breaker state
    # ----------------------------------------------------------
    def _update_circuit_breaker_state(
        self,
        changes: dict[str, dict],
        poly_probs: dict | None = None,
        ts: datetime | None = None,
    ) -> None:
        """v4 서킷브레이커 상태를 업데이트한다.

        구현 범위:
        - VIX 급등(+6%) 시 7거래일 매수/매도 정지
        - GLD 급등(+3%) 시 3거래일 매수/매도 정지
        - BTC 급락(-5%) 시 매수/매도 정지
        - BTC 급등(+5%) 시 신규 매수만 정지
        - 금리 상승 우려(rate_hike >= 50%) 시 매수/매도 정지
        - 과열 종목(+20%) 추가 매수 금지
        """
        vix_pct = changes.get("VIX", {}).get("change_pct", 0.0)
        gld_pct = changes.get("GLD", {}).get("change_pct", 0.0)
        btc_proxy_pct = changes.get("BITU", {}).get("change_pct", 0.0)
        rate_hike_raw = (poly_probs or {}).get("rate_hike", None)
        rate_hike_prob_pct = (rate_hike_raw * 100.0) if rate_hike_raw is not None else 0.0

        if vix_pct >= config.V4_CB_VIX_SPIKE_PCT:
            self._cb_vix_cooldown_days = max(
                self._cb_vix_cooldown_days,
                config.V4_CB_VIX_COOLDOWN_DAYS,
            )

        # GLD 급등 트리거 발생 시 쿨다운 리셋
        if gld_pct >= config.V4_CB_GLD_SPIKE_PCT:
            self._cb_gld_cooldown_days = max(
                self._cb_gld_cooldown_days,
                config.V4_CB_GLD_COOLDOWN_DAYS,
            )

        # 과열 종목: +20% 진입 후 고점 대비 -10% 조정 시 해제 (CB-6)
        overheated_now: set[str] = set()
        for ticker in config.CB_OVERHEAT_TICKERS:
            state = self._cb_overheat_state.setdefault(ticker, {"active": False, "peak": 0.0})
            info = changes.get(ticker, {})
            pct = info.get("change_pct")
            price = info.get("close")

            active = bool(state.get("active", False))
            peak = float(state.get("peak", 0.0))

            if price is not None and price > 0:
                if active:
                    peak = max(peak, float(price))
                    if peak > 0:
                        drawdown_pct = (float(price) - peak) / peak * 100.0
                        if drawdown_pct <= config.V4_CB_OVERHEAT_RECOVERY_PCT:
                            active = False
                            peak = float(price)
                if (not active) and pct is not None and pct >= config.V4_CB_OVERHEAT_PCT:
                    active = True
                    peak = float(price)

            state["active"] = active
            state["peak"] = peak
            if active:
                overheated_now.add(ticker)
        self._cb_overheated_tickers = overheated_now

        vix_cooldown_active = self._cb_vix_cooldown_days > 0
        gld_cooldown_active = self._cb_gld_cooldown_days > 0
        btc_crash_active = btc_proxy_pct <= config.V4_CB_BTC_CRASH_PCT
        btc_surge_active = btc_proxy_pct >= config.V4_CB_BTC_SURGE_PCT
        rate_hike_active = (
            rate_hike_raw is not None
            and rate_hike_raw != 0.5  # 0.5는 데이터 미확보 기본값으로 취급
            and rate_hike_prob_pct >= config.V4_CB_RATE_HIKE_PROB_PCT
        )

        # v4 규칙: 회피 CB는 매수·매도 모두 정지, CB-4는 신규 매수만 정지
        self._cb_block_all = (
            vix_cooldown_active
            or gld_cooldown_active
            or btc_crash_active
            or rate_hike_active
        )
        self._cb_block_new_buys = self._cb_block_all or btc_surge_active

        if ts is not None:
            self._track_cb_transition(ts, "CB-1_VIX_SPIKE", vix_cooldown_active, f"vix={vix_pct:+.2f}%")
            self._track_cb_transition(ts, "CB-2_GLD_SPIKE", gld_cooldown_active, f"gld={gld_pct:+.2f}%")
            self._track_cb_transition(ts, "CB-3_BTC_CRASH", btc_crash_active, f"btu={btc_proxy_pct:+.2f}%")
            self._track_cb_transition(ts, "CB-4_BTC_SURGE", btc_surge_active, f"btu={btc_proxy_pct:+.2f}%")
            self._track_cb_transition(
                ts,
                "CB-5_RATE_HIKE",
                rate_hike_active,
                f"rate_hike={rate_hike_prob_pct:.1f}%",
            )
            self._track_cb_transition(
                ts,
                "CB-6_OVERHEAT_ANY",
                len(self._cb_overheated_tickers) > 0,
                ",".join(sorted(self._cb_overheated_tickers)) if self._cb_overheated_tickers else "-",
            )

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    def run(self, verbose: bool = True) -> "BacktestEngineV4":
        self._verbose = verbose
        # ---- 1. Load data ----
        print("[1/4] 데이터 준비")
        df = backtest_common.load_parquet(config.DATA_DIR / "backtest_1min_v2.parquet")
        poly = backtest_common.load_polymarket_daily(config.PROJECT_ROOT / "polymarket" / "history")
        self._load_replacement_daily_prices()

        # Pre-index DataFrame for O(1) bar lookup
        ts_prices, sym_bars, day_timestamps = backtest_common.preindex_dataframe(df)
        self._intraday_tickers = set(df["symbol"].unique())
        self._prepare_conl_indicators(df)
        daily_bars = self._build_daily_bars(df)
        self._prepare_daily_risk_metrics(daily_bars)
        self._prepare_sideways_metrics(daily_bars)
        self._prepare_entry_early_metrics(df)

        # ---- 2. Prepare trading dates ----
        all_dates = sorted(day_timestamps.keys())
        bt_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        warmup_dates = [d for d in all_dates if d < self.start_date]

        prev_close: dict[str, float] = {}
        for d in warmup_dates:
            if d not in sym_bars:
                pass
            else:
                for ticker, bars in sym_bars[d].items():
                    if bars:
                        prev_close[ticker] = bars[-1]["close"]
            for repl in ("SOXX", "IREN"):
                px = self._replacement_daily_close.get((repl, d))
                if px is not None:
                    prev_close[repl] = px

        self.total_trading_days = len(bt_dates)
        if verbose:
            print(f"\n[2/4] 시뮬레이션 실행 (v4 선별 매매형)")
            print(f"  백테스트 기간: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}일)")
            print(f"  초기 자금: ${self.initial_capital_krw:,.2f}")
            print(f"  v4 주요 변경: ENTRY {config.V4_PAIR_GAP_ENTRY_THRESHOLD}% | "
                  f"DCA {config.V4_DCA_MAX_COUNT}회 | "
                  f"트리거 {config.V4_COIN_TRIGGER_PCT}% | "
                  f"쿨타임 {config.V4_SPLIT_BUY_INTERVAL_MIN}분 | "
                  f"횡보장 감지 {'ON' if config.V4_SIDEWAYS_ENABLED else 'OFF'}")
            print("  통화 단위: USD (환율 변환 미사용)\n")

        # ---- 3. Main day loop ----
        for day_idx, trading_date in enumerate(bt_dates):
            if trading_date not in ts_prices:
                continue
            day_sym = sym_bars[trading_date]
            day_ts = day_timestamps[trading_date]

            # (a) Polymarket -> market mode
            poly_probs = poly.get(trading_date, None)

            # (a-1) CB 쿨다운(거래일 기준) 감소
            if self._cb_vix_cooldown_days > 0:
                self._cb_vix_cooldown_days -= 1
            if self._cb_gld_cooldown_days > 0:
                self._cb_gld_cooldown_days -= 1
            self._advance_swing_day_counter(trading_date)

            # (b) Reset daily state
            self._sold_today.clear()
            self._traded_today.clear()
            self._gap_fail_count = 0
            self._trigger_fail_count = 0
            self._gap_entry_tickers.clear()
            self._high_vol_hits.clear()
            self._high_vol_last_above.clear()
            self._sideways_active = False
            self._sideways_last_eval = None
            self._spy_day_open = None
            self._spy_day_high = None
            self._spy_day_low = None
            self._crash_buy_active_today = False
            self._cb_block_all = self._cb_gld_cooldown_days > 0 or self._cb_vix_cooldown_days > 0
            self._cb_block_new_buys = self._cb_block_all

            # (c) Handle carry positions from previous day
            first_ts_of_day = day_ts[0] if day_ts else None
            first_prices_of_day: dict[str, float] = {}
            for sym, bars in day_sym.items():
                if bars:
                    first_prices_of_day[sym] = bars[0]["open"]
            for repl in ("SOXX", "IREN"):
                if repl not in first_prices_of_day:
                    px = self._replacement_daily_close.get((repl, trading_date))
                    if px is not None:
                        first_prices_of_day[repl] = px

            base_mode = signals_v4.determine_market_mode_v4(poly_probs, sideways_active=False)
            self._process_crash_gap_open(trading_date, first_ts_of_day, first_prices_of_day)
            self._handle_carry_positions(day_sym, day_ts, base_mode, prev_close)

            # (d) Select coin follow (carry 포지션 있으면 해당 종목 유지)
            carry_follow = None
            for cand in ["MSTU", "IRE"]:
                if cand in self.positions:
                    carry_follow = cand
                    break
            coin_follow = carry_follow if carry_follow else self._select_coin_follow(day_sym, trading_date)
            today_pairs = {}
            for pair_key, pair_cfg in config.TWIN_PAIRS_V4.items():
                if pair_key == "coin":
                    today_pairs[pair_key] = {
                        "lead": pair_cfg["lead"],
                        "follow": [coin_follow],
                        "label": f"코인 (BTC -> {coin_follow})",
                    }
                else:
                    today_pairs[pair_key] = pair_cfg

            # (e) Process each 1-min bar
            day_had_sideways = False

            for ts in day_ts:
                cur_prices = dict(ts_prices[trading_date].get(ts, {}))
                for repl in ("SOXX", "IREN"):
                    if repl not in cur_prices:
                        px = self._replacement_daily_close.get((repl, trading_date))
                        if px is not None:
                            cur_prices[repl] = px

                changes = backtest_common.calc_changes(cur_prices, prev_close)

                # v4: 서킷브레이커 업데이트 (횡보장/매매 로직보다 우선)
                self._update_circuit_breaker_state(changes, poly_probs, ts)

                # v4: 횡보장 재평가 (30분 간격)
                self._evaluate_sideways(ts, cur_prices)
                if self._sideways_active:
                    day_had_sideways = True
                self._evaluate_swing_mode(ts, changes, cur_prices)

                # 시그널 생성 (v4)
                sigs = signals_v4.generate_all_signals_v4(
                    changes,
                    poly_probs=poly_probs,
                    pairs=today_pairs,
                    sideways_active=self._sideways_active,
                    entry_threshold=config.V4_PAIR_GAP_ENTRY_THRESHOLD,
                    coin_trigger_pct=config.V4_COIN_TRIGGER_PCT,
                    coin_sell_profit_pct=config.COIN_SELL_PROFIT_PCT,
                    coin_sell_bearish_pct=config.COIN_SELL_BEARISH_PCT,
                    conl_trigger_pct=config.V4_CONL_TRIGGER_PCT,
                    conl_sell_profit_pct=config.CONL_SELL_PROFIT_PCT,
                    conl_sell_avg_pct=config.CONL_SELL_AVG_PCT,
                )

                gold_warning = sigs["gold"]["warning"]

                # v4: 갭 수렴 추적
                self._track_gap_convergence(sigs)

                # ========== SELL (항상 실행) ==========
                self._process_sells(
                    cur_prices, changes, sigs, ts, base_mode,
                )

                # ========== BUY (횡보장/시간/일일 게이트 적용) ==========
                self._process_buys(
                    cur_prices, changes, sigs, ts, base_mode,
                    gold_warning, today_pairs,
                )

            # (g) End of day
            last_prices = {sym: bars[-1]["close"] for sym, bars in day_sym.items() if bars}
            for repl in ("SOXX", "IREN"):
                if repl not in last_prices:
                    px = self._replacement_daily_close.get((repl, trading_date))
                    if px is not None:
                        last_prices[repl] = px
            last_ts = day_ts[-1] if day_ts else None

            if not self._swing_active:
                if base_mode == "bullish":
                    for ticker, pos in self.positions.items():
                        if not pos.carry:
                            pos.carry = True
                else:
                    if not self._cb_block_all:
                        for ticker in list(self.positions.keys()):
                            pos = self.positions.get(ticker)
                            if (
                                pos is not None
                                and pos.is_crash_buy
                                and pos.crash_entry_date == trading_date
                            ):
                                pos.carry = True
                                continue
                            price = last_prices.get(ticker)
                            if price is not None:
                                self._count_gap_fail_on_sell(ticker, "eod_close")
                                self._count_trigger_fail_on_sell(ticker, "eod_close")
                                self._sell_all(ticker, price, last_ts, "eod_close")

            for ticker, pos in self.positions.items():
                if (
                    pos.is_crash_buy
                    and pos.crash_entry_date == trading_date
                    and pos.crash_ref_close <= 0
                ):
                    ref_close = last_prices.get(ticker)
                    if ref_close is not None:
                        pos.crash_ref_close = float(ref_close)

            # v4: ENTRY 시그널 발생 후 미수렴 상태 flush (매수 안 된 것 포함)
            self._flush_gap_entry_eod()

            prev_close.update(last_prices)

            equity = self._calc_total_equity(cur_prices if last_prices else prev_close)
            self.equity_curve.append((trading_date, equity))

            if day_had_sideways:
                self.sideways_days += 1

            if verbose and ((day_idx + 1) % 50 == 0 or day_idx == len(bt_dates) - 1):
                mode_str = "횡보" if day_had_sideways else base_mode
                print(
                    f"  [{day_idx+1:>3}/{len(bt_dates)}] {trading_date}  "
                    f"자산: ${equity:,.2f}  "
                    f"현금: ${self.cash_krw:,.2f}  "
                    f"포지션: {len(self.positions)}개  "
                    f"모드: {mode_str}"
                )

        if verbose:
            print("\n  백테스트 완료!")
        self._dump_cb_events()
        self._dump_swing_events()
        return self

    # ----------------------------------------------------------
    # Carry position handling
    # ----------------------------------------------------------
    def _handle_carry_positions(
        self, sym_bars_day: dict[str, list[dict]], day_ts: list, market_mode: str,
        prev_close: dict[str, float] | None = None,
    ) -> None:
        if self._swing_active:
            return
        if not any(pos.carry for pos in self.positions.values()):
            return
        first_prices: dict[str, float] = {}
        for ticker in list(self.positions.keys()):
            bars = sym_bars_day.get(ticker, [])
            if bars:
                first_prices[ticker] = bars[0]["open"]
        first_ts = day_ts[0] if day_ts else None

        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.carry:
                continue
            if pos.is_crash_buy:
                continue
            price = first_prices.get(ticker)
            if price is None:
                continue
            if market_mode != "bullish":
                pos.carry = False
                continue
            net_pnl_pct = self._calc_net_profit_pct(pos, price)
            if net_pnl_pct >= config.TAKE_PROFIT_PCT:
                self._sell_all(ticker, price, first_ts, "carry_sell")
                continue
            prev_price = (prev_close or {}).get(ticker)
            is_positive_from_prev = False
            if prev_price and prev_price > 0:
                is_positive_from_prev = price > prev_price
            if is_positive_from_prev and net_pnl_pct > 0:
                self._sell_all(ticker, price, first_ts, "carry_sell")
            else:
                pos.carry = False

    # ----------------------------------------------------------
    # SELL processing
    # ----------------------------------------------------------
    def _process_sells(
        self,
        cur_prices: dict[str, float],
        changes: dict[str, dict],
        sigs: dict,
        ts: datetime,
        market_mode: str,
    ) -> None:
        if self._swing_active:
            self._process_swing_sells(cur_prices, ts)
            return

        # v4 CB-2/CB-3: 매수·매도 정지
        if self._cb_block_all:
            self.cb_sell_halt_bars += 1
            return

        # 0. 급락 역매수 다음날 보합 관찰(30분) 처리
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.is_crash_buy:
                continue
            if pos.crash_flat_monitor_end is None:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            ref_close = pos.crash_ref_close if pos.crash_ref_close > 0 else pos.initial_entry_price
            if price > ref_close:
                self._sell_all(ticker, price, ts, "crash_flat_rebound")
                continue
            if ts >= pos.crash_flat_monitor_end:
                pos.crash_flat_monitor_end = None
                pos.is_crash_buy = False

        # 1. ATR 기반 가격 손절 (일반 1.5x ATR, 강세장 2.5x ATR)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            if not self._can_use_intraday_price(pos, ts):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue

            atr_mult = 2.5 if market_mode == "bullish" else 1.5
            atr_stop = pos.initial_entry_price - (atr_mult * max(pos.entry_atr, 0.01))

            # 고변동성(최근 5거래일 10% 이상 변동 2회+)이면 -4% 고정 손절 우선 적용
            if self._is_high_vol_active(ticker, ts.date()):
                high_vol_stop = pos.initial_entry_price * (1.0 + config.V4_HIGH_VOL_STOP_LOSS_PCT / 100.0)
                atr_stop = max(atr_stop, high_vol_stop)

            if price <= atr_stop:
                self._count_gap_fail_on_sell(ticker, "stop_loss")
                self._count_trigger_fail_on_sell(ticker, "stop_loss")
                self._sell_all(ticker, price, ts, "stop_loss")

        # 2. 시간 손절 (일반: 5시간, 강세장: +2% 즉시 매도, 그 외 다음날 carry)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            if not self._can_use_intraday_price(pos, ts):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            elapsed_min = self._market_minutes_elapsed(pos.first_entry_time, ts)
            time_limit_min = config.MAX_HOLD_HOURS * 60
            if elapsed_min >= time_limit_min:
                if market_mode == "bullish":
                    net_pnl_pct = self._calc_net_profit_pct(pos, price)
                    if net_pnl_pct >= config.TAKE_PROFIT_PCT:
                        self._count_gap_fail_on_sell(ticker, "time_stop")
                        self._count_trigger_fail_on_sell(ticker, "time_stop")
                        self._sell_all(ticker, price, ts, "time_stop")
                    else:
                        pos.carry = True
                else:
                    self._count_gap_fail_on_sell(ticker, "time_stop")
                    self._count_trigger_fail_on_sell(ticker, "time_stop")
                    self._sell_all(ticker, price, ts, "time_stop")

        # 3. CONL sell
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.is_conl_conditional:
                continue
            if not self._can_use_intraday_price(pos, ts):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            net_pnl_pct = self._calc_net_profit_pct(pos, price)
            if net_pnl_pct >= config.CONL_SELL_PROFIT_PCT:
                self._sell_all(ticker, price, ts, "conl_profit")
                continue
            conl_sig = sigs.get("conditional_conl", {})
            if conl_sig.get("sell_on_avg_drop", False):
                self._trigger_fail_count += 1
                self._sell_all(ticker, price, ts, "conl_avg_drop")

        # 4. COIN sell
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.is_coin_conditional:
                continue
            if not self._can_use_intraday_price(pos, ts):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            net_pnl_pct = self._calc_net_profit_pct(pos, price)
            if market_mode == "bearish":
                if net_pnl_pct >= config.COIN_SELL_BEARISH_PCT:
                    self._sell_all(ticker, price, ts, "coin_bearish")
            else:
                if net_pnl_pct >= config.COIN_SELL_PROFIT_PCT:
                    self._sell_all(ticker, price, ts, "coin_profit")

        # 5. v4 고정 익절 (+5%): SOXL/CONL/IRE twin 포지션
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            if pos.signal_type != "twin":
                continue
            origin = pos.overheat_origin or ticker
            if origin not in config.V4_PAIR_FIXED_TP_STOCKS:
                continue
            if not self._can_use_intraday_price(pos, ts):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            net_pnl_pct = self._calc_net_profit_pct(pos, price)
            if net_pnl_pct >= config.V4_PAIR_FIXED_TP_PCT:
                self._sell_all(ticker, price, ts, "fixed_tp")

        # 6. Staged sell (일반 규칙)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.staged_sell_active:
                continue
            if not self._can_use_intraday_price(pos, ts):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            self._process_staged_sell(pos, price, ts)

        # 7. Twin SELL
        for pair_sig in sigs.get("twin_pairs", []):
            follow = pair_sig["follow"]
            signal = pair_sig["signal"]
            if signal != "SELL":
                continue
            target_tickers: list[str] = []
            if follow in self.positions:
                target_tickers.append(follow)
            for tkr, p in self.positions.items():
                if p.signal_type == "twin" and p.overheat_origin == follow:
                    target_tickers.append(tkr)
            target_tickers = list(dict.fromkeys(target_tickers))

            for target in target_tickers:
                if target not in self.positions:
                    continue
                pos = self.positions[target]
                if pos.staged_sell_active:
                    continue
                if pos.fixed_tp_active:
                    continue
                if not self._can_use_intraday_price(pos, ts):
                    continue
                price = cur_prices.get(target)
                if price is None:
                    continue

                origin = pos.overheat_origin or follow
                # v4 신규: SOXL/CONL/IRE(원본 기준)는 40% 즉시 매도 + 60% 고정 익절 대기
                if pos.signal_type == "twin" and origin in config.V4_PAIR_FIXED_TP_STOCKS:
                    first_sell_qty = pos.total_qty * config.V4_PAIR_IMMEDIATE_SELL_PCT
                    self._partial_sell(target, first_sell_qty, price, ts, "twin_converge")
                    if target in self.positions:
                        pos = self.positions[target]
                        pos.fixed_tp_active = True
                        pos.fixed_tp_target_pct = config.V4_PAIR_FIXED_TP_PCT
                        pos.staged_sell_active = False
                        pos.staged_sell_remaining = 0.0
                        pos.staged_sell_step = 0
                    continue

                first_sell_qty = pos.total_qty * config.PAIR_SELL_FIRST_PCT
                self._partial_sell(target, first_sell_qty, price, ts, "twin_converge")
                if target in self.positions:
                    pos = self.positions[target]
                    pos.staged_sell_active = True
                    pos.staged_sell_start = ts
                    pos.staged_sell_remaining = pos.total_qty
                    pos.staged_sell_step = 0

    # ----------------------------------------------------------
    # BUY processing (v4: 게이트 적용)
    # ----------------------------------------------------------
    def _process_buys(
        self,
        cur_prices: dict[str, float],
        changes: dict[str, dict],
        sigs: dict,
        ts: datetime,
        market_mode: str,
        gold_warning: bool,
        today_pairs: dict,
    ) -> None:
        if self._swing_active:
            return

        # v4 급락 역매수: 발동 시 당일 나머지 신규 매수 금지
        if self._process_crash_buy(ts, cur_prices, changes):
            return

        # v4 매수 게이트: 횡보장이면 전체 스킵
        if self._sideways_active:
            return

        gld_blocks_non_conditional = gold_warning
        if gld_blocks_non_conditional:
            self.skipped_gold_bars += 1

        # Twin ENTRY (v4: gap >= 2.2%)
        if not gld_blocks_non_conditional:
            for pair_sig in sigs.get("twin_pairs", []):
                follow = pair_sig["follow"]
                signal = pair_sig["signal"]
                if signal != "ENTRY":
                    continue
                buy_ticker = self._resolve_overheat_buy_ticker(follow, cur_prices)
                if buy_ticker is None:
                    continue
                if buy_ticker == "CONL" and not self._passes_conl_entry_filter(ts):
                    continue
                if not self._can_buy(buy_ticker, ts):
                    continue
                price = cur_prices.get(buy_ticker)
                if price is None:
                    continue
                self._buy(
                    buy_ticker, price, ts, "twin",
                    amount_krw=config.V4_INITIAL_BUY,
                    overheat_origin=follow if buy_ticker != follow else "",
                )

        # DCA (v4: max 4회, max 700만, 쿨타임 20분)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            if gld_blocks_non_conditional:
                if not pos.is_conl_conditional and not pos.is_coin_conditional:
                    continue
            if ticker == "BRKU":
                continue
            if ticker == "CONL" and not config.V4_CONL_DCA_ENABLED:
                continue
            dca_is_conditional = pos.is_conl_conditional or pos.is_coin_conditional
            if not self._can_dca(ticker, ts, is_conditional=dca_is_conditional):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            if pos.dca_count >= config.V4_DCA_MAX_COUNT:
                continue
            if pos.total_invested_krw + config.V4_DCA_BUY > config.V4_MAX_PER_STOCK:
                continue
            required_drop_pct = config.DCA_DROP_PCT * (pos.dca_count + 1)
            actual_drop_pct = (price - pos.initial_entry_price) / pos.initial_entry_price * 100
            if actual_drop_pct <= required_drop_pct:
                self._buy(
                    ticker, price, ts, pos.signal_type,
                    amount_krw=config.V4_DCA_BUY,
                    is_conl_conditional=pos.is_conl_conditional,
                    is_coin_conditional=pos.is_coin_conditional,
                )

        # CONL buy (v4: trigger 4.5%, 진입 마감 면제)
        conl_sig = sigs.get("conditional_conl", {})
        if conl_sig.get("buy_signal", False):
            target_ticker = self._resolve_overheat_buy_ticker("CONL", cur_prices)
            if target_ticker and self._can_buy(target_ticker, ts, is_conditional=True):
                is_native_conl = target_ticker == "CONL"
                if (not is_native_conl) or self._passes_conl_entry_filter(ts):
                    price = cur_prices.get(target_ticker)
                    if price is not None:
                        self._buy(
                            target_ticker,
                            price,
                            ts,
                            "conditional_conl" if is_native_conl else "conditional_conl_switch",
                            amount_krw=config.V4_CONL_FIXED_BUY,
                            is_conl_conditional=is_native_conl,
                            is_coin_conditional=(target_ticker == "COIN"),
                            overheat_origin="CONL" if target_ticker != "CONL" else "",
                        )

        # COIN buy (v4: trigger 4.5%, 진입 마감 면제)
        coin_sig = sigs.get("conditional_coin", {})
        if coin_sig.get("buy_signal", False):
            target = coin_sig.get("target", config.CONDITIONAL_TARGET_V4)
            if self._can_buy(target, ts, is_conditional=True):
                price = cur_prices.get(target)
                if price is not None:
                    self._buy(
                        target, price, ts, "conditional_coin",
                        amount_krw=config.V4_INITIAL_BUY,
                        is_coin_conditional=True,
                    )

        # Bearish
        if market_mode == "bearish":
            bearish_sig = sigs.get("bearish", {})
            if bearish_sig.get("buy_brku", False):
                if self._can_buy("BRKU", ts):
                    price = cur_prices.get("BRKU")
                    if price is not None:
                        brku_amount = self.initial_capital_krw * config.BRKU_WEIGHT_PCT / 100
                        self._buy(
                            "BRKU", price, ts, "bearish",
                            amount_krw=brku_amount,
                        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    @staticmethod
    def _get_last_prices(day_data: pd.DataFrame) -> dict[str, float]:
        last = day_data.groupby("symbol").last()
        return {sym: float(row["close"]) for sym, row in last.iterrows()}

    def _calc_total_equity(self, cur_prices: dict[str, float]) -> float:
        equity = self.cash_krw
        for ticker, pos in self.positions.items():
            price = cur_prices.get(ticker)
            if price is not None:
                equity += pos.total_qty * price
            else:
                equity += pos.total_invested_krw
        return equity

    def _dump_cb_events(self) -> None:
        """CB 이벤트를 감사용 jsonl로 저장한다."""
        if not self.cb_events:
            return
        out = config.DATA_DIR / "v4_cb_events.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in self.cb_events:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _dump_swing_events(self) -> None:
        """Swing 이벤트를 감사용 jsonl로 저장한다."""
        if not self.swing_events:
            return
        out = config.DATA_DIR / "v4_swing_events.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for row in self.swing_events:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------
    def print_report(self) -> None:
        if not self.equity_curve:
            print("  데이터 없음")
            return

        final = self.equity_curve[-1][1]
        total_ret = (final - self.initial_capital_krw) / self.initial_capital_krw * 100

        sells = [t for t in self.trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl_krw > 0]
        losses = [t for t in sells if t.pnl_krw < 0]
        breakeven = [t for t in sells if t.pnl_krw == 0]
        total_pnl = sum(t.pnl_krw for t in sells)

        mdd = backtest_common.calc_mdd(self.equity_curve)
        sharpe = backtest_common.calc_sharpe(self.equity_curve)
        total_fees = self.total_buy_fees_krw + self.total_sell_fees_krw

        # Exit reason breakdown
        exit_stats: dict[str, dict] = {}
        for t in sells:
            key = t.exit_reason or "unknown"
            if key not in exit_stats:
                exit_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            exit_stats[key]["count"] += 1
            exit_stats[key]["pnl"] += t.pnl_krw
            if t.pnl_krw > 0:
                exit_stats[key]["wins"] += 1

        # Signal-type P&L
        sig_stats: dict[str, dict] = {}
        for t in sells:
            key = t.signal_type
            if key not in sig_stats:
                sig_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            sig_stats[key]["count"] += 1
            sig_stats[key]["pnl"] += t.pnl_krw
            if t.pnl_krw > 0:
                sig_stats[key]["wins"] += 1

        # DCA statistics
        buys = [t for t in self.trades if t.side == "BUY"]
        dca_buys = [t for t in buys if t.dca_level > 0]
        initial_buys = [t for t in buys if t.dca_level == 0]

        # v2 비교용: v2의 매매 횟수 (같은 데이터 기준)
        v2_sells_count = "N/A"

        print()
        print("=" * 70)
        print("    PTJ 매매법 v4 백테스트 리포트 (선별 매매형)")
        print("=" * 70)
        print(f"  기간         : {self.equity_curve[0][0]} ~ {self.equity_curve[-1][0]}")
        print(f"  거래일       : {self.total_trading_days}일")
        print(f"  초기 자금    : ${self.initial_capital_krw:>13,.2f}")
        print(f"  최종 자산    : ${final:>13,.2f}")
        print("  통화 단위    : USD (환율 변환 미사용)")
        print(f"  총 수익률    : {total_ret:>+.2f}%")
        print(f"  총 손익      : ${total_pnl:>+13,.2f}")
        print(f"  최대 낙폭    : -{mdd:.2f}%")
        print(f"  Sharpe Ratio : {sharpe:.4f}")
        print("-" * 70)
        print(f"  총 매도 횟수 : {len(sells)}")
        print(f"  승 / 패      : {len(wins)}W / {len(losses)}L / {len(breakeven)}E")
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        print(f"  승률         : {win_rate:.1f}%")
        avg_win = np.mean([t.pnl_krw for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_krw for t in losses]) if losses else 0
        print(f"  평균 수익    : ${avg_win:>+13,.2f}")
        print(f"  평균 손실    : ${avg_loss:>+13,.2f}")
        print("-" * 70)

        # v4 전용 통계
        print("  [v4 선별 매매 효과]")
        print(f"    횡보장 감지일    : {self.sideways_days}일 / {self.total_trading_days}일")
        print(f"    횡보장 차단 매수 : {self.sideways_blocks}회")
        print(f"    진입시작 전 차단 : {self.entry_start_blocks}회")
        print(f"    시간제한 차단    : {self.entry_cutoff_blocks}회")
        print(f"    일일1회 차단     : {self.daily_limit_blocks}회")
        total_blocks = self.sideways_blocks + self.entry_start_blocks + self.entry_cutoff_blocks + self.daily_limit_blocks
        print(f"    총 차단 매수     : {total_blocks}회")
        print(f"    CB 매수 차단      : {self.cb_buy_blocks}회")
        print(f"    CB 매도 정지 바수 : {self.cb_sell_halt_bars}개")
        print(f"    CB 과열 대체 진입 : {self.cb_overheat_switches}회")
        print(f"    CB 대체불가 차단  : {self.cb_overheat_no_substitute_blocks}회")
        print(f"    CONL 필터 차단    : {self.conl_filter_blocks}회")
        print(f"    급락 역매수 실행  : {self.crash_buy_count}회")
        print(f"    스윙 진입/종료     : {self.swing_entry_count}회 / {self.swing_exit_count}회")
        print("-" * 70)

        # Fee breakdown
        print("  [수수료]")
        print(f"    매수 수수료 : ${self.total_buy_fees_krw:>11,.2f}")
        print(f"    매도 수수료 : ${self.total_sell_fees_krw:>11,.2f}")
        print(f"    총 수수료   : ${total_fees:>11,.2f}")
        print("-" * 70)

        # DCA statistics
        print("  [DCA 통계]")
        print(f"    초기 진입   : {len(initial_buys)}회")
        print(f"    물타기      : {len(dca_buys)}회")
        if dca_buys:
            max_dca = max(t.dca_level for t in dca_buys)
            total_dca_krw = sum(t.amount_krw for t in dca_buys)
            print(f"    최대 DCA 단계: {max_dca}")
            print(f"    DCA 총 투입 : ${total_dca_krw:>11,.2f}")
        print("-" * 70)

        # Exit reason breakdown
        if exit_stats:
            print("  [매도 사유별 성과]")
            for key in sorted(exit_stats.keys()):
                s = exit_stats[key]
                wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                print(
                    f"    {key:18s}  {s['count']:>4d}회  "
                    f"P&L ${s['pnl']:>+11,.2f}  승률 {wr:>5.1f}%"
                )
        print("-" * 70)

        # Signal-type P&L
        if sig_stats:
            print("  [시그널별 성과]")
            for key in sorted(sig_stats.keys()):
                s = sig_stats[key]
                wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                print(
                    f"    {key:18s}  {s['count']:>4d}회  "
                    f"P&L ${s['pnl']:>+11,.2f}  승률 {wr:>5.1f}%"
                )

        # v4 파라미터 요약
        print("-" * 70)
        print("  [v4 파라미터]")
        print(f"    ENTRY 갭          : {config.V4_PAIR_GAP_ENTRY_THRESHOLD}% (v2: 1.5%)")
        print(f"    물타기 최대        : {config.V4_DCA_MAX_COUNT}회 (v2: 7회)")
        print(f"    종목당 최대 투입   : ${config.V4_MAX_PER_STOCK:,.0f}")
        print(f"    COIN/CONL 트리거   : {config.V4_COIN_TRIGGER_PCT}% (v2: 3.0%)")
        print(f"    중복 쿨타임        : {config.V4_SPLIT_BUY_INTERVAL_MIN}분 (v2: 5분)")
        print(f"    진입 마감          : {config.V4_ENTRY_CUTOFF_HOUR}:{config.V4_ENTRY_CUTOFF_MINUTE:02d} ET")
        print(f"    일일 거래 제한     : {config.V4_MAX_DAILY_TRADES_PER_STOCK}회/종목")
        print(f"    횡보장 감지        : {'ON' if config.V4_SIDEWAYS_ENABLED else 'OFF'} ({config.V4_SIDEWAYS_MIN_SIGNALS}/5 지표)")
        print("=" * 70)

    # ----------------------------------------------------------
    # Save trade log CSV
    # ----------------------------------------------------------
    def save_trade_log(self) -> Path | None:
        if not self.trades:
            return None
        rows = []
        for t in self.trades:
            rows.append({
                "date": t.date,
                "ticker": t.ticker,
                "side": t.side,
                "price": round(t.price, 4),
                "qty": round(t.qty, 6),
                "amount_krw": round(t.amount_krw, 0),
                "pnl_krw": round(t.pnl_krw, 0),
                "pnl_pct": round(t.pnl_pct, 2),
                "net_pnl_krw": round(t.net_pnl_krw, 0),
                "signal_type": t.signal_type,
                "exit_reason": t.exit_reason,
                "dca_level": t.dca_level,
                "fees_krw": round(t.fees_krw, 0),
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            })
        out_df = pd.DataFrame(rows)
        out_path = config.DATA_DIR / "backtest_v4_trades.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  거래 로그: {out_path}")
        return out_path


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("  PTJ 매매법 v4 - 1분봉 백테스트 (선별 매매형)")
    print()

    engine = BacktestEngineV4()
    engine.run()

    print("\n[3/4] 리포트 생성")
    engine.print_report()

    print("\n[4/4] 파일 저장")
    engine.save_trade_log()
    print()


if __name__ == "__main__":
    main()
