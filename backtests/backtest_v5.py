#!/usr/bin/env python3
"""
PTJ 매매법 v5 - 1분봉 백테스트 시뮬레이션 (선별 매매형)
=======================================================
v2 엔진을 기반으로 v5 규칙 적용:
- 쌍둥이 ENTRY 갭: 1.5% → 2.2%
- 물타기: 7회 → 4회, 종목당 최대 1,000만 → 700만
- COIN/CONL 트리거: 3.0% → 4.5%
- 중복 쿨타임: 5분 → 20분
- 종목당 일일 1회 거래
- 진입 시간 제한 (V5_ENTRY_CUTOFF 이후 매수 금지)
- 횡보장 감지 → 현금 100%

Data:
  - data/backtest_1min_v2.parquet  (동일 데이터 사용)
  - polymarket/history/*.json

Dependencies:
  - config.py          : v5 parameters
  - signals_v5.py      : v5 signal functions
  - backtest_common.py : shared utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
import signals_v5
import backtest_common

# ============================================================
# Constants
# ============================================================
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
ENTRY_CUTOFF = time(config.V5_ENTRY_CUTOFF_HOUR, config.V5_ENTRY_CUTOFF_MINUTE)
ENTRY_DEFAULT_START = time(config.V5_ENTRY_DEFAULT_START_HOUR, config.V5_ENTRY_DEFAULT_START_MINUTE)
ENTRY_EARLY_START = time(config.V5_ENTRY_EARLY_START_HOUR, config.V5_ENTRY_EARLY_START_MINUTE)
UNIX_REBALANCE_TIME = time(config.V5_UNIX_REBALANCE_HOUR, config.V5_UNIX_REBALANCE_MINUTE)
DEFENSE_TICKERS = {"IAU", "GDXU"}


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
    carry: bool = False
    signal_type: str = ""
    is_conl_conditional: bool = False
    is_coin_conditional: bool = False


@dataclass
class TradeV5:
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
# Backtest Engine v5
# ============================================================
class BacktestEngineV5:
    def __init__(
        self,
        initial_capital_krw: float = config.V5_TOTAL_CAPITAL,
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
        self.trades: list[TradeV5] = []
        self.equity_curve: list[tuple[date, float]] = []
        self._sold_today: set[str] = set()
        self._traded_today: dict[str, int] = {}  # v5: 종목별 당일 트레이드 수
        self._last_buy_time: dict[str, datetime] = {}

        # v5 횡보장 상태
        self._sideways_active: bool = False
        self._sideways_last_eval: datetime | None = None
        self._gap_fail_count: int = 0      # 당일 갭 수렴 실패 횟수
        self._trigger_fail_count: int = 0  # 당일 트리거 불발 횟수
        self._gap_entry_tickers: set[str] = set()  # 당일 ENTRY 후 수렴 대기 중
        self._trigger_entry_active: bool = False    # 당일 트리거 발동 여부
        self._high_vol_hits: dict[str, int] = {}
        self._high_vol_last_above: set[str] = set()

        # v5 통계
        self.sideways_days: int = 0
        self.sideways_blocks: int = 0      # 횡보장으로 차단된 매수 수
        self.entry_start_blocks: int = 0   # 진입 시작 시각 이전 차단 수
        self.entry_cutoff_blocks: int = 0  # 시간 제한으로 차단된 매수 수
        self.daily_limit_blocks: int = 0   # 일일 1회 제한으로 차단된 매수 수
        self.cb_buy_blocks: int = 0        # 서킷브레이커로 차단된 매수 수

        # v5 서킷브레이커 상태
        self._cb_vix_cooldown_days: int = 0
        self._cb_gld_cooldown_days: int = 0
        self._cb_block_new_buys: bool = False
        self._cb_rate_leverage_cooldown_days: int = 0
        self._cb_overheated_tickers: set[str] = set()

        # v5 Unix(VIX) 방어모드 상태
        self._unix_signal_active_today: bool = False
        self._unix_signal_next_day: bool = False
        self._unix_signal_last_pct: float = 0.0
        self._iau_ban_days_remaining: int = 0
        self._gdxu_entry_day_idx: int | None = None
        self._entry_early_ok: dict[tuple[str, object], bool] = {}

        # Fee accumulators
        self.total_buy_fees_krw: float = 0.0
        self.total_sell_fees_krw: float = 0.0

        # Statistics
        self.total_trading_days: int = 0
        self.skipped_gold_bars: int = 0

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

    def _defense_mode_blocks_regular_buys(self) -> bool:
        """Unix(VIX) 방어모드 또는 방어 포지션 보유 중이면 일반 매수를 차단한다."""
        if self._unix_signal_active_today:
            return True
        return any(t in self.positions for t in DEFENSE_TICKERS)

    def _maybe_exit_gdxu(self, ts: datetime, cur_prices: dict[str, float], day_idx: int, signal_off: bool) -> None:
        """GDXU 2~3일 보유 규칙에 따라 종가 리밸런싱 시점에 청산한다."""
        if "GDXU" not in self.positions:
            self._gdxu_entry_day_idx = None
            return
        price = cur_prices.get("GDXU")
        if price is None:
            return
        bar_time = ts.time() if hasattr(ts, "time") else None
        if bar_time is None or bar_time < UNIX_REBALANCE_TIME:
            return

        if self._gdxu_entry_day_idx is None:
            self._gdxu_entry_day_idx = day_idx
        hold_days = day_idx - self._gdxu_entry_day_idx

        if hold_days >= config.V5_GDXU_HOLD_MAX_DAYS:
            self._sell_all("GDXU", price, ts, "gdxu_day3_forced_exit")
        elif hold_days >= config.V5_GDXU_HOLD_MIN_DAYS and signal_off:
            self._sell_all("GDXU", price, ts, "gdxu_day2_signal_off_exit")

    def _process_unix_defense(self, ts: datetime, cur_prices: dict[str, float], day_idx: int) -> None:
        """Unix(VIX) 방어모드(IAU 40% + GDXU 30%)를 운용한다."""
        # IAU 손절: 매수가 대비 -5%
        if "IAU" in self.positions:
            iau_price = cur_prices.get("IAU")
            if iau_price is not None:
                iau_pos = self.positions["IAU"]
                iau_net_pnl_pct = self._calc_net_profit_pct(iau_pos, iau_price)
                if iau_net_pnl_pct <= config.V5_IAU_STOP_LOSS_PCT:
                    self._sell_all("IAU", iau_price, ts, "iau_stop_loss")

        signal_off = not self._unix_signal_active_today

        # Unix 신호 해제 시 IAU는 즉시 청산, GDXU는 2~3일 룰 적용
        if signal_off:
            if "IAU" in self.positions:
                iau_price = cur_prices.get("IAU")
                if iau_price is not None:
                    self._sell_all("IAU", iau_price, ts, "unix_signal_off")
            self._maybe_exit_gdxu(ts, cur_prices, day_idx, signal_off=True)
            return

        bar_time = ts.time() if hasattr(ts, "time") else None
        can_enter_defense = bool(bar_time and ENTRY_EARLY_START <= bar_time < ENTRY_CUTOFF)

        # Unix 신호 활성 시 방어모드 진입/유지
        if self._iau_ban_days_remaining > 0:
            if "IAU" in self.positions:
                iau_price = cur_prices.get("IAU")
                if iau_price is not None:
                    self._sell_all("IAU", iau_price, ts, "iau_ban_active")
        elif can_enter_defense:
            if "IAU" not in self.positions and "IAU" not in self._sold_today:
                iau_price = cur_prices.get("IAU")
                if iau_price is not None:
                    iau_amount = self.initial_capital_krw * config.V5_UNIX_DEFENSE_IAU_WEIGHT_PCT / 100.0
                    self._buy("IAU", iau_price, ts, "unix_defense_iau", amount_krw=iau_amount)

        if can_enter_defense and "GDXU" not in self.positions and "GDXU" not in self._sold_today:
            gdxu_price = cur_prices.get("GDXU")
            if gdxu_price is not None:
                gdxu_amount = self.initial_capital_krw * config.V5_UNIX_DEFENSE_GDXU_WEIGHT_PCT / 100.0
                self._buy("GDXU", gdxu_price, ts, "unix_defense_gdxu", amount_krw=gdxu_amount)
                if "GDXU" in self.positions:
                    self._gdxu_entry_day_idx = day_idx

        self._maybe_exit_gdxu(ts, cur_prices, day_idx, signal_off=False)

    # ----------------------------------------------------------
    # v5: Buy eligibility (시간/일일제한/횡보장/쿨타임)
    # ----------------------------------------------------------
    def _can_buy(self, ticker: str, ts: datetime, is_conditional: bool = False) -> bool:
        """v5 매수 가능 여부. 횡보장/시간/일일제한/쿨타임 검증.

        Parameters
        ----------
        is_conditional : bool
            True이면 조건부 매매(COIN/CONL).
            v5 규칙상 시간 필터는 조건부에도 동일 적용한다.
        """
        # 1. v5 서킷브레이커 → 신규 매수 차단 (매도는 허용)
        if self._cb_block_new_buys:
            self.cb_buy_blocks += 1
            return False

        # 1-1. CB-6 과열 종목 추가 매수 금지
        if ticker in self._cb_overheated_tickers:
            self.cb_buy_blocks += 1
            return False

        # 1-2. CB-5 해제 후 레버리지 추가 대기
        if self._cb_rate_leverage_cooldown_days > 0 and ticker in config.LEVERAGED_TICKERS:
            self.cb_buy_blocks += 1
            return False

        # 2. 횡보장 모드 → 매수 전면 차단
        if self._sideways_active:
            self.sideways_blocks += 1
            return False

        # 3. 진입 시간 제한 (기본 08:00~10:00, 조기모드 04:00~10:00)
        if not (is_conditional and config.V5_CONDITIONAL_EXEMPT_CUTOFF):
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

        # 6. 종목당 일일 1트레이드 제한 (v5 신규)
        if self._traded_today.get(ticker, 0) >= config.V5_MAX_DAILY_TRADES_PER_STOCK:
            self.daily_limit_blocks += 1
            return False

        # 7. 쿨타임 (v5: 20분)
        last = self._last_buy_time.get(ticker)
        if last and (ts - last).total_seconds() < config.V5_SPLIT_BUY_INTERVAL_MIN * 60:
            return False

        return True

    def _can_dca(self, ticker: str, ts: datetime, is_conditional: bool = False) -> bool:
        """v5 물타기 가능 여부."""
        # 서킷브레이커 → 물타기도 차단
        if self._cb_block_new_buys:
            self.cb_buy_blocks += 1
            return False

        # 과열 종목 추가 매수 금지 (CB-6)
        if ticker in self._cb_overheated_tickers:
            self.cb_buy_blocks += 1
            return False

        # 금리 해제 후 레버리지 추가 대기
        if self._cb_rate_leverage_cooldown_days > 0 and ticker in config.LEVERAGED_TICKERS:
            self.cb_buy_blocks += 1
            return False

        # 횡보장 → 물타기도 차단
        if self._sideways_active:
            self.sideways_blocks += 1
            return False

        # 진입 시간 제한 (기본 08:00~10:00, 조기모드 04:00~10:00)
        if not (is_conditional and config.V5_CONDITIONAL_EXEMPT_CUTOFF):
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
        if last and (ts - last).total_seconds() < config.V5_SPLIT_BUY_INTERVAL_MIN * 60:
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

    def _prepare_entry_early_metrics(self, df: pd.DataFrame) -> None:
        """조기 진입(ADX + EMA + 거래량배수) 조건을 timestamp 단위로 계산한다."""
        early_ok: dict[tuple[str, object], bool] = {}

        raw = df.loc[:, ["symbol", "timestamp", "date", "high", "low", "close", "volume"]].copy()
        raw.sort_values(["symbol", "timestamp"], inplace=True)

        # 프리마켓 데이터가 없으면 조기진입 로직 비활성화.
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

            ema = close.ewm(span=config.V5_ENTRY_EMA_PERIOD, adjust=False).mean()
            ema_prev = ema.shift(max(config.V5_ENTRY_EMA_SLOPE_LOOKBACK, 1))
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
                (adx >= config.V5_ENTRY_TREND_ADX_MIN)
                & (ema_slope_pct > 0.0)
                & (vol_ratio >= config.V5_ENTRY_VOLUME_RATIO_MIN)
            )

            for ts, ok in zip(g["timestamp"], cond):
                if bool(ok):
                    early_ok[(symbol, ts)] = True

        self._entry_early_ok = early_ok

    def _is_early_entry_allowed(self, ticker: str, ts: datetime) -> bool:
        return bool(self._entry_early_ok.get((ticker, ts), False))

    def _get_entry_start_time(self, ticker: str, ts: datetime) -> time:
        if self._is_early_entry_allowed(ticker, ts):
            return ENTRY_EARLY_START
        return ENTRY_DEFAULT_START

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

        if ticker in self.positions:
            pos = self.positions[ticker]
            pos.entries.append(entry)
            pos.total_qty += qty
            pos.total_invested_krw += amount_krw
            pos.avg_price = pos.total_invested_krw / pos.total_qty
            pos.dca_count += 1
            dca_level = pos.dca_count
        else:
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
            )
            self.positions[ticker] = pos
            dca_level = 0

        self._last_buy_time[ticker] = ts

        self.trades.append(TradeV5(
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

        self.trades.append(TradeV5(
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

        # v5: 트레이드 카운트 (전체 청산 시에만 카운트)
        if pos.total_qty < 1e-9:
            self._traded_today[ticker] = self._traded_today.get(ticker, 0) + 1
            del self.positions[ticker]
            if ticker == "GDXU":
                self._gdxu_entry_day_idx = None

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

    # ----------------------------------------------------------
    # v5: 횡보장 평가
    # ----------------------------------------------------------
    def _evaluate_sideways(
        self,
        ts: datetime,
        poly_probs: dict | None,
        changes: dict,
    ) -> None:
        """30분 간격으로 횡보장 지표를 재평가한다."""
        if not config.V5_SIDEWAYS_ENABLED:
            return

        # 30분 간격 체크
        if self._sideways_last_eval is not None:
            elapsed = (ts - self._sideways_last_eval).total_seconds() / 60
            if elapsed < config.V5_SIDEWAYS_EVAL_INTERVAL_MIN:
                return

        result = signals_v5.evaluate_sideways(
            poly_probs=poly_probs,
            changes=changes,
            gap_fail_count=self._gap_fail_count,
            trigger_fail_count=self._trigger_fail_count,
            poly_low=config.V5_SIDEWAYS_POLY_LOW,
            poly_high=config.V5_SIDEWAYS_POLY_HIGH,
            gld_threshold=config.V5_SIDEWAYS_GLD_THRESHOLD,
            gap_fail_threshold=config.V5_SIDEWAYS_GAP_FAIL_COUNT,
            trigger_fail_threshold=config.V5_SIDEWAYS_TRIGGER_FAIL_COUNT,
            index_threshold=config.V5_SIDEWAYS_INDEX_THRESHOLD,
            min_signals=config.V5_SIDEWAYS_MIN_SIGNALS,
        )

        self._sideways_active = result["is_sideways"]
        self._sideways_last_eval = ts

    # ----------------------------------------------------------
    # v5: 갭/트리거 실패 추적
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
            above = pct >= config.V5_HIGH_VOL_MOVE_PCT
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

    # ----------------------------------------------------------
    # v5: Circuit breaker state
    # ----------------------------------------------------------
    def _update_circuit_breaker_state(self, changes: dict[str, dict], poly_probs: dict | None = None) -> None:
        """v5 서킷브레이커 상태를 업데이트한다.

        구현 범위:
        - VIX 급등(+6%) 시 7거래일 신규 매수 차단
        - GLD 급등(+3%) 시 3거래일 신규 매수 차단
        - BTC 급락(-5%) 시 신규 매수 차단
        - BTC 급등(+5%) 시 신규 매수 차단
        - 금리 상승 우려(rate_hike >= 50%) 시 신규 매수 차단
        - 과열 종목(+20%) 추가 매수 금지
        - 매도는 계속 허용 (v4와의 핵심 차이)
        """
        vix_pct = changes.get("VIX", {}).get("change_pct", 0.0)
        gld_pct = changes.get("GLD", {}).get("change_pct", 0.0)
        btc_proxy_pct = changes.get("BITU", {}).get("change_pct", 0.0)
        rate_hike_raw = (poly_probs or {}).get("rate_hike", None)
        rate_hike_prob_pct = (rate_hike_raw * 100.0) if rate_hike_raw is not None else 0.0

        if vix_pct >= config.V5_CB_VIX_SPIKE_PCT:
            self._cb_vix_cooldown_days = max(
                self._cb_vix_cooldown_days,
                config.V5_CB_VIX_COOLDOWN_DAYS,
            )

        # GLD 급등 트리거 발생 시 쿨다운 리셋
        if gld_pct >= config.V5_CB_GLD_SPIKE_PCT:
            self._cb_gld_cooldown_days = max(
                self._cb_gld_cooldown_days,
                config.V5_CB_GLD_COOLDOWN_DAYS,
            )

        overheated_now: set[str] = set()
        for ticker in config.CB_OVERHEAT_TICKERS:
            pct = changes.get(ticker, {}).get("change_pct")
            if pct is None:
                continue
            if pct >= config.V5_CB_OVERHEAT_PCT:
                overheated_now.add(ticker)
        self._cb_overheated_tickers = overheated_now

        rate_hike_active = (
            rate_hike_raw is not None
            and rate_hike_raw != 0.5  # 0.5는 데이터 미확보 기본값으로 취급
            and rate_hike_prob_pct >= config.V5_CB_RATE_HIKE_PROB_PCT
        )
        if rate_hike_active:
            self._cb_rate_leverage_cooldown_days = config.V5_CB_RATE_LEVERAGE_COOLDOWN_DAYS

        vix_cooldown_active = self._cb_vix_cooldown_days > 0
        gld_cooldown_active = self._cb_gld_cooldown_days > 0
        btc_crash_active = btc_proxy_pct <= config.V5_CB_BTC_CRASH_PCT
        btc_surge_active = btc_proxy_pct >= config.V5_CB_BTC_SURGE_PCT

        # v5 규칙: 신규 매수만 차단, 매도는 허용
        self._cb_block_new_buys = (
            vix_cooldown_active
            or gld_cooldown_active
            or btc_crash_active
            or btc_surge_active
            or rate_hike_active
        )

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    def run(self, verbose: bool = True) -> "BacktestEngineV5":
        self._verbose = verbose
        # ---- 1. Load data ----
        print("[1/4] 데이터 준비")
        df = backtest_common.load_parquet(config.DATA_DIR / "backtest_1min_v2.parquet")
        poly = backtest_common.load_polymarket_daily(config.PROJECT_ROOT / "polymarket" / "history")

        # Pre-index DataFrame for O(1) bar lookup
        ts_prices, sym_bars, day_timestamps = backtest_common.preindex_dataframe(df)
        self._prepare_entry_early_metrics(df)

        # ---- 2. Prepare trading dates ----
        all_dates = sorted(day_timestamps.keys())
        bt_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        warmup_dates = [d for d in all_dates if d < self.start_date]

        prev_close: dict[str, float] = {}
        for d in warmup_dates:
            if d not in sym_bars:
                continue
            for ticker, bars in sym_bars[d].items():
                if bars:
                    prev_close[ticker] = bars[-1]["close"]

        self.total_trading_days = len(bt_dates)
        if verbose:
            print(f"\n[2/4] 시뮬레이션 실행 (v5 선별 매매형)")
            print(f"  백테스트 기간: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}일)")
            print(f"  초기 자금: ${self.initial_capital_krw:,.2f}")
            print(f"  v5 주요 변경: ENTRY {config.V5_PAIR_GAP_ENTRY_THRESHOLD}% | "
                  f"DCA {config.V5_DCA_MAX_COUNT}회 | "
                  f"트리거 {config.V5_COIN_TRIGGER_PCT}% | "
                  f"쿨타임 {config.V5_SPLIT_BUY_INTERVAL_MIN}분 | "
                  f"횡보장 감지 {'ON' if config.V5_SIDEWAYS_ENABLED else 'OFF'}")
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
            if self._cb_rate_leverage_cooldown_days > 0:
                self._cb_rate_leverage_cooldown_days -= 1
            self._unix_signal_active_today = self._unix_signal_next_day

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
            self._cb_block_new_buys = self._cb_gld_cooldown_days > 0 or self._cb_vix_cooldown_days > 0
            if "GDXU" in self.positions and self._gdxu_entry_day_idx is None:
                self._gdxu_entry_day_idx = day_idx

            # (c) Handle carry positions from previous day
            first_ts_of_day = day_ts[0] if day_ts else None
            base_mode = signals_v5.determine_market_mode_v5(poly_probs, sideways_active=False)
            self._handle_carry_positions(day_sym, day_ts, base_mode, prev_close)

            # (d) Determine stop loss threshold
            if base_mode == "bullish":
                stop_loss_threshold = config.STOP_LOSS_BULLISH_PCT
            else:
                stop_loss_threshold = config.STOP_LOSS_PCT

            # (e) Select coin follow (carry 포지션 있으면 해당 종목 유지)
            carry_follow = None
            for cand in ["MSTU", "IRE"]:
                if cand in self.positions:
                    carry_follow = cand
                    break
            coin_follow = carry_follow if carry_follow else self._select_coin_follow(day_sym, trading_date)
            today_pairs = {}
            for pair_key, pair_cfg in config.TWIN_PAIRS_V5.items():
                if pair_key == "coin":
                    today_pairs[pair_key] = {
                        "lead": pair_cfg["lead"],
                        "follow": [coin_follow],
                        "label": f"코인 (BTC -> {coin_follow})",
                    }
                else:
                    today_pairs[pair_key] = pair_cfg

            # (f) Process each 1-min bar
            day_had_sideways = False

            for ts in day_ts:
                cur_prices = ts_prices[trading_date].get(ts, {})

                changes = backtest_common.calc_changes(cur_prices, prev_close)
                self._update_high_vol_state(changes)

                # v5: 서킷브레이커 업데이트 (신규 매수 차단만 적용)
                self._update_circuit_breaker_state(changes, poly_probs)

                # v5: 횡보장 재평가 (30분 간격)
                self._evaluate_sideways(ts, poly_probs, changes)
                if self._sideways_active:
                    day_had_sideways = True

                # 시그널 생성 (v5)
                sigs = signals_v5.generate_all_signals_v5(
                    changes,
                    poly_probs=poly_probs,
                    pairs=today_pairs,
                    sideways_active=self._sideways_active,
                    entry_threshold=config.V5_PAIR_GAP_ENTRY_THRESHOLD,
                    coin_trigger_pct=config.V5_COIN_TRIGGER_PCT,
                    coin_sell_profit_pct=config.COIN_SELL_PROFIT_PCT,
                    coin_sell_bearish_pct=config.COIN_SELL_BEARISH_PCT,
                    conl_trigger_pct=config.V5_CONL_TRIGGER_PCT,
                    conl_sell_profit_pct=config.CONL_SELL_PROFIT_PCT,
                    conl_sell_avg_pct=config.CONL_SELL_AVG_PCT,
                )

                gold_warning = sigs["gold"]["warning"]

                # v5: 갭 수렴 추적
                self._track_gap_convergence(sigs)

                # ========== SELL (항상 실행) ==========
                self._process_sells(
                    cur_prices, changes, sigs, ts, base_mode,
                    stop_loss_threshold,
                )

                # ========== Unix(VIX) defense ==========
                self._process_unix_defense(ts, cur_prices, day_idx)

                # ========== BUY (횡보장/시간/일일 게이트 적용) ==========
                self._process_buys(
                    cur_prices, changes, sigs, ts, base_mode,
                    gold_warning, today_pairs,
                )

            # (g) End of day
            last_prices = {sym: bars[-1]["close"] for sym, bars in day_sym.items() if bars}
            last_ts = day_ts[-1] if day_ts else None

            if base_mode == "bullish":
                for ticker, pos in self.positions.items():
                    if not pos.carry:
                        pos.carry = True
            else:
                for ticker in list(self.positions.keys()):
                    if ticker in DEFENSE_TICKERS:
                        continue
                    price = last_prices.get(ticker)
                    if price is not None:
                        self._count_gap_fail_on_sell(ticker, "eod_close")
                        self._count_trigger_fail_on_sell(ticker, "eod_close")
                        self._sell_all(ticker, price, last_ts, "eod_close")

            # v5: ENTRY 시그널 발생 후 미수렴 상태 flush (매수 안 된 것 포함)
            self._flush_gap_entry_eod()

            # Unix(VIX) 일간 신호 계산: 종가 확정 후, 다음 거래일에 적용
            prev_vix_close = prev_close.get("VIX")
            cur_vix_close = last_prices.get("VIX")
            if prev_vix_close and prev_vix_close > 0 and cur_vix_close is not None:
                self._unix_signal_last_pct = (cur_vix_close / prev_vix_close - 1.0) * 100.0
                self._unix_signal_next_day = self._unix_signal_last_pct >= config.V5_UNIX_DEFENSE_TRIGGER_PCT
            else:
                self._unix_signal_last_pct = 0.0
                self._unix_signal_next_day = False

            # IAU 금지 트리거: IAU 보유 중 GDXU 일간 -12% 이하
            iau_ban_triggered = False
            prev_gdxu_close = prev_close.get("GDXU")
            cur_gdxu_close = last_prices.get("GDXU")
            if "IAU" in self.positions and prev_gdxu_close and prev_gdxu_close > 0 and cur_gdxu_close is not None:
                gdxu_day_pct = (cur_gdxu_close / prev_gdxu_close - 1.0) * 100.0
                if gdxu_day_pct <= config.V5_IAU_BAN_TRIGGER_GDXU_DROP_PCT:
                    iau_price = last_prices.get("IAU")
                    if iau_price is not None:
                        self._sell_all("IAU", iau_price, last_ts, "iau_ban_trigger")
                    self._iau_ban_days_remaining = config.V5_IAU_BAN_DURATION_TRADING_DAYS
                    iau_ban_triggered = True

            # IAU 금지 카운트: 트리거 발생일은 Day0로 카운트 제외
            if not iau_ban_triggered and self._iau_ban_days_remaining > 0:
                self._iau_ban_days_remaining -= 1

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
        return self

    # ----------------------------------------------------------
    # Carry position handling
    # ----------------------------------------------------------
    def _handle_carry_positions(
        self, sym_bars_day: dict[str, list[dict]], day_ts: list, market_mode: str,
        prev_close: dict[str, float] | None = None,
    ) -> None:
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
        stop_loss_threshold: float,
    ) -> None:
        # 1. Stop loss
        for ticker in list(self.positions.keys()):
            if ticker in DEFENSE_TICKERS:
                continue
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            change_data = changes.get(ticker, {})
            change_pct = change_data.get("change_pct", 0.0)
            effective_stop = stop_loss_threshold
            if self._high_vol_hits.get(ticker, 0) >= config.V5_HIGH_VOL_HIT_COUNT:
                lev = config.LEVERAGE_MULTIPLIER.get(ticker, 1)
                if lev >= 3:
                    high_vol_stop = config.V5_HIGH_VOL_STOP_LOSS_3X_PCT
                elif lev == 2:
                    high_vol_stop = config.V5_HIGH_VOL_STOP_LOSS_2X_PCT
                else:
                    high_vol_stop = config.V5_HIGH_VOL_STOP_LOSS_1X_PCT
                effective_stop = max(effective_stop, high_vol_stop)
            if change_pct <= effective_stop:
                self._count_gap_fail_on_sell(ticker, "stop_loss")
                self._count_trigger_fail_on_sell(ticker, "stop_loss")
                self._sell_all(ticker, price, ts, "stop_loss")

        # 2. Time stop
        for ticker in list(self.positions.keys()):
            if ticker in DEFENSE_TICKERS:
                continue
            pos = self.positions.get(ticker)
            if pos is None:
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
                    self._count_gap_fail_on_sell(ticker, "time_stop")
                    self._count_trigger_fail_on_sell(ticker, "time_stop")
                    self._sell_all(ticker, price, ts, "time_stop")

        # 3. CONL sell
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.is_conl_conditional:
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

        # 5. Staged sell
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.staged_sell_active:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            self._process_staged_sell(pos, price, ts)

        # 6. Twin SELL
        for pair_sig in sigs.get("twin_pairs", []):
            follow = pair_sig["follow"]
            signal = pair_sig["signal"]
            if signal != "SELL":
                continue
            if follow not in self.positions:
                continue
            pos = self.positions[follow]
            if pos.staged_sell_active:
                continue
            price = cur_prices.get(follow)
            if price is None:
                continue
            first_sell_qty = pos.total_qty * config.PAIR_SELL_FIRST_PCT
            self._partial_sell(follow, first_sell_qty, price, ts, "twin_converge")
            if follow in self.positions:
                pos = self.positions[follow]
                pos.staged_sell_active = True
                pos.staged_sell_start = ts
                pos.staged_sell_remaining = pos.total_qty
                pos.staged_sell_step = 0

    # ----------------------------------------------------------
    # BUY processing (v5: 게이트 적용)
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
        # v5 매수 게이트: 횡보장이면 전체 스킵
        if self._sideways_active:
            return
        if self._defense_mode_blocks_regular_buys():
            return

        gld_blocks_non_conditional = gold_warning
        if gld_blocks_non_conditional:
            self.skipped_gold_bars += 1

        # Twin ENTRY (v5: gap >= 2.2%)
        if not gld_blocks_non_conditional:
            for pair_sig in sigs.get("twin_pairs", []):
                follow = pair_sig["follow"]
                signal = pair_sig["signal"]
                if signal != "ENTRY":
                    continue
                if not self._can_buy(follow, ts):
                    continue
                price = cur_prices.get(follow)
                if price is None:
                    continue
                self._buy(
                    follow, price, ts, "twin",
                    amount_krw=config.V5_INITIAL_BUY,
                )

        # DCA (v5: max 4회, max 700만, 쿨타임 20분)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            if gld_blocks_non_conditional:
                if not pos.is_conl_conditional and not pos.is_coin_conditional:
                    continue
            if ticker == "BRKU":
                continue
            dca_is_conditional = pos.is_conl_conditional or pos.is_coin_conditional
            if not self._can_dca(ticker, ts, is_conditional=dca_is_conditional):
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            if pos.dca_count >= config.V5_DCA_MAX_COUNT:
                continue
            if pos.total_invested_krw + config.V5_DCA_BUY > config.V5_MAX_PER_STOCK:
                continue
            required_drop_pct = config.DCA_DROP_PCT * (pos.dca_count + 1)
            actual_drop_pct = (price - pos.initial_entry_price) / pos.initial_entry_price * 100
            if actual_drop_pct <= required_drop_pct:
                self._buy(
                    ticker, price, ts, pos.signal_type,
                    amount_krw=config.V5_DCA_BUY,
                    is_conl_conditional=pos.is_conl_conditional,
                    is_coin_conditional=pos.is_coin_conditional,
                )

        # CONL buy (v5: trigger 4.5%, 진입 마감 면제)
        conl_sig = sigs.get("conditional_conl", {})
        if conl_sig.get("buy_signal", False):
            if self._can_buy("CONL", ts, is_conditional=True):
                price = cur_prices.get("CONL")
                if price is not None:
                    self._buy(
                        "CONL", price, ts, "conditional_conl",
                        amount_krw=config.V5_INITIAL_BUY,
                        is_conl_conditional=True,
                    )

        # COIN buy (v5: trigger 4.5%, 진입 마감 면제)
        coin_sig = sigs.get("conditional_coin", {})
        if coin_sig.get("buy_signal", False):
            target = coin_sig.get("target", config.CONDITIONAL_TARGET_V5)
            if self._can_buy(target, ts, is_conditional=True):
                price = cur_prices.get(target)
                if price is not None:
                    self._buy(
                        target, price, ts, "conditional_coin",
                        amount_krw=config.V5_INITIAL_BUY,
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
        print("    PTJ 매매법 v5 백테스트 리포트 (선별 매매형)")
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

        # v5 전용 통계
        print("  [v5 선별 매매 효과]")
        print(f"    횡보장 감지일    : {self.sideways_days}일 / {self.total_trading_days}일")
        print(f"    횡보장 차단 매수 : {self.sideways_blocks}회")
        print(f"    진입시작 차단    : {self.entry_start_blocks}회")
        print(f"    시간제한 차단    : {self.entry_cutoff_blocks}회")
        print(f"    일일1회 차단     : {self.daily_limit_blocks}회")
        total_blocks = self.sideways_blocks + self.entry_start_blocks + self.entry_cutoff_blocks + self.daily_limit_blocks
        print(f"    총 차단 매수     : {total_blocks}회")
        print(f"    CB 매수 차단      : {self.cb_buy_blocks}회")
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

        # v5 파라미터 요약
        print("-" * 70)
        print("  [v5 파라미터]")
        print(f"    ENTRY 갭          : {config.V5_PAIR_GAP_ENTRY_THRESHOLD}% (v2: 1.5%)")
        print(f"    물타기 최대        : {config.V5_DCA_MAX_COUNT}회 (v2: 7회)")
        print(f"    종목당 최대 투입   : ${config.V5_MAX_PER_STOCK:,.0f}")
        print(f"    COIN/CONL 트리거   : {config.V5_COIN_TRIGGER_PCT}% (v2: 3.0%)")
        print(f"    중복 쿨타임        : {config.V5_SPLIT_BUY_INTERVAL_MIN}분 (v2: 5분)")
        print(f"    진입 마감          : {config.V5_ENTRY_CUTOFF_HOUR}:{config.V5_ENTRY_CUTOFF_MINUTE:02d} ET")
        print(f"    일일 거래 제한     : {config.V5_MAX_DAILY_TRADES_PER_STOCK}회/종목")
        print(f"    횡보장 감지        : {'ON' if config.V5_SIDEWAYS_ENABLED else 'OFF'} ({config.V5_SIDEWAYS_MIN_SIGNALS}/5 지표)")
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
        out_path = config.DATA_DIR / "backtest_v5_trades.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  거래 로그: {out_path}")
        return out_path


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("  PTJ 매매법 v5 - 1분봉 백테스트 (선별 매매형)")
    print()

    engine = BacktestEngineV5()
    engine.run()

    print("\n[3/4] 리포트 생성")
    engine.print_report()

    print("\n[4/4] 파일 저장")
    engine.save_trade_log()
    print()


if __name__ == "__main__":
    main()
