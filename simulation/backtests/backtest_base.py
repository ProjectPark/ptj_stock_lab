"""
PTJ 매매법 - 백테스트 엔진 기반 클래스 (Template Method)
=========================================================
v2~v5 모든 백테스트 엔진의 공통 인프라를 제공한다.

- 데이터 로드 / pre-indexing
- warmup (prev_close 계산)
- 메인 day loop 스켈레톤
- 매수/매도 실행 및 수수료 처리
- 자산 곡선 스냅샷
- 리포트 지표 (MDD, Sharpe)

각 버전은 _init_version_state, _on_day_start, _on_bar, _on_day_end 를
구현하여 버전별 매매 로직을 정의한다.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from datetime import datetime, date, time, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.backtests import backtest_common
from simulation.strategies.params import BaseParams, FeeConfig

# ============================================================
# Constants
# ============================================================
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# ============================================================
# 프로세스 레벨 데이터 캐시 (spawn worker당 1회 로드)
# ============================================================
# key: parquet 경로 문자열
# value: (df: pd.DataFrame, poly: dict, ts_prices, sym_bars, day_timestamps)
_DATA_CACHE: dict = {}


class BacktestBase(ABC):
    """Template Method 기반 백테스트 엔진 기반 클래스.

    서브클래스는 다음 4개의 추상 메서드를 구현한다:
    - _init_version_state(): 버전별 초기 상태
    - _on_day_start(): 일 시작 처리 (carry, 쿨다운 감소 등)
    - _on_bar(): 1분봉 바 처리 (시그널, 매수/매도)
    - _on_day_end(): 일 종료 처리 (EOD 청산, carry 설정 등)
    """

    def __init__(
        self,
        params: BaseParams,
        start_date: date = date(2025, 2, 18),
        end_date: date = date(2026, 2, 17),
        use_fees: bool = True,
        signal_engine=None,
    ):
        self.params = params
        self.start_date = start_date
        self.end_date = end_date
        self.use_fees = use_fees
        self._verbose = False
        self._signal_engine = signal_engine

        # Common state
        self.cash = params.total_capital
        self.initial_capital = params.total_capital
        self.positions: dict = {}
        self.trades: list = []
        self.equity_curve: list[tuple[date, float]] = []
        self._sold_today: set[str] = set()
        self._last_buy_time: dict[str, datetime] = {}

        # Fee accumulators
        self.total_buy_fees: float = 0.0
        self.total_sell_fees: float = 0.0

        # Statistics
        self.total_trading_days: int = 0
        self.skipped_gold_bars: int = 0

        # 자금 추적 (투입원금 vs 매매수익 분리)
        self.invested_capital: float = params.total_capital
        self.trading_pnl: float = 0.0
        self.injection_log: list[tuple[date, float]] = []
        self._injection_counter: int = 0

        # Version-specific init
        self._init_version_state()

    # ============================================================
    # Abstract hooks — subclass implements
    # ============================================================

    @abstractmethod
    def _init_version_state(self) -> None:
        """버전별 추가 상태 초기화."""
        ...

    @abstractmethod
    def _load_extra_data(self, df: pd.DataFrame, poly: dict) -> None:
        """데이터 로드 후 버전별 추가 데이터 준비.

        Parameters
        ----------
        df : pd.DataFrame
            1분봉 raw DataFrame
        poly : dict
            Polymarket daily data
        """
        ...

    @abstractmethod
    def _warmup_extra(self, warmup_dates: list, sym_bars: dict, prev_close: dict) -> None:
        """warmup 기간에 버전별 추가 처리."""
        ...

    @abstractmethod
    def _on_day_start(
        self,
        trading_date: date,
        day_idx: int,
        day_sym: dict,
        day_ts: list,
        poly_probs: dict | None,
        prev_close: dict[str, float],
    ) -> dict:
        """일 시작 시 처리. 반환 dict는 _on_bar, _on_day_end에 전달된다."""
        ...

    @abstractmethod
    def _on_bar(
        self,
        ts: datetime,
        cur_prices: dict[str, float],
        changes: dict[str, dict],
        day_ctx: dict,
    ) -> None:
        """1분봉 바 단위 처리."""
        ...

    @abstractmethod
    def _on_day_end(
        self,
        trading_date: date,
        day_idx: int,
        last_prices: dict[str, float],
        last_ts: datetime | None,
        prev_close: dict[str, float],
        day_ctx: dict,
    ) -> None:
        """일 종료 시 처리."""
        ...

    @abstractmethod
    def _version_label(self) -> str:
        """버전 이름 (예: 'v3', 'v5')."""
        ...

    # ============================================================
    # Common infrastructure (NOT overridden)
    # ============================================================

    @staticmethod
    def _market_minutes_elapsed(entry_time: datetime, current_time: datetime) -> float:
        """장중 경과 시간(분)을 계산한다."""
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

    def _calc_buy_fee(self, amount: float) -> float:
        """매수 수수료를 계산한다."""
        if not self.use_fees:
            return 0.0
        return backtest_common.calc_buy_fee(amount)

    def _calc_sell_fee(self, proceeds: float) -> float:
        """매도 수수료를 계산한다."""
        if not self.use_fees:
            return 0.0
        return backtest_common.calc_sell_fee(proceeds)

    def _snapshot_equity(self, cur_prices: dict[str, float]) -> float:
        """현재 총 자산을 계산한다. 서브클래스에서 오버라이드 가능."""
        equity = self.cash
        for ticker, pos in self.positions.items():
            price = cur_prices.get(ticker)
            if price is not None:
                equity += pos.total_qty * price * self._get_fx_multiplier()
            else:
                equity += self._get_position_fallback_value(pos)
        return equity

    def _get_fx_multiplier(self) -> float:
        """통화 변환 배수. v3(KRW)는 환율, v2/v4/v5(USD)는 1.0."""
        return 1.0

    def _get_position_fallback_value(self, pos) -> float:
        """포지션의 fallback 가치. 가격을 못 받을 때 사용."""
        # v2: total_invested_usd, v3+: total_invested_krw
        # 서브클래스에서 필요 시 오버라이드
        return getattr(pos, 'total_invested_krw',
                       getattr(pos, 'total_invested_usd', 0.0))

    def _load_data(self) -> tuple[pd.DataFrame, dict]:
        """공통 데이터를 로드한다. 같은 프로세스 내 2번째 호출부터는 캐시에서 반환한다."""
        import config
        parquet_path = config.OHLCV_1MIN_DEFAULT
        cache_key = str(parquet_path)

        if cache_key not in _DATA_CACHE:
            print("[1/4] 데이터 준비")
            print(f"  OHLCV: {parquet_path.name}")
            df = backtest_common.load_parquet(parquet_path)
            if "date" in df.columns and len(df) > 0 and isinstance(df["date"].iloc[0], str):
                import pandas as _pd
                df["date"] = _pd.to_datetime(df["date"]).dt.date
            poly = backtest_common.load_polymarket_daily(config.POLY_DATA_DIR)
            # 2-tuple로 임시 저장 (preindex는 run()에서 추가)
            _DATA_CACHE[cache_key] = (df, poly)
            print(f"  Loaded: {len(df):,} rows  (RAM 캐시 저장)")
        else:
            print("[1/4] 데이터 준비 (RAM 캐시)")
            entry = _DATA_CACHE[cache_key]
            df, poly = entry[0], entry[1]

        return df, poly

    def _select_coin_follow(
        self, sym_bars_day: dict[str, list[dict]], trading_date=None
    ) -> str:
        """당일 MSTU vs IRE 코인 후행 종목을 선택한다."""
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
        if diff >= self.params.coin_follow_volatility_gap:
            return "MSTU" if mstu_vol > ire_vol else "IRE"
        else:
            return "MSTU" if stats["MSTU"]["volume"] >= stats["IRE"]["volume"] else "IRE"

    # ============================================================
    # Main loop — Template Method
    # ============================================================

    def run(self, verbose: bool = True) -> "BacktestBase":
        """메인 백테스트 루프를 실행한다."""
        self._verbose = verbose

        # 1. Load data (프로세스 내 캐시 활용)
        import config as _cfg
        cache_key = str(_cfg.OHLCV_1MIN_DEFAULT)
        df, poly = self._load_data()

        # Pre-index DataFrame for O(1) bar lookup (캐시 활용)
        if cache_key in _DATA_CACHE and len(_DATA_CACHE[cache_key]) == 5:
            _, _, ts_prices, sym_bars, day_timestamps = _DATA_CACHE[cache_key]
        else:
            ts_prices, sym_bars, day_timestamps = backtest_common.preindex_dataframe(df)
            _DATA_CACHE[cache_key] = (df, poly, ts_prices, sym_bars, day_timestamps)

        # Version-specific extra data loading
        self._load_extra_data(df, poly)

        # 2. Prepare trading dates
        all_dates = sorted(day_timestamps.keys())
        bt_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        warmup_dates = [d for d in all_dates if d < self.start_date]

        # Warmup: compute prev_close
        prev_close: dict[str, float] = {}
        for d in warmup_dates:
            if d not in sym_bars:
                continue
            for ticker, bars in sym_bars[d].items():
                if bars:
                    prev_close[ticker] = bars[-1]["close"]

        # Version-specific warmup
        self._warmup_extra(warmup_dates, sym_bars, prev_close)

        self.total_trading_days = len(bt_dates)
        if verbose:
            label = self._version_label()
            print(f"\n[2/4] 시뮬레이션 실행 ({label})")
            print(f"  백테스트 기간: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}일)")
            self._print_run_header()

        # 3. Main day loop
        for day_idx, trading_date in enumerate(bt_dates):
            if trading_date not in ts_prices:
                continue

            day_sym = sym_bars[trading_date]
            day_ts = day_timestamps[trading_date]
            poly_probs = poly.get(trading_date, None)

            # Capital injection check
            if self.params.injection_interval_days > 0 and self.params.injection_pct > 0:
                self._injection_counter += 1
                if self._injection_counter >= self.params.injection_interval_days:
                    self._injection_counter = 0
                    amount = self.invested_capital * self.params.injection_pct / 100
                    self._apply_injection(amount)
                    self.invested_capital += amount
                    self.injection_log.append((trading_date, amount))

            # Day start
            day_ctx = self._on_day_start(
                trading_date, day_idx, day_sym, day_ts, poly_probs, prev_close,
            )

            # Process each 1-min bar
            for ts in day_ts:
                cur_prices = ts_prices[trading_date].get(ts, {})
                # Allow version to augment prices
                cur_prices = self._augment_prices(cur_prices, trading_date)

                changes = backtest_common.calc_changes(cur_prices, prev_close)

                self._on_bar(ts, cur_prices, changes, day_ctx)

            # Day end
            last_prices = {sym: bars[-1]["close"] for sym, bars in day_sym.items() if bars}
            last_prices = self._augment_prices(last_prices, trading_date)
            last_ts = day_ts[-1] if day_ts else None

            self._on_day_end(
                trading_date, day_idx, last_prices, last_ts, prev_close, day_ctx,
            )

            # Update prev_close
            prev_close.update(last_prices)

            # Equity snapshot
            equity = self._snapshot_equity(
                cur_prices if last_prices else prev_close
            )
            self.equity_curve.append((trading_date, equity))

            # Progress
            if verbose and ((day_idx + 1) % 50 == 0 or day_idx == len(bt_dates) - 1):
                self._print_day_progress(day_idx, len(bt_dates), trading_date, equity, day_ctx)

        if verbose:
            print("\n  백테스트 완료!")

        self._on_run_complete()
        return self

    def _augment_prices(self, prices: dict[str, float], trading_date: date) -> dict[str, float]:
        """버전별로 가격 dict를 보강한다 (예: v4의 SOXX/IREN replacement)."""
        return prices

    def _print_run_header(self) -> None:
        """run() 시작 시 버전별 헤더를 출력한다. 서브클래스에서 오버라이드."""
        pass

    def _print_day_progress(
        self, day_idx: int, total_days: int, trading_date: date,
        equity: float, day_ctx: dict,
    ) -> None:
        """일별 진행 상황을 출력한다. 서브클래스에서 오버라이드."""
        print(
            f"  [{day_idx+1:>3}/{total_days}] {trading_date}  "
            f"자산: {equity:,.2f}  "
            f"포지션: {len(self.positions)}개"
        )

    def _apply_injection(self, amount: float) -> None:
        """자금 유입을 적용한다. 서브클래스에서 오버라이드."""
        self.cash += amount

    def _on_run_complete(self) -> None:
        """run() 완료 후 버전별 후처리. 서브클래스에서 오버라이드."""
        pass

    # ============================================================
    # Report helpers
    # ============================================================

    def calc_mdd(self) -> float:
        return backtest_common.calc_mdd(self.equity_curve)

    def calc_sharpe(self) -> float:
        return backtest_common.calc_sharpe(self.equity_curve)
