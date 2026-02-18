#!/usr/bin/env python3
"""
PTJ 매매법 - 5분봉 백테스트 시뮬레이션
======================================
Alpaca 5분봉 데이터 기반, 실제 매매 규칙을 적용한 1년간 시뮬레이션.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import config
import signals
import backtest_common

# ============================================================
# Constants
# ============================================================
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
GOLD_CHECK_BARS = 6  # 첫 30분 = 5분봉 6개
ALLOC_TWIN = 0.20     # 쌍둥이 페어당 20%
ALLOC_COND = 0.20     # 조건부 매매 20%
ALLOC_BEAR = 0.10     # 하락장 방어주 10%

# KIS 수수료 체계 (미국주식)
KIS_COMMISSION_PCT = 0.25    # 매매 수수료 0.25% (매수/매도 각각)
KIS_SEC_FEE_PCT = 0.00278   # SEC Fee 0.00278% (매도 시에만)
KIS_FX_SPREAD_PCT = 0.10    # 환전 스프레드 약 0.1% (편도)


# ============================================================
# Data Fetching (Alpaca)
# ============================================================
def fetch_backtest_data(
    start_date: date,
    end_date: date,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Alpaca에서 5분봉 데이터 수집. parquet 캐시 지원."""
    cache_path = config.DATA_DIR / "backtest_5min.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    load_dotenv(config.PROJECT_ROOT / ".env")
    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )

    tickers = list(config.TICKERS.keys())
    print(f"  Alpaca에서 {len(tickers)}개 종목 5분봉 수집 중...")

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=datetime.combine(start_date, time()),
        end=datetime.combine(end_date, time()),
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    # UTC → US/Eastern 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")

    # 정규장만 필터 (9:30 ~ 16:00 ET)
    t = df["timestamp"].dt.time
    df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()

    df["date"] = df["timestamp"].dt.date
    df = df.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"  수집 완료: {len(df):,} rows → {cache_path.name}")
    return df


def fetch_1min_data(
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Alpaca에서 1분봉 데이터 수집. 월 단위 chunking + parquet 캐시."""
    cache_path = config.DATA_DIR / "backtest_1min.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    load_dotenv(config.PROJECT_ROOT / ".env")
    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )

    tickers = list(config.TICKERS.keys())
    print(f"  Alpaca에서 {len(tickers)}개 종목 1분봉 수집 중...")
    print(f"  기간: {start_date} ~ {end_date}")

    # 월 단위 chunking (API 제한 대응)
    chunks: list[pd.DataFrame] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        # 청크 끝: 해당 월 말일 또는 end_date 중 빠른 쪽
        if chunk_start.month == 12:
            next_month = date(chunk_start.year + 1, 1, 1)
        else:
            next_month = date(chunk_start.year, chunk_start.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end_date)

        label = f"{chunk_start.strftime('%Y-%m')}"
        print(f"    [{label}] {chunk_start} ~ {chunk_end} ...", end=" ", flush=True)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=datetime.combine(chunk_start, time()),
                end=datetime.combine(chunk_end + timedelta(days=1), time()),
            )

            bars = client.get_stock_bars(request)
            chunk_df = bars.df.reset_index()
            print(f"{len(chunk_df):,} rows")
            chunks.append(chunk_df)
        except Exception as e:
            print(f"SKIP ({e})")

        chunk_start = next_month

    df = pd.concat(chunks, ignore_index=True)
    print(f"  총 수집: {len(df):,} rows")

    # UTC → US/Eastern 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")

    # 정규장만 필터 (9:30 ~ 16:00 ET)
    t = df["timestamp"].dt.time
    df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()

    df["date"] = df["timestamp"].dt.date
    df = df.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"  필터 후: {len(df):,} rows → {cache_path.name}")
    return df


def fetch_1min_data_v2(
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Alpaca에서 v2 종목 1분봉 데이터 수집. 월 단위 chunking + parquet 캐시.

    v1 대비 변경:
    - AMDL 제거, SOXL/IRE(IREN) 추가, HIMZ 제거
    - PTJ 티커 ↔ Alpaca 티커 자동 매핑 (IRE ↔ IREN)
    """
    cache_path = config.DATA_DIR / "backtest_1min_v2.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [v2] 캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    load_dotenv(config.PROJECT_ROOT / ".env")
    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
    )

    # PTJ 티커 → Alpaca 티커 변환
    ptj_tickers = list(config.TICKERS_V2.keys())
    alpaca_tickers = [config.ALPACA_TICKER_MAP.get(t, t) for t in ptj_tickers]

    print(f"  [v2] Alpaca에서 {len(alpaca_tickers)}개 종목 1분봉 수집 중...")
    print(f"  기간: {start_date} ~ {end_date}")
    print(f"  종목: {', '.join(alpaca_tickers)}")

    # 월 단위 chunking (API 제한 대응)
    chunks: list[pd.DataFrame] = []
    chunk_start = start_date
    while chunk_start <= end_date:
        if chunk_start.month == 12:
            next_month = date(chunk_start.year + 1, 1, 1)
        else:
            next_month = date(chunk_start.year, chunk_start.month + 1, 1)
        chunk_end = min(next_month - timedelta(days=1), end_date)

        label = f"{chunk_start.strftime('%Y-%m')}"
        print(f"    [{label}] {chunk_start} ~ {chunk_end} ...", end=" ", flush=True)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=alpaca_tickers,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=datetime.combine(chunk_start, time()),
                end=datetime.combine(chunk_end + timedelta(days=1), time()),
            )

            bars = client.get_stock_bars(request)
            chunk_df = bars.df.reset_index()
            print(f"{len(chunk_df):,} rows")
            chunks.append(chunk_df)
        except Exception as e:
            print(f"SKIP ({e})")

        chunk_start = next_month

    df = pd.concat(chunks, ignore_index=True)
    print(f"  [v2] 총 수집: {len(df):,} rows")

    # Alpaca 티커 → PTJ 티커 역매핑 (IREN → IRE)
    df["symbol"] = df["symbol"].replace(config.ALPACA_TICKER_REVERSE)

    # UTC → US/Eastern 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")

    # 정규장만 필터 (9:30 ~ 16:00 ET)
    t = df["timestamp"].dt.time
    df = df[(t >= MARKET_OPEN) & (t < MARKET_CLOSE)].copy()

    df["date"] = df["timestamp"].dt.date
    df = df.sort_values(["date", "symbol", "timestamp"]).reset_index(drop=True)

    # 종목별 row 수 출력
    counts = df.groupby("symbol").size()
    print(f"  [v2] 종목별 행 수:")
    for sym, cnt in counts.items():
        print(f"    {sym}: {cnt:,}")

    df.to_parquet(cache_path, index=False)
    print(f"  [v2] 필터 후: {len(df):,} rows → {cache_path.name}")
    return df


def fetch_usdkrw_hourly(
    start_date: date = date(2025, 1, 3),
    end_date: date = date(2026, 2, 17),
    use_cache: bool = True,
) -> pd.DataFrame:
    """yfinance에서 USD/KRW 1시간봉 수집. parquet 캐시 지원.

    yfinance 제한: 1분봉은 최근 7일만, 1시간봉은 최근 730일까지 가능.
    환율 백테스트 용도로 1시간봉이면 충분.
    """
    cache_path = config.DATA_DIR / "usdkrw_hourly.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  [USD/KRW] 캐시 로드: {len(df):,} rows ({cache_path.name})")
        return df

    import yfinance as yf

    print(f"  [USD/KRW] yfinance에서 1시간봉 수집 중...")
    print(f"  기간: {start_date} ~ {end_date}")

    ticker = yf.Ticker("KRW=X")
    raw = ticker.history(
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),
        interval="1h",
    )

    if raw.empty:
        print("  [USD/KRW] 데이터 없음!")
        return pd.DataFrame()

    df = raw.reset_index()
    df = df.rename(columns={
        "Datetime": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # 타임존 → US/Eastern 변환 (주식 시간과 맞추기)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

    df["date"] = df["timestamp"].dt.date
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"  [USD/KRW] 수집 완료: {len(df):,} rows → {cache_path.name}")
    print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  환율 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    return df


# ============================================================
# Data Model
# ============================================================
@dataclass
class Trade:
    """매매 기록."""
    date: date
    ticker: str
    side: str           # BUY / SELL
    price: float
    qty: float
    amount: float
    pnl: float
    pnl_pct: float
    signal_type: str
    entry_time: object = None
    exit_time: object = None


# ============================================================
# Backtest Engine
# ============================================================
class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 1000.0,
        start_date: date = date(2025, 2, 18),
        end_date: date = date(2026, 2, 17),
        stop_loss_pct: float | None = None,
        use_fees: bool = False,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else config.STOP_LOSS_PCT
        self.use_fees = use_fees

        # 수수료 누적
        self.total_commission = 0.0
        self.total_sec_fee = 0.0
        self.total_fx_cost = 0.0

        # 상태
        self.positions: dict[str, dict] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[date, float]] = []

        # 통계
        self.skipped_days = 0
        self.total_trading_days = 0
        self._stopped_today: set[str] = set()  # 당일 손절된 종목 (재진입 방지)

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    def run(self, verbose: bool = True, data: pd.DataFrame | None = None) -> "BacktestEngine":
        """백테스트 실행."""
        # 데이터가 외부에서 전달되면 재사용 (최적화 루프용)
        if data is None:
            fetch_start = self.start_date - timedelta(days=10)
            data = fetch_backtest_data(fetch_start, self.end_date)

        # Pre-index DataFrame for O(1) bar lookup
        ts_prices, sym_bars, day_timestamps = backtest_common.preindex_dataframe(data)

        all_dates = sorted(day_timestamps.keys())
        # 백테스트 기간 필터
        bt_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        # 전일 종가용 (백테스트 시작 전 데이터)
        warmup_dates = [d for d in all_dates if d < self.start_date]

        # 전일 종가 초기화
        prev_day_close: dict[str, float] = {}
        for d in warmup_dates:
            if d not in sym_bars:
                continue
            for ticker, bars in sym_bars[d].items():
                if bars:
                    prev_day_close[ticker] = bars[-1]["close"]

        self.total_trading_days = len(bt_dates)
        if verbose:
            print(f"\n  백테스트 기간: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}일)")
            print(f"  초기 자금: ${self.initial_capital:,.2f}\n")

        for i, trading_date in enumerate(bt_dates):
            if trading_date not in ts_prices:
                continue

            day_sym = sym_bars[trading_date]
            day_ts = day_timestamps[trading_date]
            self._stopped_today.clear()

            # 1) 금 시황 — 첫 30분 체크
            gold_warning = self._check_gold_30min(day_sym, prev_day_close)

            if gold_warning:
                self.skipped_days += 1
            else:
                # 2) 5분봉 순회 매매 (O(1) dict lookup per bar)
                start_idx = min(GOLD_CHECK_BARS, len(day_ts) - 1)

                for t_idx in range(start_idx, len(day_ts)):
                    ts = day_ts[t_idx]
                    cur_prices = ts_prices[trading_date].get(ts, {})

                    # 등락률 (전일 종가 대비)
                    changes = self._calc_changes(cur_prices, prev_day_close)

                    # 손절 체크 (시그널보다 먼저)
                    self._check_stop_loss(cur_prices, ts)

                    # 시그널 → 매매
                    sigs = signals.generate_all_signals(changes)
                    self._execute_signals(sigs, cur_prices, ts)

            # 3) 당일 청산 (day-trade 원칙)
            last_prices = {sym: bars[-1]["close"] for sym, bars in day_sym.items() if bars}
            last_ts = day_ts[-1] if day_ts else None
            self._close_all_positions(last_prices, last_ts, "eod_close")

            # 4) 전일 종가 업데이트
            prev_day_close.update(last_prices)

            # 5) 자산 기록
            equity = self.cash  # 당일 청산 후 포지션 없음
            self.equity_curve.append((trading_date, equity))

            # 진행률 출력
            if verbose and ((i + 1) % 50 == 0 or i == len(bt_dates) - 1):
                print(f"  [{i+1:>3}/{len(bt_dates)}] {trading_date}  자산: ${equity:,.2f}")

        if verbose:
            print("\n  백테스트 완료!")
        return self

    # ----------------------------------------------------------
    # Gold check
    # ----------------------------------------------------------
    def _check_gold_30min(
        self, sym_bars_day: dict[str, list[dict]], prev_close: dict[str, float]
    ) -> bool:
        """첫 30분간 금(GLD) 상승 여부 확인."""
        gold_bars = sym_bars_day.get(config.GOLD_TICKER, [])
        if len(gold_bars) < 2:
            return False

        first_bars = gold_bars[:GOLD_CHECK_BARS]
        gold_open = first_bars[0]["open"]
        gold_at_30m = first_bars[-1]["close"]

        ref = prev_close.get(config.GOLD_TICKER, gold_open)
        change = (gold_at_30m - ref) / ref * 100 if ref > 0 else 0
        return change > 0

    # ----------------------------------------------------------
    # Price helpers
    # ----------------------------------------------------------
    @staticmethod
    def _last_prices(day_data: pd.DataFrame) -> dict[str, float]:
        last = day_data.groupby("symbol").last()
        return {sym: float(row["close"]) for sym, row in last.iterrows()}

    @staticmethod
    def _calc_changes(
        cur_prices: dict[str, float], prev_close: dict[str, float]
    ) -> dict[str, dict]:
        changes = {}
        for ticker, price in cur_prices.items():
            prev = prev_close.get(ticker)
            if prev and prev > 0:
                pct = (price - prev) / prev * 100
            else:
                pct = 0.0
            changes[ticker] = {
                "close": price,
                "prev_close": prev or price,
                "change_pct": round(pct, 2),
            }
        return changes

    # ----------------------------------------------------------
    # Stop loss
    # ----------------------------------------------------------
    def _check_stop_loss(self, cur_prices: dict[str, float], ts) -> None:
        to_sell = []
        for ticker, pos in self.positions.items():
            price = cur_prices.get(ticker)
            if price is None:
                continue
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            if pnl_pct <= self.stop_loss_pct:
                to_sell.append((ticker, price))

        for ticker, price in to_sell:
            self._sell(ticker, price, ts, "stop_loss")
            self._stopped_today.add(ticker)

    # ----------------------------------------------------------
    # Signal execution
    # ----------------------------------------------------------
    def _execute_signals(
        self, sigs: dict, cur_prices: dict[str, float], ts
    ) -> None:
        """시그널을 실제 매매로 변환."""

        # --- 쌍둥이 페어 ---
        for pair in sigs.get("twin_pairs", []):
            follow = pair["follow"]
            gap = pair["gap"]
            signal = pair["signal"]

            if signal == "ENTRY" and gap > 0:
                # follow가 뒤처짐 → follow 매수
                if (
                    follow not in self.positions
                    and follow not in self._stopped_today
                    and follow in cur_prices
                ):
                    self._buy(follow, cur_prices[follow], ts, "twin", ALLOC_TWIN)

            elif signal == "SELL" and follow in self.positions and follow in cur_prices:
                self._sell(follow, cur_prices[follow], ts, "twin_converge")

        # --- 조건부 매매 (COIN) ---
        cond = sigs.get("conditional", {})
        target = cond.get("target", config.CONDITIONAL_TARGET)
        if (
            cond.get("all_positive")
            and target not in self.positions
            and target not in self._stopped_today
            and target in cur_prices
        ):
            self._buy(target, cur_prices[target], ts, "conditional", ALLOC_COND)

        # --- 하락장 방어주 ---
        bearish = sigs.get("bearish", {})
        if bearish.get("market_down") and not bearish.get("gold_up"):
            for pick in bearish.get("bearish_picks", []):
                t = pick["ticker"]
                if (
                    t not in self.positions
                    and t not in self._stopped_today
                    and t in cur_prices
                ):
                    self._buy(t, cur_prices[t], ts, "bearish", ALLOC_BEAR)

    # ----------------------------------------------------------
    # Order execution
    # ----------------------------------------------------------
    def _buy(
        self, ticker: str, price: float, ts, signal_type: str, alloc: float
    ) -> None:
        if ticker in self.positions or price <= 0:
            return

        amount = self.cash * alloc
        if amount < 1.0:
            return

        # 수수료 차감 (매수 시: 매매수수료 + 환전스프레드)
        if self.use_fees:
            buy_fee = amount * (KIS_COMMISSION_PCT + KIS_FX_SPREAD_PCT) / 100
            self.total_commission += amount * KIS_COMMISSION_PCT / 100
            self.total_fx_cost += amount * KIS_FX_SPREAD_PCT / 100
        else:
            buy_fee = 0.0

        net_amount = amount - buy_fee
        qty = net_amount / price
        self.cash -= amount

        self.positions[ticker] = {
            "qty": qty,
            "entry_price": price,
            "entry_time": ts,
            "signal_type": signal_type,
        }

        self.trades.append(Trade(
            date=ts.date() if hasattr(ts, "date") and callable(ts.date) else ts,
            ticker=ticker,
            side="BUY",
            price=price,
            qty=qty,
            amount=amount,
            pnl=0.0,
            pnl_pct=0.0,
            signal_type=signal_type,
            entry_time=ts,
        ))

    def _sell(self, ticker: str, price: float, ts, reason: str) -> None:
        if ticker not in self.positions:
            return

        pos = self.positions.pop(ticker)
        gross_proceeds = pos["qty"] * price
        cost = pos["qty"] * pos["entry_price"]

        # 수수료 차감 (매도 시: 매매수수료 + SEC Fee + 환전스프레드)
        if self.use_fees:
            sell_commission = gross_proceeds * KIS_COMMISSION_PCT / 100
            sec_fee = gross_proceeds * KIS_SEC_FEE_PCT / 100
            fx_cost = gross_proceeds * KIS_FX_SPREAD_PCT / 100
            sell_fee = sell_commission + sec_fee + fx_cost
            self.total_commission += sell_commission
            self.total_sec_fee += sec_fee
            self.total_fx_cost += fx_cost
        else:
            sell_fee = 0.0

        proceeds = gross_proceeds - sell_fee
        pnl = proceeds - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0.0
        self.cash += proceeds

        self.trades.append(Trade(
            date=ts.date() if hasattr(ts, "date") and callable(ts.date) else ts,
            ticker=ticker,
            side="SELL",
            price=price,
            qty=pos["qty"],
            amount=proceeds,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 2),
            signal_type=reason,
            entry_time=pos["entry_time"],
            exit_time=ts,
        ))

    def _close_all_positions(
        self, prices: dict[str, float], ts, reason: str
    ) -> None:
        for ticker in list(self.positions.keys()):
            price = prices.get(ticker, self.positions[ticker]["entry_price"])
            self._sell(ticker, price, ts, reason)

    # ----------------------------------------------------------
    # Summary dict (for optimization)
    # ----------------------------------------------------------
    def summary(self) -> dict:
        """주요 지표를 dict로 반환."""
        if not self.equity_curve:
            return {}
        final = self.equity_curve[-1][1]
        total_ret = (final - self.initial_capital) / self.initial_capital * 100
        sells = [t for t in self.trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        stop_losses = [t for t in sells if t.signal_type == "stop_loss"]
        total_pnl = sum(t.pnl for t in sells)
        sl_pnl = sum(t.pnl for t in stop_losses)

        eq_vals = [e[1] for e in self.equity_curve]
        peak = eq_vals[0]
        mdd = 0.0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > mdd:
                mdd = dd

        daily_rets = []
        for i in range(1, len(eq_vals)):
            daily_rets.append((eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1])
        avg_r = np.mean(daily_rets) if daily_rets else 0
        std_r = np.std(daily_rets) if daily_rets else 1
        sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0

        return {
            "final_equity": round(final, 2),
            "total_return_pct": round(total_ret, 2),
            "total_pnl": round(total_pnl, 2),
            "mdd_pct": round(mdd, 2),
            "sharpe": round(sharpe, 2),
            "total_sells": len(sells),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
            "stop_loss_count": len(stop_losses),
            "stop_loss_pnl": round(sl_pnl, 2),
            "skipped_days": self.skipped_days,
            "total_fees": round(self.total_commission + self.total_sec_fee + self.total_fx_cost, 2),
            "commission": round(self.total_commission, 2),
            "sec_fee": round(self.total_sec_fee, 2),
            "fx_cost": round(self.total_fx_cost, 2),
        }

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------
    def print_report(self) -> None:
        """콘솔 요약 리포트."""
        if not self.equity_curve:
            print("데이터 없음")
            return

        final = self.equity_curve[-1][1]
        total_ret = (final - self.initial_capital) / self.initial_capital * 100

        sells = [t for t in self.trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl > 0]
        losses = [t for t in sells if t.pnl < 0]
        breakeven = [t for t in sells if t.pnl == 0]
        total_pnl = sum(t.pnl for t in sells)

        # MDD
        eq_vals = [e[1] for e in self.equity_curve]
        peak = eq_vals[0]
        mdd = 0.0
        for v in eq_vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak * 100
            if dd > mdd:
                mdd = dd

        # Sharpe
        daily_rets = []
        for i in range(1, len(eq_vals)):
            daily_rets.append((eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1])
        avg_r = np.mean(daily_rets) if daily_rets else 0
        std_r = np.std(daily_rets) if daily_rets else 1
        sharpe = (avg_r / std_r) * np.sqrt(252) if std_r > 0 else 0

        # 시그널별 P&L
        sig_stats: dict[str, dict] = {}
        for t in sells:
            key = t.signal_type
            if key not in sig_stats:
                sig_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            sig_stats[key]["count"] += 1
            sig_stats[key]["pnl"] += t.pnl
            if t.pnl > 0:
                sig_stats[key]["wins"] += 1

        print()
        print("=" * 55)
        print("    PTJ 매매법 백테스트 리포트")
        print("=" * 55)
        print(f"  기간        : {self.equity_curve[0][0]} ~ {self.equity_curve[-1][0]}")
        print(f"  거래일      : {self.total_trading_days}일")
        print(f"  초기 자금   : ${self.initial_capital:,.2f}")
        print(f"  최종 자산   : ${final:,.2f}")
        print(f"  총 수익률   : {total_ret:+.2f}%")
        print(f"  총 손익     : ${total_pnl:+,.2f}")
        print(f"  최대 낙폭   : -{mdd:.2f}%")
        print(f"  Sharpe Ratio: {sharpe:.2f}")
        print("-" * 55)
        print(f"  총 매도 횟수: {len(sells)}")
        print(f"  승 / 패     : {len(wins)}W / {len(losses)}L / {len(breakeven)}E")
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        print(f"  승률        : {win_rate:.1f}%")
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        print(f"  평균 수익   : ${avg_win:+.4f}")
        print(f"  평균 손실   : ${avg_loss:+.4f}")
        print(f"  금 스킵일   : {self.skipped_days}일 / {self.total_trading_days}일")
        print("-" * 55)

        if sig_stats:
            print("  [시그널별 성과]")
            for key in sorted(sig_stats.keys()):
                s = sig_stats[key]
                wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                print(
                    f"    {key:18s}  {s['count']:>4d}회  "
                    f"P&L ${s['pnl']:>+9.2f}  승률 {wr:>5.1f}%"
                )
        print("=" * 55)

    # ----------------------------------------------------------
    # Chart
    # ----------------------------------------------------------
    def plot_equity_curve(self) -> Path | None:
        """자산 곡선 PNG 생성."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = [e[0] for e in self.equity_curve]
        vals = [e[1] for e in self.equity_curve]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
        fig.suptitle("PTJ Trading Strategy — Backtest", fontsize=14, fontweight="bold")

        # 상단: Equity curve
        ax1.plot(dates, vals, linewidth=1.2, color="#2196F3", label="Portfolio")
        ax1.axhline(
            y=self.initial_capital, color="gray", linestyle="--", alpha=0.5,
            label=f"Initial ${self.initial_capital:,.0f}",
        )
        ax1.fill_between(
            dates, self.initial_capital, vals,
            where=[v >= self.initial_capital for v in vals],
            alpha=0.12, color="green",
        )
        ax1.fill_between(
            dates, self.initial_capital, vals,
            where=[v < self.initial_capital for v in vals],
            alpha=0.12, color="red",
        )
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 하단: Drawdown
        peak = vals[0]
        dd_pcts = []
        for v in vals:
            if v > peak:
                peak = v
            dd_pcts.append(-((peak - v) / peak * 100))
        ax2.fill_between(dates, 0, dd_pcts, color="red", alpha=0.3)
        ax2.plot(dates, dd_pcts, color="red", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        for ax in (ax1, ax2):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()

        out = config.CHART_DIR / "backtest_equity.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  차트 저장: {out}")
        return out

    # ----------------------------------------------------------
    # Trade log CSV
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
                "amount": round(t.amount, 2),
                "pnl": round(t.pnl, 4),
                "pnl_pct": round(t.pnl_pct, 2),
                "signal": t.signal_type,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            })
        df = pd.DataFrame(rows)
        out = config.DATA_DIR / "backtest_trades.csv"
        df.to_csv(out, index=False)
        print(f"  거래 로그: {out}")
        return out


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║   PTJ 매매법 — 5분봉 백테스트 시뮬레이션   ║")
    print("  ╚═══════════════════════════════════════════════╝")
    print()
    print("[1/4] 데이터 준비")

    engine = BacktestEngine(
        initial_capital=1000.0,
        start_date=date(2025, 2, 18),
        end_date=date(2026, 2, 17),
    )

    print("\n[2/4] 시뮬레이션 실행")
    engine.run()

    print("\n[3/4] 리포트 생성")
    engine.print_report()

    print("\n[4/4] 파일 저장")
    engine.plot_equity_curve()
    engine.save_trade_log()
    print()


if __name__ == "__main__":
    main()
