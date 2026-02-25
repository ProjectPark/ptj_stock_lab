#!/usr/bin/env python3
"""
Phase 4B — 분봉 D2S Study
=================================================
일봉 D2S 신호 → 분봉 진입/청산 타이밍 3가지 비교.

시나리오:
  baseline_daily : 일봉 신호 → 당일 09:30 시가 진입, hold_days 마지막날 15:55 청산
  entry_opt      : 일봉 신호 → 09:30~10:30 중 최저점 봉에서 진입
  tp_realtime    : entry_opt + 분봉 high가 TP 가격 초과 시 즉시 청산

기간: WARM 2025-03-03 ~ 2026-01-30 (분봉 데이터 범위)
실거래 목표: 승률 65.5%, 평균 PnL +7.39%, 거래수 722건

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_d2s_1min.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.backtests import backtest_common
from simulation.strategies.line_c_d2s.d2s_engine import (
    D2SEngine,
    D2SPosition,
    DailySnapshot,
    TechnicalPreprocessor,
)
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE

# ============================================================
# 상수
# ============================================================
BUY_FEE_PCT = 0.35 / 100    # 매수 수수료율
SELL_FEE_PCT = 0.353 / 100   # 매도 수수료율

START_DATE = date(2025, 3, 3)   # 기술적 지표 워밍업 후
END_DATE = date(2026, 1, 30)    # 분봉 데이터 끝

# 실거래 목표치
TARGET_WIN_RATE = 65.5
TARGET_AVG_PNL = 7.39
TARGET_TRADES = 722


# ============================================================
# 포지션 컨테이너
# ============================================================
@dataclass
class MinutePosition:
    """분봉 시뮬레이션 포지션."""
    ticker: str
    entry_price: float
    qty: float
    entry_date: date
    cost_basis: float       # 총 투자 금액 (수수료 제외 순액)
    signal_info: dict       # 원본 D2S 시그널 정보
    tp_price: float = 0.0   # take profit 가격 (tp_realtime용)


@dataclass
class ClosedTrade:
    """청산 완료 거래."""
    ticker: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    hold_days: int
    reason: str


# ============================================================
# Step 1: 일봉 D2S 신호 추출
# ============================================================
def extract_daily_signals(
    params: dict,
) -> tuple[dict[date, list[dict]], dict[str, pd.DataFrame], dict]:
    """D2SBacktest 로직을 재사용하여 날짜별 BUY 신호를 추출한다.

    Returns
    -------
    signals_by_date : dict[date, list[dict]]
        {날짜: [signal_dict, ...]}  (BUY 신호만)
    tech : dict[str, pd.DataFrame]
        종목별 기술적 지표
    poly : dict
        Polymarket 데이터
    """
    data_dir = _PROJECT_ROOT / "data"
    market_path = data_dir / "market" / "daily" / "market_daily.parquet"
    poly_dir = data_dir / "polymarket"

    print("[1/4] 일봉 데이터 로드")
    df = pd.read_parquet(market_path)
    poly = backtest_common.load_polymarket_daily(poly_dir)

    # Polymarket 폴백: BITU 등락률 → pseudo btc_up
    if not poly:
        print("  Polymarket 데이터 없음 → BITU 폴백 생성")
        try:
            bitu_close = df[("BITU", "Close")]
            bitu_pct = bitu_close.pct_change() * 100
            for ts, pct in bitu_pct.items():
                if pd.isna(pct):
                    continue
                d = ts.date() if hasattr(ts, "date") else ts
                btc_up = float(np.clip(0.50 + pct / 25.0, 0.30, 0.70))
                poly[d] = {"btc_up": round(btc_up, 3)}
        except KeyError:
            pass

    print("[2/4] 기술적 지표 계산")
    preprocessor = TechnicalPreprocessor(params)
    tech = preprocessor.compute(df)
    print(f"  {len(tech)} tickers processed")

    engine = D2SEngine(params)

    # 거래일 목록
    all_dates = sorted(df.index)
    trading_dates = [
        d.date() if hasattr(d, "date") else d
        for d in all_dates
        if START_DATE <= (d.date() if hasattr(d, "date") else d) <= END_DATE
    ]

    print(f"\n[3/4] 일봉 D2S 신호 생성 ({trading_dates[0]} ~ {trading_dates[-1]}, {len(trading_dates)}일)")

    # D2SBacktest.run()과 동일한 루프로 BUY 신호만 추출
    spy_streak = 0
    riskoff_streak = 0
    positions: dict[str, D2SPosition] = {}
    signals_by_date: dict[date, list[dict]] = {}

    for td in trading_dates:
        snap = _build_snapshot(td, tech, poly, spy_streak, riskoff_streak)
        if snap is None:
            continue

        daily_buy_counts: dict[str, int] = {}

        # 시그널 생성
        sigs = engine.generate_daily_signals(snap, positions, daily_buy_counts)

        # SELL 실행 (포지션 상태 유지를 위해)
        for sig in sigs:
            if sig["action"] == "SELL":
                pos = positions.get(sig["ticker"])
                if pos:
                    del positions[sig["ticker"]]

        # BUY 시그널 수집 + 포지션 추적 (D2SBacktest._execute_buy와 유사)
        buy_signals = []
        daily_entry_used = 0.0
        daily_entry_cap = params.get("daily_new_entry_cap", 1.0)
        cash = params["total_capital"]  # 단순화: 매일 전체 자본 기준 비중 계산

        for sig in sigs:
            if sig["action"] == "BUY":
                entry_fraction = sig["size"]
                if daily_entry_used + entry_fraction > daily_entry_cap:
                    continue

                price = snap.closes.get(sig["ticker"], 0)
                if price <= 0:
                    continue

                amount = cash * sig["size"]
                fee = amount * BUY_FEE_PCT
                net_amount = amount - fee
                qty = net_amount / price

                # 포지션 추적 (일봉 엔진 상태 유지)
                existing = positions.get(sig["ticker"])
                if existing:
                    total_cost = existing.cost_basis + net_amount
                    total_qty = existing.qty + qty
                    existing.entry_price = total_cost / total_qty
                    existing.qty = total_qty
                    existing.cost_basis = total_cost
                    existing.dca_count += 1
                else:
                    positions[sig["ticker"]] = D2SPosition(
                        ticker=sig["ticker"],
                        entry_price=price,
                        qty=qty,
                        entry_date=td,
                        cost_basis=net_amount,
                    )

                daily_entry_used += entry_fraction
                daily_buy_counts[sig["ticker"]] = (
                    daily_buy_counts.get(sig["ticker"], 0) + 1
                )

                buy_signals.append({
                    "ticker": sig["ticker"],
                    "size": sig["size"],
                    "score": sig.get("score", 0),
                    "reason": sig.get("reason", ""),
                    "daily_close": price,
                })

        if buy_signals:
            signals_by_date[td] = buy_signals

        # SPY streak 업데이트
        spy_pct = snap.changes.get("SPY", 0)
        if spy_pct > 0:
            spy_streak += 1
        else:
            spy_streak = 0

        # riskoff_streak 업데이트
        gld_pct = snap.changes.get("GLD", 0)
        if gld_pct > 0 and spy_pct < 0:
            riskoff_streak += 1
        else:
            riskoff_streak = 0

    total_sigs = sum(len(v) for v in signals_by_date.values())
    print(f"  BUY 신호: {total_sigs}건 ({len(signals_by_date)}일)")

    return signals_by_date, tech, poly


def _build_snapshot(
    trading_date: date,
    tech: dict[str, pd.DataFrame],
    poly: dict,
    spy_streak: int,
    riskoff_streak: int,
) -> DailySnapshot | None:
    """D2SBacktest._build_snapshot과 동일한 로직."""
    snap = DailySnapshot(
        trading_date=trading_date,
        weekday=trading_date.weekday(),
    )

    ts = pd.Timestamp(trading_date)
    found_any = False

    for ticker, ind_df in tech.items():
        if ts not in ind_df.index:
            continue
        row = ind_df.loc[ts]
        if pd.isna(row.get("close", np.nan)):
            continue

        found_any = True
        snap.closes[ticker] = float(row["close"])
        snap.opens[ticker] = float(row.get("open", row["close"]))
        snap.highs[ticker] = float(row.get("high", row["close"]))
        snap.lows[ticker] = float(row.get("low", row["close"]))
        snap.volumes[ticker] = float(row.get("volume", 0))
        snap.changes[ticker] = float(row.get("change_pct", 0))

        rsi_val = row.get("rsi")
        if not pd.isna(rsi_val):
            snap.rsi[ticker] = float(rsi_val)
        snap.macd_bullish[ticker] = bool(row.get("macd_bullish", False))
        bb_val = row.get("bb_pct_b")
        if not pd.isna(bb_val):
            snap.bb_pct_b[ticker] = float(bb_val)
        atr_val = row.get("atr")
        if not pd.isna(atr_val):
            snap.atr[ticker] = float(atr_val)
        aq_val = row.get("atr_quantile")
        if not pd.isna(aq_val):
            snap.atr_quantile[ticker] = float(aq_val)
        rv_val = row.get("rel_volume")
        if not pd.isna(rv_val):
            snap.rel_volume[ticker] = float(rv_val)

    if not found_any:
        return None

    poly_day = poly.get(trading_date, {})
    if "btc_up" in poly_day:
        snap.poly_btc_up = poly_day["btc_up"]

    snap.spy_streak = spy_streak
    snap.riskoff_streak = riskoff_streak
    return snap


# ============================================================
# Step 2: 분봉 데이터 로드 + 인덱싱
# ============================================================
def load_minute_data() -> dict[str, pd.DataFrame]:
    """분봉 데이터를 종목별-날짜별 접근 가능하게 로드한다.

    Returns
    -------
    dict[str, pd.DataFrame]
        {date_str: DataFrame}  date_str = "YYYY-MM-DD"
        각 DataFrame은 symbol, timestamp, open, high, low, close 컬럼.
    """
    path = _PROJECT_ROOT / "data" / "market" / "ohlcv" / "backtest_1min.parquet"
    print(f"\n[4/4] 분봉 데이터 로드: {path.name}")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows, {df['date'].nunique()} days, {df['symbol'].nunique()} symbols")

    # date별 그룹
    by_date: dict[str, pd.DataFrame] = {}
    for date_str, group in df.groupby("date"):
        by_date[str(date_str)] = group
    return by_date


# ============================================================
# Step 3: 시나리오 시뮬레이션
# ============================================================
def _get_trading_dates_after(
    entry_date: date,
    all_dates: list[str],
    n_days: int,
) -> list[str]:
    """entry_date 이후 n_days 거래일 목록을 반환한다."""
    entry_str = entry_date.isoformat()
    # entry_date 다음날부터
    future = [d for d in all_dates if d > entry_str]
    return future[:n_days]


def simulate_baseline_daily(
    signals_by_date: dict[date, list[dict]],
    minute_data: dict[str, pd.DataFrame],
    params: dict,
) -> list[ClosedTrade]:
    """baseline_daily: 09:30 시가 진입, hold_days 마지막날 15:55 청산."""
    trades: list[ClosedTrade] = []
    hold_days_max = params["optimal_hold_days_max"]
    tp_pct = params["take_profit_pct"]
    all_dates = sorted(minute_data.keys())
    total_capital = params["total_capital"]

    for entry_date, sigs in sorted(signals_by_date.items()):
        entry_str = entry_date.isoformat()
        if entry_str not in minute_data:
            continue

        day_df = minute_data[entry_str]

        for sig in sigs:
            ticker = sig["ticker"]
            sym_df = day_df[day_df["symbol"] == ticker]
            if sym_df.empty:
                continue

            # 09:30 시가 진입
            first_bar = sym_df.iloc[0]
            entry_price = float(first_bar["open"])
            if entry_price <= 0:
                continue

            # 매수 실행
            amount = total_capital * sig["size"]
            buy_fee = amount * BUY_FEE_PCT
            net_amount = amount - buy_fee
            qty = net_amount / entry_price

            # 청산: hold_days 동안 일봉 TP 체크 + 마지막날 15:55 강제 청산
            exit_dates = _get_trading_dates_after(entry_date, all_dates, hold_days_max)
            exit_price = None
            exit_date_actual = None
            exit_reason = ""

            for i, ed_str in enumerate(exit_dates):
                if ed_str not in minute_data:
                    continue
                ed_df = minute_data[ed_str]
                ed_sym = ed_df[ed_df["symbol"] == ticker]
                if ed_sym.empty:
                    continue

                # 일봉 high로 TP 체크 (당일 최고가)
                day_high = float(ed_sym["high"].max())
                if day_high >= entry_price * (1 + tp_pct / 100):
                    # TP 도달 → 당일 종가 청산 (일봉 시뮬레이션 방식)
                    last_bar = ed_sym.iloc[-1]
                    exit_price = float(last_bar["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"TP({tp_pct}%) day_high={day_high:.2f}"
                    break

                # 마지막 보유일: 15:55봉(= -5번째 봉) 종가 청산
                if i == len(exit_dates) - 1:
                    # 15:55 = 마지막에서 5번째 (15:55, 15:56, 15:57, 15:58, 15:59)
                    if len(ed_sym) >= 5:
                        bar_1555 = ed_sym.iloc[-5]
                    else:
                        bar_1555 = ed_sym.iloc[-1]
                    exit_price = float(bar_1555["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"hold_max({hold_days_max}d) @15:55"

            if exit_price is None or exit_date_actual is None:
                continue

            # PnL 계산
            proceeds = qty * exit_price
            sell_fee = proceeds * SELL_FEE_PCT
            net_proceeds = proceeds - sell_fee
            gross_pnl = qty * (exit_price - entry_price)
            net_pnl = net_proceeds - net_amount
            pnl_pct = (net_pnl / net_amount) * 100
            hold = (exit_date_actual - entry_date).days

            trades.append(ClosedTrade(
                ticker=ticker, entry_date=entry_date,
                exit_date=exit_date_actual,
                entry_price=entry_price, exit_price=exit_price,
                qty=qty, gross_pnl=gross_pnl, net_pnl=net_pnl,
                pnl_pct=pnl_pct, hold_days=hold, reason=exit_reason,
            ))

    return trades


def simulate_entry_opt(
    signals_by_date: dict[date, list[dict]],
    minute_data: dict[str, pd.DataFrame],
    params: dict,
) -> list[ClosedTrade]:
    """entry_opt: 09:30~10:30 중 최저점 봉에서 진입."""
    trades: list[ClosedTrade] = []
    hold_days_max = params["optimal_hold_days_max"]
    tp_pct = params["take_profit_pct"]
    all_dates = sorted(minute_data.keys())
    total_capital = params["total_capital"]

    for entry_date, sigs in sorted(signals_by_date.items()):
        entry_str = entry_date.isoformat()
        if entry_str not in minute_data:
            continue

        day_df = minute_data[entry_str]

        for sig in sigs:
            ticker = sig["ticker"]
            sym_df = day_df[day_df["symbol"] == ticker].copy()
            if sym_df.empty:
                continue

            # 09:30~10:30 (60분, 첫 61개 봉 중 인덱스 0~60)
            first_hour = sym_df.head(61)
            if first_hour.empty:
                continue

            # 최저점 봉
            min_idx = first_hour["low"].idxmin()
            best_bar = first_hour.loc[min_idx]
            entry_price = float(best_bar["low"])
            if entry_price <= 0:
                continue

            # 매수 실행
            amount = total_capital * sig["size"]
            buy_fee = amount * BUY_FEE_PCT
            net_amount = amount - buy_fee
            qty = net_amount / entry_price

            # 청산 (baseline과 동일)
            exit_dates = _get_trading_dates_after(entry_date, all_dates, hold_days_max)
            exit_price = None
            exit_date_actual = None
            exit_reason = ""

            for i, ed_str in enumerate(exit_dates):
                if ed_str not in minute_data:
                    continue
                ed_df = minute_data[ed_str]
                ed_sym = ed_df[ed_df["symbol"] == ticker]
                if ed_sym.empty:
                    continue

                day_high = float(ed_sym["high"].max())
                if day_high >= entry_price * (1 + tp_pct / 100):
                    last_bar = ed_sym.iloc[-1]
                    exit_price = float(last_bar["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"TP({tp_pct}%) day_high={day_high:.2f}"
                    break

                if i == len(exit_dates) - 1:
                    if len(ed_sym) >= 5:
                        bar_1555 = ed_sym.iloc[-5]
                    else:
                        bar_1555 = ed_sym.iloc[-1]
                    exit_price = float(bar_1555["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"hold_max({hold_days_max}d) @15:55"

            if exit_price is None or exit_date_actual is None:
                continue

            proceeds = qty * exit_price
            sell_fee = proceeds * SELL_FEE_PCT
            net_proceeds = proceeds - sell_fee
            gross_pnl = qty * (exit_price - entry_price)
            net_pnl = net_proceeds - net_amount
            pnl_pct = (net_pnl / net_amount) * 100
            hold = (exit_date_actual - entry_date).days

            trades.append(ClosedTrade(
                ticker=ticker, entry_date=entry_date,
                exit_date=exit_date_actual,
                entry_price=entry_price, exit_price=exit_price,
                qty=qty, gross_pnl=gross_pnl, net_pnl=net_pnl,
                pnl_pct=pnl_pct, hold_days=hold, reason=exit_reason,
            ))

    return trades


def simulate_tp_realtime(
    signals_by_date: dict[date, list[dict]],
    minute_data: dict[str, pd.DataFrame],
    params: dict,
) -> list[ClosedTrade]:
    """tp_realtime: entry_opt 진입 + 분봉 high > TP 가격 시 즉시 청산."""
    trades: list[ClosedTrade] = []
    hold_days_max = params["optimal_hold_days_max"]
    tp_pct = params["take_profit_pct"]
    all_dates = sorted(minute_data.keys())
    total_capital = params["total_capital"]

    for entry_date, sigs in sorted(signals_by_date.items()):
        entry_str = entry_date.isoformat()
        if entry_str not in minute_data:
            continue

        day_df = minute_data[entry_str]

        for sig in sigs:
            ticker = sig["ticker"]
            sym_df = day_df[day_df["symbol"] == ticker].copy()
            if sym_df.empty:
                continue

            # entry_opt와 동일: 09:30~10:30 최저점 진입
            first_hour = sym_df.head(61)
            if first_hour.empty:
                continue

            min_idx = first_hour["low"].idxmin()
            best_bar = first_hour.loc[min_idx]
            entry_price = float(best_bar["low"])
            if entry_price <= 0:
                continue

            tp_price = entry_price * (1 + tp_pct / 100)

            # 매수 실행
            amount = total_capital * sig["size"]
            buy_fee = amount * BUY_FEE_PCT
            net_amount = amount - buy_fee
            qty = net_amount / entry_price

            # 진입일 나머지 봉에서도 TP 체크 (진입 봉 이후)
            entry_bar_pos = sym_df.index.get_loc(min_idx)
            if isinstance(entry_bar_pos, slice):
                entry_bar_pos = entry_bar_pos.start
            remaining_entry_day = sym_df.iloc[entry_bar_pos + 1:]

            exit_price = None
            exit_date_actual = None
            exit_reason = ""

            # 진입일 잔여 봉 TP 체크
            for _, bar in remaining_entry_day.iterrows():
                if float(bar["high"]) >= tp_price:
                    exit_price = tp_price  # TP 가격에 청산
                    exit_date_actual = entry_date
                    exit_reason = f"TP_RT({tp_pct}%) intraday @{bar['timestamp']}"
                    break

            # 진입일에 TP 안 됐으면 이후 거래일 분봉 탐색
            if exit_price is None:
                exit_dates = _get_trading_dates_after(entry_date, all_dates, hold_days_max)

                for i, ed_str in enumerate(exit_dates):
                    if ed_str not in minute_data:
                        continue
                    ed_df = minute_data[ed_str]
                    ed_sym = ed_df[ed_df["symbol"] == ticker]
                    if ed_sym.empty:
                        continue

                    # 분봉별 TP 체크
                    tp_hit = False
                    for _, bar in ed_sym.iterrows():
                        if float(bar["high"]) >= tp_price:
                            exit_price = tp_price
                            exit_date_actual = date.fromisoformat(ed_str)
                            exit_reason = f"TP_RT({tp_pct}%) @{bar['timestamp']}"
                            tp_hit = True
                            break

                    if tp_hit:
                        break

                    # 마지막 보유일: 15:55 강제 청산
                    if i == len(exit_dates) - 1:
                        if len(ed_sym) >= 5:
                            bar_1555 = ed_sym.iloc[-5]
                        else:
                            bar_1555 = ed_sym.iloc[-1]
                        exit_price = float(bar_1555["close"])
                        exit_date_actual = date.fromisoformat(ed_str)
                        exit_reason = f"hold_max({hold_days_max}d) @15:55"

            if exit_price is None or exit_date_actual is None:
                continue

            proceeds = qty * exit_price
            sell_fee = proceeds * SELL_FEE_PCT
            net_proceeds = proceeds - sell_fee
            gross_pnl = qty * (exit_price - entry_price)
            net_pnl = net_proceeds - net_amount
            pnl_pct = (net_pnl / net_amount) * 100
            hold = (exit_date_actual - entry_date).days

            trades.append(ClosedTrade(
                ticker=ticker, entry_date=entry_date,
                exit_date=exit_date_actual,
                entry_price=entry_price, exit_price=exit_price,
                qty=qty, gross_pnl=gross_pnl, net_pnl=net_pnl,
                pnl_pct=pnl_pct, hold_days=hold, reason=exit_reason,
            ))

    return trades


# ============================================================
# Step 4: 결과 분석
# ============================================================
def analyze_trades(trades: list[ClosedTrade], label: str) -> dict:
    """거래 목록을 분석하여 통계 딕셔너리를 반환한다."""
    if not trades:
        return {
            "label": label, "n_trades": 0, "win_rate": 0,
            "avg_pnl": 0, "total_return": 0, "avg_hold": 0,
        }

    wins = [t for t in trades if t.net_pnl > 0]
    pnl_pcts = [t.pnl_pct for t in trades]
    holds = [t.hold_days for t in trades]
    total_net_pnl = sum(t.net_pnl for t in trades)
    total_capital = D2S_ENGINE["total_capital"]

    return {
        "label": label,
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_pnl": round(float(np.mean(pnl_pcts)), 2),
        "median_pnl": round(float(np.median(pnl_pcts)), 2),
        "total_net_pnl": round(total_net_pnl, 2),
        "total_return": round(total_net_pnl / total_capital * 100, 2),
        "avg_hold": round(float(np.mean(holds)), 1),
    }


def print_comparison(results: list[dict]) -> None:
    """시나리오 비교표를 출력한다."""
    print("\n" + "=" * 90)
    print("  Phase 4B — 분봉 D2S Study 결과 비교")
    print("=" * 90)

    header = (
        f"  {'시나리오':<20s} | {'거래수':>6s} | {'승률':>7s} | {'평균PnL':>8s} | "
        f"{'중앙PnL':>8s} | {'총수익률':>8s} | {'평균보유':>6s}"
    )
    print(header)
    print("  " + "-" * 86)

    for r in results:
        row = (
            f"  {r['label']:<20s} | {r['n_trades']:>6d} | "
            f"{r['win_rate']:>6.1f}% | {r['avg_pnl']:>+7.2f}% | "
            f"{r.get('median_pnl', 0):>+7.2f}% | "
            f"{r['total_return']:>+7.2f}% | {r['avg_hold']:>5.1f}d"
        )
        print(row)

    # 실거래 목표 대비 격차
    print("  " + "-" * 86)
    target_row = (
        f"  {'실거래 목표':<20s} | {TARGET_TRADES:>6d} | "
        f"{TARGET_WIN_RATE:>6.1f}% | {TARGET_AVG_PNL:>+7.2f}% | "
        f"{'':>8s} | {'':>8s} | {'':>6s}"
    )
    print(target_row)

    print("\n  격차 분석:")
    for r in results:
        wr_gap = r["win_rate"] - TARGET_WIN_RATE
        pnl_gap = r["avg_pnl"] - TARGET_AVG_PNL
        trade_gap = r["n_trades"] - TARGET_TRADES
        print(
            f"    {r['label']:<20s}: 승률 {wr_gap:+.1f}%p  "
            f"평균PnL {pnl_gap:+.2f}%p  거래수 {trade_gap:+d}"
        )

    print("=" * 90)


def print_ticker_breakdown(trades: list[ClosedTrade], label: str) -> None:
    """종목별 성과를 출력한다."""
    if not trades:
        return

    print(f"\n  [{label}] 종목별 성과:")
    by_ticker: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)

    for ticker in sorted(by_ticker):
        tt = by_ticker[ticker]
        wins = sum(1 for t in tt if t.net_pnl > 0)
        wr = wins / len(tt) * 100
        avg = float(np.mean([t.pnl_pct for t in tt]))
        total = sum(t.net_pnl for t in tt)
        print(
            f"    {ticker:6s}: {len(tt):3d}건  "
            f"승률 {wr:5.1f}%  평균 {avg:+6.2f}%  PnL ${total:+,.0f}"
        )


def save_results(
    all_trades: dict[str, list[ClosedTrade]],
    results: list[dict],
) -> None:
    """결과를 CSV로 저장한다."""
    out_dir = _PROJECT_ROOT / "data" / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 시나리오별 거래 로그
    for label, trades in all_trades.items():
        records = []
        for t in trades:
            records.append({
                "ticker": t.ticker,
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "qty": round(t.qty, 4),
                "net_pnl": round(t.net_pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "hold_days": t.hold_days,
                "reason": t.reason,
            })
        df = pd.DataFrame(records)
        path = out_dir / f"d2s_1min_{label}.csv"
        df.to_csv(path, index=False)
        print(f"  저장: {path}")

    # 요약
    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "d2s_1min_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  요약: {summary_path}")


# ============================================================
# 메인
# ============================================================
def main():
    print("Phase 4B — 분봉 D2S Study")
    print("=" * 50)

    params = D2S_ENGINE.copy()

    # Step 1: 일봉 D2S 신호 추출
    signals_by_date, tech, poly = extract_daily_signals(params)

    # Step 2: 분봉 데이터 로드
    minute_data = load_minute_data()

    # Step 3: 시나리오별 시뮬레이션
    print("\n--- 시나리오 시뮬레이션 ---")

    print("\n  [1/3] baseline_daily: 09:30 시가 진입, hold_days 마지막날 15:55 청산")
    trades_baseline = simulate_baseline_daily(signals_by_date, minute_data, params)
    print(f"    -> {len(trades_baseline)} trades")

    print("\n  [2/3] entry_opt: 09:30~10:30 최저점 진입")
    trades_entry_opt = simulate_entry_opt(signals_by_date, minute_data, params)
    print(f"    -> {len(trades_entry_opt)} trades")

    print("\n  [3/3] tp_realtime: entry_opt + 분봉 TP 즉시 청산")
    trades_tp_rt = simulate_tp_realtime(signals_by_date, minute_data, params)
    print(f"    -> {len(trades_tp_rt)} trades")

    # Step 4: 결과 비교
    r_baseline = analyze_trades(trades_baseline, "baseline_daily")
    r_entry_opt = analyze_trades(trades_entry_opt, "entry_opt")
    r_tp_rt = analyze_trades(trades_tp_rt, "tp_realtime")
    results = [r_baseline, r_entry_opt, r_tp_rt]

    print_comparison(results)

    # 종목별 상세
    for label, trades in [
        ("baseline_daily", trades_baseline),
        ("entry_opt", trades_entry_opt),
        ("tp_realtime", trades_tp_rt),
    ]:
        print_ticker_breakdown(trades, label)

    # 결과 저장
    print("\n--- 결과 저장 ---")
    save_results(
        {
            "baseline_daily": trades_baseline,
            "entry_opt": trades_entry_opt,
            "tp_realtime": trades_tp_rt,
        },
        results,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
