#!/usr/bin/env python3
"""
Study D2S 1min DCA — 분봉 DCA 패턴 재현 Study
=================================================
history/ 실거래 분석 결과:
  - D2S 4종목 매수: 406건 / 102일 → 일평균 3.98건/일
  - 동일날 동일종목 2회 이상 매수: 전체의 64%
  - DCA 단가 범위: 평균 2.11%, 중앙값 1.13%

entry_opt (09:30~10:30 저점 첫 진입) + 분봉 DCA 3 임계값 비교:
  entry_opt  : DCA 없음 (비교 기준선, Phase 4B 결과)
  dca_1pct   : 직전 진입가 -1% 하락 시 DCA (최대 3회)
  dca_2pct   : 직전 진입가 -2% 하락 시 DCA (최대 3회) ← 실거래 중앙값
  dca_3pct   : 직전 진입가 -3% 하락 시 DCA (최대 3회)

청산 기준: 전체 포지션 평균단가 기준 TP hit 또는 첫 진입일 기준 hold_days 경과

기간: WARM 2025-03-03 ~ 2026-01-30 (분봉 데이터 범위)
실거래 목표: D2S 4종목 매수 406건, 승률 65.5%, 평균 PnL +7.39%

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_d2s_1min_dca.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
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
BUY_FEE_PCT  = 0.35  / 100
SELL_FEE_PCT = 0.353 / 100

START_DATE = date(2025, 3, 3)
END_DATE   = date(2026, 1, 30)   # 분봉 데이터 끝

# 실거래 목표
TARGET_WIN_RATE = 65.5
TARGET_AVG_PNL  = 7.39
TARGET_TRADES   = 406   # D2S 4종목 매수건 기준 (전체 722건 아님)

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "analysis"


# ============================================================
# 데이터 컨테이너
# ============================================================
@dataclass
class DCALayer:
    """단일 DCA 레이어."""
    price: float
    qty: float
    cost: float       # 순 투자금 (수수료 제외)
    entry_date: date


@dataclass
class DCAPosition:
    """분봉 DCA 포지션 (다중 레이어)."""
    ticker: str
    first_entry_date: date
    layers: list[DCALayer] = field(default_factory=list)

    @property
    def dca_count(self) -> int:
        return len(self.layers)

    @property
    def total_qty(self) -> float:
        return sum(l.qty for l in self.layers)

    @property
    def total_cost(self) -> float:
        return sum(l.cost for l in self.layers)

    @property
    def avg_price(self) -> float:
        return self.total_cost / self.total_qty if self.total_qty > 0 else 0.0

    @property
    def last_price(self) -> float:
        return self.layers[-1].price if self.layers else 0.0


@dataclass
class ClosedTrade:
    """청산 완료 거래."""
    ticker: str
    entry_date: date
    exit_date: date
    avg_entry_price: float
    exit_price: float
    total_qty: float
    total_cost: float
    net_pnl: float
    pnl_pct: float
    hold_days: int
    dca_layers: int
    reason: str


# ============================================================
# 일봉 D2S 신호 추출 (study_d2s_1min.py 재사용)
# ============================================================
def extract_daily_signals(params: dict) -> tuple[dict, dict, dict]:
    data_dir = _PROJECT_ROOT / "data"
    market_path = data_dir / "market" / "daily" / "market_daily.parquet"
    poly_dir    = data_dir / "polymarket"

    print("[1/4] 일봉 데이터 로드")
    df   = pd.read_parquet(market_path)
    poly = backtest_common.load_polymarket_daily(poly_dir)

    if not poly:
        print("  Polymarket 없음 → BITU 폴백 생성")
        try:
            bitu_close = df[("BITU", "Close")]
            bitu_pct   = bitu_close.pct_change() * 100
            for ts, pct in bitu_pct.items():
                if pd.isna(pct):
                    continue
                d = ts.date() if hasattr(ts, "date") else ts
                poly[d] = {"btc_up": round(float(np.clip(0.50 + pct / 25.0, 0.30, 0.70)), 3)}
        except KeyError:
            pass

    print("[2/4] 기술적 지표 계산")
    preprocessor = TechnicalPreprocessor(params)
    tech = preprocessor.compute(df)
    print(f"  {len(tech)} tickers processed")

    engine = D2SEngine(params)
    all_dates = sorted(df.index)
    trading_dates = [
        d.date() if hasattr(d, "date") else d
        for d in all_dates
        if START_DATE <= (d.date() if hasattr(d, "date") else d) <= END_DATE
    ]

    print(f"\n[3/4] 일봉 D2S 신호 생성 ({trading_dates[0]} ~ {trading_dates[-1]}, {len(trading_dates)}일)")

    spy_streak = riskoff_streak = 0
    positions: dict[str, D2SPosition] = {}
    signals_by_date: dict[date, list[dict]] = {}

    for td in trading_dates:
        snap = _build_snapshot(td, tech, poly, spy_streak, riskoff_streak)
        if snap is None:
            continue

        daily_buy_counts: dict[str, int] = {}
        sigs = engine.generate_daily_signals(snap, positions, daily_buy_counts)

        # SELL 실행 (포지션 상태 유지)
        for sig in sigs:
            if sig["action"] == "SELL" and sig["ticker"] in positions:
                del positions[sig["ticker"]]

        # BUY 수집
        buy_signals = []
        daily_entry_used = 0.0
        daily_entry_cap  = params.get("daily_new_entry_cap", 1.0)
        cash = params["total_capital"]

        for sig in sigs:
            if sig["action"] != "BUY":
                continue
            entry_fraction = sig["size"]
            if daily_entry_used + entry_fraction > daily_entry_cap:
                continue

            price = snap.closes.get(sig["ticker"], 0)
            if price <= 0:
                continue

            amount     = cash * sig["size"]
            fee        = amount * BUY_FEE_PCT
            net_amount = amount - fee
            qty        = net_amount / price

            existing = positions.get(sig["ticker"])
            if existing:
                tc = existing.cost_basis + net_amount
                tq = existing.qty + qty
                existing.entry_price = tc / tq
                existing.qty         = tq
                existing.cost_basis  = tc
                existing.dca_count  += 1
            else:
                positions[sig["ticker"]] = D2SPosition(
                    ticker=sig["ticker"], entry_price=price,
                    qty=qty, entry_date=td, cost_basis=net_amount,
                )

            daily_entry_used += entry_fraction
            daily_buy_counts[sig["ticker"]] = daily_buy_counts.get(sig["ticker"], 0) + 1
            buy_signals.append({
                "ticker": sig["ticker"], "size": sig["size"],
                "score": sig.get("score", 0), "reason": sig.get("reason", ""),
                "daily_close": price,
            })

        if buy_signals:
            signals_by_date[td] = buy_signals

        spy_pct = snap.changes.get("SPY", 0)
        spy_streak = spy_streak + 1 if spy_pct > 0 else 0
        gld_pct = snap.changes.get("GLD", 0)
        riskoff_streak = riskoff_streak + 1 if (gld_pct > 0 and spy_pct < 0) else 0

    total_sigs = sum(len(v) for v in signals_by_date.values())
    print(f"  BUY 신호: {total_sigs}건 ({len(signals_by_date)}일)")
    return signals_by_date, tech, poly


def _build_snapshot(td, tech, poly, spy_streak, riskoff_streak):
    snap = DailySnapshot(trading_date=td, weekday=td.weekday())
    ts = pd.Timestamp(td)
    found_any = False
    for ticker, ind_df in tech.items():
        if ts not in ind_df.index:
            continue
        row = ind_df.loc[ts]
        if pd.isna(row.get("close", np.nan)):
            continue
        found_any = True
        snap.closes[ticker]      = float(row["close"])
        snap.opens[ticker]       = float(row.get("open", row["close"]))
        snap.highs[ticker]       = float(row.get("high", row["close"]))
        snap.lows[ticker]        = float(row.get("low", row["close"]))
        snap.volumes[ticker]     = float(row.get("volume", 0))
        snap.changes[ticker]     = float(row.get("change_pct", 0))
        if not pd.isna(row.get("rsi")):       snap.rsi[ticker]        = float(row["rsi"])
        snap.macd_bullish[ticker] = bool(row.get("macd_bullish", False))
        if not pd.isna(row.get("bb_pct_b")):  snap.bb_pct_b[ticker]   = float(row["bb_pct_b"])
        if not pd.isna(row.get("atr")):       snap.atr[ticker]        = float(row["atr"])
        if not pd.isna(row.get("atr_quantile")): snap.atr_quantile[ticker] = float(row["atr_quantile"])
        if not pd.isna(row.get("rel_volume")): snap.rel_volume[ticker] = float(row["rel_volume"])
    if not found_any:
        return None
    poly_day = poly.get(td, {})
    if "btc_up" in poly_day:
        snap.poly_btc_up = poly_day["btc_up"]
    snap.spy_streak     = spy_streak
    snap.riskoff_streak = riskoff_streak
    return snap


# ============================================================
# 분봉 데이터 로드
# ============================================================
def load_minute_data() -> dict[str, pd.DataFrame]:
    path = _PROJECT_ROOT / "data" / "market" / "ohlcv" / "backtest_1min.parquet"
    print(f"\n[4/4] 분봉 데이터 로드: {path.name}")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} rows, {df['date'].nunique()} days, {df['symbol'].nunique()} symbols")
    by_date: dict[str, pd.DataFrame] = {}
    for date_str, group in df.groupby("date"):
        by_date[str(date_str)] = group
    return by_date


def _future_dates(entry_date: date, all_dates: list[str], n: int) -> list[str]:
    s = entry_date.isoformat()
    return [d for d in all_dates if d > s][:n]


# ============================================================
# 시나리오: entry_opt (DCA 없음, 비교 기준선)
# ============================================================
def simulate_entry_opt(
    signals_by_date: dict[date, list[dict]],
    minute_data: dict[str, pd.DataFrame],
    params: dict,
) -> list[ClosedTrade]:
    """09:30~10:30 최저점 단일 진입, hold_days 강제청산."""
    trades: list[ClosedTrade] = []
    hold_max   = params["optimal_hold_days_max"]
    tp_pct     = params["take_profit_pct"]
    all_dates  = sorted(minute_data.keys())
    total_cap  = params["total_capital"]

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

            first_hour = sym_df.head(61)
            if first_hour.empty:
                continue
            entry_price = float(first_hour.loc[first_hour["low"].idxmin(), "low"])
            if entry_price <= 0:
                continue

            amount     = total_cap * sig["size"]
            fee        = amount * BUY_FEE_PCT
            net_amount = amount - fee
            qty        = net_amount / entry_price
            tp_price   = entry_price * (1 + tp_pct / 100)

            exit_price = exit_date_actual = None
            exit_reason = ""
            for i, ed_str in enumerate(_future_dates(entry_date, all_dates, hold_max)):
                if ed_str not in minute_data:
                    continue
                ed_sym = minute_data[ed_str][minute_data[ed_str]["symbol"] == ticker]
                if ed_sym.empty:
                    continue
                if float(ed_sym["high"].max()) >= tp_price:
                    exit_price = float(ed_sym.iloc[-1]["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"TP({tp_pct}%)"
                    break
                if i == hold_max - 1:
                    bar = ed_sym.iloc[-5] if len(ed_sym) >= 5 else ed_sym.iloc[-1]
                    exit_price = float(bar["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"hold_max({hold_max}d)"

            if exit_price is None or exit_date_actual is None:
                continue

            proceeds   = qty * exit_price
            sell_fee   = proceeds * SELL_FEE_PCT
            net_pnl    = (proceeds - sell_fee) - net_amount
            pnl_pct    = net_pnl / net_amount * 100

            trades.append(ClosedTrade(
                ticker=ticker, entry_date=entry_date, exit_date=exit_date_actual,
                avg_entry_price=entry_price, exit_price=exit_price,
                total_qty=qty, total_cost=net_amount,
                net_pnl=net_pnl, pnl_pct=pnl_pct,
                hold_days=(exit_date_actual - entry_date).days,
                dca_layers=1, reason=exit_reason,
            ))
    return trades


# ============================================================
# 시나리오: entry_opt + 분봉 DCA
# ============================================================
def simulate_dca(
    signals_by_date: dict[date, list[dict]],
    minute_data: dict[str, pd.DataFrame],
    params: dict,
    dca_threshold_pct: float,   # 직전 진입가 대비 하락 임계값 (%)
    max_dca: int = 3,           # 최대 추가 매수 횟수
) -> list[ClosedTrade]:
    """entry_opt 첫 진입 후, 당일 분봉에서 -threshold% 하락 시 DCA 추가매수.

    - DCA 매수금: 첫 진입과 동일한 sig["size"] * total_capital
    - 청산: 전체 포지션 평균단가 기준 TP hit 또는 첫 진입일 기준 hold_days 강제청산
    """
    trades: list[ClosedTrade] = []
    hold_max  = params["optimal_hold_days_max"]
    tp_pct    = params["take_profit_pct"]
    all_dates = sorted(minute_data.keys())
    total_cap = params["total_capital"]

    for entry_date, sigs in sorted(signals_by_date.items()):
        entry_str = entry_date.isoformat()
        if entry_str not in minute_data:
            continue
        day_df = minute_data[entry_str]

        for sig in sigs:
            ticker = sig["ticker"]
            sym_df = day_df[day_df["symbol"] == ticker].reset_index(drop=True)
            if sym_df.empty:
                continue

            # ── 첫 진입: 09:30~10:30 최저점 ──────────────────────
            first_hour = sym_df.head(61)
            if first_hour.empty:
                continue
            first_bar_idx = int(first_hour["low"].idxmin())
            entry_price   = float(first_hour.loc[first_bar_idx, "low"])
            if entry_price <= 0:
                continue

            unit_amount = total_cap * sig["size"]   # 레이어당 매수금

            def make_layer(price: float, ed: date) -> DCALayer:
                fee = unit_amount * BUY_FEE_PCT
                net = unit_amount - fee
                return DCALayer(price=price, qty=net / price, cost=net, entry_date=ed)

            pos = DCAPosition(ticker=ticker, first_entry_date=entry_date)
            pos.layers.append(make_layer(entry_price, entry_date))

            # ── 당일 분봉 DCA: 첫 진입 봉 이후 ──────────────────
            remaining = sym_df.iloc[first_bar_idx + 1:]
            for _, bar in remaining.iterrows():
                if pos.dca_count > max_dca:
                    break
                bar_low = float(bar["low"])
                trigger = pos.last_price * (1 - dca_threshold_pct / 100)
                if bar_low <= trigger:
                    pos.layers.append(make_layer(bar_low, entry_date))

            # ── 청산: avg_price 기준 TP 또는 hold_days 강제청산 ──
            avg_entry = pos.avg_price
            tp_price  = avg_entry * (1 + tp_pct / 100)
            total_qty  = pos.total_qty
            total_cost = pos.total_cost

            exit_price = exit_date_actual = None
            exit_reason = ""

            for i, ed_str in enumerate(_future_dates(entry_date, all_dates, hold_max)):
                if ed_str not in minute_data:
                    continue
                ed_sym = minute_data[ed_str][minute_data[ed_str]["symbol"] == ticker]
                if ed_sym.empty:
                    continue

                if float(ed_sym["high"].max()) >= tp_price:
                    exit_price = float(ed_sym.iloc[-1]["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"TP({tp_pct}%) layers={pos.dca_count}"
                    break

                if i == hold_max - 1:
                    bar = ed_sym.iloc[-5] if len(ed_sym) >= 5 else ed_sym.iloc[-1]
                    exit_price = float(bar["close"])
                    exit_date_actual = date.fromisoformat(ed_str)
                    exit_reason = f"hold_max({hold_max}d) layers={pos.dca_count}"

            if exit_price is None or exit_date_actual is None:
                continue

            proceeds = total_qty * exit_price
            sell_fee = proceeds * SELL_FEE_PCT
            net_pnl  = (proceeds - sell_fee) - total_cost
            pnl_pct  = net_pnl / total_cost * 100

            trades.append(ClosedTrade(
                ticker=ticker, entry_date=entry_date, exit_date=exit_date_actual,
                avg_entry_price=avg_entry, exit_price=exit_price,
                total_qty=total_qty, total_cost=total_cost,
                net_pnl=net_pnl, pnl_pct=pnl_pct,
                hold_days=(exit_date_actual - entry_date).days,
                dca_layers=pos.dca_count, reason=exit_reason,
            ))
    return trades


# ============================================================
# 결과 분석 및 출력
# ============================================================
def analyze_trades(trades: list[ClosedTrade], label: str) -> dict:
    if not trades:
        return {"label": label, "n_trades": 0, "win_rate": 0,
                "avg_pnl": 0, "median_pnl": 0, "total_return": 0,
                "avg_hold": 0, "avg_layers": 1}
    wins     = [t for t in trades if t.net_pnl > 0]
    pnl_pcts = [t.pnl_pct for t in trades]
    total_cap = D2S_ENGINE["total_capital"]
    return {
        "label":        label,
        "n_trades":     len(trades),
        "win_rate":     round(len(wins) / len(trades) * 100, 1),
        "avg_pnl":      round(float(np.mean(pnl_pcts)), 2),
        "median_pnl":   round(float(np.median(pnl_pcts)), 2),
        "total_net_pnl":round(sum(t.net_pnl for t in trades), 2),
        "total_return": round(sum(t.net_pnl for t in trades) / total_cap * 100, 2),
        "avg_hold":     round(float(np.mean([t.hold_days for t in trades])), 1),
        "avg_layers":   round(float(np.mean([t.dca_layers for t in trades])), 2),
    }


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  Study D2S 1min DCA — 시나리오 비교")
    print("=" * 100)
    hdr = (f"  {'시나리오':<18} | {'거래수':>5} | {'승률':>6} | {'평균PnL':>7} | "
           f"{'중앙PnL':>7} | {'총수익률':>7} | {'평균보유':>5} | {'평균DCA':>5}")
    print(hdr)
    print("  " + "-" * 95)
    for r in results:
        print(f"  {r['label']:<18} | {r['n_trades']:>5d} | "
              f"{r['win_rate']:>5.1f}% | {r['avg_pnl']:>+6.2f}% | "
              f"{r.get('median_pnl',0):>+6.2f}% | {r['total_return']:>+6.2f}% | "
              f"{r['avg_hold']:>4.1f}d | {r.get('avg_layers',1):>4.2f}x")
    print("  " + "-" * 95)
    print(f"  {'실거래 목표(D2S4종목)':<18} | {TARGET_TRADES:>5d} | "
          f"{TARGET_WIN_RATE:>5.1f}% | {TARGET_AVG_PNL:>+6.2f}% | "
          f"{'':>7} | {'':>7} | {'':>5} | {'':>5}")
    print("=" * 100)
    print("\n  격차 (vs 실거래 D2S 4종목 기준):")
    for r in results:
        print(f"    {r['label']:<18}: 거래수 {r['n_trades']-TARGET_TRADES:+d}  "
              f"승률 {r['win_rate']-TARGET_WIN_RATE:+.1f}%p  "
              f"평균PnL {r['avg_pnl']-TARGET_AVG_PNL:+.2f}%p")


def print_dca_stats(trades: list[ClosedTrade], label: str) -> None:
    if not trades:
        return
    layer_dist = {}
    for t in trades:
        layer_dist[t.dca_layers] = layer_dist.get(t.dca_layers, 0) + 1
    print(f"\n  [{label}] DCA 레이어 분포:")
    for k in sorted(layer_dist):
        pct = layer_dist[k] / len(trades) * 100
        bar = "█" * int(pct / 3)
        print(f"    {k}레이어: {layer_dist[k]:3d}건 ({pct:4.1f}%) {bar}")


def print_ticker_breakdown(trades: list[ClosedTrade], label: str) -> None:
    if not trades:
        return
    print(f"\n  [{label}] 종목별 성과:")
    by_ticker: dict[str, list] = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)
    for ticker in sorted(by_ticker):
        tt = by_ticker[ticker]
        wins = sum(1 for t in tt if t.net_pnl > 0)
        avg  = float(np.mean([t.pnl_pct for t in tt]))
        total = sum(t.net_pnl for t in tt)
        print(f"    {ticker:6s}: {len(tt):3d}건  승률 {wins/len(tt)*100:5.1f}%  "
              f"평균 {avg:+6.2f}%  PnL ${total:+,.0f}")


def save_results(all_trades: dict[str, list[ClosedTrade]], results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for label, trades in all_trades.items():
        records = [
            {"ticker": t.ticker, "entry_date": str(t.entry_date),
             "exit_date": str(t.exit_date),
             "avg_entry_price": round(t.avg_entry_price, 4),
             "exit_price": round(t.exit_price, 4),
             "total_qty": round(t.total_qty, 4),
             "net_pnl": round(t.net_pnl, 2),
             "pnl_pct": round(t.pnl_pct, 2),
             "hold_days": t.hold_days,
             "dca_layers": t.dca_layers,
             "reason": t.reason}
            for t in trades
        ]
        pd.DataFrame(records).to_csv(RESULTS_DIR / f"d2s_1min_dca_{label}.csv", index=False)
    with open(RESULTS_DIR / "d2s_1min_dca_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  저장 완료: {RESULTS_DIR}")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 65)
    print("  Study D2S 1min DCA — 분봉 DCA 패턴 재현")
    print("=" * 65)
    print(f"  실거래 D2S 4종목 매수 목표: {TARGET_TRADES}건")
    print(f"  DCA 패턴: 일평균 3.98건/일, 중앙값 범위 1.13%")

    params = D2S_ENGINE.copy()

    signals_by_date, _, _ = extract_daily_signals(params)
    minute_data = load_minute_data()

    print("\n--- 시나리오 시뮬레이션 ---")

    print("\n  [1/4] entry_opt: DCA 없음 (비교 기준선)")
    t_entry = simulate_entry_opt(signals_by_date, minute_data, params)
    print(f"    -> {len(t_entry)}건")

    print("\n  [2/4] dca_1pct: -1% 하락 시 DCA (최대 3회)")
    t_dca1 = simulate_dca(signals_by_date, minute_data, params, dca_threshold_pct=1.0, max_dca=3)
    print(f"    -> {len(t_dca1)}건")

    print("\n  [3/4] dca_2pct: -2% 하락 시 DCA (최대 3회, 실거래 중앙값)")
    t_dca2 = simulate_dca(signals_by_date, minute_data, params, dca_threshold_pct=2.0, max_dca=3)
    print(f"    -> {len(t_dca2)}건")

    print("\n  [4/4] dca_3pct: -3% 하락 시 DCA (최대 3회)")
    t_dca3 = simulate_dca(signals_by_date, minute_data, params, dca_threshold_pct=3.0, max_dca=3)
    print(f"    -> {len(t_dca3)}건")

    results = [
        analyze_trades(t_entry, "entry_opt"),
        analyze_trades(t_dca1,  "dca_1pct"),
        analyze_trades(t_dca2,  "dca_2pct"),
        analyze_trades(t_dca3,  "dca_3pct"),
    ]

    print_comparison(results)

    for label, trades in [
        ("entry_opt", t_entry),
        ("dca_1pct",  t_dca1),
        ("dca_2pct",  t_dca2),
        ("dca_3pct",  t_dca3),
    ]:
        print_dca_stats(trades, label)
        print_ticker_breakdown(trades, label)

    print("\n--- 결과 저장 ---")
    save_results({
        "entry_opt": t_entry,
        "dca_1pct":  t_dca1,
        "dca_2pct":  t_dca2,
        "dca_3pct":  t_dca3,
    }, results)


if __name__ == "__main__":
    main()
