#!/usr/bin/env python3
"""
Study D2S Entry Cap — daily_new_entry_cap 완화 실험
=====================================================
거래수 격차 원인: daily_new_entry_cap=0.30 → 하루 신규 진입 총합 ≤ 자본 30%
  → 실거래 대비 하루 진입 종목 수 억제 (약 2종목 상한)

dca_1pct 고정 + cap 4레벨 비교:
  cap_30pct: 0.30  (현재값, 비교 기준선)
  cap_50pct: 0.50
  cap_80pct: 0.80
  cap_off:   1.00  (사실상 제한 없음)

기간: WARM 2025-03-03 ~ 2026-01-30 (분봉 데이터 범위)
실거래 목표: D2S 4종목 매수 406건, 승률 65.5%, 평균 PnL +7.39%

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_d2s_entry_cap.py
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

TARGET_WIN_RATE = 65.5
TARGET_AVG_PNL  = 7.39
TARGET_TRADES   = 406   # D2S 4종목 매수 기준

DCA_THRESHOLD_PCT = 1.0  # dca_1pct 고정
MAX_DCA           = 3

SCENARIOS = [
    {"label": "cap_30pct", "cap": 0.30, "note": "현재값 (비교 기준선)"},
    {"label": "cap_50pct", "cap": 0.50, "note": "완화 1단계"},
    {"label": "cap_80pct", "cap": 0.80, "note": "완화 2단계"},
    {"label": "cap_off",   "cap": 1.00, "note": "사실상 제한 없음"},
]

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "analysis"


# ============================================================
# 데이터 컨테이너
# ============================================================
@dataclass
class DCALayer:
    price: float
    qty: float
    cost: float
    entry_date: date


@dataclass
class DCAPosition:
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
# 일봉 D2S 신호 추출 (entry_cap 파라미터화)
# ============================================================
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
        snap.closes[ticker]       = float(row["close"])
        snap.opens[ticker]        = float(row.get("open", row["close"]))
        snap.highs[ticker]        = float(row.get("high", row["close"]))
        snap.lows[ticker]         = float(row.get("low", row["close"]))
        snap.volumes[ticker]      = float(row.get("volume", 0))
        snap.changes[ticker]      = float(row.get("change_pct", 0))
        if not pd.isna(row.get("rsi")):          snap.rsi[ticker]        = float(row["rsi"])
        snap.macd_bullish[ticker] = bool(row.get("macd_bullish", False))
        if not pd.isna(row.get("bb_pct_b")):     snap.bb_pct_b[ticker]   = float(row["bb_pct_b"])
        if not pd.isna(row.get("atr")):          snap.atr[ticker]        = float(row["atr"])
        if not pd.isna(row.get("atr_quantile")): snap.atr_quantile[ticker] = float(row["atr_quantile"])
        if not pd.isna(row.get("rel_volume")):   snap.rel_volume[ticker] = float(row["rel_volume"])
    if not found_any:
        return None
    poly_day = poly.get(td, {})
    if "btc_up" in poly_day:
        snap.poly_btc_up = poly_day["btc_up"]
    snap.spy_streak     = spy_streak
    snap.riskoff_streak = riskoff_streak
    return snap


def extract_daily_signals(params: dict, entry_cap: float) -> dict[date, list[dict]]:
    """일봉 D2S 신호 추출 — daily_new_entry_cap을 entry_cap으로 오버라이드."""
    data_dir    = _PROJECT_ROOT / "data"
    market_path = data_dir / "market" / "daily" / "market_daily.parquet"
    poly_dir    = data_dir / "polymarket"

    df   = pd.read_parquet(market_path)
    poly = backtest_common.load_polymarket_daily(poly_dir)

    if not poly:
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

    preprocessor = TechnicalPreprocessor(params)
    tech = preprocessor.compute(df)

    engine      = D2SEngine(params)
    all_dates   = sorted(df.index)
    trading_dates = [
        d.date() if hasattr(d, "date") else d
        for d in all_dates
        if START_DATE <= (d.date() if hasattr(d, "date") else d) <= END_DATE
    ]

    spy_streak = riskoff_streak = 0
    positions: dict[str, D2SPosition] = {}
    signals_by_date: dict[date, list[dict]] = {}
    cash = params["total_capital"]

    for td in trading_dates:
        snap = _build_snapshot(td, tech, poly, spy_streak, riskoff_streak)
        if snap is None:
            continue

        daily_buy_counts: dict[str, int] = {}
        sigs = engine.generate_daily_signals(snap, positions, daily_buy_counts)

        for sig in sigs:
            if sig["action"] == "SELL" and sig["ticker"] in positions:
                del positions[sig["ticker"]]

        buy_signals = []
        daily_entry_used = 0.0

        for sig in sigs:
            if sig["action"] != "BUY":
                continue
            entry_fraction = sig["size"]
            if daily_entry_used + entry_fraction > entry_cap:  # ← cap 오버라이드
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
                "score":  sig.get("score", 0), "reason": sig.get("reason", ""),
                "daily_close": price,
            })

        if buy_signals:
            signals_by_date[td] = buy_signals

        spy_pct = snap.changes.get("SPY", 0)
        spy_streak = spy_streak + 1 if spy_pct > 0 else 0
        gld_pct = snap.changes.get("GLD", 0)
        riskoff_streak = riskoff_streak + 1 if (gld_pct > 0 and spy_pct < 0) else 0

    total_sigs = sum(len(v) for v in signals_by_date.values())
    print(f"  cap={entry_cap:.2f}  BUY 신호: {total_sigs}건 ({len(signals_by_date)}일)")
    return signals_by_date


# ============================================================
# 분봉 데이터 로드
# ============================================================
def load_minute_data() -> dict[str, pd.DataFrame]:
    path = _PROJECT_ROOT / "data" / "market" / "ohlcv" / "backtest_1min.parquet"
    print(f"\n[분봉] 로드: {path.name}")
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
# 시나리오: entry_opt + dca_1pct
# ============================================================
def simulate_dca(
    signals_by_date: dict[date, list[dict]],
    minute_data: dict[str, pd.DataFrame],
    params: dict,
    dca_threshold_pct: float = DCA_THRESHOLD_PCT,
    max_dca: int = MAX_DCA,
) -> list[ClosedTrade]:
    """entry_opt 첫 진입(09:30~10:30 최저점) + 당일 분봉 DCA."""
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

            first_hour = sym_df.head(61)
            if first_hour.empty:
                continue
            first_bar_idx = int(first_hour["low"].idxmin())
            entry_price   = float(first_hour.loc[first_bar_idx, "low"])
            if entry_price <= 0:
                continue

            unit_amount = total_cap * sig["size"]

            def make_layer(price: float, ed: date) -> DCALayer:
                fee = unit_amount * BUY_FEE_PCT
                net = unit_amount - fee
                return DCALayer(price=price, qty=net / price, cost=net, entry_date=ed)

            pos = DCAPosition(ticker=ticker, first_entry_date=entry_date)
            pos.layers.append(make_layer(entry_price, entry_date))

            remaining = sym_df.iloc[first_bar_idx + 1:]
            for _, bar in remaining.iterrows():
                if pos.dca_count > max_dca:
                    break
                bar_low = float(bar["low"])
                trigger = pos.last_price * (1 - dca_threshold_pct / 100)
                if bar_low <= trigger:
                    pos.layers.append(make_layer(bar_low, entry_date))

            avg_entry  = pos.avg_price
            tp_price   = avg_entry * (1 + tp_pct / 100)
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
# 결과 분석
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
        "label":         label,
        "n_trades":      len(trades),
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "avg_pnl":       round(float(np.mean(pnl_pcts)), 2),
        "median_pnl":    round(float(np.median(pnl_pcts)), 2),
        "total_net_pnl": round(sum(t.net_pnl for t in trades), 2),
        "total_return":  round(sum(t.net_pnl for t in trades) / total_cap * 100, 2),
        "avg_hold":      round(float(np.mean([t.hold_days for t in trades])), 1),
        "avg_layers":    round(float(np.mean([t.dca_layers for t in trades])), 2),
    }


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 105)
    print("  Study D2S Entry Cap — daily_new_entry_cap 완화 실험 (dca_1pct 고정)")
    print("=" * 105)
    hdr = (f"  {'시나리오':<14} | {'거래수':>5} | {'승률':>6} | {'평균PnL':>7} | "
           f"{'중앙PnL':>7} | {'총수익률':>8} | {'평균보유':>5} | {'평균DCA':>5}")
    print(hdr)
    print("  " + "-" * 100)
    for r in results:
        print(f"  {r['label']:<14} | {r['n_trades']:>5d} | "
              f"{r['win_rate']:>5.1f}% | {r['avg_pnl']:>+6.2f}% | "
              f"{r.get('median_pnl', 0):>+6.2f}% | {r['total_return']:>+7.2f}% | "
              f"{r['avg_hold']:>4.1f}d | {r.get('avg_layers', 1):>4.2f}x")
    print("  " + "-" * 100)
    print(f"  {'실거래 목표':<14} | {TARGET_TRADES:>5d} | "
          f"{TARGET_WIN_RATE:>5.1f}% | {TARGET_AVG_PNL:>+6.2f}% | "
          f"{'':>7} | {'':>8} | {'':>5} | {'':>5}")
    print("=" * 105)

    print("\n  격차 (vs 실거래 D2S 4종목 기준):")
    for r in results:
        trade_gap = r['n_trades'] - TARGET_TRADES
        wr_gap    = r['win_rate'] - TARGET_WIN_RATE
        pnl_gap   = r['avg_pnl']  - TARGET_AVG_PNL
        print(f"    {r['label']:<14}: 거래수 {trade_gap:+4d}  승률 {wr_gap:+.1f}%p  평균PnL {pnl_gap:+.2f}%p")


def print_ticker_breakdown(trades: list[ClosedTrade], label: str) -> None:
    if not trades:
        return
    print(f"\n  [{label}] 종목별:")
    by_ticker: dict[str, list] = {}
    for t in trades:
        by_ticker.setdefault(t.ticker, []).append(t)
    for ticker in sorted(by_ticker):
        tt = by_ticker[ticker]
        wins = sum(1 for t in tt if t.net_pnl > 0)
        avg  = float(np.mean([t.pnl_pct for t in tt]))
        pnl  = sum(t.net_pnl for t in tt)
        print(f"    {ticker:6s}: {len(tt):3d}건  승률 {wins/len(tt)*100:5.1f}%  "
              f"평균 {avg:+6.2f}%  PnL ${pnl:+,.0f}")


def save_results(all_trades: dict[str, list[ClosedTrade]], results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for label, trades in all_trades.items():
        records = [
            {"ticker": t.ticker, "entry_date": str(t.entry_date),
             "exit_date": str(t.exit_date),
             "avg_entry_price": round(t.avg_entry_price, 4),
             "exit_price": round(t.exit_price, 4),
             "net_pnl": round(t.net_pnl, 2),
             "pnl_pct": round(t.pnl_pct, 2),
             "hold_days": t.hold_days,
             "dca_layers": t.dca_layers,
             "reason": t.reason}
            for t in trades
        ]
        pd.DataFrame(records).to_csv(RESULTS_DIR / f"d2s_entry_cap_{label}.csv", index=False)
    with open(RESULTS_DIR / "d2s_entry_cap_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  저장 완료: {RESULTS_DIR}")


# ============================================================
# 메인
# ============================================================
def main():
    print("=" * 65)
    print("  Study D2S Entry Cap — daily_new_entry_cap 완화 실험")
    print("=" * 65)
    print(f"  실거래 D2S 4종목 매수 목표: {TARGET_TRADES}건")
    print(f"  DCA: dca_1pct (-{DCA_THRESHOLD_PCT}% 하락 시 추가매수, 최대 {MAX_DCA}회)")
    print(f"  시나리오: {[s['label'] for s in SCENARIOS]}")

    params = D2S_ENGINE.copy()
    minute_data = load_minute_data()

    print("\n--- 신호 추출 + 시뮬레이션 ---")
    all_trades: dict[str, list[ClosedTrade]] = {}
    results: list[dict] = []

    for i, scen in enumerate(SCENARIOS, 1):
        label = scen["label"]
        cap   = scen["cap"]
        note  = scen["note"]
        print(f"\n  [{i}/{len(SCENARIOS)}] {label} (cap={cap:.2f}, {note})")

        signals_by_date = extract_daily_signals(params, cap)
        trades = simulate_dca(signals_by_date, minute_data, params)
        print(f"    → 청산 거래: {len(trades)}건")

        all_trades[label] = trades
        results.append(analyze_trades(trades, label))

    print_comparison(results)

    for scen in SCENARIOS:
        print_ticker_breakdown(all_trades[scen["label"]], scen["label"])

    print("\n--- 결과 저장 ---")
    save_results(all_trades, results)


if __name__ == "__main__":
    main()
