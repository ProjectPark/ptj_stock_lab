#!/usr/bin/env python3
"""
Study 10 — Look-ahead Bias 수정 전후 비교
==========================================
목적: T일 종가 체결 → T+1일 시가 체결(슬리피지 포함) 수정 후
     수익률 변화 정량화 및 전략 재평가 기준 확인.

검증 기간:
  IS   : 2024-09-18 ~ 2025-05-31
  OOS  : 2025-06-01 ~ 2026-02-17
  FULL : 전체 1.5년

비교 대상:
  biased   : 수정 전 — T일 종가 체결 (편향됨, 현재 엔진에서는 impossible)
  corrected: 수정 후 — T+1일 시가 체결 + 슬리피지 0.05%

※ biased 재현을 위해 D2SBacktestV3를 slippage_pct=0.0으로 실행한 뒤
  _execute_buy/_execute_sell을 종가 사용 모드로 바꾼 별도 클래스를 사용.

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_10_bias_corrected_v3.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from simulation.backtests.backtest_d2s_v3 import D2SBacktestV3
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V3_NO_ROBN
from simulation.strategies.line_c_d2s.d2s_engine import DailySnapshot

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "backtests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 기간 정의 ──────────────────────────────────────────────
IS_START   = date(2024, 9, 18)
IS_END     = date(2025, 5, 31)
OOS_START  = date(2025, 6, 1)
OOS_END    = date(2026, 2, 17)
FULL_START = date(2024, 9, 18)
FULL_END   = date(2026, 2, 17)

PERIODS = [
    ("IS",   IS_START,   IS_END),
    ("OOS",  OOS_START,  OOS_END),
    ("FULL", FULL_START, FULL_END),
]


# ============================================================
# Biased 버전 — T일 종가로 결정+체결 (수정 전 동작 재현)
# ============================================================

class D2SBacktestV3Biased(D2SBacktestV3):
    """Look-ahead bias 있는 버전 — T일 종가로 결정+즉시 체결 (수정 전 동작 재현).

    run()을 오버라이드하여 pending_signals 없이 같은 날 종가로 즉시 체결.
    slippage=0, 체결가=closes.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("slippage_pct", 0.0)
        super().__init__(**kwargs)

    def run(self, verbose: bool = True) -> "D2SBacktestV3Biased":
        """원래 편향 동작 재현 — T종가로 결정+즉시 체결 (pending 없음)."""
        df, tech, poly = self._load_data()

        all_dates = sorted(df.index)
        trading_dates = [
            d.date() if hasattr(d, "date") else d
            for d in all_dates
            if self.start_date <= (d.date() if hasattr(d, "date") else d) <= self.end_date
        ]

        if verbose:
            print(f"\n[3/3] D2S v3 biased (look-ahead bias 있음 — 종가 즉시 체결)")
            print(f"  기간: {trading_dates[0]} ~ {trading_dates[-1]} ({len(trading_dates)}일)")

        spy_streak = 0

        for i, td in enumerate(trading_dates):
            snap = self._build_snapshot(td, tech, poly, spy_streak)
            if snap is None:
                continue

            self._current_td = td

            spy_close = snap.closes.get("SPY", None)
            if spy_close:
                self._spy_closes.append(float(spy_close))

            poly_btc = snap.poly_btc_up
            regime = self._detect_regime(spy_streak, spy_close, poly_btc)
            self._current_regime = regime
            self._regime_days[regime] = self._regime_days.get(regime, 0) + 1

            daily_buy_counts: dict[str, int] = {}

            # 시그널 생성 (T 종가 기반)
            signals = self.engine.generate_daily_signals(snap, self.positions, daily_buy_counts)

            # R20/R21 레짐 조건부 청산 오버라이드
            non_sell = [s for s in signals if s["action"] != "SELL"]
            regime_sell = []
            for ticker, pos in list(self.positions.items()):
                exit_ctx = self._check_regime_exit(ticker, pos, snap, regime)
                if exit_ctx["should_exit"]:
                    regime_sell.append({
                        "action": "SELL", "ticker": ticker, "size": 1.0,
                        "reason": exit_ctx["reason"], "score": 0,
                        "signal_regime": regime,
                    })
                    self._r20_r21_applied[regime] = self._r20_r21_applied.get(regime, 0) + 1
            signals = non_sell + regime_sell

            # R18 조기 손절
            already_selling = {s["ticker"] for s in signals if s["action"] == "SELL"}
            for ticker, pos in list(self.positions.items()):
                if ticker in already_selling:
                    continue
                r18_reason = self._check_r18_exit(ticker, pos, snap)
                if r18_reason:
                    signals.append({"action": "SELL", "ticker": ticker, "size": 1.0,
                                    "reason": r18_reason, "score": 0, "signal_regime": regime})
                    self._r18_count += 1

            # T 종가로 즉시 체결 (편향 동작)
            for sig in signals:
                if sig["action"] == "SELL":
                    pnl = self._execute_sell_close(sig["ticker"], sig["size"], snap, sig["reason"])
                    if sig["ticker"] not in self.positions:
                        self.position_meta.pop(sig["ticker"], None)
                        sig_regime = sig.get("signal_regime", "neutral")
                        self._regime_exits[sig_regime].append(pnl.pnl_pct if pnl is not None else 0)

            daily_entry_cap = self.params.get("daily_new_entry_cap", 1.0)
            daily_entry_used = 0.0
            for sig in signals:
                if sig["action"] == "BUY":
                    if daily_entry_used + sig["size"] > daily_entry_cap:
                        continue
                    trade = self._execute_buy_close(sig["ticker"], sig["size"], snap,
                                                    sig["reason"], sig.get("score", 0))
                    if trade:
                        daily_entry_used += sig["size"]
                        daily_buy_counts[sig["ticker"]] = daily_buy_counts.get(sig["ticker"], 0) + 1

            # SPY streak 업데이트
            spy_pct = snap.changes.get("SPY", 0)
            if spy_pct > 0:
                spy_streak += 1
                self._spy_down_streak = 0
            else:
                spy_streak = 0
                self._spy_down_streak += 1

            # 자산 스냅샷 (종가 기준)
            equity = self.cash + sum(
                snap.closes.get(t, pos.entry_price) * pos.qty
                for t, pos in self.positions.items()
            )
            self.equity_curve.append((td, equity))

        return self

    def _execute_buy_close(self, ticker: str, size: float, snap: DailySnapshot,
                           reason: str, score: float):
        """T 종가로 즉시 매수 (편향 동작 재현)."""
        from simulation.backtests.backtest_d2s import BUY_FEE_PCT, TradeRecord
        from simulation.strategies.line_c_d2s.d2s_engine import D2SPosition

        price = snap.closes.get(ticker, 0)
        if price <= 0 or self.cash <= 0:
            return None

        amount = self.cash * size
        amount = min(amount, self.cash)
        if amount < 1.0:
            return None

        fee = amount * BUY_FEE_PCT / 100 if self.use_fees else 0
        net_amount = amount - fee
        qty = net_amount / price

        existing = self.positions.get(ticker)
        if existing:
            total_cost = existing.cost_basis + net_amount
            total_qty = existing.qty + qty
            existing.entry_price = total_cost / total_qty
            existing.qty = total_qty
            existing.cost_basis = total_cost
            existing.dca_count += 1
        else:
            self.positions[ticker] = D2SPosition(
                ticker=ticker, entry_price=price, qty=qty,
                entry_date=snap.trading_date, cost_basis=net_amount,
            )
        self.cash -= amount

        trade = TradeRecord(
            date=snap.trading_date, ticker=ticker, side="BUY",
            price=price, qty=qty, amount=amount, fee=fee,
            pnl=0, pnl_pct=0, reason=reason, score=score,
        )
        self.trades.append(trade)
        return trade

    def _execute_sell_close(self, ticker: str, size: float, snap: DailySnapshot, reason: str):
        """T 종가로 즉시 매도 (편향 동작 재현)."""
        from simulation.backtests.backtest_d2s import SELL_FEE_PCT, TradeRecord

        pos = self.positions.get(ticker)
        if not pos:
            return None

        price = snap.closes.get(ticker, 0)
        if price <= 0:
            return None

        sell_qty = pos.qty * size
        proceeds = sell_qty * price
        fee = proceeds * SELL_FEE_PCT / 100 if self.use_fees else 0
        net_proceeds = proceeds - fee

        cost_basis = pos.entry_price * sell_qty
        pnl = net_proceeds - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0

        pos.qty -= sell_qty
        pos.cost_basis -= cost_basis
        if pos.qty < 0.001:
            del self.positions[ticker]
        self.cash += net_proceeds

        trade = TradeRecord(
            date=snap.trading_date, ticker=ticker, side="SELL",
            price=price, qty=sell_qty, amount=proceeds, fee=fee,
            pnl=pnl, pnl_pct=pnl_pct, reason=reason,
        )
        self.trades.append(trade)
        return trade


# ============================================================
# 실행 함수
# ============================================================

def run_backtest(label: str, cls, params: dict, start: date, end: date) -> dict:
    """단일 백테스트 실행 후 report dict 반환."""
    bt = cls(params=params, start_date=start, end_date=end)
    bt.run(verbose=False)
    r = bt.report()
    r["label"] = label
    return r


def print_comparison_table(results_by_period: dict[str, list[dict]]) -> None:
    """기간별 biased vs corrected 비교 테이블 출력."""
    print("\n" + "=" * 80)
    print("  Study 10: Look-ahead Bias 수정 전후 비교")
    print("=" * 80)

    for period_label, (biased, corrected) in results_by_period.items():
        print(f"\n  ── {period_label} ─────────────────────────────────────────")
        header = f"  {'구분':<12} {'수익률':>9} {'MDD':>8} {'Sharpe':>8} {'승률':>7} {'청산':>6} {'avg_pnl':>8}"
        print(header)
        print("  " + "-" * 60)

        for r in [biased, corrected]:
            mark = "⚠️ biased " if r["label"] == "biased" else "✅ corrected"
            print(
                f"  {mark:<12} "
                f"{r['total_return_pct']:>8.2f}% "
                f"{r['mdd_pct']:>7.2f}% "
                f"{r['sharpe_ratio']:>8.3f} "
                f"{r['win_rate']:>6.1f}% "
                f"{r['sell_trades']:>6} "
                f"{r['avg_pnl_pct']:>7.2f}%"
            )

        # 차이
        diff_ret = corrected["total_return_pct"] - biased["total_return_pct"]
        diff_mdd = corrected["mdd_pct"] - biased["mdd_pct"]
        diff_sharpe = corrected["sharpe_ratio"] - biased["sharpe_ratio"]
        print(f"\n  {'Δ (corrected-biased)':<12} "
              f"{diff_ret:>8.2f}%p "
              f"{diff_mdd:>7.2f}%p "
              f"{diff_sharpe:>8.3f}")

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("  Study 10 — Look-ahead Bias 수정 전후 비교 (D2S_ENGINE_V3_NO_ROBN)")
    print(f"  실행 시각: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)

    params = dict(D2S_ENGINE_V3_NO_ROBN)
    results_by_period: dict[str, list[dict]] = {}
    all_results: list[dict] = []

    for period_label, start, end in PERIODS:
        print(f"\n  [{period_label}] {start} ~ {end}")

        # Biased (수정 전 재현)
        print(f"    biased (종가 체결)...")
        r_biased = run_backtest("biased", D2SBacktestV3Biased, params, start, end)
        r_biased["period"] = period_label

        # Corrected (T+1 시가 체결)
        print(f"    corrected (시가 체결 + 슬리피지)...")
        r_corrected = run_backtest("corrected", D2SBacktestV3, params, start, end)
        r_corrected["period"] = period_label

        results_by_period[period_label] = [r_biased, r_corrected]
        all_results.extend([r_biased, r_corrected])

        print(f"    biased:    수익률={r_biased['total_return_pct']:+.2f}%  "
              f"MDD={r_biased['mdd_pct']:.2f}%  Sharpe={r_biased['sharpe_ratio']:.3f}")
        print(f"    corrected: 수익률={r_corrected['total_return_pct']:+.2f}%  "
              f"MDD={r_corrected['mdd_pct']:.2f}%  Sharpe={r_corrected['sharpe_ratio']:.3f}")

    print_comparison_table(results_by_period)

    # ── OOS 변화량으로 Study 13 판단 ─────────────────────────────
    oos_biased    = next(r for r in all_results if r["period"] == "OOS" and r["label"] == "biased")
    oos_corrected = next(r for r in all_results if r["period"] == "OOS" and r["label"] == "corrected")
    oos_change = abs(oos_corrected["total_return_pct"] - oos_biased["total_return_pct"])

    print(f"\n  OOS 수익률 변화량: {oos_change:.1f}%p")
    if oos_change > 10:
        print("  → ⚡ Study 13 발동 조건 충족 (변화 > 10%p): Optuna 재최적화 권장")
    else:
        print("  → Study 13 불필요 (변화 ≤ 10%p): 기존 파라미터 유효")

    # ── 저장 ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"study_10_bias_corrected_{datetime.now():%Y%m%d}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  결과 저장: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
