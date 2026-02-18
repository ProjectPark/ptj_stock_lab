#!/usr/bin/env python3
"""
PTJ v2 백테스트 — 핵심 파라미터 변경 테스트
=============================================
config 모듈 값을 임시로 변경한 후 BacktestEngineV2.run() 호출,
결과를 수집하고 config 복원. 6개 시나리오 비교.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import config
import backtest_common
from backtest_v2 import BacktestEngineV2


# ============================================================
# Result dataclass
# ============================================================
@dataclass
class TestResult:
    name: str
    final_equity: float
    total_return_pct: float
    mdd: float
    total_fees: float
    total_sells: int
    total_buys: int
    sharpe: float
    # breakdown
    buy_fees: float
    sell_fees: float
    stop_loss_count: int
    stop_loss_pnl: float
    staged_sell_count: int
    conl_pnl: float


# ============================================================
# Helper: run engine and extract metrics
# ============================================================
def run_and_collect(label: str) -> TestResult:
    """Run backtest engine and return structured result."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    t0 = time.time()
    engine = BacktestEngineV2()
    engine.run(verbose=False)
    elapsed = time.time() - t0

    # Basic metrics
    initial = engine.initial_capital_krw
    final = engine.equity_curve[-1][1] if engine.equity_curve else initial
    total_ret = (final - initial) / initial * 100
    mdd = backtest_common.calc_mdd(engine.equity_curve)
    sharpe = backtest_common.calc_sharpe(engine.equity_curve)
    total_fees = engine.total_buy_fees_krw + engine.total_sell_fees_krw

    sells = [t for t in engine.trades if t.side == "SELL"]
    buys = [t for t in engine.trades if t.side == "BUY"]

    # Stop loss breakdown
    stop_loss_sells = [t for t in sells if t.exit_reason == "stop_loss"]
    stop_loss_count = len(stop_loss_sells)
    stop_loss_pnl = sum(t.pnl_krw for t in stop_loss_sells)

    # Staged sell count
    staged_sells = [t for t in sells if t.exit_reason == "staged_sell"]
    staged_sell_count = len(staged_sells)

    # CONL P&L
    conl_sells = [t for t in sells if t.ticker == "CONL"]
    conl_pnl = sum(t.pnl_krw for t in conl_sells)

    result = TestResult(
        name=label,
        final_equity=final,
        total_return_pct=total_ret,
        mdd=mdd,
        total_fees=total_fees,
        total_sells=len(sells),
        total_buys=len(buys),
        sharpe=sharpe,
        buy_fees=engine.total_buy_fees_krw,
        sell_fees=engine.total_sell_fees_krw,
        stop_loss_count=stop_loss_count,
        stop_loss_pnl=stop_loss_pnl,
        staged_sell_count=staged_sell_count,
        conl_pnl=conl_pnl,
    )

    print(f"\n  결과 요약 ({elapsed:.1f}초)")
    print(f"  최종 자산    : {final:>14,.0f}원")
    print(f"  총 수익률    : {total_ret:>+.2f}%")
    print(f"  MDD          : -{mdd:.2f}%")
    print(f"  총 수수료    : {total_fees:>12,.0f}원")
    print(f"  매도 횟수    : {len(sells)}")
    print(f"  매수 횟수    : {len(buys)}")
    print(f"  손절 횟수    : {stop_loss_count} (P&L {stop_loss_pnl:>+12,.0f}원)")
    print(f"  분할매도 횟수: {staged_sell_count}")
    print(f"  CONL P&L     : {conl_pnl:>+12,.0f}원")
    print(f"  Sharpe       : {sharpe:.4f}")

    return result


# ============================================================
# Config backup / restore helpers
# ============================================================
def backup_config() -> dict:
    """Backup all config values we'll modify."""
    return {
        "PAIR_SELL_REMAINING_PCT": config.PAIR_SELL_REMAINING_PCT,
        "PAIR_SELL_FIRST_PCT": config.PAIR_SELL_FIRST_PCT,
        "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
        "CONL_TRIGGER_PCT": config.CONL_TRIGGER_PCT,
        "COIN_TRIGGER_PCT": config.COIN_TRIGGER_PCT,
        "DCA_MAX_COUNT": config.DCA_MAX_COUNT,
    }


def restore_config(backup: dict) -> None:
    """Restore config values from backup."""
    for key, val in backup.items():
        setattr(config, key, val)


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("  PTJ v2 백테스트 — 파라미터 최적화 테스트")
    print("  " + "=" * 55)
    print()

    results: list[TestResult] = []
    cfg_backup = backup_config()

    # ---- Test 0: Baseline (기본값) ----
    restore_config(cfg_backup)
    r0 = run_and_collect("Test 0: Baseline (기본값)")
    results.append(r0)

    # ---- Test 1: 분할매도 비율 100% ----
    restore_config(cfg_backup)
    config.PAIR_SELL_REMAINING_PCT = 1.0  # 기본 0.30
    r1 = run_and_collect("Test 1: 분할매도 100% (PAIR_SELL_REMAINING_PCT=1.0)")
    results.append(r1)

    # ---- Test 2: 1차 매도 100% ----
    restore_config(cfg_backup)
    config.PAIR_SELL_FIRST_PCT = 1.0  # 기본 0.80
    r2 = run_and_collect("Test 2: 1차 전량매도 (PAIR_SELL_FIRST_PCT=1.0)")
    results.append(r2)

    # ---- Test 3: 손절 라인 완화 -5% ----
    restore_config(cfg_backup)
    config.STOP_LOSS_PCT = -5.0  # 기본 -3.0
    r3 = run_and_collect("Test 3: 손절 완화 (STOP_LOSS_PCT=-5.0)")
    results.append(r3)

    # ---- Test 4: CONL/COIN 트리거 상향 +5% ----
    restore_config(cfg_backup)
    config.CONL_TRIGGER_PCT = 5.0  # 기본 3.0
    config.COIN_TRIGGER_PCT = 5.0  # 기본 3.0
    r4 = run_and_collect("Test 4: 트리거 상향 (CONL/COIN_TRIGGER_PCT=5.0)")
    results.append(r4)

    # ---- Test 5: DCA 최대 3회 ----
    restore_config(cfg_backup)
    config.DCA_MAX_COUNT = 3  # 기본 7
    r5 = run_and_collect("Test 5: DCA 제한 (DCA_MAX_COUNT=3)")
    results.append(r5)

    # ---- Test 6: 복합 최적화 (Test 1 + Test 3 + Test 4) ----
    restore_config(cfg_backup)
    config.PAIR_SELL_REMAINING_PCT = 1.0
    config.STOP_LOSS_PCT = -5.0
    config.CONL_TRIGGER_PCT = 5.0
    config.COIN_TRIGGER_PCT = 5.0
    r6 = run_and_collect("Test 6: 복합 최적화 (T1+T3+T4)")
    results.append(r6)

    # Restore config
    restore_config(cfg_backup)

    # ============================================================
    # Comparison Table
    # ============================================================
    baseline = results[0]

    print()
    print()
    print("=" * 120)
    print("  종합 비교표")
    print("=" * 120)
    print(
        f"  {'테스트':<40s}  {'최종자산':>14s}  {'수익률':>8s}  {'MDD':>8s}  "
        f"{'수수료':>12s}  {'매도':>6s}  {'손절':>6s}  {'Sharpe':>8s}"
    )
    print("-" * 120)

    for r in results:
        ret_delta = r.total_return_pct - baseline.total_return_pct
        fee_delta = r.total_fees - baseline.total_fees
        sell_delta = r.total_sells - baseline.total_sells

        delta_str = ""
        if r.name != baseline.name:
            delta_str = (
                f"  (수익률 {ret_delta:>+.2f}pp, "
                f"수수료 {fee_delta:>+,.0f}원, "
                f"매도 {sell_delta:>+d}건)"
            )

        print(
            f"  {r.name:<40s}  "
            f"{r.final_equity:>14,.0f}  "
            f"{r.total_return_pct:>+7.2f}%  "
            f"-{r.mdd:>6.2f}%  "
            f"{r.total_fees:>12,.0f}  "
            f"{r.total_sells:>6d}  "
            f"{r.stop_loss_count:>6d}  "
            f"{r.sharpe:>8.4f}"
        )
        if delta_str:
            print(f"  {'':40s}  {delta_str}")

    print("=" * 120)

    # ============================================================
    # Detailed delta table
    # ============================================================
    print()
    print("  [기본값 대비 변화량 상세]")
    print("-" * 100)
    print(
        f"  {'테스트':<40s}  {'수익률Δ':>9s}  {'수수료Δ':>12s}  "
        f"{'매도Δ':>7s}  {'손절Δ':>7s}  {'분할매도Δ':>9s}  {'CONL P&L Δ':>12s}"
    )
    print("-" * 100)

    for r in results[1:]:
        print(
            f"  {r.name:<40s}  "
            f"{r.total_return_pct - baseline.total_return_pct:>+8.2f}%  "
            f"{r.total_fees - baseline.total_fees:>+12,.0f}  "
            f"{r.total_sells - baseline.total_sells:>+7d}  "
            f"{r.stop_loss_count - baseline.stop_loss_count:>+7d}  "
            f"{r.staged_sell_count - baseline.staged_sell_count:>+9d}  "
            f"{r.conl_pnl - baseline.conl_pnl:>+12,.0f}"
        )

    print("-" * 100)
    print()

    # Best result
    best = max(results, key=lambda r: r.total_return_pct)
    print(f"  ★ 최고 수익률: {best.name}")
    print(f"    수익률 {best.total_return_pct:>+.2f}% | 수수료 {best.total_fees:>,.0f}원 | MDD -{best.mdd:.2f}%")
    print()


if __name__ == "__main__":
    main()
