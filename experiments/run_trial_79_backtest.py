#!/usr/bin/env python3
"""
Trial #79 파라미터로 상세 백테스트 실행
Train/Test 양 기간에 대한 상세 지표 출력
"""

import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "backtests"), str(_ROOT / "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import backtest_common
from backtest_v3 import BacktestEngineV3

# Trial #79 파라미터
TRIAL_79_PARAMS = {
    "V3_PAIR_GAP_ENTRY_THRESHOLD": 9.5,
    "V3_DCA_MAX_COUNT": 7,
    "V3_MAX_PER_STOCK_KRW": 5_000_000,
    "V3_COIN_TRIGGER_PCT": 4.0,
    "V3_CONL_TRIGGER_PCT": 9.5,
    "V3_SPLIT_BUY_INTERVAL_MIN": 50,
    "V3_ENTRY_CUTOFF_HOUR": 12,
    "V3_ENTRY_CUTOFF_MINUTE": 30,
    "V3_SIDEWAYS_MIN_SIGNALS": 3,
    "V3_SIDEWAYS_POLY_LOW": 0.5,
    "V3_SIDEWAYS_POLY_HIGH": 0.55,
    "V3_SIDEWAYS_GLD_THRESHOLD": 0.3,
    "V3_SIDEWAYS_INDEX_THRESHOLD": 0.4,
    "STOP_LOSS_PCT": -1.5,
    "STOP_LOSS_BULLISH_PCT": -14.0,
    "COIN_SELL_PROFIT_PCT": 6.5,
    "CONL_SELL_PROFIT_PCT": 7.5,
    "DCA_DROP_PCT": -0.3,
    "MAX_HOLD_HOURS": 5,
    "TAKE_PROFIT_PCT": 7.5,
    "PAIR_GAP_SELL_THRESHOLD_V2": 6.5,
    "PAIR_SELL_FIRST_PCT": 0.85,
}

# Train/Test 기간
TRAIN_START = date(2025, 1, 3)
TRAIN_END = date(2025, 12, 31)
TEST_START = date(2026, 1, 1)
TEST_END = date(2026, 2, 17)


def run_backtest(params: dict, start_date: date, end_date: date) -> dict:
    """지정된 기간에 대해 백테스트를 실행하고 결과를 반환"""
    # 파라미터 임시 적용
    originals = {}
    for key, value in params.items():
        if hasattr(config, key):
            originals[key] = getattr(config, key)
            setattr(config, key, value)

    try:
        # 백테스트 실행
        engine = BacktestEngineV3(start_date=start_date, end_date=end_date)
        engine.run(verbose=False)

        # 결과 수집
        initial = engine.initial_capital_krw
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial
        total_ret = (final - initial) / initial * 100
        mdd = backtest_common.calc_mdd(engine.equity_curve)
        sharpe = backtest_common.calc_sharpe(engine.equity_curve)

        # 매수/매도 분리
        buys = [t for t in engine.trades if t.side == "BUY"]
        sells = [t for t in engine.trades if t.side == "SELL"]

        # 승률 계산
        win_count = sum(1 for t in sells if t.pnl_pct > 0)
        total_sells = len(sells)
        win_rate = (win_count / total_sells * 100) if total_sells > 0 else 0

        # 손절/시간손절 카운트
        stop_loss_count = sum(1 for t in sells if "손절" in t.exit_reason)
        time_stop_count = sum(1 for t in sells if "시간" in t.exit_reason)
        profit_target_count = sum(1 for t in sells if "익절" in t.exit_reason or "목표" in t.exit_reason)

        # 평균 수익/손실
        winning_trades = [t for t in sells if t.pnl_pct > 0]
        losing_trades = [t for t in sells if t.pnl_pct < 0]

        avg_win = sum(t.pnl_pct for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl_pct for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # 최대 수익/손실
        max_win = max((t.pnl_pct for t in sells), default=0)
        max_loss = min((t.pnl_pct for t in sells), default=0)

        return {
            "return_pct": total_ret,
            "mdd": mdd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "total_buys": len(buys),
            "total_sells": total_sells,
            "win_count": win_count,
            "loss_count": len(losing_trades),
            "stop_loss_count": stop_loss_count,
            "time_stop_count": time_stop_count,
            "profit_target_count": profit_target_count,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "sideways_days": engine.sideways_days,
            "trades": engine.trades,
        }
    finally:
        # 파라미터 복원
        for key, value in originals.items():
            setattr(config, key, value)


def print_results(period_name: str, results: dict):
    """결과 출력"""
    print(f"\n{'=' * 80}")
    print(f"{period_name} 상세 결과")
    print(f"{'=' * 80}\n")

    print(f"📊 수익성 지표")
    print(f"  - 총 수익률: {results['return_pct']:+.2f}%")
    print(f"  - MDD: {results['mdd']:.2f}%")
    print(f"  - Sharpe Ratio: {results['sharpe']:.3f}")
    print()

    print(f"📈 거래 통계")
    print(f"  - 총 매수: {results['total_buys']}회")
    print(f"  - 총 매도: {results['total_sells']}회")
    print(f"  - 승률: {results['win_rate']:.1f}% ({results['win_count']}승 {results['loss_count']}패)")
    print()

    print(f"🎯 청산 사유 분해")
    print(f"  - 익절: {results['profit_target_count']}회")
    print(f"  - 손절: {results['stop_loss_count']}회")
    print(f"  - 시간손절: {results['time_stop_count']}회")
    print()

    print(f"💰 수익/손실 분석")
    print(f"  - 평균 수익: {results['avg_win']:+.2f}%")
    print(f"  - 평균 손실: {results['avg_loss']:+.2f}%")
    print(f"  - 최대 수익: {results['max_win']:+.2f}%")
    print(f"  - 최대 손실: {results['max_loss']:+.2f}%")
    if results['avg_loss'] != 0:
        profit_factor = abs(results['avg_win'] / results['avg_loss'])
        print(f"  - Profit Factor: {profit_factor:.2f}x")
    print()

    print(f"🔍 기타")
    print(f"  - 횡보장 일수: {results['sideways_days']}일")
    print()


def show_sample_trades(trades, n=10):
    """샘플 거래 내역 출력"""
    print(f"\n{'=' * 80}")
    print(f"📋 최근 거래 내역 (최대 {n}건)")
    print(f"{'=' * 80}\n")

    # 매도 거래만 (완료된 거래)
    sells = [t for t in trades if t.side == "SELL"][-n:]

    if not sells:
        print("매도 거래가 없습니다.")
        return

    for i, t in enumerate(sells, 1):
        print(f"[{i}] {t.ticker}")
        print(f"    수익률: {t.pnl_pct:+.2f}% | 청산: {t.exit_reason}")
        print(f"    매도: {t.exit_time} @ {t.price:.2f}원")
        print()


if __name__ == "__main__":
    print("=" * 80)
    print("🏆 Trial #79 상세 백테스트")
    print("=" * 80)
    print()
    print("파라미터:")
    for key, value in TRIAL_79_PARAMS.items():
        print(f"  {key} = {value}")
    print()

    # Train 기간 백테스트
    print("\n[1/2] Train 기간 백테스트 실행 중...")
    train_results = run_backtest(TRIAL_79_PARAMS, TRAIN_START, TRAIN_END)
    print_results(f"Train 기간 ({TRAIN_START} ~ {TRAIN_END})", train_results)
    show_sample_trades(train_results['trades'], n=10)

    # Test 기간 백테스트
    print("\n[2/2] Test 기간 백테스트 실행 중...")
    test_results = run_backtest(TRIAL_79_PARAMS, TEST_START, TEST_END)
    print_results(f"Test 기간 ({TEST_START} ~ {TEST_END})", test_results)
    show_sample_trades(test_results['trades'], n=10)

    # 비교 요약
    print("\n" + "=" * 80)
    print("📊 Train vs Test 비교")
    print("=" * 80)
    print()
    print(f"{'지표':<20} {'Train':>15} {'Test':>15} {'차이':>15}")
    print("-" * 80)
    print(f"{'수익률':<20} {train_results['return_pct']:>14.2f}% {test_results['return_pct']:>14.2f}% {train_results['return_pct'] - test_results['return_pct']:>14.2f}%p")
    print(f"{'MDD':<20} {train_results['mdd']:>14.2f}% {test_results['mdd']:>14.2f}% {test_results['mdd'] - train_results['mdd']:>14.2f}%p")
    print(f"{'Sharpe Ratio':<20} {train_results['sharpe']:>15.3f} {test_results['sharpe']:>15.3f} {test_results['sharpe'] - train_results['sharpe']:>15.3f}")
    print(f"{'승률':<20} {train_results['win_rate']:>14.1f}% {test_results['win_rate']:>14.1f}% {test_results['win_rate'] - train_results['win_rate']:>14.1f}%p")
    print(f"{'총 거래':<20} {train_results['total_sells']:>15} {test_results['total_sells']:>15} {test_results['total_sells'] - train_results['total_sells']:>15}")
    print()

    print("✅ 백테스트 완료!")
