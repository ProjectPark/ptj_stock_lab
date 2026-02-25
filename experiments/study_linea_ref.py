#!/usr/bin/env python3
"""
Line A (backtest_v5, 1분봉) 참고 성과 추출
==========================================
D2S v3 (일봉 기반)와의 기간 비교용 참고 데이터.

※ 타임프레임이 상이하므로 직접 비교 불가 — 참고용만.

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_linea_ref.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from simulation.backtests.backtest_v5 import BacktestEngineV5


def main():
    # ------------------------------------------------------------------
    # 기간 설정: D2S v3와 겹치는 최대 기간
    # backtest_v5 기본: 2025-02-18 ~ 2026-02-17
    # D2S v3 전체 기간: 2024-09-18 ~ 2026-02-17 (데이터 의존)
    # 1분봉 데이터 가용 범위 내에서 최대한 맞춤
    # ------------------------------------------------------------------
    start = date(2025, 2, 18)
    end = date(2026, 2, 17)

    print()
    print("=" * 60)
    print("  Line A (signals_v5, 1분봉) 참고 성과 추출")
    print("  ※ D2S(일봉)와 타임프레임 상이 — 직접 비교 불가, 참고용만")
    print("=" * 60)
    print()

    engine = BacktestEngineV5(
        start_date=start,
        end_date=end,
    )
    engine.run(verbose=True)

    # ------------------------------------------------------------------
    # 지표 추출
    # ------------------------------------------------------------------
    if not engine.equity_curve:
        print("  [ERROR] equity_curve 없음 — 데이터 문제")
        return

    initial = engine.initial_capital_krw
    final = engine.equity_curve[-1][1]
    total_return_pct = (final - initial) / initial * 100

    mdd_pct = engine.calc_mdd()
    sharpe = engine.calc_sharpe()

    sells = [t for t in engine.trades if t.side == "SELL"]
    wins = [t for t in sells if t.pnl_krw > 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0.0

    actual_start = engine.equity_curve[0][0]
    actual_end = engine.equity_curve[-1][0]

    # ------------------------------------------------------------------
    # 출력
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  === Line A (signals_v5, 1분봉 기반) 참고 성과 ===")
    print("  ※ D2S(일봉)와 타임프레임 상이 — 직접 비교 불가, 참고용만")
    print()
    print(f"  기간: {actual_start} ~ {actual_end}")
    print(f"  총 수익률: {total_return_pct:+.2f}%")
    print(f"  MDD: -{mdd_pct:.2f}%")
    print(f"  Sharpe: {sharpe:.4f}")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  총 매도: {len(sells)}회  (승 {len(wins)} / 패 {len(sells) - len(wins)})")
    print(f"  초기자금: ${initial:,.2f}  →  최종자산: ${final:,.2f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 전체 리포트 (상세)
    # ------------------------------------------------------------------
    print()
    engine.print_report()

    # ------------------------------------------------------------------
    # JSON 저장
    # ------------------------------------------------------------------
    today_str = date.today().strftime("%Y%m%d")
    result = {
        "study": "study_linea_ref",
        "engine": "BacktestEngineV5 (Line A, signals_v5, 1분봉)",
        "note": "D2S(일봉)와 타임프레임 상이 — 직접 비교 불가, 참고용만",
        "period": f"{actual_start} ~ {actual_end}",
        "start_date": str(actual_start),
        "end_date": str(actual_end),
        "trading_days": engine.total_trading_days,
        "initial_capital": initial,
        "final_equity": round(final, 2),
        "total_return_pct": round(total_return_pct, 2),
        "mdd_pct": round(mdd_pct, 2),
        "sharpe_ratio": round(sharpe, 4),
        "win_rate": round(win_rate, 1),
        "total_sells": len(sells),
        "total_wins": len(wins),
    }

    out_dir = config.RESULTS_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"study_linea_ref_{today_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
