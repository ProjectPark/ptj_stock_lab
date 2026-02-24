"""
Line D — JUN 매매법 v2.1 백테스트 실행 스크립트
=================================================
실행: pyenv shell ptj_stock_lab && python simulation/backtests/backtest_line_d.py
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 path에 추가
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation.strategies.line_d_history.engine import JunTradeEngine
from simulation.strategies.line_d_history.params import JunTradeParams

RESULTS_DIR = ROOT / "data" / "results" / "backtests"


def print_report(result: dict) -> None:
    """백테스트 결과 리포트 출력."""
    summary = result["summary"]
    trades = result["closed_trades"]

    print("\n" + "=" * 70)
    print("  JUN 매매법 v2.1 백테스트 결과")
    print("=" * 70)

    # 기본 통계
    print(f"\n{'거래 수:':<20} {summary.get('trades', 0)}")
    print(f"{'승리:':<20} {summary.get('wins', 0)}")
    print(f"{'패배:':<20} {summary.get('losses', 0)}")
    print(f"{'승률:':<20} {summary.get('win_rate', 0):.1f}%")
    print(f"{'평균 손익:':<20} {summary.get('avg_pnl_pct', 0):+.2f}%")
    print(f"{'총 손익(USD):':<20} ${summary.get('total_pnl_usd', 0):+.2f}")

    # 청산 사유별 통계
    if trades:
        print(f"\n{'─' * 40}")
        print("청산 사유별 통계:")
        exit_reasons = Counter(t.exit_reason for t in trades)
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            reason_trades = [t for t in trades if t.exit_reason == reason]
            avg_pnl = sum(t.pnl_pct for t in reason_trades) / len(reason_trades)
            wins = sum(1 for t in reason_trades if t.pnl_pct > 0)
            print(f"  {reason:<15} {count:>3}건  승률={wins/len(reason_trades)*100:.0f}%  평균={avg_pnl:+.1f}%")

        # 종목별 성과
        print(f"\n{'─' * 40}")
        print("종목별 성과:")
        ticker_trades: dict[str, list] = {}
        for t in trades:
            ticker_trades.setdefault(t.ticker, []).append(t)
        for ticker in sorted(ticker_trades.keys()):
            tt = ticker_trades[ticker]
            avg = sum(t.pnl_pct for t in tt) / len(tt)
            wins = sum(1 for t in tt if t.pnl_pct > 0)
            total_usd = sum(t.pnl_usd for t in tt)
            print(f"  {ticker:<8} {len(tt):>3}건  승률={wins/len(tt)*100:.0f}%  평균={avg:+.1f}%  총=${total_usd:+.2f}")

        # 피라미딩 통계
        print(f"\n{'─' * 40}")
        print("피라미딩 통계:")
        for n_adds in sorted(set(t.add_count for t in trades)):
            grp = [t for t in trades if t.add_count == n_adds]
            avg = sum(t.pnl_pct for t in grp) / len(grp)
            wins = sum(1 for t in grp if t.pnl_pct > 0)
            print(f"  추가매수 {n_adds}회: {len(grp):>3}건  승률={wins/len(grp)*100:.0f}%  평균={avg:+.1f}%")

        # 자산곡선 MDD / Sharpe
        equity = result.get("equity_curve", [])
        if equity:
            vals = [v for _, v in equity if v > 0]
            if vals:
                peak = vals[0]
                mdd = 0.0
                for v in vals:
                    if v > peak:
                        peak = v
                    dd = (v - peak) / peak * 100
                    if dd < mdd:
                        mdd = dd
                print(f"\n{'─' * 40}")
                print(f"자산곡선 MDD: {mdd:.1f}%")

    print("=" * 70)


def save_trade_log(result: dict, filename: str = "line_d_trades.csv") -> None:
    """거래 로그를 CSV로 저장."""
    trades = result["closed_trades"]
    if not trades:
        print("[save] 저장할 거래 없음")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in trades:
        rows.append({
            "ticker": t.ticker,
            "open_date": t.open_date,
            "close_date": t.close_date,
            "avg_price_usd": round(t.avg_price_usd, 4),
            "close_price_usd": round(t.close_price_usd, 4),
            "shares": round(t.shares, 6),
            "add_count": t.add_count,
            "pnl_pct": t.pnl_pct,
            "pnl_usd": t.pnl_usd,
            "exit_reason": t.exit_reason,
        })

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"\n[save] 거래 로그 저장: {path} ({len(df)} 건)")


def main() -> None:
    params = JunTradeParams()
    engine = JunTradeEngine(
        params=params,
        start_date=date(2023, 10, 1),
        end_date=date(2026, 2, 17),
    )
    result = engine.run(verbose=True)
    print_report(result)
    save_trade_log(result)


if __name__ == "__main__":
    main()
