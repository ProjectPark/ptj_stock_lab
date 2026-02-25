#!/usr/bin/env python3
"""
Line C (D2S) 백테스트 평가
===========================
D2SBacktest (v1) 및 D2SBacktestV2 (v2)를 실행하고
핵심 지표를 비교한다.

Usage:
    pyenv shell ptj_stock_lab && python experiments/d2s_backtest_evaluation.py

Output:
    - 콘솔: 결과 요약
    - docs/reports/backtest/d2s_evaluation.md
"""
from __future__ import annotations

import sys
import traceback
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ============================================================
# 설정
# ============================================================
START_DATE = date(2025, 3, 3)    # 기술적 지표 워밍업 후
END_DATE   = date(2026, 2, 17)

OUTPUT_PATH = _PROJECT_ROOT / "docs" / "reports" / "backtest" / "d2s_evaluation.md"

DATA_PATH = _PROJECT_ROOT / "data" / "market" / "daily" / "market_daily.parquet"
POLY_DIR  = _PROJECT_ROOT / "data" / "polymarket"


# ============================================================
# Monkey-patch: backtest_d2s.py에 _ROOT 변수 주입
# ============================================================
# backtest_d2s.py가 _ROOT를 참조하지만 _PROJECT_ROOT만 정의되어 있음.
# 모듈 로드 전에 sys.modules에 직접 주입하는 대신
# 서브클래스를 통해 _load_data를 오버라이드하여 우회한다.


def _load_d2s_data_patched(backtest_instance, verbose: bool = True):
    """_ROOT 버그를 우회하기 위한 데이터 로더."""
    import pandas as pd
    from simulation.backtests import backtest_common

    market_path = DATA_PATH
    poly_dir = POLY_DIR

    if verbose:
        print("[1/3] 데이터 로드")

    if not market_path.exists():
        raise FileNotFoundError(
            f"market_daily.parquet 없음: {market_path}\n"
            f"데이터를 수집하세요: fetchers/fetch_market_daily.py"
        )

    df = pd.read_parquet(market_path)
    if verbose:
        print(f"  market_daily: {df.shape[0]} rows")

    if verbose:
        print("[2/3] 기술적 지표 계산")
    tech = backtest_instance.preprocessor.compute(df)
    if verbose:
        print(f"  {len(tech)} tickers processed")

    poly = backtest_common.load_polymarket_daily(poly_dir)

    return df, tech, poly


# ============================================================
# D2S v1 실행
# ============================================================

def run_d2s_v1(verbose: bool = True) -> dict:
    """D2SBacktest (v1)를 실행하고 지표를 반환한다."""
    import numpy as np
    from simulation.backtests.backtest_d2s import D2SBacktest
    from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE

    class D2SBacktestPatched(D2SBacktest):
        def _load_data(self):
            return _load_d2s_data_patched(self, verbose=verbose)

    bt = D2SBacktestPatched(
        params=D2S_ENGINE,
        start_date=START_DATE,
        end_date=END_DATE,
        use_fees=True,
    )
    bt.run(verbose=verbose)
    return bt.report(), bt


def run_d2s_v2(verbose: bool = True) -> dict:
    """D2SBacktestV2를 실행하고 지표를 반환한다."""
    import numpy as np
    from simulation.backtests.backtest_d2s import D2SBacktest, TradeRecord, BUY_FEE_PCT, SELL_FEE_PCT
    from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2
    from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2

    class D2SBacktestV2Patched(D2SBacktestV2):
        def _load_data(self):
            return _load_d2s_data_patched(self, verbose=verbose)

    bt = D2SBacktestV2Patched(
        params=D2S_ENGINE_V2,
        start_date=START_DATE,
        end_date=END_DATE,
        use_fees=True,
    )
    bt.run(verbose=verbose)
    return bt.report(), bt


# ============================================================
# 보고서 생성
# ============================================================

def build_report(results_v1: dict | None, results_v2: dict | None,
                 err_v1: str | None, err_v2: str | None) -> str:
    lines = []
    lines.append("# Line C (D2S) 백테스트 평가 리포트")
    lines.append("")
    lines.append(f"- 백테스트 기간: {START_DATE} ~ {END_DATE}")
    lines.append(f"- 실행일: 2026-02-23")
    lines.append(f"- 데이터: market_daily.parquet (일봉 기반)")
    lines.append(f"- 수수료: 포함 (KIS 미국주식 기준)")
    lines.append("")

    lines.append("## D2S 전략 개요")
    lines.append("")
    lines.append("D2S(D2S행동추출)는 실거래 행동 패턴을 추출하여 규칙화한 Line C 전략이다.")
    lines.append("일봉 기반으로 작동하며 RSI, MACD, Bollinger Band, ATR 등 기술적 지표를 활용한다.")
    lines.append("")
    lines.append("| 버전 | 규칙 | 핵심 특징 |")
    lines.append("|------|------|-----------|")
    lines.append("| v1 | trading_rules_attach_v1.md (R1~R16) | 쌍둥이 갭, 기술지표, 시황필터, 캘린더 효과 |")
    lines.append("| v2 | trading_rules_attach_v2.md (R17~R18 추가) | V-바운스 2x 포지션, 조기 손절, DCA 레이어 제한 |")
    lines.append("")

    lines.append("## 핵심 성과 비교")
    lines.append("")
    lines.append("| 지표 | D2S v1 | D2S v2 |")
    lines.append("|------|--------|--------|")

    def r(label: str, key: str, fmt_str: str = ".2f", suffix: str = "") -> str:
        v1_val = "ERROR" if err_v1 else (
            format(results_v1.get(key, "N/A"), fmt_str) + suffix
            if isinstance(results_v1.get(key), (int, float)) else str(results_v1.get(key, "N/A"))
        )
        v2_val = "ERROR" if err_v2 else (
            format(results_v2.get(key, "N/A"), fmt_str) + suffix
            if isinstance(results_v2.get(key), (int, float)) else str(results_v2.get(key, "N/A"))
        )
        return f"| {label} | {v1_val} | {v2_val} |"

    if results_v1 and not err_v1:
        lines.append(f"| 기간 | {results_v1.get('period', 'N/A')} | " +
                     (f"{results_v2.get('period', 'N/A')}" if results_v2 and not err_v2 else "ERROR") + " |")
        lines.append(f"| 거래일 | {results_v1.get('trading_days', 'N/A')} | " +
                     (str(results_v2.get('trading_days', 'N/A')) if results_v2 and not err_v2 else "ERROR") + " |")
    else:
        lines.append(f"| 기간 | ERROR | " +
                     (f"{results_v2.get('period', 'N/A')}" if results_v2 and not err_v2 else "ERROR") + " |")

    lines.append(r("초기 자본 (USD)", "initial_capital", ",.0f"))
    lines.append(r("최종 자산 (USD)", "final_equity", ",.2f"))
    lines.append(r("총 수익률 (%)", "total_return_pct", "+.2f", "%"))
    lines.append(r("MDD (%)", "mdd_pct", ".2f", "%"))
    lines.append(r("Sharpe Ratio", "sharpe_ratio", ".3f"))
    lines.append(r("총 손익 (USD)", "total_pnl", "+,.2f"))
    lines.append("")

    lines.append("## 거래 통계")
    lines.append("")
    lines.append("| 지표 | D2S v1 | D2S v2 |")
    lines.append("|------|--------|--------|")
    lines.append(r("총 거래 횟수", "total_trades", "d"))
    lines.append(r("매수 횟수", "buy_trades", "d"))
    lines.append(r("매도 횟수", "sell_trades", "d"))
    lines.append(r("승 횟수", "win_trades", "d"))
    lines.append(r("패 횟수", "lose_trades", "d"))
    lines.append(r("승률 (%)", "win_rate", ".1f", "%"))
    lines.append(r("평균 매도 수익률 (%)", "avg_pnl_pct", "+.2f", "%"))
    lines.append(r("총 수수료 (USD)", "total_fees", ",.2f"))
    lines.append(r("잔여 포지션", "remaining_positions", "d"))
    lines.append("")

    lines.append("## 에러 내역")
    lines.append("")
    if err_v1:
        lines.append(f"- **D2S v1 오류**: `{err_v1}`")
    if err_v2:
        lines.append(f"- **D2S v2 오류**: `{err_v2}`")
    if not err_v1 and not err_v2:
        lines.append("- 오류 없음")
    lines.append("")

    lines.append("## 관측 사항")
    lines.append("")

    ok_results = {}
    if results_v1 and not err_v1:
        ok_results["v1"] = results_v1
    if results_v2 and not err_v2:
        ok_results["v2"] = results_v2

    if ok_results:
        for ver, r in ok_results.items():
            ret = r.get("total_return_pct", 0)
            wr = r.get("win_rate", 0)
            mdd = r.get("mdd_pct", 0)
            lines.append(f"- **D2S {ver}**: 수익률 {ret:+.2f}%, 승률 {wr:.1f}%, MDD {mdd:.2f}%")
    else:
        lines.append("- 실행 실패로 관측 불가")

    if "v1" in ok_results and "v2" in ok_results:
        delta_ret = ok_results["v2"]["total_return_pct"] - ok_results["v1"]["total_return_pct"]
        delta_wr = ok_results["v2"]["win_rate"] - ok_results["v1"]["win_rate"]
        delta_mdd = ok_results["v2"]["mdd_pct"] - ok_results["v1"]["mdd_pct"]
        lines.append(f"- v1 대비 v2: 수익률 {delta_ret:+.2f}%p, 승률 {delta_wr:+.1f}%p, MDD {delta_mdd:+.2f}%p")

    lines.append("")
    lines.append("## 비고")
    lines.append("")
    lines.append("- D2S는 1분봉 기반 Line A v2~v5와 달리 **일봉** 기반으로 동작한다.")
    lines.append("- 동일 비교를 위해 D2S 백테스트 기간은 기술적 지표 워밍업(약 2주) 이후부터 적용한다.")
    lines.append("- 자본 단위: USD (D2S params의 total_capital 기준)")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by experiments/d2s_backtest_evaluation.py*")

    return "\n".join(lines)


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 70)
    print("  Line C (D2S) 백테스트 평가")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print("=" * 70)

    # 데이터 파일 존재 여부 확인
    if not DATA_PATH.exists():
        print(f"\n[ERROR] market_daily.parquet 없음: {DATA_PATH}")
        print("  데이터를 먼저 수집하세요.")
        # 오류 보고서 저장
        err_report = build_report(None, None,
                                   f"market_daily.parquet 없음: {DATA_PATH}",
                                   f"market_daily.parquet 없음: {DATA_PATH}")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(err_report, encoding="utf-8")
        print(f"오류 보고서 저장: {OUTPUT_PATH}")
        return

    results_v1 = None
    results_v2 = None
    err_v1 = None
    err_v2 = None

    # D2S v1
    print("\n[D2S v1] 백테스트 실행 중...")
    try:
        results_v1, bt_v1 = run_d2s_v1(verbose=True)
        bt_v1.print_report()
        print(f"  완료: 수익률 {results_v1.get('total_return_pct', 0):+.2f}%, "
              f"MDD {results_v1.get('mdd_pct', 0):.2f}%, "
              f"Sharpe {results_v1.get('sharpe_ratio', 0):.3f}")
    except Exception as e:
        err_v1 = str(e)
        print(f"  [ERROR] D2S v1: {err_v1}")
        traceback.print_exc()

    # D2S v2
    print("\n[D2S v2] 백테스트 실행 중...")
    try:
        results_v2, bt_v2 = run_d2s_v2(verbose=True)
        bt_v2.print_report()
        print(f"  완료: 수익률 {results_v2.get('total_return_pct', 0):+.2f}%, "
              f"MDD {results_v2.get('mdd_pct', 0):.2f}%, "
              f"Sharpe {results_v2.get('sharpe_ratio', 0):.3f}")
    except Exception as e:
        err_v2 = str(e)
        print(f"  [ERROR] D2S v2: {err_v2}")
        traceback.print_exc()

    # 보고서 저장
    report_md = build_report(results_v1, results_v2, err_v1, err_v2)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report_md, encoding="utf-8")
    print(f"\n보고서 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
