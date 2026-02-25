#!/usr/bin/env python3
"""
Line A v2~v5 백테스트 비교 평가
================================
pipeline.py의 run_backtest()를 사용하여 v2~v5 각 버전을 동일 기간으로
백테스트 실행하고 핵심 지표를 비교 테이블로 출력한다.

Usage:
    pyenv shell ptj_stock_lab && python experiments/backtest_evaluation.py

Output:
    - 콘솔: 비교 테이블
    - docs/reports/backtest/version_comparison_table.md
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
START_DATE = date(2025, 2, 18)
END_DATE   = date(2026, 2, 17)

VERSIONS = ["v2", "v3", "v4", "v5"]

OUTPUT_PATH = _PROJECT_ROOT / "docs" / "reports" / "backtest" / "version_comparison_table.md"


# ============================================================
# 지표 추출 헬퍼
# ============================================================

def _get_pnl_field(trade) -> float:
    """버전별 pnl 필드명 차이 처리."""
    return getattr(trade, "pnl_krw", getattr(trade, "pnl_usd", 0.0))


def extract_metrics(version: str, engine) -> dict:
    """엔진에서 비교 지표를 추출한다."""
    from simulation.backtests import backtest_common

    # 자본 (v2는 USD 기반, v3~v5는 USD 기반이지만 _krw alias 사용)
    if hasattr(engine, "initial_capital_usd") and not hasattr(engine, "initial_capital_krw"):
        initial = engine.initial_capital_usd
        currency = "USD"
    elif hasattr(engine, "initial_capital_krw"):
        initial = engine.initial_capital_krw
        currency = "USD"  # v3~v5도 실제론 USD 단위
    else:
        initial = engine.initial_capital
        currency = "USD"

    if not engine.equity_curve:
        return {
            "version": version, "currency": currency,
            "initial": initial, "final": initial,
            "return_pct": 0.0, "mdd": 0.0, "sharpe": 0.0,
            "total_sells": 0, "total_buys": 0,
            "win_rate": 0.0, "win_count": 0, "loss_count": 0,
            "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0,
            "stop_loss_count": 0, "total_fees": 0.0,
            "trading_days": 0,
            "sideways_days": 0, "sideways_blocks": 0,
            "entry_cutoff_blocks": 0, "daily_limit_blocks": 0,
            "cb_buy_blocks": 0,
            "error": None,
        }

    final = engine.equity_curve[-1][1]
    total_ret = (final - initial) / initial * 100
    mdd = backtest_common.calc_mdd(engine.equity_curve)
    sharpe = backtest_common.calc_sharpe(engine.equity_curve)

    sells = [t for t in engine.trades if t.side == "SELL"]
    buys  = [t for t in engine.trades if t.side == "BUY"]
    wins  = [t for t in sells if _get_pnl_field(t) > 0]
    losses = [t for t in sells if _get_pnl_field(t) < 0]
    stop_losses = [t for t in sells if getattr(t, "exit_reason", "") == "stop_loss"]

    total_pnl = sum(_get_pnl_field(t) for t in sells)
    avg_win  = sum(_get_pnl_field(t) for t in wins)  / len(wins)  if wins  else 0.0
    avg_loss = sum(_get_pnl_field(t) for t in losses) / len(losses) if losses else 0.0

    # 수수료
    if hasattr(engine, "total_buy_fees_krw"):
        total_fees = engine.total_buy_fees_krw + engine.total_sell_fees_krw
    elif hasattr(engine, "total_buy_fees_usd"):
        total_fees = engine.total_buy_fees_usd + engine.total_sell_fees_usd
    else:
        total_fees = getattr(engine, "total_buy_fees", 0.0) + getattr(engine, "total_sell_fees", 0.0)

    return {
        "version": version,
        "currency": currency,
        "initial": initial,
        "final": final,
        "return_pct": total_ret,
        "mdd": mdd,
        "sharpe": sharpe,
        "total_sells": len(sells),
        "total_buys": len(buys),
        "win_rate": len(wins) / len(sells) * 100 if sells else 0.0,
        "win_count": len(wins),
        "loss_count": len(losses),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": total_pnl,
        "stop_loss_count": len(stop_losses),
        "total_fees": total_fees,
        "trading_days": getattr(engine, "total_trading_days", len(engine.equity_curve)),
        "sideways_days": getattr(engine, "sideways_days", 0),
        "sideways_blocks": getattr(engine, "sideways_blocks", 0),
        "entry_cutoff_blocks": getattr(engine, "entry_cutoff_blocks", 0),
        "daily_limit_blocks": getattr(engine, "daily_limit_blocks", 0),
        "cb_buy_blocks": getattr(engine, "cb_buy_blocks", 0),
        "error": None,
    }


# ============================================================
# 보고서 생성
# ============================================================

def _fmt(value, fmt: str, fallback: str = "N/A") -> str:
    try:
        return format(value, fmt)
    except (TypeError, ValueError):
        return fallback


def build_report(results: list[dict]) -> str:
    lines = []
    lines.append("# Line A v2~v5 백테스트 비교 리포트")
    lines.append("")
    lines.append(f"- 백테스트 기간: {START_DATE} ~ {END_DATE}")
    lines.append(f"- 실행일: 2026-02-23")
    lines.append(f"- 수수료: 포함 (KIS 미국주식 기준)")
    lines.append("")

    lines.append("## 핵심 성과 비교")
    lines.append("")
    lines.append("| 지표 | v2 | v3 | v4 | v5 |")
    lines.append("|------|----|----|----|----|")

    def row(label: str, key: str, fmt: str = ".2f", suffix: str = "") -> str:
        vals = []
        for r in results:
            if r.get("error"):
                vals.append("ERROR")
            elif key in r:
                vals.append(f"{r[key]:{fmt}}{suffix}")
            else:
                vals.append("N/A")
        return f"| {label} | " + " | ".join(vals) + " |"

    for r in results:
        if r.get("error"):
            curr_label = "ERROR"
        else:
            curr_label = f"${r['initial']:,.0f}"

    lines.append("| 초기 자금 (USD) | " + " | ".join(
        "ERROR" if r.get("error") else f"${r['initial']:,.0f}"
        for r in results
    ) + " |")
    lines.append("| 최종 자산 (USD) | " + " | ".join(
        "ERROR" if r.get("error") else f"${r['final']:,.0f}"
        for r in results
    ) + " |")
    lines.append(row("총 수익률 (%)", "return_pct", "+.2f", "%"))
    lines.append(row("최대 낙폭 MDD (%)", "mdd", ".2f", "%"))
    lines.append(row("Sharpe Ratio", "sharpe", ".4f"))
    lines.append(row("총 손익 (USD)", "total_pnl", "+,.2f"))
    lines.append("")

    lines.append("## 거래 통계")
    lines.append("")
    lines.append("| 지표 | v2 | v3 | v4 | v5 |")
    lines.append("|------|----|----|----|----|")
    lines.append(row("매수 횟수", "total_buys", "d"))
    lines.append(row("매도 횟수", "total_sells", "d"))
    lines.append(row("승률 (%)", "win_rate", ".1f", "%"))
    lines.append(row("승 횟수", "win_count", "d"))
    lines.append(row("패 횟수", "loss_count", "d"))
    lines.append(row("평균 수익 (USD)", "avg_win", "+,.2f"))
    lines.append(row("평균 손실 (USD)", "avg_loss", "+,.2f"))
    lines.append(row("손절 횟수", "stop_loss_count", "d"))
    lines.append(row("총 수수료 (USD)", "total_fees", ",.2f"))
    lines.append(row("거래일", "trading_days", "d"))
    lines.append("")

    lines.append("## v3~v5 선별 매매 효과 (v2는 N/A)")
    lines.append("")
    lines.append("| 지표 | v2 | v3 | v4 | v5 |")
    lines.append("|------|----|----|----|----|")
    lines.append(row("횡보장 감지일", "sideways_days", "d"))
    lines.append(row("횡보장 매수 차단", "sideways_blocks", "d"))
    lines.append(row("진입마감 차단", "entry_cutoff_blocks", "d"))
    lines.append(row("일일한도 차단", "daily_limit_blocks", "d"))
    lines.append(row("CB 매수 차단", "cb_buy_blocks", "d"))
    lines.append("")

    lines.append("## 버전별 주요 변경 사항")
    lines.append("")
    lines.append("| 버전 | 핵심 변경 |")
    lines.append("|------|----------|")
    lines.append("| v2 | 기본 전략 — 쌍둥이 갭 1.5%, DCA 7회, 트리거 3%, 쿨타임 5분 |")
    lines.append("| v3 | 선별 매매형 — 갭 2.2%, DCA 4회, 트리거 4.5%, 쿨타임 20분, 일일 1회 제한, 횡보장 감지 |")
    lines.append("| v4 | v3 + 조기진입(ADX+EMA+거래량), 서킷브레이커, 스윙 이벤트, 급락 역매수 |")
    lines.append("| v5 | v4 + Unix(VIX) 방어모드(IAU/GDXU), 진입 시작 시각 04:00, CB 강화 |")
    lines.append("")

    lines.append("## 관측 사항")
    lines.append("")

    # 에러 발생 버전 기록
    error_versions = [r["version"] for r in results if r.get("error")]
    ok_versions = [r for r in results if not r.get("error")]

    if error_versions:
        lines.append(f"- **실행 오류 발생 버전**: {', '.join(error_versions)}")
        for r in results:
            if r.get("error"):
                lines.append(f"  - {r['version']}: `{r['error']}`")
        lines.append("")

    if ok_versions:
        best = max(ok_versions, key=lambda r: r["return_pct"])
        lowest_mdd = min(ok_versions, key=lambda r: r["mdd"])
        best_sharpe = max(ok_versions, key=lambda r: r["sharpe"])
        best_wr = max(ok_versions, key=lambda r: r["win_rate"])

        lines.append(f"- **최고 수익률**: {best['version']} ({best['return_pct']:+.2f}%)")
        lines.append(f"- **최소 MDD**: {lowest_mdd['version']} ({lowest_mdd['mdd']:.2f}%)")
        lines.append(f"- **최고 Sharpe**: {best_sharpe['version']} ({best_sharpe['sharpe']:.4f})")
        lines.append(f"- **최고 승률**: {best_wr['version']} ({best_wr['win_rate']:.1f}%)")
        lines.append("")

        # 버전간 비교 코멘트
        v2 = next((r for r in ok_versions if r["version"] == "v2"), None)
        for ver in ["v3", "v4", "v5"]:
            vr = next((r for r in ok_versions if r["version"] == ver), None)
            if v2 and vr:
                delta = vr["return_pct"] - v2["return_pct"]
                lines.append(
                    f"- v2 대비 {ver}: 수익률 {delta:+.2f}%p, "
                    f"거래 수 {vr['total_sells'] - v2['total_sells']:+d}회, "
                    f"승률 {vr['win_rate'] - v2['win_rate']:+.1f}%p"
                )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by experiments/backtest_evaluation.py*")

    return "\n".join(lines)


# ============================================================
# 메인
# ============================================================

def main():
    from simulation.pipeline import run_backtest

    print("=" * 70)
    print("  Line A v2~v5 백테스트 비교 평가")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print("=" * 70)

    results: list[dict] = []

    for version in VERSIONS:
        print(f"\n[{version}] 백테스트 실행 중...")
        try:
            engine = run_backtest(
                version=version,
                start_date=START_DATE,
                end_date=END_DATE,
                use_fees=True,
                verbose=False,
            )
            metrics = extract_metrics(version, engine)
            results.append(metrics)
            print(f"  완료: 수익률 {metrics['return_pct']:+.2f}%, "
                  f"MDD {metrics['mdd']:.2f}%, "
                  f"Sharpe {metrics['sharpe']:.4f}, "
                  f"승률 {metrics['win_rate']:.1f}%")
        except Exception as e:
            err_msg = str(e)
            print(f"  [ERROR] {version}: {err_msg}")
            traceback.print_exc()
            results.append({
                "version": version, "currency": "USD",
                "initial": 0, "final": 0,
                "return_pct": 0.0, "mdd": 0.0, "sharpe": 0.0,
                "total_sells": 0, "total_buys": 0,
                "win_rate": 0.0, "win_count": 0, "loss_count": 0,
                "avg_win": 0.0, "avg_loss": 0.0, "total_pnl": 0.0,
                "stop_loss_count": 0, "total_fees": 0.0,
                "trading_days": 0,
                "sideways_days": 0, "sideways_blocks": 0,
                "entry_cutoff_blocks": 0, "daily_limit_blocks": 0,
                "cb_buy_blocks": 0,
                "error": err_msg,
            })

    # 콘솔 출력
    print("\n" + "=" * 70)
    print("  비교 결과")
    print("=" * 70)
    header = f"{'지표':<25} " + " ".join(f"{v:>12}" for v in VERSIONS)
    print(header)
    print("-" * 70)

    metrics_to_print = [
        ("총 수익률 (%)", "return_pct", "+.2f"),
        ("최대 낙폭 MDD (%)", "mdd", ".2f"),
        ("Sharpe Ratio", "sharpe", ".4f"),
        ("매도 횟수", "total_sells", "d"),
        ("승률 (%)", "win_rate", ".1f"),
        ("손절 횟수", "stop_loss_count", "d"),
        ("총 수수료 (USD)", "total_fees", ",.0f"),
        ("거래일", "trading_days", "d"),
    ]

    for label, key, fmt in metrics_to_print:
        vals = []
        for r in results:
            if r.get("error"):
                vals.append("ERROR".rjust(12))
            else:
                try:
                    vals.append(format(r[key], fmt).rjust(12))
                except (KeyError, TypeError, ValueError):
                    vals.append("N/A".rjust(12))
        print(f"{label:<25} " + " ".join(vals))

    # 보고서 저장
    report_md = build_report(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report_md, encoding="utf-8")
    print(f"\n보고서 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
