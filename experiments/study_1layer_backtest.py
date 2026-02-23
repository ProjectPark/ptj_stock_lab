#!/usr/bin/env python3
"""
Axis 1 백테스트 검증 — 1-layer 실패 패턴 필터 효과 측정
=========================================================
study_1layer_failures.py EDA 결과를 v2 백테스트로 검증.

EDA 핵심 발견:
  - BB < 0.20: 17건, 강제청산 0건 (100% 승률)
  - score ≥ 0.70: 38건, 실패 3건 (92% 승률, vs 기준 67%)

테스트 시나리오:
  baseline : v2 기준 (contrarian score≥0.65, bb_entry_max=0.60 보너스)
  F1       : contrarian entry threshold → score ≥ 0.70 (상향)
  F2       : BB 진입 하드 제한 → bb ≤ 0.30 만 허용
  F1+F2    : 두 필터 동시 적용

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_1layer_backtest.py
"""
from __future__ import annotations

import sys
from copy import deepcopy
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.strategies.line_c_d2s.d2s_engine import D2SEngine, DailySnapshot
from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE_V2
from simulation.backtests.backtest_d2s_v2 import D2SBacktestV2


# ============================================================
# StudyD2SEngine — 파라미터화된 진입 임계값 + BB 하드 필터
# ============================================================

class StudyD2SEngine(D2SEngine):
    """연구용 엔진 — score 임계값 파라미터화 + BB 하드 진입 제한.

    추가 파라미터 (params dict에 설정)
    ------------------------------------
    score_contrarian_min  float  역발상 단독 진입 임계값 (기본 0.65)
    score_riskoff_min     float  리스크오프 강제 진입 임계값 (기본 0.60)
    bb_entry_hard_max     float | None  BB %B 진입 상한 (None = 제한 없음)
    """

    def generate_daily_signals(
        self,
        snap: DailySnapshot,
        positions: dict,
        daily_buy_counts: dict,
    ) -> list[dict]:
        """score / BB 임계값을 파라미터로 제어하는 시그널 생성."""
        score_contrarian_min = self.p.get("score_contrarian_min", 0.65)
        score_riskoff_min    = self.p.get("score_riskoff_min", 0.60)
        bb_hard_max          = self.p.get("bb_entry_hard_max", None)

        signals = []

        # ── 청산 스캔 ──
        for ticker, pos in list(positions.items()):
            exit_ctx = self.check_exit(pos, snap)
            if exit_ctx["should_exit"]:
                signals.append({
                    "action": "SELL",
                    "ticker": ticker,
                    "size": 1.0,
                    "reason": exit_ctx["reason"],
                    "score": 0,
                })

        # ── 시황 필터 ──
        market_ctx = self.check_market_filter(snap)
        if market_ctx["blocked"]:
            return signals

        # ── 쌍둥이 갭 후보 ──
        gap_candidates = self.check_twin_gaps(snap)
        buy_candidates = []
        seen_tickers = set()

        for gc in gap_candidates:
            ticker = gc["follow"]
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            if daily_buy_counts.get(ticker, 0) >= self.p["dca_max_daily"]:
                continue

            # BB 하드 필터
            if bb_hard_max is not None:
                bb = snap.bb_pct_b.get(ticker)
                if bb is not None and bb > bb_hard_max:
                    continue

            tech_ctx = self.check_technical_filter(ticker, snap)
            if tech_ctx["blocked"]:
                continue

            quality = self.check_entry_quality(ticker, snap, market_ctx, tech_ctx)
            buy_candidates.append({
                "ticker": ticker,
                "source": f"gap:{gc['pair']}({gc['gap']:+.2f}%)",
                "score": quality["score"],
                "size_hint": quality["size_hint"],
                "reasons": quality["reasons"],
            })

        # ── 단독 역발상 진입 ──
        for ticker in self._tickers:
            if ticker in seen_tickers:
                continue
            if daily_buy_counts.get(ticker, 0) >= self.p["dca_max_daily"]:
                continue
            ticker_pct = snap.changes.get(ticker, 0.0)
            if ticker_pct >= self.p["contrarian_entry_threshold"]:
                continue

            # BB 하드 필터
            if bb_hard_max is not None:
                bb = snap.bb_pct_b.get(ticker)
                if bb is not None and bb > bb_hard_max:
                    continue

            tech_ctx = self.check_technical_filter(ticker, snap)
            if tech_ctx["blocked"]:
                continue

            quality = self.check_entry_quality(ticker, snap, market_ctx, tech_ctx)
            if quality["score"] >= score_contrarian_min:  # ← 파라미터화
                seen_tickers.add(ticker)
                buy_candidates.append({
                    "ticker": ticker,
                    "source": f"contrarian({ticker_pct:+.1f}%)",
                    "score": quality["score"],
                    "size_hint": quality["size_hint"],
                    "reasons": quality["reasons"],
                })

        # ── 리스크오프 강제 매수 ──
        if market_ctx.get("riskoff_boost") and not buy_candidates:
            for ticker in self._tickers:
                if ticker in seen_tickers:
                    continue
                if daily_buy_counts.get(ticker, 0) >= self.p["dca_max_daily"]:
                    continue

                # BB 하드 필터
                if bb_hard_max is not None:
                    bb = snap.bb_pct_b.get(ticker)
                    if bb is not None and bb > bb_hard_max:
                        continue

                tech_ctx = self.check_technical_filter(ticker, snap)
                if tech_ctx["blocked"]:
                    continue

                quality = self.check_entry_quality(ticker, snap, market_ctx, tech_ctx)
                if quality["score"] >= score_riskoff_min:  # ← 파라미터화
                    buy_candidates.append({
                        "ticker": ticker,
                        "source": "riskoff_forced",
                        "score": quality["score"],
                        "size_hint": quality["size_hint"],
                        "reasons": quality["reasons"],
                    })

        # ── 점수 순 정렬, 상위 3개 선택 ──
        buy_candidates.sort(key=lambda x: x["score"], reverse=True)
        for cand in buy_candidates[:3]:
            size = (
                self.p["buy_size_large"]
                if cand["size_hint"] == "large"
                else self.p["buy_size_small"]
            )
            signals.append({
                "action": "BUY",
                "ticker": cand["ticker"],
                "size": size,
                "reason": f"{cand['source']} | {', '.join(cand['reasons'])}",
                "score": cand["score"],
            })

        return signals


# ============================================================
# StudyD2SBacktestV2 — StudyD2SEngine 주입
# ============================================================

class StudyD2SBacktestV2(D2SBacktestV2):
    """연구용 v2 백테스트 — StudyD2SEngine 사용."""

    def __init__(self, params: dict, **kwargs):
        super().__init__(params=params, **kwargs)
        # 부모 __init__에서 생성된 D2SEngine을 StudyD2SEngine으로 교체
        self.engine = StudyD2SEngine(params)


# ============================================================
# 시나리오 정의
# ============================================================

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "label": "v2 기준 (score≥0.65, bb_bonus≤0.60)",
        "params": {},
    },
    "F1_score070": {
        "label": "F1: score ≥ 0.70 (contrarian/riskoff threshold 상향)",
        "params": {
            "score_contrarian_min": 0.70,
            "score_riskoff_min": 0.65,
        },
    },
    "F2_bb030": {
        "label": "F2: BB ≤ 0.30 하드 제한",
        "params": {
            "bb_entry_hard_max": 0.30,
        },
    },
    "F1_F2": {
        "label": "F1+F2: score ≥ 0.70 + BB ≤ 0.30",
        "params": {
            "score_contrarian_min": 0.70,
            "score_riskoff_min": 0.65,
            "bb_entry_hard_max": 0.30,
        },
    },
}


# ============================================================
# 시나리오 실행
# ============================================================

def run_scenario(
    name: str,
    config: dict,
    start_date: date = date(2025, 3, 3),
    end_date: date = date(2026, 2, 17),
) -> dict:
    """단일 시나리오를 실행하고 결과 dict를 반환한다."""
    params = deepcopy(D2S_ENGINE_V2)
    params.update(config["params"])

    bt = StudyD2SBacktestV2(
        params=params,
        start_date=start_date,
        end_date=end_date,
        use_fees=True,
    )
    bt.run(verbose=False)
    r = bt.report()

    # 강제청산 건수 카운트
    forced_liq = sum(
        1 for t in bt.trades
        if t.side == "SELL" and (
            "forced" in t.reason.lower()
            or "force" in t.reason.lower()
            or "강제" in t.reason
        )
    )

    # 1-layer 승/패 분리 (dca_count==1 포지션)
    buy_scores = [t.score for t in bt.trades if t.side == "BUY"]
    avg_buy_score = sum(buy_scores) / len(buy_scores) if buy_scores else 0.0

    return {
        "scenario": name,
        "label": config["label"],
        "total_return": r["total_return_pct"],
        "win_rate": r["win_rate"],
        "mdd": r["mdd_pct"],
        "sharpe": r["sharpe_ratio"],
        "total_trades": r["total_trades"],
        "buy_trades": r["buy_trades"],
        "sell_trades": r["sell_trades"],
        "win_trades": r["win_trades"],
        "lose_trades": r["lose_trades"],
        "avg_pnl": r["avg_pnl_pct"],
        "total_pnl": r["total_pnl"],
        "forced_liquidations": forced_liq,
        "r17_count": bt._r17_count,
        "r18_count": bt._r18_count,
        "avg_buy_score": avg_buy_score,
    }


# ============================================================
# 비교 출력
# ============================================================

def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  Axis 1 백테스트 검증 — score / BB 하드 필터 효과")
    print("  (v2 백테스트 엔진 기반, 2025-03-03 ~ 2026-02-17)")
    print("=" * 80)

    hdr = (
        f"  {'시나리오':<16} {'수익률':>8} {'승률':>6} {'MDD':>7}"
        f" {'Sharpe':>7} {'매수':>5} {'평균PnL':>8} {'강제청산':>6} {'Δ수익률':>8}"
    )
    print(hdr)
    print("  " + "-" * 76)

    baseline = results[0]
    for r in results:
        delta = r["total_return"] - baseline["total_return"]
        delta_str = f"{delta:+.2f}%p" if r["scenario"] != "baseline" else "  —"
        print(
            f"  {r['scenario']:<16}"
            f" {r['total_return']:>+7.2f}%"
            f" {r['win_rate']:>5}%"
            f" {r['mdd']:>6.1f}%"
            f" {r['sharpe']:>7.3f}"
            f" {r['buy_trades']:>5}건"
            f" {r['avg_pnl']:>+7.2f}%"
            f" {r['forced_liquidations']:>5}건"
            f" {delta_str:>8}"
        )

    print("  " + "=" * 76)

    # 상세 해설
    print("\n[시나리오별 변화]")
    b = baseline
    for r in results[1:]:
        ret_d = r["total_return"] - b["total_return"]
        buy_d = r["buy_trades"] - b["buy_trades"]
        wr_d  = r["win_rate"] - b["win_rate"]
        fl_d  = r["forced_liquidations"] - b["forced_liquidations"]
        print(
            f"  {r['scenario']:>12}: "
            f"수익률 {ret_d:+.2f}%p │ "
            f"매수 {buy_d:+d}건 │ "
            f"승률 {wr_d:+.1f}%p │ "
            f"강제청산 {fl_d:+d}건 │ "
            f"R17 {r['r17_count']}회 R18 {r['r18_count']}회"
        )

    print("\n[EDA 예측 vs 실제]")
    print("  EDA 예측: score ≥ 0.70 → 92% 승률 (17→38건 중 3실패)")
    print("  EDA 예측: BB ≤ 0.30 → 100% 승률 (17건 0실패)")
    for r in results[1:]:
        print(f"  {r['scenario']:>12}: 실제 승률 {r['win_rate']}%, 강제청산 {r['forced_liquidations']}건")


# ============================================================
# 메인
# ============================================================

# ============================================================
# IS / OOS 기간 정의
# EDA 기반: IS = 전체 기간 전반부, OOS = 후반부
# ============================================================

# 전체 백테스트 범위: 2025-03-03 ~ 2026-02-17
# IS:  2025-03-03 ~ 2025-09-30  (7개월, EDA 파생 기간과 동일)
# OOS: 2025-10-01 ~ 2026-02-17  (4.5개월, 미래 검증)

DATE_FULL_START = date(2025, 3, 3)
DATE_FULL_END   = date(2026, 2, 17)
DATE_IS_END     = date(2025, 9, 30)
DATE_OOS_START  = date(2025, 10, 1)


def run_period_set(label: str, start: date, end: date) -> list[dict]:
    results = []
    for name, config in SCENARIOS.items():
        r = run_scenario(name, config, start_date=start, end_date=end)
        r["period_label"] = label
        results.append(r)
        print(
            f"    [{label}] {name}: 수익률 {r['total_return']:+.2f}%, "
            f"승률 {r['win_rate']}%, MDD {r['mdd']:.1f}%, 매수 {r['buy_trades']}건"
        )
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  Axis 1 백테스트 검증 — IS / OOS 분리 실행")
    print("=" * 70)
    print(f"  IS  기간: {DATE_FULL_START} ~ {DATE_IS_END}")
    print(f"  OOS 기간: {DATE_OOS_START} ~ {DATE_FULL_END}")
    print(f"  전체기간: {DATE_FULL_START} ~ {DATE_FULL_END}")
    print()

    print("▶ [전체 기간 실행]")
    results_full = run_period_set("FULL", DATE_FULL_START, DATE_FULL_END)
    print()

    print("▶ [IS 기간 실행]")
    results_is = run_period_set("IS", DATE_FULL_START, DATE_IS_END)
    print()

    print("▶ [OOS 기간 실행]")
    results_oos = run_period_set("OOS", DATE_OOS_START, DATE_FULL_END)
    print()

    # 비교 출력
    print_comparison(results_full)

    print("\n" + "=" * 80)
    print("  IS / OOS 비교 — F2_bb030 중심")
    print("=" * 80)
    hdr = f"  {'기간':<6} {'시나리오':<16} {'수익률':>8} {'승률':>6} {'MDD':>7} {'Sharpe':>7} {'매수':>5} {'Δ수익률':>8}"
    print(hdr)
    print("  " + "-" * 72)

    for results, tag in [(results_is, "IS"), (results_oos, "OOS"), (results_full, "FULL")]:
        baseline_ret = results[0]["total_return"]
        for r in results:
            delta = r["total_return"] - baseline_ret
            delta_str = f"{delta:+.2f}%p" if r["scenario"] != "baseline" else "  —"
            print(
                f"  {tag:<6} {r['scenario']:<16}"
                f" {r['total_return']:>+7.2f}%"
                f" {r['win_rate']:>5}%"
                f" {r['mdd']:>6.1f}%"
                f" {r['sharpe']:>7.3f}"
                f" {r['buy_trades']:>5}건"
                f" {delta_str:>8}"
            )
        print("  " + "-" * 72)
