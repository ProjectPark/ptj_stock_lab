#!/usr/bin/env python3
"""
Phase 2 — attach v1 규칙 기여도 분석 (Ablation)
=================================================
attach v1의 핵심 규칙 그룹을 하나씩 OFF 했을 때 성과 변화를 측정.
"어떤 규칙이 실거래 재현에 가장 크게 기여하는가"

시나리오 (8개):
  1. baseline       : 모든 규칙 ON (D2S_ENGINE 기본값)
  2. no_market_score: 0차 게이트 OFF (Study C)
  3. no_riskoff     : R14 리스크오프 역발상 OFF
  4. no_ticker_diff : Study B 종목별 역발상 차별화 OFF
  5. no_tech_filter : R7(RSI)/R8(BB)/R9(MACD) 기술적 필터 OFF
  6. no_calendar    : R15 금요일 부스트 OFF
  7. no_gld         : R1 GLD 매수 억제 OFF
  8. no_spy_streak  : R13 SPY 연속 상승 금지 OFF

기간: WARM (2025-03-03 ~ 2026-02-12) — 기술적 지표 워밍업 확보
실거래 목표: 승률 65.5%, 평균 +7.39%

Usage:
    pyenv shell ptj_stock_lab && python experiments/study_ablation_rules.py
    pyenv shell ptj_stock_lab && python experiments/study_ablation_rules.py --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import date
from multiprocessing import Pool
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from simulation.strategies.line_c_d2s.params_d2s import D2S_ENGINE

# ── 기간 설정 ──────────────────────────────────────────────────
START_DATE = date(2025, 3, 3)
END_DATE   = date(2026, 2, 12)

# ── 실거래 목표 ────────────────────────────────────────────────
ACTUAL_WIN_RATE = 65.5
ACTUAL_AVG_PNL  = 7.39

RESULTS_DIR = _PROJECT_ROOT / "data" / "results" / "analysis"
N_JOBS = int(os.environ.get("N_JOBS", "8"))

# ============================================================
# 시나리오 정의
# ============================================================

def build_scenarios() -> dict[str, dict]:
    base = deepcopy(D2S_ENGINE)

    scenarios = {}

    # 1. 기준선
    scenarios["baseline"] = deepcopy(base)

    # 2. 0차 게이트 OFF — market_score 게이트를 0으로 낮춰 사실상 비활성화
    s = deepcopy(base)
    s["market_score_suppress"] = 0.0
    s["market_score_entry_b"]  = 0.0
    s["market_score_entry_a"]  = 0.0
    scenarios["no_market_score"] = s

    # 3. R14 리스크오프 OFF
    s = deepcopy(base)
    s["riskoff_gld_up_spy_down"] = False
    scenarios["no_riskoff"] = s

    # 4. Study B 종목별 역발상 차별화 OFF (모두 기본 역발상으로)
    s = deepcopy(base)
    s["mstu_riskoff_contrarian_only"]   = False
    s["robn_riskoff_momentum_boost"]    = False
    s["conl_contrarian_require_riskoff"] = False
    s["amdl_friday_contrarian_threshold"] = base["contrarian_entry_threshold"]
    scenarios["no_ticker_diff"] = s

    # 5. R7~R9 기술적 필터 OFF — RSI/BB/Vol 범위 최대 개방
    s = deepcopy(base)
    s["rsi_entry_min"]   = 0
    s["rsi_entry_max"]   = 100
    s["rsi_danger_zone"] = 100
    s["bb_entry_max"]    = 2.0
    s["bb_danger_zone"]  = 2.0
    s["vol_entry_min"]   = 0.0
    s["vol_entry_max"]   = 99.9
    scenarios["no_tech_filter"] = s

    # 6. R15 금요일 부스트 OFF
    s = deepcopy(base)
    s["friday_boost"] = False
    scenarios["no_calendar"] = s

    # 7. R1 GLD 매수 억제 OFF (임계값을 극단값으로)
    s = deepcopy(base)
    s["gld_suppress_threshold"] = 99.9
    scenarios["no_gld"] = s

    # 8. R13 SPY 연속 상승 금지 OFF
    s = deepcopy(base)
    s["spy_streak_max"] = 999
    scenarios["no_spy_streak"] = s

    return scenarios


# ============================================================
# 단일 시나리오 실행 (Pool 호환)
# ============================================================

def _run_scenario(args: tuple) -> dict:
    name, params = args

    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from datetime import date as _date
    from simulation.backtests.backtest_d2s import D2SBacktest

    bt = D2SBacktest(
        params=params,
        start_date=_date(2025, 3, 3),
        end_date=_date(2026, 2, 12),
        use_fees=True,
    )
    bt.run(verbose=False)
    rpt = bt.report()
    rpt["scenario"] = name
    return rpt


# ============================================================
# 비교표 출력
# ============================================================

SCENARIO_LABELS = {
    "baseline":       "기준선 (모든 규칙 ON)",
    "no_market_score":"0차 게이트 OFF (Study C)",
    "no_riskoff":     "R14 리스크오프 OFF",
    "no_ticker_diff": "Study B 종목차별화 OFF",
    "no_tech_filter": "R7~R9 기술적 필터 OFF",
    "no_calendar":    "R15 금요일 부스트 OFF",
    "no_gld":         "R1 GLD 억제 OFF",
    "no_spy_streak":  "R13 SPY streak OFF",
}

SCENARIO_ORDER = list(SCENARIO_LABELS.keys())


def print_ablation_table(results: list[dict]) -> None:
    result_map = {r["scenario"]: r for r in results}
    base = result_map.get("baseline", {})
    base_wr  = base.get("win_rate", 0)
    base_pnl = base.get("avg_pnl_pct", 0)
    base_ret = base.get("total_return_pct", 0)

    print(f"\n{'=' * 85}")
    print("  Phase 2 — attach v1 규칙 기여도 분석 (Ablation)")
    print(f"  실거래 목표: 승률 {ACTUAL_WIN_RATE}%  평균PnL {ACTUAL_AVG_PNL}%")
    print(f"{'=' * 85}")
    print(f"  {'시나리오':<28}  {'승률':>6}  {'Δwr':>6}  "
          f"{'평균%':>6}  {'Δpnl':>6}  {'수익률':>7}  {'거래':>5}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}")

    for key in SCENARIO_ORDER:
        r = result_map.get(key)
        if not r:
            continue
        label = SCENARIO_LABELS[key]
        wr    = r.get("win_rate", 0)
        pnl   = r.get("avg_pnl_pct", 0)
        ret   = r.get("total_return_pct", 0)
        sells = r.get("sell_trades", 0)
        d_wr  = wr - base_wr
        d_pnl = pnl - base_pnl
        mark  = " ◀" if key == "baseline" else ""
        print(f"  {label:<28}  {wr:>6.1f}  {d_wr:>+6.1f}  "
              f"{pnl:>6.2f}  {d_pnl:>+6.2f}  {ret:>7.2f}%  {sells:>5}{mark}")

    print(f"\n  실거래 목표{'':16}  {ACTUAL_WIN_RATE:>6.1f}  {'':>6}  "
          f"{ACTUAL_AVG_PNL:>6.2f}  {'':>6}  {'':>7}  {'':>5}")
    print(f"{'=' * 85}")

    # 규칙 기여 순위
    ranked = []
    for key in SCENARIO_ORDER:
        if key == "baseline":
            continue
        r = result_map.get(key)
        if not r:
            continue
        delta_wr = base_wr - r.get("win_rate", 0)  # OFF 시 하락폭 = 기여도
        ranked.append((key, delta_wr, r.get("win_rate", 0)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    print("\n  규칙 기여도 순위 (OFF 시 승률 하락폭 기준):")
    for i, (key, delta, wr_off) in enumerate(ranked, 1):
        direction = "▼" if delta > 0 else "▲"
        print(f"    {i}. {SCENARIO_LABELS[key]:<28}  "
              f"승률 {wr_off:.1f}% ({direction}{abs(delta):.1f}%p)")


# ============================================================
# 메인
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="attach v1 Ablation 분석")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    scenarios = build_scenarios()
    tasks = list(scenarios.items())

    print("=" * 65)
    print("  Phase 2 — attach v1 규칙 기여도 분석 (Ablation)")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  시나리오: {len(tasks)}개  병렬 workers: {args.n_jobs}")
    print("=" * 65)

    n_workers = min(args.n_jobs, len(tasks))
    with Pool(n_workers) as pool:
        results = pool.map(_run_scenario, tasks)

    print_ablation_table(results)

    # 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "d2s_v1_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  [저장] {out_path}")


if __name__ == "__main__":
    main()
