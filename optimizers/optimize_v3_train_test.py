#!/usr/bin/env python3
"""
PTJ v3 — Train/Test 분리 최적화
=================================
Train 기간에서 파라미터 최적화 후 Test 기간에서 검증

Usage:
    pyenv shell market && python optimize_v3_train_test.py --n-trials 300 --n-jobs 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

# ── 경로 상수 ─────────────────────────────────────────────────

DOCS_DIR = Path(__file__).resolve().parent / "docs"
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_TEST_REPORT = DOCS_DIR / "v3_train_test_report.md"

# ── Train/Test 기간 설정 ──────────────────────────────────────

TRAIN_START = date(2025, 1, 3)
TRAIN_END = date(2025, 12, 31)
TEST_START = date(2026, 1, 1)
TEST_END = date(2026, 2, 17)


# ── 백테스트 실행 함수 ─────────────────────────────────────────

def _run_backtest(params: dict, start_date: date, end_date: date) -> dict:
    """지정된 기간에 대해 백테스트를 실행하고 결과를 반환한다."""
    import config
    import backtest_common
    from backtest_v3 import BacktestEngineV3

    # 파라미터 임시 적용
    originals = {}
    for key, value in params.items():
        if hasattr(config, key):
            originals[key] = getattr(config, key)
            setattr(config, key, value)

    try:
        engine = BacktestEngineV3(start_date=start_date, end_date=end_date)
        engine.run(verbose=False)

        initial = engine.initial_capital_krw
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial
        total_ret = (final - initial) / initial * 100
        mdd = backtest_common.calc_mdd(engine.equity_curve)
        sharpe = backtest_common.calc_sharpe(engine.equity_curve)

        # 매수/매도 카운트
        buys = [t for t in engine.trades if t.side == "BUY"]
        sells = [t for t in engine.trades if t.side == "SELL"]

        # 승률 계산 (매도 기준)
        win_count = sum(1 for t in sells if t.pnl_pct > 0)
        total_sells = len(sells)
        win_rate = (win_count / total_sells * 100) if total_sells > 0 else 0

        # 손절 카운트
        stop_loss_count = sum(1 for t in sells if "손절" in t.exit_reason)
        time_stop_count = sum(1 for t in sells if "시간" in t.exit_reason)

        return {
            "return_pct": total_ret,
            "mdd": mdd,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "total_buys": len(buys),
            "total_sells": total_sells,
            "total_trades": total_sells,  # backward compat
            "stop_loss_count": stop_loss_count,
            "time_stop_count": time_stop_count,
            "sideways_days": engine.sideways_days,
        }
    finally:
        # 원래 값 복원
        for key, value in originals.items():
            setattr(config, key, value)


# ── Optuna Objective ──────────────────────────────────────────


class TrainTestObjective:
    """Train 기간에서 최적화, Test 기간에서 검증 (매 Trial마다)."""

    def __init__(self, gap_max: float = 10.0):
        self.gap_max = gap_max
        self.trial_count = 0

    def __call__(self, trial: optuna.Trial) -> float:
        """Train 기간 수익률을 반환 (최대화 목표). Test 결과는 user_attrs에 저장."""
        self.trial_count += 1

        # 넓은 범위 탐색 (과최적화 방지)
        params = {
            # GAP 임계값: 매우 넓은 범위
            "V3_PAIR_GAP_ENTRY_THRESHOLD": trial.suggest_float(
                "V3_PAIR_GAP_ENTRY_THRESHOLD", 1.0, self.gap_max, step=0.5
            ),
            # DCA 설정: 1~10회
            "V3_DCA_MAX_COUNT": trial.suggest_int("V3_DCA_MAX_COUNT", 1, 10),
            # 종목당 투자: 300만~1000만원
            "V3_MAX_PER_STOCK_KRW": trial.suggest_int(
                "V3_MAX_PER_STOCK_KRW", 3_000_000, 10_000_000, step=1_000_000
            ),
            # 조건부 트리거: 2~10%
            "V3_COIN_TRIGGER_PCT": trial.suggest_float(
                "V3_COIN_TRIGGER_PCT", 2.0, 10.0, step=0.5
            ),
            "V3_CONL_TRIGGER_PCT": trial.suggest_float(
                "V3_CONL_TRIGGER_PCT", 2.0, 10.0, step=0.5
            ),
            # 분할매수 간격: 5~60분
            "V3_SPLIT_BUY_INTERVAL_MIN": trial.suggest_int(
                "V3_SPLIT_BUY_INTERVAL_MIN", 5, 60, step=5
            ),
            # 진입 시간 제한
            "V3_ENTRY_CUTOFF_HOUR": trial.suggest_int("V3_ENTRY_CUTOFF_HOUR", 10, 14),
            "V3_ENTRY_CUTOFF_MINUTE": trial.suggest_int(
                "V3_ENTRY_CUTOFF_MINUTE", 0, 30, step=30
            ),
            # 횡보장 감지: 넓은 범위
            "V3_SIDEWAYS_MIN_SIGNALS": trial.suggest_int(
                "V3_SIDEWAYS_MIN_SIGNALS", 1, 5
            ),
            "V3_SIDEWAYS_POLY_LOW": trial.suggest_float(
                "V3_SIDEWAYS_POLY_LOW", 0.2, 0.5, step=0.05
            ),
            "V3_SIDEWAYS_POLY_HIGH": trial.suggest_float(
                "V3_SIDEWAYS_POLY_HIGH", 0.4, 0.7, step=0.05
            ),
            "V3_SIDEWAYS_GLD_THRESHOLD": trial.suggest_float(
                "V3_SIDEWAYS_GLD_THRESHOLD", 0.1, 0.7, step=0.1
            ),
            "V3_SIDEWAYS_INDEX_THRESHOLD": trial.suggest_float(
                "V3_SIDEWAYS_INDEX_THRESHOLD", 0.3, 1.0, step=0.1
            ),
            # 손절: -10% ~ -1%
            "STOP_LOSS_PCT": trial.suggest_float(
                "STOP_LOSS_PCT", -10.0, -1.0, step=0.5
            ),
            "STOP_LOSS_BULLISH_PCT": trial.suggest_float(
                "STOP_LOSS_BULLISH_PCT", -20.0, -5.0, step=1.0
            ),
            # 익절: 1~10%
            "COIN_SELL_PROFIT_PCT": trial.suggest_float(
                "COIN_SELL_PROFIT_PCT", 1.0, 10.0, step=0.5
            ),
            "CONL_SELL_PROFIT_PCT": trial.suggest_float(
                "CONL_SELL_PROFIT_PCT", 1.0, 8.0, step=0.5
            ),
            # DCA 하락폭: -3% ~ -0.2%
            "DCA_DROP_PCT": trial.suggest_float(
                "DCA_DROP_PCT", -3.0, -0.2, step=0.1
            ),
            # 보유 시간: 1~8시간
            "MAX_HOLD_HOURS": trial.suggest_int("MAX_HOLD_HOURS", 1, 8),
            # 익절 목표: 1~8%
            "TAKE_PROFIT_PCT": trial.suggest_float(
                "TAKE_PROFIT_PCT", 1.0, 8.0, step=0.5
            ),
            # 페어 GAP 청산: 0.5 ~ max
            "PAIR_GAP_SELL_THRESHOLD_V2": trial.suggest_float(
                "PAIR_GAP_SELL_THRESHOLD_V2", 0.5, self.gap_max, step=0.5
            ),
            # 분할 청산 비율: 0.5 ~ 1.0
            "PAIR_SELL_FIRST_PCT": trial.suggest_float(
                "PAIR_SELL_FIRST_PCT", 0.5, 1.0, step=0.05
            ),
        }

        # Train 기간 평가 (최적화 목표)
        train_result = _run_backtest(params, TRAIN_START, TRAIN_END)

        # Test 기간 평가 (검증용)
        test_result = _run_backtest(params, TEST_START, TEST_END)

        # Test 결과를 trial attributes에 저장
        trial.set_user_attr("test_return", test_result["return_pct"])
        trial.set_user_attr("test_mdd", test_result["mdd"])
        trial.set_user_attr("test_sharpe", test_result["sharpe"])
        trial.set_user_attr("test_win_rate", test_result["win_rate"])
        trial.set_user_attr("train_return", train_result["return_pct"])
        degradation = train_result["return_pct"] - test_result["return_pct"]
        trial.set_user_attr("degradation", degradation)

        # 매 Trial마다 Train/Test 결과 출력
        import sys

        print(
            f"[T{trial.number:3d}] "
            f"Train: {train_result['return_pct']:+6.2f}% | "
            f"Test: {test_result['return_pct']:+6.2f}% | "
            f"차이: {degradation:+5.2f}%p",
            flush=True,
        )
        sys.stdout.flush()

        return train_result["return_pct"]


# ── 메인 실행 ────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PTJ v3 Train/Test 최적화")
    parser.add_argument("--n-trials", type=int, default=300, help="Trial 수")
    parser.add_argument("--n-jobs", type=int, default=10, help="병렬 Worker 수")
    parser.add_argument("--gap-max", type=float, default=10.0, help="GAP 최대값")
    parser.add_argument(
        "--study-name", type=str, default="ptj_v3_train_test", help="Study 이름"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///data/optuna_v3_train_test.db",
        help="Optuna DB",
    )
    args = parser.parse_args()

    n_trials = args.n_trials
    n_jobs = args.n_jobs
    gap_max = args.gap_max
    study_name = args.study_name
    storage = args.db

    print("=" * 70)
    print("  PTJ v3 — Train/Test 분리 최적화")
    print("=" * 70)
    print(f"  Train 기간: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  Test 기간:  {TEST_START} ~ {TEST_END}")
    print()
    print("=" * 70)
    print(f"  Optuna 최적화 ({n_trials} trials, {n_jobs} workers)")
    print("=" * 70)

    # 콜백: 매 Trial마다 Train/Test 결과 출력
    def _log_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            train_ret = trial.user_attrs.get("train_return", trial.value)
            test_ret = trial.user_attrs.get("test_return", 0)
            degradation = trial.user_attrs.get("degradation", 0)
            print(
                f"[Trial {trial.number:3d}] "
                f"Train: {train_ret:+6.2f}% | "
                f"Test: {test_ret:+6.2f}% | "
                f"차이: {degradation:+5.2f}%p"
            )

    # Optuna Study 생성
    sampler = TPESampler(seed=42, multivariate=True, group=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    start_time = time.time()

    # 병렬 실행: subprocess 방식
    if n_jobs > 1:
        processes = []
        for i in range(n_jobs):
            if i == 0:
                continue  # 메인 프로세스가 worker 0 역할
            cmd = [
                sys.executable,
                "-c",
                f"""
import optuna
from optimize_v3_train_test import TrainTestObjective

study = optuna.load_study(study_name='{study_name}', storage='{storage}')
objective = TrainTestObjective(gap_max={gap_max})
study.optimize(objective, n_trials={n_trials // n_jobs + 1}, show_progress_bar=False)
""",
            ]
            p = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
            processes.append(p)

        # 메인 프로세스도 최적화 참여
        objective = TrainTestObjective(gap_max=gap_max)
        study.optimize(
            objective,
            n_trials=n_trials // n_jobs,
            show_progress_bar=False,
            callbacks=[_log_callback],
        )

        # 모든 워커 종료 대기
        for p in processes:
            p.wait()
    else:
        # 단일 프로세스
        objective = TrainTestObjective(gap_max=gap_max)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    elapsed = time.time() - start_time

    # 최적 Trial 가져오기
    best_trial = study.best_trial
    best_params = best_trial.params

    print()
    print("=" * 70)
    print("  Train 기간 최적화 완료")
    print("=" * 70)
    print(f"  실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print(f"  Trial당 평균: {elapsed/n_trials:.1f}초")
    print()
    print(f"  BEST Trial #{best_trial.number}")
    print(f"  Train 수익률: {best_trial.value:+.2f}%")
    print()

    # Test 기간 결과 (이미 계산됨 - user_attrs에서 가져오기)
    print("=" * 70)
    print("  Test 기간 검증")
    print("=" * 70)
    test_return = best_trial.user_attrs.get("test_return", 0)
    test_mdd = best_trial.user_attrs.get("test_mdd", 0)
    test_sharpe = best_trial.user_attrs.get("test_sharpe", 0)
    test_win_rate = best_trial.user_attrs.get("test_win_rate", 0)
    degradation = best_trial.user_attrs.get("degradation", 0)

    print(f"  Test 수익률: {test_return:+.2f}%")
    print(f"  Test MDD:    {test_mdd:.2f}%")
    print(f"  Test Sharpe: {test_sharpe:.4f}")
    print(f"  Test 승률:   {test_win_rate:.1f}%")
    print(f"  성능 차이:   {degradation:+.2f}%p")
    print()

    # Train 기간 상세 결과 (재실행)
    train_result = _run_backtest(best_params, TRAIN_START, TRAIN_END)

    # Test 기간 상세 결과 (재실행)
    test_result = _run_backtest(best_params, TEST_START, TEST_END)

    # Top 5 trials 정보 (이미 계산된 Test 결과 사용)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
    top_test_results = []
    for trial in top_trials:
        top_test_results.append(
            {
                "trial_number": trial.number,
                "train_return": trial.value,
                "test_return": trial.user_attrs.get("test_return", 0),
                "test_mdd": trial.user_attrs.get("test_mdd", 0),
                "test_sharpe": trial.user_attrs.get("test_sharpe", 0),
                "params": trial.params,
            }
        )

    # 리포트 생성
    _generate_report(
        train_result=train_result,
        test_result=test_result,
        best_params=best_params,
        best_trial_number=best_trial.number,
        top_test_results=top_test_results,
        n_trials=n_trials,
        elapsed=elapsed,
    )

    print(f"  리포트 저장: {TRAIN_TEST_REPORT}")
    print()
    print("  완료!")


def _generate_report(
    train_result: dict,
    test_result: dict,
    best_params: dict,
    best_trial_number: int,
    top_test_results: list[dict],
    n_trials: int,
    elapsed: float,
):
    """Train/Test 리포트를 마크다운으로 생성한다."""
    TRAIN_TEST_REPORT.parent.mkdir(exist_ok=True)

    lines = [
        "# PTJ v3 Train/Test 분리 최적화 리포트",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 📊 데이터 기간",
        "",
        "| 구분 | 기간 | 용도 |",
        "|---|---|---|",
        f"| **Train** | {TRAIN_START} ~ {TRAIN_END} | 파라미터 최적화 |",
        f"| **Test** | {TEST_START} ~ {TEST_END} | 성능 검증 (Out-of-Sample) |",
        "",
        "---",
        "",
        "## 1. 실행 정보",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 총 Trial | {n_trials} |",
        f"| 실행 시간 | {elapsed:.1f}초 ({elapsed/60:.1f}분) |",
        f"| Trial당 평균 | {elapsed/n_trials:.1f}초 |",
        "| Sampler | TPE (seed=42) |",
        "",
        "---",
        "",
        "## 2. Train vs Test 성능 비교 (Best Trial)",
        "",
        "| 지표 | Train | Test | 차이 |",
        "|---|---|---|---|",
        f"| **수익률** | **{train_result['return_pct']:+.2f}%** | **{test_result['return_pct']:+.2f}%** | {test_result['return_pct'] - train_result['return_pct']:+.2f}%p |",
        f"| MDD | {train_result['mdd']:.2f}% | {test_result['mdd']:.2f}% | {test_result['mdd'] - train_result['mdd']:+.2f}%p |",
        f"| Sharpe | {train_result['sharpe']:.4f} | {test_result['sharpe']:.4f} | {test_result['sharpe'] - train_result['sharpe']:+.4f} |",
        f"| 승률 | {train_result['win_rate']:.1f}% | {test_result['win_rate']:.1f}% | {test_result['win_rate'] - train_result['win_rate']:+.1f}%p |",
        f"| 매수 횟수 | {train_result['total_buys']} | {test_result['total_buys']} | - |",
        f"| 매도 횟수 | {train_result['total_sells']} | {test_result['total_sells']} | - |",
        f"| 손절 횟수 | {train_result['stop_loss_count']} | {test_result['stop_loss_count']} | - |",
        f"| 횡보일 | {train_result['sideways_days']} | {test_result['sideways_days']} | - |",
        "",
        "### 📈 과최적화 평가",
        "",
    ]

    # 과최적화 평가
    train_ret = train_result["return_pct"]
    test_ret = test_result["return_pct"]
    degradation = train_ret - test_ret

    if degradation < 2:
        verdict = "✅ **우수**: Test 성능이 Train과 유사 → 강건한 전략"
    elif degradation < 5:
        verdict = "⚠️ **주의**: Test 성능이 소폭 하락 → 모니터링 필요"
    else:
        verdict = "❌ **과최적화**: Test 성능이 크게 하락 → 파라미터 재조정 권장"

    lines.extend(
        [
            f"- Train 수익률: **{train_ret:+.2f}%**",
            f"- Test 수익률: **{test_ret:+.2f}%**",
            f"- 성능 하락: **{degradation:+.2f}%p**",
            "",
            f"**{verdict}**",
            "",
            "---",
            "",
            "## 3. Top 5 Trials - Test 성능",
            "",
            "| # | Train 수익률 | Test 수익률 | Test MDD | Test Sharpe | 성능 차이 |",
            "|---|---|---|---|---|---|",
        ]
    )

    for res in top_test_results:
        diff = res["train_return"] - res["test_return"]
        lines.append(
            f"| {res['trial_number']} | {res['train_return']:+.2f}% | {res['test_return']:+.2f}% | {res['test_mdd']:.2f}% | {res['test_sharpe']:.4f} | {diff:+.2f}%p |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 4. 최적 파라미터 (Trial #{best_trial_number})",
            "",
            "```python",
        ]
    )

    for key, value in sorted(best_params.items()):
        if isinstance(value, float):
            lines.append(f"{key} = {value:.2f}")
        else:
            lines.append(f"{key} = {value}")

    lines.extend(
        [
            "```",
            "",
            "---",
            "",
            "## 5. 결론",
            "",
            "### ✅ 강점",
            f"- Train 기간 수익률: **{train_ret:+.2f}%**",
            f"- Test 기간 검증: **{test_ret:+.2f}%**",
            f"- Out-of-Sample 검증 완료",
            "",
            "### ⚠️ 주의사항",
            "- Test 기간이 짧음 (약 1.5개월) → 추가 검증 권장",
            "- 시장 환경 변화에 따른 전략 재조정 필요",
            "",
            "### 📋 다음 단계",
            "1. 2026년 2월 18일 이후 데이터로 Forward Test",
            "2. 파라미터 민감도 분석",
            "3. Paper Trading 1개월 실시",
            "",
        ]
    )

    TRAIN_TEST_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
