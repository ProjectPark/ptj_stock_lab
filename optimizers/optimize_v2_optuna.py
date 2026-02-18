#!/usr/bin/env python3
"""
PTJ v2 — Optuna 파라미터 최적화 (개선판)
==========================================
개선점:
  1. Walk-forward: Train(180일) 최적화 → Test(60일) 검증 — 과적합 방지
  2. 파라미터 축소: 14개 → 5개 (fANOVA 중요도 기반)
  3. Multi-objective: 수익률 ↑ + MDD ↓ (NSGA-II Pareto front)

Usage:
    pyenv shell market && python optimize_v2_optuna.py --n-trials 100 --n-jobs 10
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import os
import sys
import time
from datetime import date
from io import StringIO

import optuna
from optuna.samplers import NSGAIISampler


# ── Walk-forward 기간 설정 ─────────────────────────────────────────
TRAIN_START = date(2025, 2, 18)
TRAIN_END   = date(2025, 11, 14)   # ~180 거래일
TEST_START  = date(2025, 11, 15)
TEST_END    = date(2026, 1, 30)    # ~60 거래일


# ── stdout 억제 ────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_stdout():
    """데이터 로딩 출력을 억제한다."""
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old


# ── 탐색 공간 (5개 핵심 파라미터) ──────────────────────────────────

def define_search_space(trial: optuna.Trial) -> dict:
    """확장된 7개 파라미터 탐색 공간."""
    return {
        # 조건부 매매 트리거 (확장: 1.0~12.0%)
        "COIN_TRIGGER_PCT": trial.suggest_float(
            "COIN_TRIGGER_PCT", 1.0, 12.0, step=0.5
        ),
        "CONL_TRIGGER_PCT": trial.suggest_float(
            "CONL_TRIGGER_PCT", 1.0, 12.0, step=0.5
        ),
        # 손절 (확장: -12.0~-1.5%)
        "STOP_LOSS_PCT": trial.suggest_float(
            "STOP_LOSS_PCT", -12.0, -1.5, step=0.5
        ),
        # 강세장 손절 (신규 추가: -15.0~-5.0%)
        "STOP_LOSS_BULLISH_PCT": trial.suggest_float(
            "STOP_LOSS_BULLISH_PCT", -15.0, -5.0, step=0.5
        ),
        # DCA 횟수 (확장: 1~15)
        "DCA_MAX_COUNT": trial.suggest_int(
            "DCA_MAX_COUNT", 1, 15
        ),
        # 쌍둥이 매도 갭 (확장: 0.5~15.0%)
        "PAIR_GAP_SELL_THRESHOLD_V2": trial.suggest_float(
            "PAIR_GAP_SELL_THRESHOLD_V2", 0.5, 15.0, step=0.5
        ),
        # 시간 손절 (신규 추가: 2~10시간)
        "MAX_HOLD_HOURS": trial.suggest_int(
            "MAX_HOLD_HOURS", 2, 10
        ),
    }


# ── Objective (Train 기간만 평가) ──────────────────────────────────

def objective(trial: optuna.Trial) -> tuple[float, float]:
    """Multi-objective: (수익률 maximize, MDD minimize).

    Train 기간에서만 평가하여 과적합을 방지한다.
    """
    import config
    import backtest_common
    from backtest_v2 import BacktestEngineV2

    params = define_search_space(trial)

    originals = {}
    for key, value in params.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        with suppress_stdout():
            engine = BacktestEngineV2(
                start_date=TRAIN_START,
                end_date=TRAIN_END,
            )
            engine.run(verbose=False)

        initial = engine.initial_capital_usd
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial
        train_ret = (final - initial) / initial * 100
        train_mdd = backtest_common.calc_mdd(engine.equity_curve)
        sharpe = backtest_common.calc_sharpe(engine.equity_curve)
        total_fees = engine.total_buy_fees_usd + engine.total_sell_fees_usd

        sells = [t for t in engine.trades if t.side == "SELL"]
        stop_losses = [t for t in sells if t.exit_reason == "stop_loss"]

        trial.set_user_attr("train_return", round(train_ret, 2))
        trial.set_user_attr("train_mdd", round(train_mdd, 2))
        trial.set_user_attr("final_equity", round(final, 2))
        trial.set_user_attr("total_fees", round(total_fees, 2))
        trial.set_user_attr("total_sells", len(sells))
        trial.set_user_attr("stop_loss_count", len(stop_losses))
        trial.set_user_attr("sharpe", round(sharpe, 4))

        # directions=["maximize", "minimize"] → return ↑, MDD ↓
        return (train_ret, train_mdd)

    finally:
        for key, value in originals.items():
            setattr(config, key, value)


# ── 워커 프로세스 ──────────────────────────────────────────────────

def _worker_process(
    worker_id: int,
    study_name: str,
    storage_url: str,
    n_trials: int,
    timeout: int | None,
) -> None:
    """독립 프로세스에서 study.optimize()를 실행한다."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)


# ── Pareto front 최적해 선택 ──────────────────────────────────────

def select_best_trial(pareto_trials: list) -> optuna.trial.FrozenTrial | None:
    """Pareto front에서 균형 잡힌 최적해를 선택한다.

    Score = 수익률 - 0.5 * MDD
    수익률을 중시하되 과도한 낙폭에 페널티를 준다.
    """
    if not pareto_trials:
        return None
    best = None
    best_score = float("-inf")
    for t in pareto_trials:
        ret = t.values[0]
        mdd = t.values[1]
        score = ret - 0.5 * mdd
        if score > best_score:
            best_score = score
            best = t
    return best


# ── Test 기간 검증 ─────────────────────────────────────────────────

def validate_on_test(params: dict) -> dict:
    """최적 파라미터로 Test 기간 백테스트를 1회 실행한다 (out-of-sample)."""
    import config
    import backtest_common
    from backtest_v2 import BacktestEngineV2

    originals = {}
    for key, value in params.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        with suppress_stdout():
            engine = BacktestEngineV2(
                start_date=TEST_START,
                end_date=TEST_END,
            )
            engine.run(verbose=False)

        initial = engine.initial_capital_usd
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial
        test_ret = (final - initial) / initial * 100
        test_mdd = backtest_common.calc_mdd(engine.equity_curve)
        sharpe = backtest_common.calc_sharpe(engine.equity_curve)
        total_fees = engine.total_buy_fees_usd + engine.total_sell_fees_usd
        sells = [t for t in engine.trades if t.side == "SELL"]
        stop_losses = [t for t in sells if t.exit_reason == "stop_loss"]

        return {
            "test_return": round(test_ret, 2),
            "test_mdd": round(test_mdd, 2),
            "final_equity": round(final, 2),
            "total_fees": round(total_fees, 2),
            "total_sells": len(sells),
            "stop_loss_count": len(stop_losses),
            "sharpe": round(sharpe, 4),
        }
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def run_full_period(params: dict) -> dict:
    """최적 파라미터로 전체 기간 백테스트를 실행한다."""
    import config
    import backtest_common
    from backtest_v2 import BacktestEngineV2

    originals = {}
    for key, value in params.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        with suppress_stdout():
            engine = BacktestEngineV2(
                start_date=TRAIN_START,
                end_date=TEST_END,
            )
            engine.run(verbose=False)

        initial = engine.initial_capital_usd
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial
        ret = (final - initial) / initial * 100
        mdd = backtest_common.calc_mdd(engine.equity_curve)

        return {"return": round(ret, 2), "mdd": round(mdd, 2), "final_equity": round(final, 2)}
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


# ── 결과 출력 ───────────────────────────────────────────────────────

def print_results(study: optuna.Study) -> None:
    """최적화 결과 + Walk-forward 검증을 출력한다."""
    print("\n" + "=" * 70)
    print("  Optuna 최적화 결과 (Walk-forward + Multi-objective)")
    print("=" * 70)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n  총 Trial: {len(study.trials)} (완료: {len(completed)} | 실패: {len(failed)})")

    if not completed:
        print("  완료된 trial이 없습니다.")
        return

    # ── Pareto front ──
    pareto = study.best_trials
    print(f"  Pareto front: {len(pareto)}개 해")

    best = select_best_trial(pareto)
    if best is None:
        print("  Pareto front이 비어 있습니다.")
        return

    # ── Best Trial (Train) ──
    print(f"\n{'─' * 70}")
    print(f"  BEST Trial #{best.number}  (Train: {TRAIN_START} ~ {TRAIN_END})")
    print(f"{'─' * 70}")
    print(f"  Train 수익률  : {best.values[0]:+.2f}%")
    print(f"  Train MDD     : -{best.values[1]:.2f}%")
    print(f"  최종 자산      : ${best.user_attrs.get('final_equity', 0):,.2f}")
    print(f"  수수료         : ${best.user_attrs.get('total_fees', 0):,.2f}")
    print(f"  Sharpe        : {best.user_attrs.get('sharpe', 0):.4f}")
    print(f"  매도 횟수      : {best.user_attrs.get('total_sells', 0)}")
    print(f"  손절 횟수      : {best.user_attrs.get('stop_loss_count', 0)}")

    print(f"\n  최적 파라미터:")
    for key, value in sorted(best.params.items()):
        if isinstance(value, int):
            print(f"    {key:40s} = {value}")
        else:
            print(f"    {key:40s} = {value:.2f}")

    # ── Test 검증 (out-of-sample, 1회만 실행) ──
    print(f"\n{'─' * 70}")
    print(f"  Walk-forward 검증  (Test: {TEST_START} ~ {TEST_END})")
    print(f"{'─' * 70}")

    test = validate_on_test(best.params)
    print(f"  Test 수익률   : {test['test_return']:+.2f}%")
    print(f"  Test MDD      : -{test['test_mdd']:.2f}%")
    print(f"  Test 수수료    : ${test['total_fees']:,.2f}")
    print(f"  Test Sharpe   : {test['sharpe']:.4f}")
    print(f"  Test 매도     : {test['total_sells']}")
    print(f"  Test 손절     : {test['stop_loss_count']}")

    gap = abs(best.values[0] - test["test_return"])
    print(f"\n  Train/Test 수익률 갭: {gap:.2f}pp")
    if gap > 15:
        print(f"  >> 갭이 크므로 과적합 가능성 있음")
    elif test["test_return"] >= best.values[0]:
        print(f"  >> Test >= Train — 견고한 파라미터")
    else:
        print(f"  >> 수용 가능한 범위")

    # ── 전체 기간 비교 ──
    print(f"\n{'─' * 70}")
    print(f"  전체 기간 비교  ({TRAIN_START} ~ {TEST_END})")
    print(f"{'─' * 70}")
    full = run_full_period(best.params)
    baseline_ret = -22.87
    print(f"  기본값 수익률  : {baseline_ret:+.2f}%")
    print(f"  최적화 수익률  : {full['return']:+.2f}%")
    print(f"  최적화 MDD     : -{full['mdd']:.2f}%")
    print(f"  개선           : {full['return'] - baseline_ret:+.2f}pp")

    # ── Pareto Front Top 10 ──
    scored = [(t, t.values[0] - 0.5 * t.values[1]) for t in pareto]
    scored.sort(key=lambda x: -x[1])
    top10 = scored[:10]

    print(f"\n{'─' * 70}")
    print(f"  Pareto Front Top 10 (score = 수익률 - 0.5*MDD)")
    print(f"{'─' * 70}")
    print(f"  {'#':>4s}  {'수익률':>8s}  {'MDD':>8s}  {'Score':>8s}  {'수수료':>12s}  {'Sharpe':>8s}  {'매도':>5s}  {'손절':>4s}")
    for t, score in top10:
        print(
            f"  {t.number:4d}  {t.values[0]:+7.2f}%"
            f"  -{t.values[1]:6.2f}%"
            f"  {score:+7.2f}"
            f"  {t.user_attrs.get('total_fees', 0):>11,.0f}"
            f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
            f"  {t.user_attrs.get('total_sells', 0):>5d}"
            f"  {t.user_attrs.get('stop_loss_count', 0):>4d}"
        )

    # ── config.py 복사용 ──
    print(f"\n{'─' * 70}")
    print(f"  config.py 적용 코드 (복사용)")
    print(f"{'─' * 70}")
    for key, value in sorted(best.params.items()):
        if isinstance(value, int):
            print(f"{key} = {value}")
        else:
            print(f"{key} = {value:.2f}")


# ── 메인 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PTJ v2 Optuna 최적화 (개선판)")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="총 trial 수 (기본: 100)")
    parser.add_argument("--n-jobs", type=int, default=10,
                        help="병렬 프로세스 수 (기본: 10)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="최대 실행 시간(초)")
    parser.add_argument("--study-name", type=str, default="ptj_v2_wf",
                        help="study 이름 (기본: ptj_v2_wf)")
    parser.add_argument("--db", type=str, default=None,
                        help="Optuna DB URL")
    args = parser.parse_args()

    if args.db:
        storage_url = args.db
    else:
        db_path = os.path.join(os.path.dirname(__file__), "data", "optuna_v2.db")
        storage_url = f"sqlite:///{db_path}"

    n_jobs = max(1, args.n_jobs)
    trials_per_worker = max(1, args.n_trials // n_jobs)
    remainder = args.n_trials - trials_per_worker * n_jobs

    print("=" * 70)
    print("  PTJ v2 — Optuna 파라미터 최적화 (개선판)")
    print("=" * 70)
    print(f"  Walk-forward : Train {TRAIN_START}~{TRAIN_END} / Test {TEST_START}~{TEST_END}")
    print(f"  Multi-obj    : 수익률 maximize + MDD minimize (NSGA-II)")
    print(f"  파라미터 (7개): COIN/CONL_TRIGGER, STOP_LOSS (normal+bullish), DCA, PAIR_GAP, MAX_HOLD_HOURS")
    print(f"  Trials       : {args.n_trials} | Workers: {n_jobs}")
    print(f"  DB           : {storage_url}")
    print("=" * 70)

    sampler = NSGAIISampler(seed=42, population_size=50)
    study = optuna.create_study(
        study_name=args.study_name,
        directions=["maximize", "minimize"],  # 수익률 ↑, MDD ↓
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    t0 = time.time()

    if n_jobs == 1:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
        )
    else:
        ctx = mp.get_context("spawn")
        processes = []

        for i in range(n_jobs):
            n = trials_per_worker + (1 if i < remainder else 0)
            p = ctx.Process(
                target=_worker_process,
                args=(i, args.study_name, storage_url, n, args.timeout),
                name=f"optuna-worker-{i}",
            )
            processes.append(p)

        print(f"\n  {n_jobs}개 워커 프로세스 시작...")
        for p in processes:
            p.start()

        while any(p.is_alive() for p in processes):
            time.sleep(10)
            study_refresh = optuna.load_study(
                study_name=args.study_name,
                storage=storage_url,
            )
            done = len([t for t in study_refresh.trials
                        if t.state == optuna.trial.TrialState.COMPLETE])
            elapsed = time.time() - t0
            alive = sum(1 for p in processes if p.is_alive())

            pareto = study_refresh.best_trials if done > 0 else []
            if pareto:
                best_t = select_best_trial(pareto)
                if best_t:
                    print(
                        f"  [{elapsed:6.0f}s] 완료: {done:>3}/{args.n_trials}"
                        f" | 워커: {alive}/{n_jobs}"
                        f" | Pareto: {len(pareto)}"
                        f" | Best: {best_t.values[0]:+.1f}% / MDD -{best_t.values[1]:.1f}%"
                    )
                else:
                    print(f"  [{elapsed:6.0f}s] 완료: {done:>3}/{args.n_trials} | 워커: {alive}/{n_jobs}")
            else:
                print(f"  [{elapsed:6.0f}s] 완료: {done:>3}/{args.n_trials} | 워커: {alive}/{n_jobs}")

        for p in processes:
            p.join()

        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage_url,
        )

    elapsed = time.time() - t0
    completed_count = len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  총 실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    if completed_count > 0:
        print(f"  Trial당 평균: {elapsed/completed_count:.1f}초 (병렬)")
        print(f"  실효 처리량: {completed_count/elapsed*60:.1f} trials/분")

    print_results(study)


if __name__ == "__main__":
    main()
