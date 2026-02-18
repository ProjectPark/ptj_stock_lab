#!/usr/bin/env python3
"""
PTJ v4 — Optuna 기반 파라미터 최적화
=====================================
Stage 1: 현재 config 가중치로 baseline 실행 → 리포트 저장
Stage 2: Optuna TPE sampler로 최적 파라미터 탐색 → 리포트 저장

Usage:
    pyenv shell market && python optimize_v4_optuna.py --stage 1
    pyenv shell market && python optimize_v4_optuna.py --stage 2 [--n-trials 20] [--n-jobs 6]
    pyenv shell market && python optimize_v4_optuna.py              # 1 → 2 연속 실행
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import date, datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

# ── 경로 상수 ─────────────────────────────────────────────────

DOCS_DIR = Path(__file__).resolve().parent / "docs"
DATA_DIR = Path(__file__).resolve().parent / "data"
BASELINE_JSON = DATA_DIR / "v4_baseline_result.json"
BASELINE_REPORT = DOCS_DIR / "v4_baseline_report.md"
OPTUNA_REPORT = DOCS_DIR / "v4_optuna_report.md"


# ── 현재 config baseline 값 ──────────────────────────────────


def _get_baseline_params() -> dict:
    """현재 config.py에 설정된 v4 파라미터를 반환한다."""
    import config

    return {
        # v4 고유
        "V4_PAIR_GAP_ENTRY_THRESHOLD": config.V4_PAIR_GAP_ENTRY_THRESHOLD,
        "V4_DCA_MAX_COUNT": config.V4_DCA_MAX_COUNT,
        "V4_MAX_PER_STOCK": config.V4_MAX_PER_STOCK,
        "V4_COIN_TRIGGER_PCT": config.V4_COIN_TRIGGER_PCT,
        "V4_CONL_TRIGGER_PCT": config.V4_CONL_TRIGGER_PCT,
        "V4_SPLIT_BUY_INTERVAL_MIN": config.V4_SPLIT_BUY_INTERVAL_MIN,
        "V4_ENTRY_CUTOFF_HOUR": config.V4_ENTRY_CUTOFF_HOUR,
        "V4_ENTRY_CUTOFF_MINUTE": config.V4_ENTRY_CUTOFF_MINUTE,
        "V4_INITIAL_BUY": config.V4_INITIAL_BUY,
        "V4_DCA_BUY": config.V4_DCA_BUY,
        # v4 횡보장
        "V4_SIDEWAYS_MIN_SIGNALS": config.V4_SIDEWAYS_MIN_SIGNALS,
        "V4_SIDEWAYS_POLY_LOW": config.V4_SIDEWAYS_POLY_LOW,
        "V4_SIDEWAYS_POLY_HIGH": config.V4_SIDEWAYS_POLY_HIGH,
        "V4_SIDEWAYS_GLD_THRESHOLD": config.V4_SIDEWAYS_GLD_THRESHOLD,
        "V4_SIDEWAYS_GAP_FAIL_COUNT": config.V4_SIDEWAYS_GAP_FAIL_COUNT,
        "V4_SIDEWAYS_TRIGGER_FAIL_COUNT": config.V4_SIDEWAYS_TRIGGER_FAIL_COUNT,
        "V4_SIDEWAYS_INDEX_THRESHOLD": config.V4_SIDEWAYS_INDEX_THRESHOLD,
        "V4_CB_GLD_SPIKE_PCT": config.V4_CB_GLD_SPIKE_PCT,
        "V4_CB_GLD_COOLDOWN_DAYS": config.V4_CB_GLD_COOLDOWN_DAYS,
        "V4_CB_BTC_CRASH_PCT": config.V4_CB_BTC_CRASH_PCT,
        "V4_CB_BTC_SURGE_PCT": config.V4_CB_BTC_SURGE_PCT,
        # v2 공유
        "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
        "STOP_LOSS_BULLISH_PCT": config.STOP_LOSS_BULLISH_PCT,
        "COIN_SELL_PROFIT_PCT": config.COIN_SELL_PROFIT_PCT,
        "COIN_SELL_BEARISH_PCT": config.COIN_SELL_BEARISH_PCT,
        "CONL_SELL_PROFIT_PCT": config.CONL_SELL_PROFIT_PCT,
        "CONL_SELL_AVG_PCT": config.CONL_SELL_AVG_PCT,
        "DCA_DROP_PCT": config.DCA_DROP_PCT,
        "MAX_HOLD_HOURS": config.MAX_HOLD_HOURS,
        "TAKE_PROFIT_PCT": config.TAKE_PROFIT_PCT,
        "PAIR_GAP_SELL_THRESHOLD_V2": config.PAIR_GAP_SELL_THRESHOLD_V2,
        "PAIR_SELL_FIRST_PCT": config.PAIR_SELL_FIRST_PCT,
    }


# ── 워커 프로세스 ─────────────────────────────────────────────


def _run_single_trial(params: dict) -> dict:
    """v4 백테스트 1회를 실행하고 지표를 반환한다."""
    import config
    import backtest_common
    from backtest_v4 import BacktestEngineV4

    originals = {}
    for key, value in params.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    try:
        engine = BacktestEngineV4()
        engine.run(verbose=False)

        initial = engine.initial_capital_krw
        final = engine.equity_curve[-1][1] if engine.equity_curve else initial
        total_ret = (final - initial) / initial * 100
        mdd = backtest_common.calc_mdd(engine.equity_curve)
        sharpe = backtest_common.calc_sharpe(engine.equity_curve)
        total_fees = engine.total_buy_fees_krw + engine.total_sell_fees_krw

        sells = [t for t in engine.trades if t.side == "SELL"]
        buys = [t for t in engine.trades if t.side == "BUY"]
        stop_losses = [t for t in sells if t.exit_reason == "stop_loss"]
        time_stops = [t for t in sells if t.exit_reason == "time_stop"]
        eod_closes = [t for t in sells if t.exit_reason == "eod_close"]
        wins = [t for t in sells if t.pnl_krw > 0]
        losses = [t for t in sells if t.pnl_krw < 0]

        # 시그널별 통계
        sig_stats = {}
        for t in sells:
            key = t.signal_type
            if key not in sig_stats:
                sig_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            sig_stats[key]["count"] += 1
            sig_stats[key]["pnl"] += t.pnl_krw
            if t.pnl_krw > 0:
                sig_stats[key]["wins"] += 1

        # 매도 사유별 통계
        exit_stats = {}
        for t in sells:
            key = t.exit_reason or "unknown"
            if key not in exit_stats:
                exit_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            exit_stats[key]["count"] += 1
            exit_stats[key]["pnl"] += t.pnl_krw
            if t.pnl_krw > 0:
                exit_stats[key]["wins"] += 1

        return {
            "final_equity": final,
            "total_return_pct": total_ret,
            "mdd": mdd,
            "sharpe": sharpe,
            "total_fees": total_fees,
            "total_sells": len(sells),
            "total_buys": len(buys),
            "win_rate": len(wins) / len(sells) * 100 if sells else 0,
            "win_count": len(wins),
            "loss_count": len(losses),
            "avg_win": sum(t.pnl_krw for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.pnl_krw for t in losses) / len(losses) if losses else 0,
            "total_pnl": sum(t.pnl_krw for t in sells),
            "stop_loss_count": len(stop_losses),
            "stop_loss_pnl": sum(t.pnl_krw for t in stop_losses),
            "time_stop_count": len(time_stops),
            "eod_close_count": len(eod_closes),
            "sideways_days": engine.sideways_days,
            "sideways_blocks": engine.sideways_blocks,
            "entry_cutoff_blocks": engine.entry_cutoff_blocks,
            "daily_limit_blocks": engine.daily_limit_blocks,
            "cb_buy_blocks": getattr(engine, "cb_buy_blocks", 0),
            "cb_sell_halt_bars": getattr(engine, "cb_sell_halt_bars", 0),
            "total_trading_days": engine.total_trading_days,
            "sig_stats": sig_stats,
            "exit_stats": exit_stats,
        }
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def _worker(params: dict) -> dict:
    """mp.Pool에서 호출되는 최상위 함수."""
    return _run_single_trial(params)


# =====================================================================
# Stage 1: Baseline
# =====================================================================


def _save_baseline_json(result: dict, params: dict) -> None:
    """baseline 결과를 JSON으로 저장 (stage 2에서 재사용)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"result": result, "params": params, "timestamp": datetime.now().isoformat()}
    with open(BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Baseline JSON: {BASELINE_JSON}")


def _load_baseline_json() -> tuple[dict, dict]:
    """저장된 baseline JSON을 로드한다."""
    if not BASELINE_JSON.exists():
        raise FileNotFoundError(
            f"Baseline 결과 없음: {BASELINE_JSON}\n"
            "  먼저 --stage 1을 실행하세요."
        )
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["result"], data["params"]


def _generate_baseline_report(result: dict, params: dict) -> str:
    """Baseline 마크다운 리포트를 생성한다."""
    r = result
    total_blocks = r["sideways_blocks"] + r["entry_cutoff_blocks"] + r["daily_limit_blocks"]

    lines = [
        "# PTJ v4 Baseline 시뮬레이션 리포트",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 개요",
        "",
        "현재 `config.py`에 설정된 v4 파라미터로 백테스트를 실행한 결과입니다.",
        "",
        "## 2. 핵심 지표",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 총 수익률 | **{r['total_return_pct']:+.2f}%** |",
        f"| 최종 자산 | {r['final_equity']:,.0f}원 |",
        f"| MDD | -{r['mdd']:.2f}% |",
        f"| Sharpe Ratio | {r['sharpe']:.4f} |",
        f"| 총 손익 | {r['total_pnl']:+,.0f}원 |",
        f"| 총 수수료 | {r['total_fees']:,.0f}원 |",
        "",
        "## 3. 매매 통계",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 거래일 | {r['total_trading_days']}일 |",
        f"| 매수 횟수 | {r['total_buys']}회 |",
        f"| 매도 횟수 | {r['total_sells']}회 |",
        f"| 승/패 | {r['win_count']}W / {r['loss_count']}L |",
        f"| 승률 | {r['win_rate']:.1f}% |",
        f"| 평균 수익 | {r['avg_win']:+,.0f}원 |",
        f"| 평균 손실 | {r['avg_loss']:+,.0f}원 |",
        "",
        "## 4. v4 선별 매매 효과",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 횡보장 감지일 | {r['sideways_days']}일 / {r['total_trading_days']}일 |",
        f"| 횡보장 차단 매수 | {r['sideways_blocks']}회 |",
        f"| 시간제한 차단 | {r['entry_cutoff_blocks']}회 |",
        f"| 일일1회 차단 | {r['daily_limit_blocks']}회 |",
        f"| **총 차단 매수** | **{total_blocks}회** |",
        "",
        "## 5. 매도 사유별 성과",
        "",
        "| 사유 | 횟수 | P&L | 승률 |",
        "|---|---|---|---|",
    ]
    for key in sorted(r.get("exit_stats", {}).keys()):
        s = r["exit_stats"][key]
        wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        lines.append(f"| {key} | {s['count']}회 | {s['pnl']:+,.0f}원 | {wr:.1f}% |")

    lines += [
        "",
        "## 6. 시그널별 성과",
        "",
        "| 시그널 | 횟수 | P&L | 승률 |",
        "|---|---|---|---|",
    ]
    for key in sorted(r.get("sig_stats", {}).keys()):
        s = r["sig_stats"][key]
        wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        lines.append(f"| {key} | {s['count']}회 | {s['pnl']:+,.0f}원 | {wr:.1f}% |")

    lines += [
        "",
        "## 7. 현재 파라미터 (config.py)",
        "",
        "### v4 고유",
        "",
        "| 파라미터 | 값 |",
        "|---|---|",
    ]
    v4_keys = [k for k in sorted(params.keys()) if k.startswith("V4_")]
    shared_keys = [k for k in sorted(params.keys()) if not k.startswith("V4_")]
    for k in v4_keys:
        v = params[k]
        if isinstance(v, float):
            lines.append(f"| `{k}` | {v:.2f} |")
        elif isinstance(v, int) and v >= 1_000_000:
            lines.append(f"| `{k}` | {v:,} |")
        else:
            lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "### v2 공유",
        "",
        "| 파라미터 | 값 |",
        "|---|---|",
    ]
    for k in shared_keys:
        v = params[k]
        if isinstance(v, float):
            lines.append(f"| `{k}` | {v:.2f} |")
        elif isinstance(v, int) and v >= 1_000_000:
            lines.append(f"| `{k}` | {v:,} |")
        else:
            lines.append(f"| `{k}` | {v} |")

    lines.append("")
    return "\n".join(lines)


def run_stage1() -> tuple[dict, dict]:
    """Stage 1: baseline 실행 → JSON 저장 → 마크다운 리포트 생성."""
    print("\n" + "=" * 70)
    print("  [Stage 1] Baseline — 현재 config 가중치 시뮬레이션")
    print("=" * 70)

    params = _get_baseline_params()

    print("\n  실행 중...")
    t0 = time.time()
    result = _run_single_trial(params)
    elapsed = time.time() - t0
    print(f"  완료 ({elapsed:.1f}초)")

    # 콘솔 요약
    print(f"\n  수익률  : {result['total_return_pct']:+.2f}%")
    print(f"  자산    : {result['final_equity']:,.0f}원")
    print(f"  MDD     : -{result['mdd']:.2f}%")
    print(f"  Sharpe  : {result['sharpe']:.4f}")
    print(f"  승률    : {result['win_rate']:.1f}%  ({result['win_count']}W / {result['loss_count']}L)")
    print(f"  매수/매도: {result['total_buys']} / {result['total_sells']}")
    total_blocks = result["sideways_blocks"] + result["entry_cutoff_blocks"] + result["daily_limit_blocks"]
    print(f"  차단    : {total_blocks}회 (횡보 {result['sideways_blocks']} / 시간 {result['entry_cutoff_blocks']} / 일일 {result['daily_limit_blocks']})")

    # JSON 저장
    _save_baseline_json(result, params)

    # 마크다운 리포트
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md = _generate_baseline_report(result, params)
    BASELINE_REPORT.write_text(md, encoding="utf-8")
    print(f"  Baseline 리포트: {BASELINE_REPORT}")

    return result, params


# =====================================================================
# Stage 2: Optuna
# =====================================================================


class BacktestObjectiveV4:
    """Optuna objective — v4 파라미터 탐색 공간 정의."""

    def __init__(self, pool: mp.Pool | None = None, gap_max: float = 4.0):
        self.pool = pool
        self.gap_max = gap_max

    def __call__(self, trial: optuna.Trial) -> float:
        # ---- v4 고유 파라미터 ----
        params = {
            "V4_PAIR_GAP_ENTRY_THRESHOLD": trial.suggest_float(
                "V4_PAIR_GAP_ENTRY_THRESHOLD", 1.0, self.gap_max, step=0.2
            ),
            "V4_DCA_MAX_COUNT": trial.suggest_int(
                "V4_DCA_MAX_COUNT", 1, 7
            ),
            "V4_MAX_PER_STOCK": trial.suggest_int(
                "V4_MAX_PER_STOCK", 3_000, 10_000, step=250
            ),
            "V4_COIN_TRIGGER_PCT": trial.suggest_float(
                "V4_COIN_TRIGGER_PCT", 2.0, 7.0, step=0.5
            ),
            "V4_CONL_TRIGGER_PCT": trial.suggest_float(
                "V4_CONL_TRIGGER_PCT", 2.0, 7.0, step=0.5
            ),
            "V4_SPLIT_BUY_INTERVAL_MIN": trial.suggest_int(
                "V4_SPLIT_BUY_INTERVAL_MIN", 5, 30, step=5
            ),
            "V4_ENTRY_CUTOFF_HOUR": trial.suggest_int(
                "V4_ENTRY_CUTOFF_HOUR", 10, 14
            ),
            "V4_ENTRY_CUTOFF_MINUTE": trial.suggest_categorical(
                "V4_ENTRY_CUTOFF_MINUTE", [0, 30]
            ),
            "V4_INITIAL_BUY": trial.suggest_int(
                "V4_INITIAL_BUY", 1_000, 3_500, step=250
            ),
            "V4_DCA_BUY": trial.suggest_int(
                "V4_DCA_BUY", 250, 1_500, step=125
            ),
        }

        # ---- v4 횡보장 ----
        params.update({
            "V4_SIDEWAYS_MIN_SIGNALS": trial.suggest_int(
                "V4_SIDEWAYS_MIN_SIGNALS", 2, 5
            ),
            "V4_SIDEWAYS_POLY_LOW": trial.suggest_float(
                "V4_SIDEWAYS_POLY_LOW", 0.30, 0.50, step=0.05
            ),
            "V4_SIDEWAYS_POLY_HIGH": trial.suggest_float(
                "V4_SIDEWAYS_POLY_HIGH", 0.50, 0.70, step=0.05
            ),
            "V4_SIDEWAYS_GLD_THRESHOLD": trial.suggest_float(
                "V4_SIDEWAYS_GLD_THRESHOLD", 0.1, 0.8, step=0.1
            ),
            "V4_SIDEWAYS_INDEX_THRESHOLD": trial.suggest_float(
                "V4_SIDEWAYS_INDEX_THRESHOLD", 0.2, 1.0, step=0.1
            ),
            "V4_CB_GLD_SPIKE_PCT": trial.suggest_float(
                "V4_CB_GLD_SPIKE_PCT", 2.0, 5.0, step=0.5
            ),
            "V4_CB_GLD_COOLDOWN_DAYS": trial.suggest_int(
                "V4_CB_GLD_COOLDOWN_DAYS", 1, 5
            ),
            "V4_CB_BTC_CRASH_PCT": trial.suggest_float(
                "V4_CB_BTC_CRASH_PCT", -8.0, -3.0, step=0.5
            ),
            "V4_CB_BTC_SURGE_PCT": trial.suggest_float(
                "V4_CB_BTC_SURGE_PCT", 3.0, 8.0, step=0.5
            ),
        })

        # ---- v2 공유 ----
        params.update({
            "STOP_LOSS_PCT": trial.suggest_float(
                "STOP_LOSS_PCT", -6.0, -1.5, step=0.5
            ),
            "STOP_LOSS_BULLISH_PCT": trial.suggest_float(
                "STOP_LOSS_BULLISH_PCT", -12.0, -5.0, step=0.5
            ),
            "COIN_SELL_PROFIT_PCT": trial.suggest_float(
                "COIN_SELL_PROFIT_PCT", 1.0, 5.0, step=0.5
            ),
            "CONL_SELL_PROFIT_PCT": trial.suggest_float(
                "CONL_SELL_PROFIT_PCT", 1.0, 5.0, step=0.5
            ),
            "DCA_DROP_PCT": trial.suggest_float(
                "DCA_DROP_PCT", -2.0, -0.3, step=0.1
            ),
            "MAX_HOLD_HOURS": trial.suggest_int(
                "MAX_HOLD_HOURS", 2, 8
            ),
            "TAKE_PROFIT_PCT": trial.suggest_float(
                "TAKE_PROFIT_PCT", 1.0, 5.0, step=0.5
            ),
            "PAIR_GAP_SELL_THRESHOLD_V2": trial.suggest_float(
                "PAIR_GAP_SELL_THRESHOLD_V2", 2.0, 10.0, step=0.1
            ),
            "PAIR_SELL_FIRST_PCT": trial.suggest_float(
                "PAIR_SELL_FIRST_PCT", 0.5, 1.0, step=0.05
            ),
        })

        # ---- 실행 ----
        if self.pool is not None:
            result = self.pool.apply(_worker, (params,))
        else:
            result = _run_single_trial(params)

        # ---- 지표 기록 ----
        for attr_key in [
            "final_equity", "mdd", "sharpe", "total_fees",
            "total_sells", "total_buys", "win_rate",
            "stop_loss_count", "time_stop_count",
            "sideways_days", "sideways_blocks",
            "entry_cutoff_blocks", "daily_limit_blocks",
            "cb_buy_blocks", "cb_sell_halt_bars",
        ]:
            trial.set_user_attr(attr_key, result[attr_key])

        return result["total_return_pct"]


def _generate_optuna_report(
    study: optuna.Study,
    baseline: dict,
    baseline_params: dict,
    elapsed: float,
    n_jobs: int,
) -> str:
    """Optuna 마크다운 리포트를 생성한다."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    best = study.best_trial

    bl = baseline
    diff_ret = best.value - bl["total_return_pct"]

    lines = [
        "# PTJ v4 Optuna 최적화 리포트",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 실행 정보",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 총 Trial | {len(study.trials)} (완료: {len(completed)}, 실패: {len(failed)}) |",
        f"| 병렬 Worker | {n_jobs} |",
        f"| 실행 시간 | {elapsed:.1f}초 ({elapsed / 60:.1f}분) |",
        f"| Trial당 평균 | {elapsed / len(study.trials):.1f}초 |" if study.trials else "",
        f"| Sampler | TPE (seed=42) |",
        "",
        "## 2. Baseline vs Best 비교",
        "",
        "| 지표 | Baseline | Best (#{best.number}) | 차이 |",
        "|---|---|---|---|",
        f"| **수익률** | {bl['total_return_pct']:+.2f}% | **{best.value:+.2f}%** | {diff_ret:+.2f}% |",
        f"| MDD | -{bl['mdd']:.2f}% | -{best.user_attrs.get('mdd', 0):.2f}% | {best.user_attrs.get('mdd', 0) - bl['mdd']:+.2f}% |",
        f"| Sharpe | {bl['sharpe']:.4f} | {best.user_attrs.get('sharpe', 0):.4f} | {best.user_attrs.get('sharpe', 0) - bl['sharpe']:+.4f} |",
        f"| 승률 | {bl['win_rate']:.1f}% | {best.user_attrs.get('win_rate', 0):.1f}% | {best.user_attrs.get('win_rate', 0) - bl['win_rate']:+.1f}% |",
        f"| 매도 횟수 | {bl['total_sells']} | {best.user_attrs.get('total_sells', 0)} | {best.user_attrs.get('total_sells', 0) - bl['total_sells']:+d} |",
        f"| 손절 횟수 | {bl['stop_loss_count']} | {best.user_attrs.get('stop_loss_count', 0)} | {best.user_attrs.get('stop_loss_count', 0) - bl['stop_loss_count']:+d} |",
        f"| 시간손절 | {bl['time_stop_count']} | {best.user_attrs.get('time_stop_count', 0)} | {best.user_attrs.get('time_stop_count', 0) - bl['time_stop_count']:+d} |",
        f"| 횡보장 일수 | {bl['sideways_days']} | {best.user_attrs.get('sideways_days', 0)} | {best.user_attrs.get('sideways_days', 0) - bl['sideways_days']:+d} |",
        f"| 수수료 | {bl['total_fees']:,.0f}원 | {best.user_attrs.get('total_fees', 0):,.0f}원 | {best.user_attrs.get('total_fees', 0) - bl['total_fees']:+,.0f}원 |",
        "",
        "## 3. 최적 파라미터 (Best Trial #{})".format(best.number),
        "",
        "| 파라미터 | 최적값 | Baseline | 변경 |",
        "|---|---|---|---|",
    ]
    for key, value in sorted(best.params.items()):
        bl_val = baseline_params.get(key, "N/A")
        changed = ""
        if isinstance(bl_val, (int, float)):
            if isinstance(value, float):
                changed = f"{value - bl_val:+.2f}" if value != bl_val else "-"
            else:
                changed = f"{value - bl_val:+d}" if value != bl_val else "-"
        if isinstance(value, float):
            bl_str = f"{bl_val:.2f}" if isinstance(bl_val, float) else str(bl_val)
            lines.append(f"| `{key}` | **{value:.2f}** | {bl_str} | {changed} |")
        elif isinstance(value, int) and value >= 1_000_000:
            bl_str = f"{bl_val:,}" if isinstance(bl_val, int) else str(bl_val)
            lines.append(f"| `{key}` | **{value:,}** | {bl_str} | {changed} |")
        else:
            lines.append(f"| `{key}` | **{value}** | {bl_val} | {changed} |")

    # Top 5
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    lines += [
        "",
        "## 4. Top 5 Trials",
        "",
        "| # | 수익률 | MDD | Sharpe | 승률 | 매도 | 손절 | 횡보일 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in top5:
        lines.append(
            f"| {t.number} | {t.value:+.2f}% "
            f"| -{t.user_attrs.get('mdd', 0):.2f}% "
            f"| {t.user_attrs.get('sharpe', 0):.4f} "
            f"| {t.user_attrs.get('win_rate', 0):.1f}% "
            f"| {t.user_attrs.get('total_sells', 0)} "
            f"| {t.user_attrs.get('stop_loss_count', 0)} "
            f"| {t.user_attrs.get('sideways_days', 0)} |"
        )

    # Parameter importance
    if len(completed) >= 5:
        try:
            importance = optuna.importance.get_param_importances(study)
            lines += [
                "",
                "## 5. 파라미터 중요도 (fANOVA)",
                "",
                "| 파라미터 | 중요도 |",
                "|---|---|",
            ]
            for param, score in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "█" * int(score * 30)
                lines.append(f"| `{param}` | {score:.4f} {bar} |")
        except Exception:
            pass

    # Top 5 파라미터 상세
    lines += [
        "",
        "## 6. Top 5 파라미터 상세",
        "",
    ]
    for rank, t in enumerate(top5, 1):
        lines.append(f"### #{rank} — Trial {t.number} ({t.value:+.2f}%)")
        lines.append("")
        lines.append("```")
        for key, value in sorted(t.params.items()):
            if isinstance(value, float):
                lines.append(f"{key} = {value:.2f}")
            elif isinstance(value, int) and value >= 1_000_000:
                lines.append(f"{key} = {value:_}")
            else:
                lines.append(f"{key} = {value}")
        lines.append("```")
        lines.append("")

    # config.py 적용 코드
    lines += [
        "## 7. config.py 적용 코드 (Best Trial)",
        "",
        "```python",
    ]
    for key, value in sorted(best.params.items()):
        if isinstance(value, int):
            if value >= 1_000_000:
                lines.append(f"{key} = {value:_}")
            else:
                lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {value:.2f}" if abs(value) >= 0.01 else f"{key} = {value}")
    lines += ["```", ""]

    return "\n".join(lines)


def run_stage2(
    n_trials: int,
    n_jobs: int,
    timeout: int | None,
    study_name: str,
    db: str | None,
    baseline: dict | None = None,
    baseline_params: dict | None = None,
    gap_max: float = 4.0,
) -> None:
    """Stage 2: Optuna 최적화 실행 → 마크다운 리포트 생성."""
    # baseline 로드
    if baseline is None or baseline_params is None:
        baseline, baseline_params = _load_baseline_json()
        print(f"  Baseline 로드: {BASELINE_JSON}")
        print(f"  Baseline 수익률: {baseline['total_return_pct']:+.2f}%")

    print(f"\n{'=' * 70}")
    print(f"  [Stage 2] Optuna 최적화 ({n_trials} trials, {n_jobs} workers, GAP max {gap_max}%)")
    print(f"{'=' * 70}")

    sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials))
    storage = db if db else None

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    # baseline 파라미터를 첫 trial로 enqueue
    study.enqueue_trial(baseline_params)

    t0 = time.time()

    if n_jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            objective = BacktestObjectiveV4(pool=pool, gap_max=gap_max)
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )
    else:
        objective = BacktestObjectiveV4(pool=None, gap_max=gap_max)
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

    elapsed = time.time() - t0

    # 콘솔 요약
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        diff = best.value - baseline["total_return_pct"]
        print(f"\n  실행 시간: {elapsed:.1f}초 ({elapsed / 60:.1f}분)")
        print(f"  Trial당 평균: {elapsed / len(study.trials):.1f}초")
        print(f"\n  BEST Trial #{best.number}")
        print(f"  수익률  : {best.value:+.2f}%  (baseline 대비 {diff:+.2f}%)")
        print(f"  MDD     : -{best.user_attrs.get('mdd', 0):.2f}%")
        print(f"  Sharpe  : {best.user_attrs.get('sharpe', 0):.4f}")
        print(f"  승률    : {best.user_attrs.get('win_rate', 0):.1f}%")

        # Top 5 콘솔 출력
        top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
        print(f"\n  Top 5:")
        print(f"  {'#':>4s}  {'수익률':>8s}  {'MDD':>8s}  {'Sharpe':>8s}  {'승률':>6s}")
        for t in top5:
            print(
                f"  {t.number:4d}  {t.value:+7.2f}%"
                f"  -{t.user_attrs.get('mdd', 0):6.2f}%"
                f"  {t.user_attrs.get('sharpe', 0):>8.4f}"
                f"  {t.user_attrs.get('win_rate', 0):>5.1f}%"
            )
    else:
        print("\n  완료된 trial 없음")
        return

    # 마크다운 리포트
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md = _generate_optuna_report(study, baseline, baseline_params, elapsed, n_jobs)
    OPTUNA_REPORT.write_text(md, encoding="utf-8")
    print(f"\n  Optuna 리포트: {OPTUNA_REPORT}")


# ── 메인 ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PTJ v4 Optuna 최적화")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None,
                        help="실행 단계 (1: baseline, 2: Optuna, 미지정: 둘 다)")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trial 수 (기본: 20)")
    parser.add_argument("--n-jobs", type=int, default=6, help="병렬 프로세스 수 (기본: 6)")
    parser.add_argument("--timeout", type=int, default=None, help="최대 실행 시간(초)")
    parser.add_argument("--study-name", type=str, default="ptj_v4_opt", help="study 이름")
    parser.add_argument("--db", type=str, default=None, help="Optuna DB URL (기본: in-memory)")
    parser.add_argument("--gap-max", type=float, default=4.0, help="V4_PAIR_GAP_ENTRY_THRESHOLD 탐색 상한 (%%, 기본: 4.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("  PTJ v4 — Optuna 파라미터 최적화")
    print("=" * 70)

    if args.stage == 1:
        run_stage1()

    elif args.stage == 2:
        run_stage2(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            study_name=args.study_name,
            db=args.db,
            gap_max=args.gap_max,
        )

    else:
        # 둘 다 실행
        result, params = run_stage1()
        print()
        run_stage2(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            study_name=args.study_name,
            db=args.db,
            baseline=result,
            baseline_params=params,
            gap_max=args.gap_max,
        )

    print("\n  완료!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
