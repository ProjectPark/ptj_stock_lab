"""Optuna 리포트 섹션 빌더 함수들.

각 함수는 list[str] (마크다운 라인)을 반환.
BaseOptimizer._generate_optuna_report()에서 조합하여 사용.
서브클래스는 _get_report_sections()를 오버라이드하여 섹션 교체/추가 가능.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import optuna


# ============================================================
# Section 1: 실행 정보
# ============================================================


def section_execution_info(
    study: optuna.Study,
    elapsed: float,
    n_jobs: int,
    version: str,
    extra_rows: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Section: 실행 정보 테이블."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    lines = [
        "## 1. 실행 정보",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 총 Trial | {len(study.trials)} (완료: {len(completed)}, 실패: {len(failed)}) |",
        f"| 병렬 Worker | {n_jobs} |",
        f"| 실행 시간 | {elapsed:.1f}초 ({elapsed / 60:.1f}분) |",
    ]
    if study.trials:
        lines.append(f"| Trial당 평균 | {elapsed / len(study.trials):.1f}초 |")
    lines.append("| Sampler | TPE (seed=42) |")
    if extra_rows:
        for key, val in extra_rows:
            lines.append(f"| {key} | {val} |")
    lines.append("")
    return lines


# ============================================================
# Section 2: Baseline vs Best 비교
# ============================================================


def section_baseline_vs_best(
    study: optuna.Study,
    baseline: dict,
    extra_rows: list[tuple[str, Any, Any]] | None = None,
) -> list[str]:
    """Section: Baseline vs Best 비교 테이블.

    extra_rows: [(label, baseline_key, user_attr_key)] — 추가 비교 행.
    """
    best = study.best_trial
    bl = baseline
    diff_ret = best.value - bl["total_return_pct"]

    lines = [
        "## 2. Baseline vs Best 비교",
        "",
        f"| 지표 | Baseline | Best (#{best.number}) | 차이 |",
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
    ]
    if extra_rows:
        for label, bl_key, ua_key in extra_rows:
            bl_val = bl.get(bl_key, 0)
            best_val = best.user_attrs.get(ua_key, 0)
            lines.append(f"| {label} | {bl_val} | {best_val} | {best_val - bl_val:+d} |")
    lines.append("")
    return lines


# ============================================================
# Section 3: 최적 파라미터
# ============================================================


def section_best_params(study: optuna.Study, baseline_params: dict) -> list[str]:
    """Section: 최적 파라미터 테이블."""
    best = study.best_trial
    lines = [
        f"## 3. 최적 파라미터 (Best Trial #{best.number})",
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
    lines.append("")
    return lines


# ============================================================
# Section 4: Top 5 Trials
# ============================================================


def section_top5_table(
    study: optuna.Study,
    extra_columns: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Section: Top 5 Trials 테이블.

    extra_columns: [(header, user_attr_key)] — v4 CB차단 등 추가 컬럼.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    # 헤더 구성
    headers = ["#", "수익률", "MDD", "Sharpe", "승률", "매도", "손절", "횡보일"]
    if extra_columns:
        headers += [h for h, _ in extra_columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|" + "|".join(["---"] * len(headers)) + "|"

    lines = [
        "",
        "## 4. Top 5 Trials",
        "",
        header_line,
        sep_line,
    ]
    for t in top5:
        row = (
            f"| {t.number} | {t.value:+.2f}% "
            f"| -{t.user_attrs.get('mdd', 0):.2f}% "
            f"| {t.user_attrs.get('sharpe', 0):.4f} "
            f"| {t.user_attrs.get('win_rate', 0):.1f}% "
            f"| {t.user_attrs.get('total_sells', 0)} "
            f"| {t.user_attrs.get('stop_loss_count', 0)} "
            f"| {t.user_attrs.get('sideways_days', 0)} "
        )
        if extra_columns:
            for _, attr_key in extra_columns:
                row += f"| {t.user_attrs.get(attr_key, 0)} "
        row += "|"
        lines.append(row)
    lines.append("")
    return lines


# ============================================================
# Section 5: 파라미터 중요도 (fANOVA)
# ============================================================


def section_importance(study: optuna.Study) -> list[str]:
    """Section: fANOVA 파라미터 중요도. trial 5개 미만이면 빈 리스트."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 5:
        return []
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        return []
    lines = [
        "",
        "## 5. 파라미터 중요도 (fANOVA)",
        "",
        "| 파라미터 | 중요도 |",
        "|---|---|",
    ]
    for param, score in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "\u2588" * int(score * 30)
        lines.append(f"| `{param}` | {score:.4f} {bar} |")
    lines.append("")
    return lines


# ============================================================
# Section 6: Top 5 파라미터 상세
# ============================================================


def section_top5_detail(study: optuna.Study) -> list[str]:
    """Section: Top 5 파라미터 상세 코드 블록."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    lines = [
        "",
        "## 6. Top 5 파라미터 상세",
        "",
    ]
    for rank, t in enumerate(top5, 1):
        lines.append(f"### #{rank} \u2014 Trial {t.number} ({t.value:+.2f}%)")
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
    return lines


# ============================================================
# Section 7: config.py 적용 코드
# ============================================================


def section_config_code(study: optuna.Study) -> list[str]:
    """Section: config.py 적용 코드 (복사용)."""
    best = study.best_trial
    lines = [
        f"## 7. config.py 적용 코드 (Best Trial #{best.number})",
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
    return lines


# ============================================================
# Section 8: Train/Test 성과 비교 (v4 전용)
# ============================================================


def section_train_test(
    test_results: list[dict],
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> list[str]:
    """Section: Train/Test 성과 비교 (과적합 검증)."""
    lines = [
        "## 8. Train/Test 성과 비교 (과적합 검증)",
        "",
        f"> Train 기간: {train_end or '전체'} 이전  |  "
        f"Test 기간: {test_start or 'N/A'} ~ {test_end or '전체'}",
        "",
        "| Rank | Trial# | Train Score | Test 수익률 | Test MDD | Test Sharpe | Test 승률 | 판정 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in test_results:
        tr = item["test_result"]
        if tr is None:
            lines.append(
                f"| {item['rank']} | #{item['trial_number']} "
                f"| {item['train_score']:+.2f} | - | - | - | - | 실패 |"
            )
            continue
        test_ret = tr["total_return_pct"]
        test_mdd = tr["mdd"]
        if test_ret > 10.0 and test_mdd < 12.0:
            verdict = "양호"
        elif test_ret > 0.0:
            verdict = "보통"
        else:
            verdict = "과적합"
        lines.append(
            f"| {item['rank']} | #{item['trial_number']} "
            f"| {item['train_score']:+.2f} "
            f"| {test_ret:+.2f}% "
            f"| -{test_mdd:.2f}% "
            f"| {tr['sharpe']:.4f} "
            f"| {tr['win_rate']:.1f}% "
            f"| {verdict} |"
        )
    lines.append("")
    return lines


# ============================================================
# 리포트 빌더
# ============================================================


def build_report(version: str, sections: list[list[str]]) -> str:
    """섹션들을 합쳐서 최종 리포트 문자열을 생성한다."""
    lines = [
        f"# PTJ {version} Optuna 최적화 리포트",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for section_lines in sections:
        lines += section_lines
    return "\n".join(lines)
