"""
엔진 이식: {엔진명}
소스: ptj_stock_lab/simulation/strategies/{경로}
대상: ptj_stock/backend/app/core/signals.py
생성일: {날짜}

[삽입 위치]
  - DEFAULT 상수: 파일 상단 상수 영역에 추가
  - 함수: signals.py 하단, generate_all_signals() 위에 추가
  - generate_all_signals(): result dict에 키 추가
"""

from __future__ import annotations


# ── 이 블록을 signals.py 상단 상수 영역에 추가 ──────────────
DEFAULT_XXX_PARAMS: dict = {
    # "key": value,   # 출처: lab config.py V5_XXX
}


# ── 이 함수를 signals.py에 추가 ────────────────────────────
def check_xxx_signal(
    changes: dict[str, dict],
    params: dict | None = None,
) -> list[dict]:
    """TODO: 시그널 로직 구현.

    Args:
        changes: 종목별 등락률 dict
        params: 파라미터 dict (None이면 DEFAULT_XXX_PARAMS 사용)

    Returns:
        시그널 dict 리스트
    """
    if params is None:
        params = DEFAULT_XXX_PARAMS
    results: list[dict] = []
    # TODO: lab 엔진에서 로직 이식
    return results


# ── generate_all_signals()에 아래 라인 추가 ─────────────────
# result["xxx"] = check_xxx_signal(changes, params=xxx_params)
