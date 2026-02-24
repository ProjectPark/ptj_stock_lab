"""
엔진 이식: {엔진명}
소스: ptj_stock_lab/simulation/strategies/{경로}
대상: ptj_stock/backend/app/services/auto_trader.py
생성일: {날짜}

[삽입 위치]
  - 엔진 함수: auto_trader.py 내 기존 엔진 함수들 아래에 추가
  - evaluate_and_execute(): actions.extend() 호출 추가 (우선순위 위치에)
"""

from __future__ import annotations


# ── 이 함수를 auto_trader.py에 추가 ────────────────────────
def _engine_xxx(
    signals: dict,
    latest: dict,
    session: object,  # MarketSession
    balance_cache: object,  # _BalanceCache
) -> list:  # list[_OrderAction]
    """TODO: 엔진 로직 구현.

    Args:
        signals: generate_all_signals() 전체 결과
        latest: Redis ptj:latest (현재가 조회용)
        session: 현재 마켓 세션
        balance_cache: 잔고 캐시

    Returns:
        _OrderAction 리스트
    """
    actions = []
    signal_data = signals.get("xxx", [])
    if not signal_data:
        return actions

    # TODO: lab 엔진에서 로직 이식
    # _OrderAction 필드:
    #   engine, symbol, side, order_type, quantity, price, signal_data

    return actions


# ── evaluate_and_execute()에 아래 라인 추가 ─────────────────
# 우선순위: N번째 (사용자 확인 필요)
# actions.extend(_engine_xxx(signals, latest, session, balance_cache))
