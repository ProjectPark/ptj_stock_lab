"""
초단위 실행 판단 레이어
대상: ptj_stock/backend/app/services/execution_layer.py (신규 파일 또는 기존에 추가)
엔진: {엔진명}
생성일: {날짜}

[배포 방법]
  이 파일을 ptj_stock/backend/app/services/ 에 복사
  또는 기존 execution_layer.py에 해당 시그널용 로직 추가
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ActiveSignal:
    """분봉 시그널이 발생한 후 초단위 실행 대기 상태."""

    signal_key: str          # "twin_entry:MSTU", "stop_loss:BITU"
    signal_type: str         # 시그널 타입
    action: str              # BUY / SELL
    ticker: str              # 대상 종목
    target_price: float      # 시그널 발생 시점 가격
    created_at: datetime     # 시그널 발생 시각
    ttl_sec: int             # 유효 기간 (초)
    executed: bool = False

    # 실행 조건
    price_tolerance_pct: float = 0.5
    min_tick_count: int = 3
    max_spread_pct: float = 1.0


class ExecutionLayer:
    """분봉 시그널을 초단위 tick으로 실행 여부 판단.

    TODO: 엔진별 시그널 타입에 맞게 구현
    """

    def __init__(self) -> None:
        self.active_signals: dict[str, ActiveSignal] = {}
        self._tick_confirmations: dict[str, int] = {}

    def update_signals(self, signals: dict) -> None:
        """1분 주기: 시그널 엔진 결과 → active_signals 갱신."""
        # TODO: 새 시그널 등록, 소멸된 시그널 제거
        ...

    def on_tick(
        self,
        ticker: str,
        price: float,
        volume: float,
        timestamp: datetime,
    ) -> list:
        """1초 주기: tick 도착 → 실행 판단.

        판단 기준:
        1. active_signal 존재하는가?
        2. 가격이 허용 범위 내인가?
        3. N틱 연속 조건 충족하는가? (chattering 방지)
        4. TTL 이내인가?
        5. 급격한 가격 변동 중이 아닌가? (volatility guard)

        Returns:
            실행할 주문 액션 리스트 (빈 리스트 = 실행 안 함)
        """
        # TODO: 구현
        return []

    def expire_stale(self) -> list[str]:
        """TTL 만료된 시그널 정리."""
        expired = []
        now = datetime.now()
        for key, sig in list(self.active_signals.items()):
            elapsed = (now - sig.created_at).total_seconds()
            if sig.ttl_sec > 0 and elapsed > sig.ttl_sec:
                expired.append(key)
                del self.active_signals[key]
                self._tick_confirmations.pop(key, None)
        return expired
