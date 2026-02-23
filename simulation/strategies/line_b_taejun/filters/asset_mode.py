"""
자산 모드 시스템 (Asset Mode)
================================
전략을 공격/방어/조심/이머전시 모드로 분류하여 자본 배분 관리.

모드 우선순위: 이머전시 > 공격 > 방어 > 조심

- 이머전시: 모든 모드 오버라이드, Polymarket 급변 대응
- 공격: 전액 매수/매도 (잽, 저가매수, VIX 등)
- 방어: 적립식 (섹터 로테이션)
- 조심: 공격자산 레버리지 50% 제한 (REIT 트리거)

출처: kakaotalk_trading_notes_2026-02-19.csv — 자산 모드 시스템
"""
from __future__ import annotations

from enum import IntEnum

from ..common.params import ASSET_MODE


class AssetMode(IntEnum):
    """자산 모드 (숫자가 낮을수록 우선순위 높음)."""
    EMERGENCY = 0    # 이머전시 — 최우선, 모든 모드 오버라이드
    ATTACK = 10      # 공격 — 전액 매수/매도 (잽, 저가매수)
    DEFENSE = 20     # 방어 — 적립식 (섹터 로테이션)
    CAUTIOUS = 30    # 조심 — 공격자산 레버리지 50%


class AssetModeManager:
    """자산 모드 관리.

    - 이머전시 > 공격 > 방어 > 조심
    - 공격/방어 자본 분리
    - 조심모드: 공격자산 레버리지 50% 제한
    - 이머전시: 모든 모드 오버라이드

    Parameters
    ----------
    params : dict | None
        ASSET_MODE 파라미터. None이면 기본값.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or ASSET_MODE
        self._current_mode: AssetMode = AssetMode.ATTACK
        self._attack_strategies: set[str] = set(
            self.params.get("attack_strategies", [])
        )
        self._defense_strategies: set[str] = set(
            self.params.get("defense_strategies", [])
        )
        self._cautious_leverage_pct: int = self.params.get(
            "cautious_leverage_pct", 50
        )

    @property
    def mode(self) -> AssetMode:
        return self._current_mode

    def set_mode(self, mode: AssetMode) -> None:
        self._current_mode = mode

    def activate_emergency(self) -> None:
        """이머전시 모드 활성화."""
        self._current_mode = AssetMode.EMERGENCY

    def activate_cautious(self, attack_leverage_pct: int | None = None) -> None:
        """조심모드 활성화 (REIT 트리거)."""
        self._current_mode = AssetMode.CAUTIOUS
        if attack_leverage_pct is not None:
            self._cautious_leverage_pct = attack_leverage_pct

    def deactivate_emergency(self) -> None:
        """이머전시 해제 → 공격모드로 복귀."""
        if self._current_mode == AssetMode.EMERGENCY:
            self._current_mode = AssetMode.ATTACK

    def deactivate_cautious(self) -> None:
        """조심모드 해제 → 공격모드로 복귀."""
        if self._current_mode == AssetMode.CAUTIOUS:
            self._current_mode = AssetMode.ATTACK

    # ------------------------------------------------------------------
    # 전략 분류
    # ------------------------------------------------------------------

    def is_attack_strategy(self, strategy_name: str) -> bool:
        return strategy_name in self._attack_strategies

    def is_defense_strategy(self, strategy_name: str) -> bool:
        return strategy_name in self._defense_strategies

    # ------------------------------------------------------------------
    # 자본 배분 조정
    # ------------------------------------------------------------------

    def get_leverage_multiplier(self, strategy_name: str) -> float:
        """현재 모드에서 해당 전략의 레버리지 배율을 반환한다.

        - 이머전시: 1.0 (제한 없음)
        - 공격: 1.0
        - 방어: 1.0
        - 조심: 공격전략이면 cautious_leverage_pct/100, 방어전략이면 1.0
        """
        if self._current_mode == AssetMode.CAUTIOUS:
            if self.is_attack_strategy(strategy_name):
                return self._cautious_leverage_pct / 100.0
        return 1.0

    def is_emergency_active(self) -> bool:
        return self._current_mode == AssetMode.EMERGENCY

    def is_cautious_active(self) -> bool:
        return self._current_mode == AssetMode.CAUTIOUS
