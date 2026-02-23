"""
taejun_attach_pattern - 전략 기본 클래스
========================================
박태준 매매 전략 패턴의 공통 인터페이스.
모든 개별 전략은 BaseStrategy를 상속한다.

데이터 단위: 현재 1분봉 기준, 향후 초봉(tick) 전환 가능.
MarketData는 단위(granularity)에 무관하게 설계.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Action(Enum):
    """시그널 행동 유형"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"


class ExitReason(Enum):
    """청산 사유"""
    TARGET_HIT = "target_hit"           # 목표 수익률 도달
    STOP_LOSS = "stop_loss"             # 손절
    TIME_LIMIT = "time_limit"           # 시간 제한 초과
    CONDITION_BREAK = "condition_break"  # 조건 이탈
    MANUAL = "manual"                   # 수동


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketData:
    """전략에 전달되는 시장 데이터 스냅샷.

    Parameters
    ----------
    changes : dict[str, float]
        종목별 당일 변동률 (%). {"GLD": 0.1, "BITU": -0.4, ...}
    prices : dict[str, float]
        종목별 현재가 (USD). {"GLD": 238.5, ...}
    poly : dict[str, float] | None
        Polymarket 확률. {"btc_up": 0.63, "ndx_up": 0.55, "eth_up": 0.40}
    time : datetime
        현재 시각 (KST 기준).
    history : dict[str, dict] | None
        종목별 히스토리. {"CONL": {"high_3y": 120.0, "low_3y": 8.5, ...}}
    volumes : dict[str, float] | None
        종목별 거래량.
    crypto : dict[str, float] | None
        크립토 스팟 가격 변동률 (%). {"BTC": 0.9, "ETH": 0.9, "SOL": 2.0, "XRP": 5.0}
    poly_timestamps : dict[str, datetime] | None
        각 Polymarket 값의 마지막 갱신 시각. 품질 필터에 사용.
    poly_prev : dict[str, float] | None
        이전 시점의 Polymarket 확률. 이머전시 모드 변동 감지에 사용.
    ohlcv : dict[str, pd.DataFrame] | None
        종목별 OHLCV 히스토리 (pd.DataFrame, columns: open/high/low/close/volume).
        SidewaysDetector v5 기술지표 계산, CONL ADX/EMA, SOXL 독립 매매 등에 사용.
        예: {"SPY": df, "QQQ": df, "CONL": df}
    luld_counts : dict[str, int] | None
        종목별 LULD(Limit Up-Limit Down) 거래중단 발생 횟수.
        급락 역매수(5-5절) 트리거 조건에 사용. 예: {"SOXL": 3, "CONL": 0}
    """
    changes: dict[str, float]
    prices: dict[str, float]
    poly: dict[str, float] | None
    time: datetime
    history: dict[str, dict] | None = None
    volumes: dict[str, float] | None = None
    crypto: dict[str, float] | None = None
    poly_timestamps: dict[str, datetime] | None = None
    poly_prev: dict[str, float] | None = None
    ohlcv: "dict[str, pd.DataFrame] | None" = None
    luld_counts: dict[str, int] | None = None

    @classmethod
    def from_backtest_bar(
        cls,
        changes: dict[str, dict],
        prices: dict[str, float] | None = None,
        poly_probs: dict[str, float] | None = None,
        ts: "datetime | None" = None,
    ) -> "MarketData":
        """Legacy 백테스트 데이터 → MarketData 변환.

        Parameters
        ----------
        changes : dict[str, dict]
            {ticker: {"change_pct": float, ...}, ...} 형식의 Legacy 변동률.
        prices : dict[str, float] | None
            현재가 dict.
        poly_probs : dict[str, float] | None
            Polymarket 확률.
        ts : datetime | None
            타임스탬프.
        """
        flat_changes = {k: v.get("change_pct", 0.0) for k, v in changes.items()}
        return cls(
            changes=flat_changes,
            prices=prices or {},
            poly=poly_probs,
            time=ts or datetime.now(),
        )


@dataclass
class Signal:
    """전략이 반환하는 시그널.

    Parameters
    ----------
    action : Action
        BUY / SELL / HOLD / SKIP
    ticker : str
        대상 종목.
    size : float
        자본 대비 매수/매도 비율 (0.0~1.0). 1.0 = 전액.
    target_pct : float
        목표 수익률 (%). 매도 시그널이면 0.
    reason : str
        로그용 사유 문자열.
    exit_reason : ExitReason | None
        매도 시그널일 때 청산 사유.
    timestamp : datetime | None
        시그널 생성 시각.
    metadata : dict
        추가 정보 (분할매도 횟수, 재투자 대상 등).
    """
    action: Action
    ticker: str
    size: float
    target_pct: float
    reason: str
    exit_reason: ExitReason | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """보유 포지션 정보.

    Parameters
    ----------
    ticker : str
        종목.
    avg_price : float
        평균 매수가 (USD).
    qty : float
        보유 수량.
    entry_time : datetime
        최초 진입 시각.
    strategy_name : str
        진입 전략 이름.
    stage : int
        매수 단계 (1=초기, 2=추가매수, ...).
    confirmed : bool
        MT_VNQ3 §5: FILLED 또는 Fill Window 종료로 확정된 상태.
        False이면 아직 pending/partial 상태.
    """
    ticker: str
    avg_price: float
    qty: float
    entry_time: datetime
    strategy_name: str
    stage: int = 1
    confirmed: bool = False

    def pnl_pct(self, current_price: float) -> float | None:
        """매수 평균가 대비 현재 손익률(%)을 반환한다.

        Returns None if current_price or avg_price is non-positive.
        """
        if current_price <= 0 or self.avg_price <= 0:
            return None
        return (current_price - self.avg_price) / self.avg_price * 100


# ---------------------------------------------------------------------------
# Base strategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """모든 전략의 공통 인터페이스.

    서브클래스는 반드시 name, check_entry, check_exit, generate_signal을 구현.
    """

    name: str = ""
    version: str = "1.0"
    description: str = ""

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    @abstractmethod
    def check_entry(self, market: MarketData) -> bool:
        """진입 조건 충족 여부를 반환한다."""
        ...

    @abstractmethod
    def check_exit(self, market: MarketData, position: Position) -> bool:
        """청산 조건 충족 여부를 반환한다."""
        ...

    @abstractmethod
    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        """시장 데이터와 보유 포지션을 받아 시그널을 생성한다.

        - 포지션 없음 → 진입 검토 (BUY or SKIP)
        - 포지션 있음 → 청산 검토 (SELL or HOLD)
        """
        ...

    def generate_signals(self, market: MarketData,
                         positions: dict[str, "Position"] | None = None,
                         ) -> list[Signal]:
        """복수 시그널 생성 (IAU+GDXU 동시 매수 등 다종목 전략용).

        기본 구현은 generate_signal()을 래핑하여 단일 시그널 반환.
        복수 종목을 다루는 전략(예: VixGold)에서 오버라이드.

        인터페이스 사용 패턴:
        - 단일 종목 전략: generate_signal() 구현, generate_signals()는 기본 래퍼 사용
        - 다종목 전략 (VixGold 등): generate_signals() 오버라이드
        - evaluate() 전략 (TwinPair, ConditionalCoin 등):
          generate_signal()은 SKIP 반환, composite_signal_engine이 evaluate() 직접 호출

        Parameters
        ----------
        market : MarketData
        positions : dict[str, Position] | None
            보유 포지션 dict. 키는 ticker.
        """
        pos = None
        if positions:
            # 첫 번째 관련 포지션을 가져온다 (단일 포지션 전략 호환)
            for p in positions.values():
                if p.strategy_name == self.name:
                    pos = p
                    break
        sig = self.generate_signal(market, pos)
        if sig.action in (Action.BUY, Action.SELL):
            return [sig]
        return []

    def validate_params(self) -> list[str]:
        """파라미터 유효성 검증. 에러 메시지 리스트를 반환한다.

        기본 구현은 빈 리스트 (에러 없음).
        서브클래스에서 필요시 오버라이드.
        """
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} v{self.version}>"
