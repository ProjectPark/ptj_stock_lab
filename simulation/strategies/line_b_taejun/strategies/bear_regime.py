"""
급락 체제 감지기 (Bear Regime Detector) — todd_fuck_v1
========================================================
Polymarket 절대값 기반으로 급락 체제를 선언하고
롱 레버리지 축소 + 인버스 ETF 진입 신호를 생성한다.

기존 Emergency Mode (±30pp 스윙)와의 차이:
  - Emergency: 단발성 급변 감지 → 즉시 대응
  - BearRegime: 절대값 수준 기반 → 지속 체제 관리

체제 ON 조건 (AND):
  btc_up < 0.43          ← 시장 컨센서스 비관 (v2 OOS: 0.38→0.43)
  btc_monthly_dip > 0.30 ← 이달 저점 도달 확률 높음

체제 OFF (히스테리시스):
  btc_up > 0.57 회복 시 해제 (v2 OOS: 0.50→0.57)

체제 ON 효과:
  - 모든 롱 전략 unit_mul × 0.5 축소
  - btc_up < 0.35 → BITI 진입 허용
  - ndx_up < 0.38 → SOXS 진입 허용

Polymarket 확률 기반 포지션 연속화 (scale_unit_mul_by_poly):
  btc_up 구간별로 unit_mul을 연속적으로 조정.
  이진 게이트(ON/OFF) 대신 부드러운 스케일링 적용.
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from ..common.params import BEAR_REGIME, POLY_POSITION_SCALE
from ..common.registry import register


# ============================================================
# 유틸리티: 포지션 크기 연속화
# ============================================================

def scale_unit_mul_by_poly(
    base_mul: float,
    btc_up: float,
    params: dict | None = None,
    is_bear_regime: bool = False,
) -> float:
    """btc_up 확률 기반 unit_mul 연속 스케일링.

    Parameters
    ----------
    base_mul : float
        기본 unit_mul (파라미터에 정의된 값)
    btc_up : float
        Polymarket btc_up 확률 (0~1)
    params : dict | None
        POLY_POSITION_SCALE 파라미터. None이면 기본값 사용.
    is_bear_regime : bool
        BearRegime ON 상태일 때 True → btc_up 차단 면제 (scale=1.0).
        regime_factor만으로 포지션 축소를 관리한다.

    Returns
    -------
    float
        조정된 unit_mul. 0.0이면 진입 차단.

    Examples
    --------
    >>> scale_unit_mul_by_poly(3.0, 0.75)  # btc_up 75% → 1.5x → 4.5
    4.5
    >>> scale_unit_mul_by_poly(3.0, 0.50)  # btc_up 50% → 1.0x → 3.0
    3.0
    >>> scale_unit_mul_by_poly(3.0, 0.48)  # btc_up 48% → 0.7x → 2.1
    2.1
    >>> scale_unit_mul_by_poly(3.0, 0.40)  # btc_up 40% → 차단 → 0.0
    0.0
    >>> scale_unit_mul_by_poly(3.0, 0.35, is_bear_regime=True)  # 면제 → 1.0x → 3.0
    3.0
    """
    p = params or POLY_POSITION_SCALE
    if not p.get("enabled", True):
        return base_mul

    thresholds: list[float] = p["btc_up_thresholds"]   # [0.45, 0.55, 0.70]
    factors: list[float] = p["unit_mul_factors"]         # [0.0, 0.7, 1.0, 1.5]

    if btc_up < thresholds[0]:
        if is_bear_regime:
            return base_mul * 1.0      # BearRegime 면제: 차단 대신 1.0x
        return base_mul * factors[0]   # 0.0 → 차단
    elif btc_up < thresholds[1]:
        return base_mul * factors[1]   # 0.7x
    elif btc_up < thresholds[2]:
        return base_mul * factors[2]   # 1.0x (기본)
    else:
        return base_mul * factors[3]   # 1.5x (확대)


# ============================================================
# Bear Regime 감지기
# ============================================================

class BearRegimeDetector:
    """급락 체제 감지기.

    체제 상태를 유지하며 (히스테리시스) ON/OFF 전환을 관리한다.
    전략 클래스가 아닌 유틸리티 클래스 — BaseStrategy를 상속하지 않는다.

    Parameters
    ----------
    params : dict | None
        BEAR_REGIME 파라미터. None이면 기본값 사용.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or BEAR_REGIME
        self._is_bear: bool = False

    # ------------------------------------------------------------------
    # 체제 점수
    # ------------------------------------------------------------------

    def get_regime_score(self, market: MarketData) -> float:
        """급락 체제 점수 계산 (0.0=강세, 1.0=강한 약세).

        세 가지 Polymarket 신호를 가중 평균:
          - btc_up_score (50%): btc_up이 낮을수록 높음
          - dip_score (30%): btc_monthly_dip이 높을수록 높음
          - pressure_score (20%): btc_upside_pressure가 낮을수록 높음
        """
        poly = market.poly or {}

        btc_up = poly.get("btc_up", 0.5)
        btc_monthly_dip = poly.get("btc_monthly_dip", 0.0)
        btc_upside_pressure = poly.get("btc_upside_pressure", 0.5)

        # btc_up: 0.5 기준 반전 (낮을수록 약세)
        btc_up_score = max(0.0, min(1.0, (0.5 - btc_up) / 0.5))

        # btc_monthly_dip: 그대로 (높을수록 약세)
        dip_score = min(1.0, btc_monthly_dip)

        # btc_upside_pressure: 0.5 기준 반전
        pressure_score = max(0.0, min(1.0, (0.5 - btc_upside_pressure) / 0.5))

        score = (btc_up_score * 0.5 + dip_score * 0.3 + pressure_score * 0.2)
        return round(min(1.0, max(0.0, score)), 4)

    # ------------------------------------------------------------------
    # 체제 판정 (히스테리시스)
    # ------------------------------------------------------------------

    def is_bear_regime(self, market: MarketData) -> bool:
        """급락 체제 여부 판정.

        ON 조건 (AND):
          btc_up < btc_up_min (0.43)
          btc_monthly_dip > monthly_dip_min (0.30)

        OFF 조건 (히스테리시스):
          btc_up >= recovery_threshold (0.57)
        """
        poly = market.poly or {}
        btc_up = poly.get("btc_up", 0.5)
        btc_monthly_dip = poly.get("btc_monthly_dip", 0.0)

        btc_up_min: float = self.params.get("btc_up_min", 0.38)
        monthly_dip_min: float = self.params.get("monthly_dip_min", 0.30)
        recovery: float = self.params.get("recovery_threshold", 0.50)

        if self._is_bear:
            if btc_up >= recovery:
                self._is_bear = False
        else:
            if btc_up < btc_up_min and btc_monthly_dip > monthly_dip_min:
                self._is_bear = True

        return self._is_bear

    # ------------------------------------------------------------------
    # 롱 레버리지 조정
    # ------------------------------------------------------------------

    def get_long_leverage_factor(self, market: MarketData) -> float:
        """체제별 롱 unit_mul 배율 (4단계).

        Returns
        -------
        float
            NORMAL=1.0, WARN=0.8, BEAR=0.5, STRONG_BEAR=0.3
        """
        score = self.get_regime_score(market)
        is_bear = self.is_bear_regime(market)

        bear_thresh = self.params.get("regime_score_bear", 0.55)
        warn_thresh = self.params.get("regime_score_warn", 0.40)

        if score >= bear_thresh:
            return self.params.get("strong_bear_leverage", 0.3)
        if score >= warn_thresh:
            return self.params.get("warn_leverage", 0.8)
        if is_bear:
            return self.params.get("cautious_leverage", 0.5)
        return 1.0

    # ------------------------------------------------------------------
    # 인버스 ETF 진입 신호
    # ------------------------------------------------------------------

    def get_inverse_ticker(self, market: MarketData) -> str | None:
        """급락 체제에서 매수할 인버스 ETF를 반환.

        체제 OFF이면 None.
        btc_up < inverse_btc_up_max → BITI
        ndx_up < inverse_ndx_up_max → SOXS
        둘 다 해당하면 btc_up 기준 우선.
        """
        if not self.is_bear_regime(market):
            return None

        poly = market.poly or {}
        btc_up = poly.get("btc_up", 0.5)
        ndx_up = poly.get("ndx_up", 0.5)

        btc_max: float = self.params.get("inverse_btc_up_max", 0.35)
        ndx_max: float = self.params.get("inverse_ndx_up_max", 0.38)

        if btc_up < btc_max:
            return self.params.get("btc_bear_ticker", "BITI")
        if ndx_up < ndx_max:
            return self.params.get("ndx_bear_ticker", "SOXS")
        return None

    # ------------------------------------------------------------------
    # 요약 정보
    # ------------------------------------------------------------------

    def get_status_dict(self, market: MarketData) -> dict:
        """현재 체제 상태를 딕셔너리로 반환 (로깅/디버깅용)."""
        poly = market.poly or {}
        score = self.get_regime_score(market)
        is_bear = self.is_bear_regime(market)
        inverse = self.get_inverse_ticker(market)
        warn_threshold: float = self.params.get("regime_score_warn", 0.40)
        bear_threshold: float = self.params.get("regime_score_bear", 0.55)

        if score >= bear_threshold:
            label = "STRONG_BEAR"
        elif score >= warn_threshold:
            label = "WARN"
        elif is_bear:
            label = "BEAR"
        else:
            label = "NORMAL"

        # 4단계 leverage factor
        if score >= bear_threshold:
            lev_factor = self.params.get("strong_bear_leverage", 0.3)
        elif score >= warn_threshold:
            lev_factor = self.params.get("warn_leverage", 0.8)
        elif is_bear:
            lev_factor = self.params.get("cautious_leverage", 0.5)
        else:
            lev_factor = 1.0

        return {
            "is_bear": is_bear,
            "score": score,
            "label": label,
            "btc_up": poly.get("btc_up", 0.5),
            "btc_monthly_dip": poly.get("btc_monthly_dip", 0.0),
            "btc_upside_pressure": poly.get("btc_upside_pressure", 0.5),
            "inverse_ticker": inverse,
            "long_leverage_factor": lev_factor,
        }


# ============================================================
# BearRegimeLong 전략 — 인버스 ETF 진입
# ============================================================

@register
class BearRegimeLong(BaseStrategy):
    """급락 체제에서 인버스 ETF를 매수하는 전략.

    Emergency Mode가 단발성 스윙 대응이라면,
    BearRegimeLong은 지속적 급락 체제에서 인버스 포지션을 관리한다.

    진입: BearRegimeDetector.get_inverse_ticker() 반환 시
    청산: 체제 해제 OR 목표/손절 도달
    """

    name = "bear_regime_long"
    version = "1.0"
    description = "급락 체제(btc_up<43%, monthly_dip>30%) → 인버스 ETF 매수"

    def __init__(self, params: dict | None = None):
        super().__init__(params or BEAR_REGIME)
        self._detector = BearRegimeDetector(params or BEAR_REGIME)

    def check_entry(self, market: MarketData) -> bool:
        """인버스 ETF 진입 가능 여부."""
        return self._detector.get_inverse_ticker(market) is not None

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """인버스 청산 (4단계 우선순위).

        1. 체제 해제 (BearRegime OFF)
        2. 목표수익률 달성 (INV_TARGET_PCT)
        3. 손절 (INV_STOP_LOSS_PCT)
        4. 보유 기한 초과 (INV_MAX_HOLD_DAYS)
        """
        # P1: 체제 해제
        if not self._detector.is_bear_regime(market):
            return True

        # P2-P3: 목표/손절
        current = market.prices.get(position.ticker, 0)
        pnl_pct = position.pnl_pct(current)
        if pnl_pct is not None:
            target = self.params.get("inv_target_pct", 8.0)
            stoploss = self.params.get("inv_stop_loss_pct", -5.0)
            if pnl_pct >= target:
                return True
            if pnl_pct <= -abs(stoploss):
                return True

        # P4: 보유 기한
        max_days = self.params.get("inv_max_hold_days", 30)
        if position.entry_time and market.time:
            hold_days = (market.time - position.entry_time).days
            if hold_days > max_days:
                return True

        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        # 청산
        if position is not None:
            if self.check_exit(market, position):
                # 청산 사유 판별
                exit_reason = ExitReason.CONDITION_BREAK
                reason_text = "체제 해제"

                if not self._detector.is_bear_regime(market):
                    exit_reason = ExitReason.CONDITION_BREAK
                    reason_text = "체제 해제"
                else:
                    current = market.prices.get(position.ticker, 0)
                    pnl_pct = position.pnl_pct(current)
                    if pnl_pct is not None:
                        target = self.params.get("inv_target_pct", 8.0)
                        stoploss = self.params.get("inv_stop_loss_pct", -5.0)
                        if pnl_pct >= target:
                            exit_reason = ExitReason.TARGET_HIT
                            reason_text = f"목표달성 {pnl_pct:.1f}%"
                        elif pnl_pct <= -abs(stoploss):
                            exit_reason = ExitReason.STOP_LOSS
                            reason_text = f"손절 {pnl_pct:.1f}%"
                        else:
                            exit_reason = ExitReason.TIME_LIMIT
                            reason_text = "보유기한 초과"
                    else:
                        exit_reason = ExitReason.TIME_LIMIT
                        reason_text = "보유기한 초과"

                return Signal(
                    action=Action.SELL,
                    ticker=position.ticker,
                    size=1.0,
                    target_pct=0,
                    reason=f"bear_regime: {reason_text} → 인버스 청산",
                    exit_reason=exit_reason,
                    metadata={"bear_regime_exit": True},
                )
            return Signal(Action.HOLD, position.ticker, 0, 0, "bear_regime: 체제 유지")

        # 진입
        inverse_ticker = self._detector.get_inverse_ticker(market)
        if inverse_ticker is None:
            return Signal(Action.SKIP, "", 0, 0, "bear_regime: 체제 미진입")

        poly = market.poly or {}
        status = self._detector.get_status_dict(market)
        label = status["label"]
        target_pct: float = self.params.get("inv_target_pct", 8.0)

        # 체제별 인버스 투입 비율: STRONG_BEAR=0.8, BEAR=0.3
        if label == "STRONG_BEAR":
            size = self.params.get("inv_size_strong_bear", 0.8)
        elif label == "BEAR":
            size = self.params.get("inv_size_bear", 0.3)
        else:
            size = self.params.get("inv_size_bear", 0.3)

        return Signal(
            action=Action.BUY,
            ticker=inverse_ticker,
            size=size,
            target_pct=target_pct,
            reason=(
                f"bear_regime: {inverse_ticker} 진입 "
                f"(btc_up={poly.get('btc_up', 0):.2f}, "
                f"dip={poly.get('btc_monthly_dip', 0):.2f}, "
                f"score={status['score']:.3f}, "
                f"label={label}, size={size:.1f})"
            ),
            metadata={
                "bear_regime": True,
                "regime_score": status["score"],
                "regime_label": label,
                "inv_target_pct": target_pct,
                "inv_stop_loss_pct": self.params.get("inv_stop_loss_pct", -5.0),
            },
        )

    def validate_params(self) -> list[str]:
        errors = []
        if self.params.get("btc_up_min", 0) >= self.params.get("recovery_threshold", 1):
            errors.append("btc_up_min must be < recovery_threshold")
        return errors
