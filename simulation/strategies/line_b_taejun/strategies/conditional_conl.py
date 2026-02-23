"""
조건부 CONL 매수 전략 (Conditional-CONL)
========================================
ETHU/XXRP/SOLT 각각 +N% 이상이면 CONL 매수 시그널.
Legacy: signals_v2.check_conditional_conl_v2()

v5 변경 (6-2절):
- trigger_pct: 3.0% → 4.5%
- ADX(14) >= 18 진입 필터 추가
- 20 EMA 기울기 양수 진입 필터 추가
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.base import Action, BaseStrategy, MarketData, Position, Signal
from ..common.indicators import compute_adx, ema_slope_positive
from ..common.registry import register

if TYPE_CHECKING:
    import pandas as pd


@register
class ConditionalConlStrategy(BaseStrategy):
    """ETHU/XXRP/SOLT 트리거 기반 CONL 조건부 매수.

    v5 변경:
    - trigger_pct: 4.5% (v4: 3.0%)
    - ADX(14) >= 18 + EMA 기울기 양수 진입 필터 (6-2절)

    Note: 이 전략은 evaluate() 인터페이스를 사용합니다.
    generate_signal()은 항상 SKIP을 반환하며,
    composite_signal_engine에서 evaluate()를 직접 호출합니다.
    """

    name = "conditional_conl"
    version = "1.1"
    description = "ETHU/XXRP/SOLT 각각 4.5%+ & ADX>=18 & EMA양수이면 CONL 매수"

    def __init__(self, params: dict | None = None):
        defaults = {
            "trigger_pct": 4.5,         # v5: 4.5% (v4: 3.0%)
            "sell_profit_pct": 2.8,     # 순수익 +2.8% 매도
            "sell_avg_pct": 1.0,        # 트리거 평균 <1% → 매도
            "adx_min": 18,              # ADX(14) 최소 (6-2절)
            "ema_slope_positive": True,  # 20 EMA 기울기 양수 필수
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    # ------------------------------------------------------------------
    # ADX / EMA 필터
    # ------------------------------------------------------------------

    def check_adx_ema(
        self,
        ohlcv_conl: "pd.DataFrame | None",
    ) -> tuple[bool, str]:
        """ADX + EMA 기울기 필터 검증.

        Parameters
        ----------
        ohlcv_conl : pd.DataFrame | None
            CONL의 OHLCV DataFrame. None이면 필터 스킵 (통과로 처리).

        Returns
        -------
        (passed, reason)
        """
        if ohlcv_conl is None:
            # 데이터 없으면 필터 통과로 처리 (백테스트 미지원 환경 호환)
            return True, "no ohlcv data — filter skipped"

        if len(ohlcv_conl) < 20:
            return True, "insufficient ohlcv rows — filter skipped"

        # ADX 체크
        adx_min = self.params.get("adx_min", 18)
        adx_val = compute_adx(ohlcv_conl)
        if adx_val < adx_min:
            return False, f"ADX {adx_val:.1f} < {adx_min} (추세 없음)"

        # EMA 기울기 체크
        if self.params.get("ema_slope_positive", True):
            ema_ok = ema_slope_positive(ohlcv_conl)
            if not ema_ok:
                return False, "20 EMA 기울기 음수 (하락 추세)"

        return True, f"ADX {adx_val:.1f} >= {adx_min}, EMA 양수"

    # ------------------------------------------------------------------
    # Legacy evaluate() — 백테스트/신호 생성 공용
    # ------------------------------------------------------------------

    def evaluate(
        self,
        changes: dict,
        ohlcv_conl: "pd.DataFrame | None" = None,
    ) -> dict:
        """CONL 매수 시그널 판정.

        Parameters
        ----------
        changes : dict
            {ticker: {"change_pct": float, ...}, ...}
        ohlcv_conl : pd.DataFrame | None
            CONL의 OHLCV DataFrame. ADX/EMA 필터에 사용.

        Returns
        -------
        dict
            triggers, all_above_threshold, adx_ema_passed, trigger_avg_pct,
            sell_on_avg_drop, conl_pct, buy_signal, message
        """
        trigger_pct = self.params.get("trigger_pct", 4.5)
        sell_avg_pct = self.params.get("sell_avg_pct", 1.0)
        trigger_tickers = ["ETHU", "XXRP", "SOLT"]

        trigger_info: dict[str, dict] = {}
        all_above = True
        pct_sum = 0.0

        for ticker in trigger_tickers:
            data = changes.get(ticker, {})
            pct = data.get("change_pct", 0.0) if isinstance(data, dict) else float(data)
            above = pct >= trigger_pct
            if not above:
                all_above = False
            pct_sum += pct
            trigger_info[ticker] = {"change_pct": pct, "above_threshold": above}

        trigger_avg = pct_sum / len(trigger_tickers) if trigger_tickers else 0.0
        sell_on_avg_drop = trigger_avg < sell_avg_pct

        conl_data = changes.get("CONL", {})
        conl_pct = (
            conl_data.get("change_pct", 0.0) if isinstance(conl_data, dict)
            else float(conl_data)
        )

        # ADX + EMA 필터 (v5 신규)
        adx_ema_passed, adx_ema_reason = self.check_adx_ema(ohlcv_conl)

        buy_signal = all_above and adx_ema_passed

        if all_above and adx_ema_passed:
            message = (
                f"ETHU/XXRP/SOLT 각각 ≥ +{trigger_pct:.1f}% + {adx_ema_reason} "
                f"→ CONL 매수 (평균 {trigger_avg:+.2f}%)"
            )
        elif all_above and not adx_ema_passed:
            message = (
                f"트리거 충족이나 ADX/EMA 필터 차단: {adx_ema_reason} "
                f"→ CONL 매수 보류"
            )
        else:
            below = [
                t for t in trigger_tickers
                if not trigger_info[t]["above_threshold"]
            ]
            message = (
                f"{', '.join(below)} 미달 (기준 +{trigger_pct:.1f}%) "
                f"→ CONL 매수 보류 (평균 {trigger_avg:+.2f}%)"
            )

        return {
            "triggers": trigger_info,
            "all_above_threshold": all_above,
            "adx_ema_passed": adx_ema_passed,
            "adx_ema_reason": adx_ema_reason,
            "trigger_avg_pct": round(trigger_avg, 4),
            "sell_on_avg_drop": sell_on_avg_drop,
            "conl_pct": conl_pct,
            "buy_signal": buy_signal,
            "message": message,
        }

    def check_entry(self, market: MarketData) -> bool:
        ohlcv_conl = (market.ohlcv or {}).get("CONL") if market.ohlcv else None
        result = self.evaluate(
            {t: {"change_pct": market.changes.get(t, 0.0)} for t in ["ETHU", "XXRP", "SOLT", "CONL"]},
            ohlcv_conl=ohlcv_conl,
        )
        return result["buy_signal"]

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        return Signal(Action.SKIP, "CONL", 0, 0,
                      "use evaluate() for conditional conl signals")
