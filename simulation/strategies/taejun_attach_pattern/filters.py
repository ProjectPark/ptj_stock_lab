"""
Legacy 시그널 필터 — OOP 래퍼
==============================
signals_v2의 determine_market_mode, check_gold_signal_v2, evaluate_sideways를 래핑.
"""
from __future__ import annotations
from dataclasses import dataclass


class MarketModeFilter:
    """Polymarket 확률 기반 시황 판정 필터.

    Legacy: signals_v2.determine_market_mode()
    v3+: sideways_active 추가
    """
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def evaluate(self, poly_probs: dict | None, sideways_active: bool = False) -> str:
        """시황 모드를 반환한다: "sideways" | "bearish" | "bullish" | "normal"

        우선순위: sideways > bearish > bullish > normal
        """
        if sideways_active:
            return "sideways"

        if poly_probs is None:
            return "normal"

        btc_up = poly_probs.get("btc_up", 0.0)
        ndx_up = poly_probs.get("ndx_up", 0.0)
        eth_up = poly_probs.get("eth_up", 0.0)

        if btc_up >= 0.70:
            return "bullish"
        if btc_up <= 0.20 and ndx_up <= 0.20 and eth_up <= 0.20:
            return "bearish"

        return "normal"


class GoldFilter:
    """GLD 변동률 기반 매매 허용 필터.

    Legacy: signals_v2.check_gold_signal_v2()
    """
    def evaluate(self, gld_pct: float) -> dict:
        """GLD 변동률 기반 매매 허용 여부를 반환한다.

        Returns: dict with ticker, change_pct, warning, allow_extra_buy, message
        """
        warning = gld_pct > 0
        allow_extra_buy = gld_pct < 0

        if warning:
            message = f"GLD +{gld_pct:.2f}% — 금 양봉, 전체 매매 금지"
        elif allow_extra_buy:
            message = f"GLD {gld_pct:.2f}% — 금 음봉, 현금 추가 매수 허용"
        else:
            message = "GLD 0.00% — 변동 없음, 기존 포지션 유지"

        return {
            "ticker": "GLD",
            "change_pct": gld_pct,
            "warning": warning,
            "allow_extra_buy": allow_extra_buy,
            "message": message,
        }


class SidewaysDetector:
    """횡보장 5개 지표 평가 필터.

    Legacy: signals_v3/v5.evaluate_sideways()
    v4 dual-path: indicators dict를 직접 전달 가능
    """
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def evaluate(
        self,
        poly_probs: dict | None = None,
        changes: dict | None = None,
        gap_fail_count: int = 0,
        trigger_fail_count: int = 0,
        indicators: dict[str, bool] | None = None,
    ) -> dict:
        """횡보장 여부를 판정한다.

        v4 dual-path: indicators가 주어지면 해당 dict로 직접 판정.
        그 외: poly_probs, changes에서 5개 지표를 계산.

        Returns: dict with indicators, count, is_sideways, message
        """
        poly_low = self.params.get("poly_low", 0.40)
        poly_high = self.params.get("poly_high", 0.60)
        gld_threshold = self.params.get("gld_threshold", 0.3)
        gap_fail_threshold = self.params.get("gap_fail_threshold", 2)
        trigger_fail_threshold = self.params.get("trigger_fail_threshold", 2)
        index_threshold = self.params.get("index_threshold", 0.5)
        min_signals = self.params.get("min_signals", 3)

        # v4 dual-path: 외부에서 직접 indicators dict를 전달
        if indicators is not None:
            total = len(indicators) if indicators else 0
            count = sum(1 for v in indicators.values() if v)
            is_sideways = count >= min_signals
            met = [k for k, v in indicators.items() if v]
            if is_sideways:
                message = f"횡보장 감지 ({count}/{total} 충족: {', '.join(met)}) → 현금 100%"
            else:
                message = f"횡보장 아님 ({count}/{total} 충족, 기준 {min_signals}개)"
            return {
                "indicators": indicators,
                "count": count,
                "is_sideways": is_sideways,
                "message": message,
            }

        # 기본 경로: 5개 지표 계산
        ind: dict[str, bool] = {}
        changes = changes or {}

        # 1. Polymarket 확률 40~60% 범위
        if poly_probs is not None:
            btc_up = poly_probs.get("btc_up", 0.5)
            poly_in_range = poly_low <= btc_up <= poly_high
        else:
            poly_in_range = False
        ind["poly_range"] = poly_in_range

        # 2. GLD |등락률| <= threshold
        gld_data = changes.get("GLD", {})
        gld_pct = abs(gld_data.get("change_pct", 0.0))
        ind["gld_flat"] = gld_pct <= gld_threshold

        # 3. 쌍둥이 갭 수렴 실패
        ind["gap_fail"] = gap_fail_count >= gap_fail_threshold

        # 4. COIN/CONL 트리거 불발
        ind["trigger_fail"] = trigger_fail_count >= trigger_fail_threshold

        # 5. SPY·QQQ 모두 |등락률| <= threshold
        spy_pct = abs(changes.get("SPY", {}).get("change_pct", 0.0))
        qqq_pct = abs(changes.get("QQQ", {}).get("change_pct", 0.0))
        ind["index_flat"] = spy_pct <= index_threshold and qqq_pct <= index_threshold

        count = sum(1 for v in ind.values() if v)
        is_sideways = count >= min_signals

        met = [k for k, v in ind.items() if v]
        if is_sideways:
            message = f"횡보장 감지 ({count}/5 충족: {', '.join(met)}) → 현금 100%"
        else:
            message = f"횡보장 아님 ({count}/5 충족, 기준 {min_signals}개)"

        return {
            "indicators": ind,
            "count": count,
            "is_sideways": is_sideways,
            "message": message,
        }
