"""
M200 즉시매도 킬스위치
========================
7개 조건 병렬 평가 → 1개라도 True → 즉시 매도 (SCHD 제외).

파이프라인 (MT_VNQ3 §15):
전역 락 ON → OrderQueue.cancel_all() → 취소 ACK 대기 →
reserved_cash 재계산 → 청산 실행 (is_urgent=True) → 전역 락 OFF

출처: MT_VNQ3.md §15 (M200)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..common.base import MarketData


@dataclass
class M200Result:
    """M200 평가 결과."""
    triggered: bool
    conditions: list[str] = field(default_factory=list)
    reason: str = ""


class M200KillSwitch:
    """M200 즉시매도 킬스위치.

    7개 조건:
    1. 거래대금 15% 급감
    2. Polymarket BTC (OFF — P-5 미확인)
    3. GLD +6%
    4. VIX +10%
    5. 20일선 이탈
    6. 기한 만기
    7. REIT_MIX +5%
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        # Thresholds with defaults
        self._volume_crash_pct = self.config.get("volume_crash_pct", -15.0)
        self._gld_surge_pct = self.config.get("gld_surge_pct", 6.0)
        self._vix_spike_pct = self.config.get("vix_spike_pct", 10.0)
        self._reit_mix_surge_pct = self.config.get("reit_mix_surge_pct", 5.0)
        self._poly_btc_enabled = self.config.get("poly_btc_enabled", False)  # P-5: OFF

    def evaluate(self, market: "MarketData") -> dict:
        """7개 조건 병렬 평가.

        Parameters
        ----------
        market : MarketData
            시장 데이터 스냅샷.

        Returns
        -------
        dict
            {"triggered": bool, "conditions": list[str], "reason": str}
        """
        triggered_conditions: list[str] = []

        # 1. 거래대금 15% 급감
        vol_result = self._check_volume_crash(market)
        if vol_result:
            triggered_conditions.append(vol_result)

        # 2. Polymarket BTC (OFF by default — P-5 미확인)
        if self._poly_btc_enabled:
            poly_result = self._check_poly_btc(market)
            if poly_result:
                triggered_conditions.append(poly_result)

        # 3. GLD +6%
        gld_result = self._check_gld_surge(market)
        if gld_result:
            triggered_conditions.append(gld_result)

        # 4. VIX +10%
        vix_result = self._check_vix_spike(market)
        if vix_result:
            triggered_conditions.append(vix_result)

        # 5. 20일선 이탈
        ma_result = self._check_ma20_break(market)
        if ma_result:
            triggered_conditions.append(ma_result)

        # 6. 기한 만기
        deadline_result = self._check_deadline(market)
        if deadline_result:
            triggered_conditions.append(deadline_result)

        # 7. REIT_MIX +5%
        reit_result = self._check_reit_mix_surge(market)
        if reit_result:
            triggered_conditions.append(reit_result)

        triggered = len(triggered_conditions) > 0

        return {
            "triggered": triggered,
            "conditions": triggered_conditions,
            "reason": " | ".join(triggered_conditions) if triggered else "",
        }

    def _check_volume_crash(self, market: "MarketData") -> str | None:
        """1. 거래대금 15% 급감 체크.

        market.volumes에서 주요 종목 거래대금 변화 확인.
        """
        if not market.volumes:
            return None

        # 전체 거래대금 변화율 (구현: volumes dict에 "_total_change_pct" 키 기대)
        total_change = market.volumes.get("_total_change_pct")
        if total_change is not None and total_change <= self._volume_crash_pct:
            return f"volume_crash: {total_change:.1f}% (threshold: {self._volume_crash_pct}%)"
        return None

    def _check_poly_btc(self, market: "MarketData") -> str | None:
        """2. Polymarket BTC 조건 (P-5: OFF by default)."""
        # P-5 미확인 → 비활성
        return None

    def _check_gld_surge(self, market: "MarketData") -> str | None:
        """3. GLD +6% 급등."""
        gld = market.changes.get("GLD", 0)
        if gld >= self._gld_surge_pct:
            return f"gld_surge: GLD +{gld:.1f}% (threshold: +{self._gld_surge_pct}%)"
        return None

    def _check_vix_spike(self, market: "MarketData") -> str | None:
        """4. VIX +10% 급등."""
        vix = market.changes.get("VIX", market.changes.get("^VIX", 0))
        if vix >= self._vix_spike_pct:
            return f"vix_spike: VIX +{vix:.1f}% (threshold: +{self._vix_spike_pct}%)"
        return None

    def _check_ma20_break(self, market: "MarketData") -> str | None:
        """5. 20일선 이탈.

        SPY 또는 QQQ가 20일선 하회 시 발동.
        """
        if not market.history:
            return None

        for ticker in ("SPY", "QQQ"):
            hist = market.history.get(ticker, {})
            ma_20 = hist.get("ma_20")
            if ma_20 is None:
                continue
            price = market.prices.get(ticker, 0)
            if price > 0 and price < ma_20:
                drop_pct = (price - ma_20) / ma_20 * 100
                return f"ma20_break: {ticker} {price:.2f} < MA20 {ma_20:.2f} ({drop_pct:+.1f}%)"
        return None

    def _check_deadline(self, market: "MarketData") -> str | None:
        """6. 기한 만기.

        포지션별 기한 체크는 Orchestrator 레벨에서 수행.
        여기서는 전역 기한 플래그만 체크.
        """
        # 전역 기한 만기는 orchestrator에서 처리
        # M200에서는 시장 수준 조건만 체크
        return None

    def _check_reit_mix_surge(self, market: "MarketData") -> str | None:
        """7. REIT_MIX +5% 급등.

        VNQ + KR리츠 평균 수익률이 +5% 이상.
        """
        if not market.history:
            return None

        reit_tickers = ["VNQ", "SK리츠", "TIGER 리츠부동산인프라", "롯데리츠"]
        returns = []
        for reit in reit_tickers:
            hist = market.history.get(reit, {})
            ret_7d = hist.get("return_7d")
            if ret_7d is not None:
                returns.append(ret_7d)

        if not returns:
            return None

        reit_mix = sum(returns) / len(returns)
        if reit_mix >= self._reit_mix_surge_pct:
            return f"reit_mix_surge: REIT_MIX +{reit_mix:.1f}% (threshold: +{self._reit_mix_surge_pct}%)"
        return None
