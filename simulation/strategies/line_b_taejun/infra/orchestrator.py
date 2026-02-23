"""
Orchestrator — 통합 진입점 + 우선순위 파이프라인
================================================
M200 → M201 → SCHD 매도차단 → 리스크/이머전시 → 거래시간 →
M28 게이트 → M5 배분 → M6 리츠 감산 → 금지티커 → OrderQueue.submit()

출처: MT_VNQ3.md (전체 통합)
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from .limit_order import LimitOrder, OrderQueue, OrderStatus
from .m28_poly_gate import M28PolyGate
from .m201_mode import M201ImmediateMode
from .m5_weight_manager import M5WeightManager
from .schd_master import SCHDMaster, SIGNAL_ONLY_TICKERS

if TYPE_CHECKING:
    from ..common.base import MarketData, Signal


class Orchestrator:
    """통합 시그널 파이프라인.

    우선순위:
    1. M200 즉시매도 체크 → 발동: 전역 락 + cancel_all + 매도 파이프라인
    2. M201 즉시전환 체크 → BTC 확률 급변 시 포지션 전환
    3. SCHD 매도 차단 → is_sell_blocked() 확인
    4. 리스크/이머전시 모드
    5. M3 거래시간/휴장 게이트
    6. M28 포지션 게이트 (LONG/SHORT/NEUTRAL)
    7. M5 T1~T4 비중 일괄 배분
    8. M6 리츠 감산 적용
    9. 금지/오타/대체 티커 처리 (CI-0-15)
    10. OrderQueue.submit()
    """

    def __init__(
        self,
        order_queue: OrderQueue | None = None,
        m201: M201ImmediateMode | None = None,
        m28: M28PolyGate | None = None,
        m5: M5WeightManager | None = None,
        schd: SCHDMaster | None = None,
        m200_checker: Any = None,
    ):
        self.order_queue = order_queue or OrderQueue()
        self.m201 = m201 or M201ImmediateMode()
        self.m28 = m28 or M28PolyGate()
        self.m5 = m5 or M5WeightManager()
        self.schd = schd or SCHDMaster()
        self.m200_checker = m200_checker  # M200KillSwitch (optional, set later)

        self._global_lock = False
        self._last_m200_result: dict | None = None

    @property
    def is_locked(self) -> bool:
        return self._global_lock

    def process_signals(
        self,
        signals: list[tuple[str, "Signal"]],
        market: "MarketData",
        positions: dict[str, Any] | None = None,
        cash: float = 0.0,
        total_assets: float = 0.0,
        reit_risk_state: dict | None = None,
    ) -> list[dict]:
        """전체 파이프라인 실행.

        Parameters
        ----------
        signals : list[tuple[str, Signal]]
            (strategy_name, Signal) 리스트.
        market : MarketData
        positions : dict | None
            현재 보유 포지션.
        cash : float
        total_assets : float
        reit_risk_state : dict | None
            REIT 리스크 상태 (cautious_mode 등).

        Returns
        -------
        list[dict]
            처리된 주문 결과 리스트.
        """
        results: list[dict] = []
        positions = positions or {}

        # ──────────────────────────────────────────────────────
        # Step 1: M200 즉시매도 체크
        # ──────────────────────────────────────────────────────
        if self.m200_checker is not None:
            m200_result = self.m200_checker.evaluate(market)
            if m200_result.get("triggered", False):
                self._global_lock = True
                self._last_m200_result = m200_result

                # 모든 대기 주문 취소
                cancel_events = self.order_queue.cancel_all(market.time)
                for ce in cancel_events:
                    results.append({
                        "type": "cancel",
                        "source": "m200",
                        "order_id": ce.order_id,
                        "ticker": ce.ticker,
                        "released_cash": ce.released_cash,
                    })

                # 즉시 매도 신호 생성 (SCHD 제외)
                for ticker in list(positions.keys()):
                    if self.schd.is_sell_blocked(ticker):
                        continue
                    results.append({
                        "type": "urgent_sell",
                        "source": "m200",
                        "ticker": ticker,
                        "reason": m200_result.get("reason", "M200 kill switch"),
                        "is_urgent": True,
                    })

                self._global_lock = False
                return results

        # ──────────────────────────────────────────────────────
        # Step 2: M201 즉시전환 체크
        # ──────────────────────────────────────────────────────
        if market.poly and market.poly_prev:
            btc_p = market.poly.get("btc_up", 0.5)
            btc_p_prev = market.poly_prev.get("btc_up", 0.5)

            # LONG 포지션 체크
            m201_long = self.m201.check_long(btc_p, btc_p_prev)
            if m201_long is not None:
                results.append({
                    "type": "m201",
                    "action": m201_long.action.value,
                    "reason": m201_long.reason,
                    "p": m201_long.p,
                    "delta_pp": m201_long.delta_pp,
                })
                # 청산 후 전환
                post = self.m201.post_close_action(btc_p, "long")
                if post.value != "none":
                    results.append({
                        "type": "m201_post",
                        "action": post.value,
                        "p": btc_p,
                    })

            # SHORT 포지션 체크
            m201_short = self.m201.check_short(btc_p, btc_p_prev)
            if m201_short is not None:
                results.append({
                    "type": "m201",
                    "action": m201_short.action.value,
                    "reason": m201_short.reason,
                    "p": m201_short.p,
                    "delta_pp": m201_short.delta_pp,
                })
                post = self.m201.post_close_action(btc_p, "short")
                if post.value != "none":
                    results.append({
                        "type": "m201_post",
                        "action": post.value,
                        "p": btc_p,
                    })

        # ──────────────────────────────────────────────────────
        # Step 3: SCHD 매도 차단 필터
        # ──────────────────────────────────────────────────────
        filtered_signals = []
        for name, sig in signals:
            if sig.action.value == "SELL" and self.schd.is_sell_blocked(sig.ticker):
                results.append({
                    "type": "blocked",
                    "source": "schd",
                    "ticker": sig.ticker,
                    "reason": "SCHD sell blocked",
                })
                continue
            filtered_signals.append((name, sig))

        # ──────────────────────────────────────────────────────
        # Step 4~5: 리스크/이머전시 모드 + 거래시간 게이트
        # (이 단계는 CompositeSignalEngine에서 처리 — pass through)
        # ──────────────────────────────────────────────────────

        # ──────────────────────────────────────────────────────
        # Step 6: M28 포지션 게이트
        # ──────────────────────────────────────────────────────
        gate = self.m28.evaluate(market.poly)
        results.append({
            "type": "gate",
            "source": "m28",
            "btc_direction": gate["btc_direction"],
            "ndx_direction": gate["ndx_direction"],
        })

        # ──────────────────────────────────────────────────────
        # Step 7: M5 T1~T4 비중 배분
        # ──────────────────────────────────────────────────────
        buy_signals = [
            (name, sig) for name, sig in filtered_signals
            if sig.action.value == "BUY"
        ]
        sell_signals = [
            (name, sig) for name, sig in filtered_signals
            if sig.action.value == "SELL"
        ]

        if buy_signals:
            allocations = self.m5.allocate(
                [sig for _, sig in buy_signals], cash, total_assets,
            )
            for i, alloc in enumerate(allocations):
                if i >= len(buy_signals):
                    break
                name, sig = buy_signals[i]
                results.append({
                    "type": "buy",
                    "source": name,
                    "ticker": sig.ticker,
                    "weight": alloc.weight,
                    "amount": alloc.amount,
                    "tier": alloc.tier,
                    "target_pct": sig.target_pct,
                    "reason": sig.reason,
                })

        # ──────────────────────────────────────────────────────
        # Step 8: M6 리츠 감산 적용
        # ──────────────────────────────────────────────────────
        if reit_risk_state and reit_risk_state.get("cautious_mode", False):
            for r in results:
                if r.get("type") == "buy":
                    # target_net 최소 하한 0.8%
                    weight = r.get("weight", 0)
                    # 감산: target × 1/2
                    reduced = weight * 0.5
                    # 감산 후 하한 재확인
                    if reduced <= 0.008:
                        r["weight"] = 0
                        r["buy_stop"] = True
                        r["reason"] = f"{r.get('reason', '')} [M6 감산 → BUY_STOP]"
                    else:
                        r["weight"] = reduced
                        r["m6_reduced"] = True

        # ──────────────────────────────────────────────────────
        # Step 9: 금지/오타/대체 티커 처리 (CI-0-15)
        # ──────────────────────────────────────────────────────
        final_results = []
        for r in results:
            ticker = r.get("ticker", "")
            # SIGNAL_ONLY_TICKERS → 매수 신호 drop
            if r.get("type") == "buy" and ticker in SIGNAL_ONLY_TICKERS:
                continue
            final_results.append(r)

        # ──────────────────────────────────────────────────────
        # Step 10: Sell signals pass through
        # ──────────────────────────────────────────────────────
        for name, sig in sell_signals:
            final_results.append({
                "type": "sell",
                "source": name,
                "ticker": sig.ticker,
                "size": sig.size,
                "reason": sig.reason,
            })

        return final_results
