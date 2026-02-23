"""
LimitOrder + OrderQueue — 지정가 주문 관리 레이어
=================================================
CI-0-12 의도 기반 주문 식별 + CI-17 Fill Window + CI-18 추격매수 금지.
백테스트에서는 다음 봉 OHLCV로 체결 판단.

출처: MT_VNQ3.md §3, §5, §6, §7
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class FillEvent:
    """체결 이벤트."""
    order_id: str
    ticker: str
    side: str
    filled_qty: float
    fill_price: float
    timestamp: datetime
    remaining_qty: float = 0.0


@dataclass
class CancelEvent:
    """취소 이벤트."""
    order_id: str
    ticker: str
    reason: str
    released_cash: float
    timestamp: datetime


@dataclass
class LimitOrder:
    """지정가 주문.

    Parameters
    ----------
    order_id : str
        고유 주문 ID (UUID).
    intent_id : tuple[str, str, str, str, int]
        CI-0-12: (strategy, ticker, side, signal_bar_ts, seq_no)
    ticker : str
        종목 코드.
    side : Literal["BUY", "SELL"]
        매수/매도.
    limit_price : float
        지정가.
    qty : float
        주문 수량.
    reference_price : float
        CI-18: 추격매수 금지 기준가 (신호 발생 시점 가격).
    status : OrderStatus
        주문 상태.
    reserved_cash : float
        예약된 현금 (BUY 주문).
    filled_qty : float
        체결된 수량.
    avg_fill_price : float
        평균 체결가.
    is_urgent : bool
        CI-19: 매도 즉시성 플래그 (M200 등).
    created_at : datetime
        주문 생성 시각.
    fill_window_sec : int
        CI-17: Fill Window (기본 10초).
    ttl_sec : int
        CI-0-3: 주문 TTL (기본 120초).
    retry_count : int
        재시도 횟수.
    max_retries : int
        최대 재시도 (기본 3).
    """
    order_id: str
    intent_id: tuple
    ticker: str
    side: Literal["BUY", "SELL"]
    limit_price: float
    qty: float
    reference_price: float
    status: OrderStatus
    reserved_cash: float
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    is_urgent: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    fill_window_sec: int = 10
    ttl_sec: int = 120
    retry_count: int = 0
    max_retries: int = 3

    @property
    def remaining_qty(self) -> float:
        return self.qty - self.filled_qty

    @property
    def is_expired(self) -> bool:
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed >= self.ttl_sec

    def is_expired_at(self, current_time: datetime) -> bool:
        elapsed = (current_time - self.created_at).total_seconds()
        return elapsed >= self.ttl_sec

    def fill_window_expired_at(self, current_time: datetime) -> bool:
        elapsed = (current_time - self.created_at).total_seconds()
        return elapsed >= self.fill_window_sec


class OrderQueue:
    """주문 큐 — 주문 접수, 체결 판단, 취소 관리.

    백테스트 체결 규칙:
    - BUY: 다음 봉 low <= limit_price → filled (체결가 = limit_price)
    - SELL: 다음 봉 high >= limit_price → filled
    - SELL (is_urgent): bid 기반 0.2% 이내 → filled 확률 높음 (백테스트에서는 시뮬레이션)
    """

    def __init__(self):
        self._orders: dict[str, LimitOrder] = {}  # order_id -> LimitOrder
        self._intent_index: dict[tuple, str] = {}  # intent_id -> order_id

    @property
    def pending_orders(self) -> list[LimitOrder]:
        return [o for o in self._orders.values()
                if o.status in (OrderStatus.PENDING, OrderStatus.PARTIAL)]

    def submit(self, order: LimitOrder) -> str:
        """주문 접수. intent_id 중복 시 기존 주문 반환.

        Returns
        -------
        str
            order_id
        """
        # CI-0-12: intent_id 중복 체크
        if order.intent_id in self._intent_index:
            existing_id = self._intent_index[order.intent_id]
            existing = self._orders.get(existing_id)
            if existing and existing.status in (OrderStatus.PENDING, OrderStatus.PARTIAL):
                return existing_id  # 중복 주문 무시

        if not order.order_id:
            order.order_id = str(uuid.uuid4())[:8]

        self._orders[order.order_id] = order
        self._intent_index[order.intent_id] = order.order_id
        return order.order_id

    def on_tick(self, current_time: datetime,
                prices: dict[str, dict]) -> list[FillEvent]:
        """틱/봉 데이터로 체결 판단.

        Parameters
        ----------
        current_time : datetime
        prices : dict[str, dict]
            {ticker: {"low": float, "high": float, "close": float, "bid": float?}}

        Returns
        -------
        list[FillEvent]
        """
        fills: list[FillEvent] = []

        for order in list(self.pending_orders):
            # TTL 만료 체크
            if order.is_expired_at(current_time):
                if order.retry_count < order.max_retries:
                    order.retry_count += 1
                    order.created_at = current_time  # 리셋
                    continue
                order.status = OrderStatus.EXPIRED
                continue

            bar = prices.get(order.ticker)
            if bar is None:
                continue

            filled = False
            fill_price = order.limit_price

            if order.side == "BUY":
                # 다음 봉 low <= limit_price → filled
                low = bar.get("low", float("inf"))
                if low <= order.limit_price:
                    filled = True
            else:  # SELL
                if order.is_urgent:
                    # 매도 즉시성: bid 기반 0.2% 이내
                    bid = bar.get("bid", bar.get("close", 0))
                    if bid > 0:
                        slip = abs(order.limit_price - bid) / bid
                        if slip <= 0.002:
                            filled = True
                            fill_price = bid
                        else:
                            # 백테스트 fallback: high 기준
                            high = bar.get("high", 0)
                            if high >= order.limit_price:
                                filled = True
                else:
                    high = bar.get("high", 0)
                    if high >= order.limit_price:
                        filled = True

            if filled:
                fill_qty = order.remaining_qty
                # 평균 체결가 갱신
                total_filled = order.filled_qty + fill_qty
                if total_filled > 0:
                    order.avg_fill_price = (
                        (order.avg_fill_price * order.filled_qty + fill_price * fill_qty)
                        / total_filled
                    )
                order.filled_qty = total_filled
                order.status = OrderStatus.FILLED

                fills.append(FillEvent(
                    order_id=order.order_id,
                    ticker=order.ticker,
                    side=order.side,
                    filled_qty=fill_qty,
                    fill_price=fill_price,
                    timestamp=current_time,
                    remaining_qty=0.0,
                ))

        return fills

    def on_fill_window_expire(self, order_id: str,
                              current_time: datetime) -> CancelEvent | None:
        """Fill Window 만료 → 잔량 취소.

        Parameters
        ----------
        order_id : str
        current_time : datetime

        Returns
        -------
        CancelEvent | None
        """
        order = self._orders.get(order_id)
        if order is None or order.status not in (OrderStatus.PENDING, OrderStatus.PARTIAL):
            return None

        if not order.fill_window_expired_at(current_time):
            return None

        released = order.reserved_cash * (order.remaining_qty / order.qty) if order.qty > 0 else 0
        order.status = OrderStatus.CANCELLED

        return CancelEvent(
            order_id=order.order_id,
            ticker=order.ticker,
            reason="fill_window_expired",
            released_cash=released,
            timestamp=current_time,
        )

    def cancel_all(self, current_time: datetime | None = None) -> list[CancelEvent]:
        """모든 대기 주문 취소 (M200 파이프라인용).

        Returns
        -------
        list[CancelEvent]
        """
        ts = current_time or datetime.now()
        events: list[CancelEvent] = []
        for order in list(self.pending_orders):
            released = order.reserved_cash * (order.remaining_qty / order.qty) if order.qty > 0 else 0
            order.status = OrderStatus.CANCELLED
            events.append(CancelEvent(
                order_id=order.order_id,
                ticker=order.ticker,
                reason="cancel_all",
                released_cash=released,
                timestamp=ts,
            ))
        return events

    def cancel_by_ticker(self, ticker: str,
                         current_time: datetime | None = None) -> list[CancelEvent]:
        """특정 종목 대기 주문 취소.

        Returns
        -------
        list[CancelEvent]
        """
        ts = current_time or datetime.now()
        events: list[CancelEvent] = []
        for order in list(self.pending_orders):
            if order.ticker != ticker:
                continue
            released = order.reserved_cash * (order.remaining_qty / order.qty) if order.qty > 0 else 0
            order.status = OrderStatus.CANCELLED
            events.append(CancelEvent(
                order_id=order.order_id,
                ticker=order.ticker,
                reason=f"cancel_by_ticker:{ticker}",
                released_cash=released,
                timestamp=ts,
            ))
        return events

    def get_reserved_cash(self) -> float:
        """현재 대기 주문에 예약된 총 현금."""
        return sum(
            o.reserved_cash * (o.remaining_qty / o.qty) if o.qty > 0 else 0
            for o in self.pending_orders
        )

    def get_order(self, order_id: str) -> LimitOrder | None:
        return self._orders.get(order_id)

    def get_orders_by_ticker(self, ticker: str) -> list[LimitOrder]:
        return [o for o in self._orders.values() if o.ticker == ticker]

    def get_active_count(self) -> int:
        return len(self.pending_orders)
