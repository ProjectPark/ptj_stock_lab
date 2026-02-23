"""
taejun_attach_pattern - 포트폴리오 매니저
==========================================
다중 전략 동시 운용시 자금배분, 포지션 추적, 시그널 충돌 해소.

역할:
1. 전체 자본 관리 (현금 + 포지션 합산)
2. 전략별 자금 한도 관리
3. 동시 BUY 시그널 충돌시 우선순위 해소
4. 포지션 추적 (진입가, 수량, 전략명)
5. 수수료 반영된 체결 시뮬레이션

v5 포트폴리오 제약 (추가):
- 동시 보유 한도: 최대 $12,000 (총자본 80%), 최소 현금 $3,000 (20%)
- CONL 단일 전략 보유 원칙: 전략 간 CONL 포지션 중복 금지
- 일일 1회 거래 제한: 종목별 하루 1회 매수
- 예외: vix_gold (VIX 방어), crash_buy (급락 역매수)는 한도 초과 허용
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .asset_mode import AssetModeManager
from .base import Action, ExitReason, MarketData, Position, Signal
from .fees import FeeCalculator, FeeConfig


# ============================================================
# 거래 기록
# ============================================================

@dataclass
class Trade:
    """체결된 거래 기록."""
    timestamp: datetime
    ticker: str
    side: str               # "BUY" or "SELL"
    price: float
    qty: float
    amount: float            # 거래대금 (USD)
    fee: float               # 수수료 (USD)
    net_amount: float        # 순 금액 (USD)
    strategy: str            # 전략 이름
    pnl: float = 0.0        # 손익 (매도시만)
    pnl_pct: float = 0.0    # 손익률 (매도시만)
    reason: str = ""


# ============================================================
# 전략별 한도 설정
# ============================================================

@dataclass
class StrategyAllocation:
    """전략별 자금 배분 한도."""
    max_pct: float = 1.0          # 전체 자본 대비 최대 비율 (1.0 = 100%)
    max_amount_usd: float = 0     # 절대 금액 한도 (0 = 무제한)
    max_positions: int = 0        # 최대 동시 포지션 수 (0 = 무제한)
    priority: int = 5             # 충돌시 우선순위 (낮을수록 높음)


# 기본 전략 우선순위
DEFAULT_ALLOCATIONS: dict[str, StrategyAllocation] = {
    # 이머전시 (최최우선)
    "emergency_mode":   StrategyAllocation(max_pct=1.0, priority=0),

    # 매크로 전략 (최우선)
    "short_macro":      StrategyAllocation(max_pct=1.0, priority=1),
    "reit_risk":        StrategyAllocation(max_pct=1.0, priority=1),

    # 이벤트 드리븐 (높음)
    "vix_gold":         StrategyAllocation(max_pct=1.0, priority=2),

    # 잽모드 단타 (중간)
    "jab_soxl":         StrategyAllocation(max_pct=1.0, priority=3),
    "jab_bitu":         StrategyAllocation(max_pct=1.0, priority=3),
    "jab_tsll":         StrategyAllocation(max_pct=1.0, max_amount_usd=1500, priority=3),
    "jab_seth":         StrategyAllocation(max_pct=1.0, priority=3),

    # 저가매수 (중간-낮음)
    "bargain_buy":      StrategyAllocation(max_pct=0.5, max_positions=3, priority=4),

    # 기타 (낮음)
    "sp500_entry":      StrategyAllocation(max_pct=1.0, priority=5),
    "sector_rotate":    StrategyAllocation(max_pct=0.1, priority=6),
    "bank_conditional": StrategyAllocation(max_pct=0.2, max_amount_usd=2300, priority=5),
}


class PortfolioManager:
    """포트폴리오 매니저.

    Parameters
    ----------
    total_capital_usd : float
        총 투자 자본 (USD).
    fee_config : FeeConfig | None
        수수료 설정. None이면 KIS 기본값.
    allocations : dict[str, StrategyAllocation] | None
        전략별 한도. None이면 기본값 사용.
    """

    def __init__(
        self,
        total_capital_usd: float = 15_000,
        fee_config: FeeConfig | None = None,
        allocations: dict[str, StrategyAllocation] | None = None,
        mode_manager: AssetModeManager | None = None,
    ):
        self.total_capital = total_capital_usd
        self.cash = total_capital_usd
        self.fee_calc = FeeCalculator(fee_config)
        self.allocations = allocations or DEFAULT_ALLOCATIONS
        self.mode_manager = mode_manager

        # 상태
        self.positions: dict[str, Position] = {}  # ticker -> Position
        self.trades: list[Trade] = []
        self.total_fees: float = 0.0

        # CI-0-2: 매수 주문 시 현금 예약 (pending 상태에서 over-invest 방지)
        # 출처: MT_VNQ2.md L2852~2862
        self.reserved_cash: float = 0.0

        # MT_VNQ3 §5: 부분체결 누적 기록
        self.fills_ledger: list[dict] = []

        # v5 포트폴리오 제약
        self._max_simultaneous_hold_usd: float = 12_000.0     # 동시 보유 한도 (80%)
        self._min_cash_reserve_usd: float = 3_000.0           # 최소 현금 유지 (20%)
        self._simultaneous_cap_exempt: frozenset[str] = frozenset({"vix_gold", "crash_buy"})
        self._daily_bought: set[str] = set()                   # 일일 매수 기록
        self._daily_date: datetime.date | None = None          # 기록 기준 날짜
        self._last_buy_time: dict[str, datetime] = {}          # ticker -> 마지막 매수 시각

    # ------------------------------------------------------------------
    # 잔고 조회
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        """현재 총 자산 (현금 + 포지션 평가액). prices 필요시 update_equity 호출."""
        return self.cash + sum(
            p.avg_price * p.qty for p in self.positions.values()
        )

    def free_cash(self) -> float:
        """CI-0-2: 예약금 제외 사용 가능 현금."""
        return self.cash - self.reserved_cash

    def reserve(self, amount: float) -> None:
        """CI-0-3: 매수 주문 pending 시 현금 예약 (over-invest 방지)."""
        self.reserved_cash += amount

    def unreserve(self, amount: float) -> None:
        """CI-0-3: 주문 취소/만료 시 예약 해제."""
        self.reserved_cash = max(0.0, self.reserved_cash - amount)

    # ------------------------------------------------------------------
    # MT_VNQ3 §5: 부분체결 / 포지션 확정
    # ------------------------------------------------------------------

    def partial_fill(self, ticker: str, qty_delta: float, price: float,
                     ts: datetime | None = None) -> None:
        """MT_VNQ3 §5: 부분체결 즉시 반영.

        fills_ledger에 기록하고, position.qty를 qty_delta만큼 반영하며,
        reserved_cash를 체결된 만큼 즉시 해제한다.
        """
        self.fills_ledger.append({
            "ticker": ticker,
            "qty_delta": qty_delta,
            "price": price,
            "ts": ts or datetime.now(),
        })

        position = self.positions.get(ticker)
        if position is not None:
            # 평균가 재계산 (기존 cost + 신규 cost) / (기존 qty + 신규 qty)
            total_cost = position.avg_price * position.qty + price * qty_delta
            position.qty += qty_delta
            if position.qty > 0:
                position.avg_price = total_cost / position.qty

        # 체결된 만큼 예약금 해제
        filled_amount = qty_delta * price
        self.unreserve(filled_amount)

    def confirm_position(self, ticker: str) -> None:
        """MT_VNQ3 §5: Fill Window 종료 후 포지션 확정."""
        position = self.positions.get(ticker)
        if position is not None:
            position.confirmed = True

    def update_equity(self, prices: dict[str, float]) -> float:
        """현재 시가로 총 자산을 재계산한다."""
        pos_value = sum(
            prices.get(p.ticker, p.avg_price) * p.qty
            for p in self.positions.values()
        )
        return self.cash + pos_value

    def get_position(self, ticker: str) -> Position | None:
        return self.positions.get(ticker)

    def get_strategy_positions(self, strategy_name: str) -> list[Position]:
        return [p for p in self.positions.values() if p.strategy_name == strategy_name]

    def get_strategy_exposure(self, strategy_name: str,
                              prices: dict[str, float]) -> float:
        """특정 전략의 현재 투입 금액(USD)."""
        return sum(
            prices.get(p.ticker, p.avg_price) * p.qty
            for p in self.positions.values()
            if p.strategy_name == strategy_name
        )

    # ------------------------------------------------------------------
    # 자금 한도 체크
    # ------------------------------------------------------------------

    def _get_allocation(self, strategy_name: str) -> StrategyAllocation:
        return self.allocations.get(strategy_name, StrategyAllocation())

    def check_buy_allowed(self, signal: Signal, strategy_name: str,
                          prices: dict[str, float],
                          market_time: datetime | None = None) -> tuple[bool, str]:
        """매수 가능 여부를 확인한다.

        Returns (allowed, reason).
        """
        if self.cash <= 0:
            return False, "no cash"

        alloc = self._get_allocation(strategy_name)

        # 최대 포지션 수 체크
        if alloc.max_positions > 0:
            current_count = len(self.get_strategy_positions(strategy_name))
            if current_count >= alloc.max_positions:
                return False, "max_positions reached (%d)" % alloc.max_positions

        # 비율 한도 체크
        equity = self.update_equity(prices)
        max_by_pct = equity * alloc.max_pct
        current_exposure = self.get_strategy_exposure(strategy_name, prices)
        remaining_by_pct = max_by_pct - current_exposure

        # 절대 금액 한도
        if alloc.max_amount_usd > 0:
            remaining_by_amount = alloc.max_amount_usd - current_exposure
            remaining = min(remaining_by_pct, remaining_by_amount)
        else:
            remaining = remaining_by_pct

        # 조심모드: 공격전략 remaining × leverage_multiplier
        if self.mode_manager:
            multiplier = self.mode_manager.get_leverage_multiplier(strategy_name)
            if multiplier < 1.0:
                remaining = remaining * multiplier

        if remaining <= 0:
            return False, "allocation limit reached"

        # ── v5 추가 제약 ─────────────────────────────────────────────

        # (1) 일일 1회 매수 제한
        if signal.ticker in self._daily_bought:
            return False, f"daily buy limit: {signal.ticker} already bought today"

        # (2) 동시 보유 한도 $12,000 / 최소 현금 $3,000 (vix_gold·crash_buy 제외)
        if strategy_name not in self._simultaneous_cap_exempt:
            pos_value = sum(
                prices.get(p.ticker, p.avg_price) * p.qty
                for p in self.positions.values()
            )
            if pos_value >= self._max_simultaneous_hold_usd:
                return False, (
                    f"simultaneous hold cap: ${pos_value:.0f} >= "
                    f"${self._max_simultaneous_hold_usd:.0f}"
                )
            if self.cash <= self._min_cash_reserve_usd:
                return False, f"min cash reserve: ${self.cash:.0f} <= ${self._min_cash_reserve_usd:.0f}"

        # (3) CONL 단일 전략 보유 원칙
        if signal.ticker == "CONL":
            existing_conl = self.positions.get("CONL")
            if existing_conl and existing_conl.strategy_name != strategy_name:
                return False, (
                    f"CONL single-strategy: already held by {existing_conl.strategy_name}"
                )

        # (4) 종목당 최대 투입 한도
        _max_per_stock = {"CONL": 6_750.0}
        _default_max = 5_250.0
        ticker_max = _max_per_stock.get(signal.ticker, _default_max)
        ticker_exposure = sum(
            prices.get(p.ticker, p.avg_price) * p.qty
            for p in self.positions.values()
            if p.ticker == signal.ticker
        )
        if ticker_exposure >= ticker_max:
            return False, f"per-stock limit: {signal.ticker} ${ticker_exposure:.0f} >= ${ticker_max:.0f}"

        # (5) 물타기 횟수 제한
        existing_pos = self.positions.get(signal.ticker)
        if existing_pos and existing_pos.strategy_name == strategy_name:
            conl_max_stage = 1  # CONL: 1회 진입만 (물타기 금지)
            default_max_stage = 5  # 일반: 초기+4회 물타기
            max_stage = conl_max_stage if signal.ticker == "CONL" else default_max_stage
            if existing_pos.stage >= max_stage:
                return False, f"DCA limit: {signal.ticker} stage {existing_pos.stage} >= {max_stage}"

        # (6) 중복 주문 20분 간격
        if market_time is not None:
            last_buy = self._last_buy_time.get(signal.ticker)
            if last_buy is not None:
                if market_time - last_buy < timedelta(minutes=20):
                    return False, (
                        f"duplicate order: {signal.ticker} last bought at {last_buy}, "
                        f"only {(market_time - last_buy).total_seconds() / 60:.1f}min ago"
                    )

        # (7) 자금 우선순위
        _PRIORITY_ORDER = {
            "vix_gold": 1, "crash_buy": 2, "conditional_conl": 3,
            "soxl_independent": 4, "twin_pair": 5, "bargain_buy": 5,
        }
        my_priority = _PRIORITY_ORDER.get(strategy_name, 99)
        for pos in self.positions.values():
            pos_priority = _PRIORITY_ORDER.get(pos.strategy_name, 99)
            if pos_priority < my_priority:
                return False, (
                    f"fund priority: {pos.strategy_name}(pri={pos_priority}) active, "
                    f"{strategy_name}(pri={my_priority}) blocked"
                )

        # (8) 진입 시간 필터 (ET 08:00~10:00)
        if market_time is not None:
            et_hour = market_time.hour  # market_time은 ET 기준으로 가정
            if et_hour < 8 or et_hour >= 10:
                if strategy_name not in self._simultaneous_cap_exempt:
                    return False, f"entry time filter: ET {et_hour}:00 outside 08:00-10:00"

        return True, ""

    # ------------------------------------------------------------------
    # 충돌 해소
    # ------------------------------------------------------------------

    def resolve_conflicts(self, signals: list[tuple[str, Signal]],
                          prices: dict[str, float]) -> list[tuple[str, Signal]]:
        """동시 시그널 충돌을 해소한다.

        Parameters
        ----------
        signals : list[tuple[str, Signal]]
            [(strategy_name, signal), ...] 형태.
        prices : dict[str, float]
            현재가.

        Returns
        -------
        list[tuple[str, Signal]]
            실행 가능한 시그널 (우선순위 순 정렬).

        충돌 해소 규칙:
        1. SELL 시그널은 항상 먼저 실행 (현금 확보)
        2. BUY 시그널은 우선순위 순 정렬
        3. 자금이 부족하면 낮은 우선순위 시그널 제거
        """
        sells = []
        emergency_sells = []
        buys = []

        for name, sig in signals:
            if sig.action == Action.SELL:
                if sig.metadata.get("emergency_mode"):
                    emergency_sells.append((name, sig))
                else:
                    sells.append((name, sig))
            elif sig.action == Action.BUY:
                buys.append((name, sig))
            # HOLD, SKIP은 무시

        # 이머전시 SELL → 최우선, 그 다음 일반 SELL (자금 확보)
        result = emergency_sells + sells

        # BUY는 우선순위 순 정렬
        buys.sort(key=lambda x: self._get_allocation(x[0]).priority)

        for name, sig in buys:
            allowed, reason = self.check_buy_allowed(sig, name, prices)
            if allowed:
                result.append((name, sig))

        return result

    # ------------------------------------------------------------------
    # 체결 (매수)
    # ------------------------------------------------------------------

    def execute_buy(self, signal: Signal, strategy_name: str,
                    prices: dict[str, float], timestamp: datetime) -> Trade | None:
        """매수 시그널을 체결한다.

        Parameters
        ----------
        signal : Signal
        strategy_name : str
        prices : dict[str, float]
            현재가.
        timestamp : datetime

        Returns
        -------
        Trade | None
            체결되면 Trade 반환, 실패시 None.
        """
        ticker = signal.ticker
        price = prices.get(ticker, 0)
        if price <= 0:
            return None

        # 매수 금액 결정
        alloc = self._get_allocation(strategy_name)
        equity = self.update_equity(prices)

        if signal.size > 0:
            desired = self.cash * signal.size
        else:
            # size=0이면 metadata의 qty 기반
            qty = signal.metadata.get("qty", 1)
            desired = price * qty

        # 한도 적용
        max_by_pct = equity * alloc.max_pct
        current_exposure = self.get_strategy_exposure(strategy_name, prices)
        remaining = max_by_pct - current_exposure

        if alloc.max_amount_usd > 0:
            remaining = min(remaining, alloc.max_amount_usd - current_exposure)

        # metadata의 max_amount_krw → USD 변환 (환율 1350 가정)
        if signal.metadata.get("max_amount_krw"):
            max_usd = signal.metadata["max_amount_krw"] / 1350
            desired = min(desired, max_usd)

        # v5: 최소 현금 유지 — 매수 금액 클램프
        if strategy_name not in self._simultaneous_cap_exempt:
            max_spend = max(0.0, self.cash - self._min_cash_reserve_usd)
            amount = min(desired, remaining, self.cash, max_spend)
        else:
            amount = min(desired, remaining, self.cash)
        if amount <= 0:
            return None

        # 수수료 계산
        fee_result = self.fee_calc.calc_buy(amount)
        net_for_shares = fee_result.net_amount
        qty = net_for_shares / price

        # 포지션 업데이트
        existing = self.positions.get(ticker)
        if existing and existing.strategy_name == strategy_name:
            # 추가매수 → 평균가 재계산
            total_cost = existing.avg_price * existing.qty + net_for_shares
            total_qty = existing.qty + qty
            existing.avg_price = total_cost / total_qty
            existing.qty = total_qty
            existing.stage += 1
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                avg_price=price,
                qty=qty,
                entry_time=timestamp,
                strategy_name=strategy_name,
                stage=1,
            )

        # 현금 차감
        self.cash -= amount
        self.total_fees += fee_result.fee

        # v5: 일일 매수 기록
        current_date = timestamp.date()
        if self._daily_date != current_date:
            self._daily_bought.clear()
            self._daily_date = current_date
        self._daily_bought.add(ticker)

        # 마지막 매수 시각 기록 (중복 주문 20분 간격 체크용)
        self._last_buy_time[ticker] = timestamp

        trade = Trade(
            timestamp=timestamp, ticker=ticker, side="BUY",
            price=price, qty=qty, amount=amount,
            fee=fee_result.fee, net_amount=net_for_shares,
            strategy=strategy_name, reason=signal.reason,
        )
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # 체결 (매도)
    # ------------------------------------------------------------------

    def execute_sell(self, signal: Signal, strategy_name: str,
                     prices: dict[str, float], timestamp: datetime) -> Trade | None:
        """매도 시그널을 체결한다."""
        ticker = signal.ticker

        # 와일드카드 매도 (short_macro의 전체 롱 매도)
        if ticker in ("*", "*leveraged"):
            return self._execute_bulk_sell(signal, strategy_name, prices, timestamp)

        position = self.positions.get(ticker)
        if not position:
            return None

        price = prices.get(ticker, 0)
        if price <= 0:
            return None

        # 매도 수량 결정
        sell_qty = position.qty * signal.size
        if sell_qty <= 0:
            return None

        proceeds = sell_qty * price

        # 수수료 계산
        fee_result = self.fee_calc.calc_sell(proceeds)

        # 손익 계산
        cost_basis = position.avg_price * sell_qty
        pnl = fee_result.net_amount - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        # 포지션 업데이트
        position.qty -= sell_qty
        if position.qty <= 0.001:  # 소수점 오차 처리
            del self.positions[ticker]

        # 현금 증가
        self.cash += fee_result.net_amount
        self.total_fees += fee_result.fee

        trade = Trade(
            timestamp=timestamp, ticker=ticker, side="SELL",
            price=price, qty=sell_qty, amount=proceeds,
            fee=fee_result.fee, net_amount=fee_result.net_amount,
            strategy=strategy_name, pnl=pnl, pnl_pct=pnl_pct,
            reason=signal.reason,
        )
        self.trades.append(trade)
        return trade

    def _execute_bulk_sell(self, signal: Signal, strategy_name: str,
                           prices: dict[str, float],
                           timestamp: datetime) -> Trade | None:
        """전체 또는 레버리지 포지션 일괄 매도."""
        keep = set(signal.metadata.get("keep_tickers", []))
        ban_except = set(signal.metadata.get("ban_except", []))
        keep_all = keep | ban_except | {"cash"}

        sold_count = 0
        total_pnl = 0.0

        tickers_to_sell = [
            t for t in list(self.positions.keys())
            if t not in keep_all
        ]

        for ticker in tickers_to_sell:
            sell_sig = Signal(
                action=Action.SELL, ticker=ticker, size=1.0,
                target_pct=0, reason=signal.reason,
                exit_reason=ExitReason.CONDITION_BREAK,
            )
            trade = self.execute_sell(sell_sig, strategy_name, prices, timestamp)
            if trade:
                sold_count += 1
                total_pnl += trade.pnl

        if sold_count == 0:
            return None

        # 요약 Trade 반환
        return Trade(
            timestamp=timestamp, ticker="*BULK",
            side="SELL", price=0, qty=0,
            amount=0, fee=0, net_amount=0,
            strategy=strategy_name,
            pnl=total_pnl,
            reason="bulk_sell: %d positions, keep=%s" % (sold_count, list(keep_all)),
        )

    # ------------------------------------------------------------------
    # 상태 리포트
    # ------------------------------------------------------------------

    def summary(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        """포트폴리오 현재 상태 요약."""
        if prices:
            equity = self.update_equity(prices)
        else:
            equity = self.equity

        pos_value = sum(
            (prices.get(p.ticker, p.avg_price) if prices else p.avg_price) * p.qty
            for p in self.positions.values()
        )
        return {
            "total_capital": self.total_capital,
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "return_pct": round((equity / self.total_capital - 1) * 100, 2),
            "positions": {
                t: {
                    "avg_price": round(p.avg_price, 4),
                    "qty": round(p.qty, 4),
                    "value": round(p.avg_price * p.qty, 2),
                    "strategy": p.strategy_name,
                    "stage": p.stage,
                }
                for t, p in self.positions.items()
            },
            "total_trades": len(self.trades),
            "total_fees": round(self.total_fees, 2),
            "win_trades": sum(1 for t in self.trades if t.side == "SELL" and t.pnl > 0),
            "lose_trades": sum(1 for t in self.trades if t.side == "SELL" and t.pnl <= 0),
            # v5 포트폴리오 제약 상태
            "v5_simultaneous_hold_usd": round(pos_value, 2),
            "v5_simultaneous_cap_usd": self._max_simultaneous_hold_usd,
            "v5_daily_bought": sorted(self._daily_bought),
        }

    def trade_log(self) -> list[dict]:
        """거래 기록을 dict 리스트로 반환."""
        return [
            {
                "timestamp": str(t.timestamp),
                "ticker": t.ticker,
                "side": t.side,
                "price": round(t.price, 4),
                "qty": round(t.qty, 4),
                "amount": round(t.amount, 2),
                "fee": round(t.fee, 2),
                "net": round(t.net_amount, 2),
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "strategy": t.strategy,
                "reason": t.reason,
            }
            for t in self.trades
        ]
