"""
저가매수 전략 (Bargain-Buy)
===========================
3년 최고가 대비 대폭락시 진입, 목표 수익률 도달시 매도.
종목별 파라미터 테이블로 11개 종목 독립 관리.

출처: kakaotalk_trading_notes_2026-02-19.csv — 6️⃣ 폭락시 레버리지 롱 / 저가매수모드 1~5
"""
from __future__ import annotations

from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal
from .params import BARGAIN_BUY
from .registry import register


@register
class BargainBuy(BaseStrategy):
    """저가매수 — 3년 최고가 대비 N% 하락시 진입."""

    name = "bargain_buy"
    version = "1.0"
    description = "3년 최고가 대비 폭락 종목 저가매수 + 목표수익률 매도"

    def __init__(self, params: dict | None = None):
        super().__init__(params or BARGAIN_BUY)
        self._ticker_params = self.params.get("tickers", {})
        self._block_rules = self.params.get("block_rules", {})

    # ------------------------------------------------------------------
    # 금지 조건
    # ------------------------------------------------------------------

    def _is_blocked(self, market: MarketData) -> tuple[bool, str]:
        """저가매수 금지 조건 체크.

        Returns (blocked, reason).
        """
        # 금 하락시 금지
        if self._block_rules.get("gld_decline"):
            gld = market.changes.get("GLD", 0)
            if gld < 0:
                return True, f"GLD decline ({gld:.2f}%)"

        # Polymarket 상승 기대 49% 이하시 금지
        poly_min = self._block_rules.get("poly_ndx_min", 0.49)
        if market.poly:
            ndx_up = market.poly.get("ndx_up", 0.5)
            if ndx_up < poly_min:
                return True, f"Polymarket NDX too low ({ndx_up:.2f})"

        # 거래량 감소 체크 (volumes 데이터가 있을 때만)
        # 향후 volume_decline_days 기반 로직 추가 가능

        return False, ""

    # ------------------------------------------------------------------
    # 3년 최고가 대비 하락률 계산
    # ------------------------------------------------------------------

    def _calc_drop_from_high(self, ticker: str, market: MarketData) -> float | None:
        """3년 최고가 대비 현재가 하락률(%)을 계산한다.

        Returns None if data unavailable.
        """
        if not market.history or ticker not in market.history:
            return None
        high_3y = market.history[ticker].get("high_3y", 0)
        if high_3y <= 0:
            return None
        current = market.prices.get(ticker, 0)
        if current <= 0:
            return None
        return (current - high_3y) / high_3y * 100

    # ------------------------------------------------------------------
    # 진입 검토
    # ------------------------------------------------------------------

    def check_entry(self, market: MarketData) -> bool:
        """하나라도 진입 가능한 종목이 있으면 True."""
        blocked, _ = self._is_blocked(market)
        if blocked:
            return False

        for ticker in self._ticker_params:
            if self._check_ticker_entry(ticker, market):
                return True
        return False

    def _check_ticker_entry(self, ticker: str, market: MarketData) -> bool:
        """특정 종목의 초기 진입 조건 충족 여부."""
        cfg = self._ticker_params.get(ticker)
        if not cfg:
            return False

        drop = self._calc_drop_from_high(ticker, market)
        if drop is None:
            return False

        return drop <= cfg["drop_pct"]

    def _check_ticker_add(self, ticker: str, market: MarketData,
                          position: Position) -> bool:
        """추가매수 조건 충족 여부.

        진입 이후 add_drop% 추가 하락시 추가매수.
        """
        cfg = self._ticker_params.get(ticker)
        if not cfg:
            return False

        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False

        pnl_pct = (current - position.avg_price) / position.avg_price * 100
        return pnl_pct <= cfg["add_drop"]

    # ------------------------------------------------------------------
    # 청산 검토
    # ------------------------------------------------------------------

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """목표 수익률 도달 여부."""
        ticker = position.ticker
        cfg = self._ticker_params.get(ticker)
        if not cfg:
            return False

        current = market.prices.get(ticker, 0)
        if current <= 0 or position.avg_price <= 0:
            return False

        pnl_pct = (current - position.avg_price) / position.avg_price * 100

        # 보유 기한 초과 체크
        if cfg.get("hold_days", 0) > 0:
            elapsed = (market.time - position.entry_time).days
            if elapsed >= cfg["hold_days"]:
                return True

        return pnl_pct >= cfg["target_pct"]

    # ------------------------------------------------------------------
    # 시그널 생성
    # ------------------------------------------------------------------

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        """시장 데이터를 받아 시그널을 생성한다."""

        # 포지션이 있으면 청산 검토
        if position is not None:
            return self._generate_exit_signal(market, position)

        # 포지션 없으면 진입 검토
        return self._generate_entry_signal(market)

    def _generate_entry_signal(self, market: MarketData) -> Signal:
        """신규 진입 시그널."""
        blocked, reason = self._is_blocked(market)
        if blocked:
            return Signal(Action.SKIP, "", 0, 0, f"blocked: {reason}")

        for ticker, cfg in self._ticker_params.items():
            if ticker not in market.prices:
                continue

            if not self._check_ticker_entry(ticker, market):
                continue

            drop = self._calc_drop_from_high(ticker, market)
            return Signal(
                action=Action.BUY,
                ticker=ticker,
                size=0.5,  # 초기 진입 50% (CONL 80% 제외 — 대부분 50%)
                target_pct=cfg["target_pct"],
                reason=f"bargain_entry: {ticker} dropped {drop:.1f}% from 3y high "
                       f"(threshold: {cfg['drop_pct']}%)",
                metadata={
                    "drop_from_high": drop,
                    "add_drop": cfg["add_drop"],
                    "sell_splits": cfg["sell_splits"],
                    "reinvest": cfg["reinvest"],
                    "split_days": cfg["split_days"],
                },
            )

        return Signal(Action.SKIP, "", 0, 0, "no bargain opportunity")

    def _generate_exit_signal(self, market: MarketData,
                              position: Position) -> Signal:
        """보유 포지션 청산 시그널."""
        ticker = position.ticker
        cfg = self._ticker_params.get(ticker)
        if not cfg:
            return Signal(Action.HOLD, ticker, 0, 0, "unknown ticker config")

        current = market.prices.get(ticker, 0)
        if current <= 0:
            return Signal(Action.HOLD, ticker, 0, 0, "no price data")

        pnl_pct = (current - position.avg_price) / position.avg_price * 100

        # 추가매수 검토 (아직 stage 1이고 추가 하락시)
        if position.stage == 1 and self._check_ticker_add(ticker, market, position):
            return Signal(
                action=Action.BUY,
                ticker=ticker,
                size=cfg["add_size"],
                target_pct=cfg["target_pct"],
                reason=f"bargain_add: {ticker} dropped {pnl_pct:.1f}% from entry "
                       f"(add threshold: {cfg['add_drop']}%)",
                metadata={"stage": 2},
            )

        # 보유 기한 초과
        if cfg.get("hold_days", 0) > 0:
            elapsed = (market.time - position.entry_time).days
            if elapsed >= cfg["hold_days"]:
                return Signal(
                    action=Action.SELL,
                    ticker=ticker,
                    size=1.0,
                    target_pct=0,
                    reason=f"bargain_exit: {ticker} hold limit {cfg['hold_days']}d reached",
                    exit_reason=ExitReason.TIME_LIMIT,
                    metadata={"reinvest": cfg["reinvest"]},
                )

        # 목표 수익률 도달
        if pnl_pct >= cfg["target_pct"]:
            sell_splits = cfg.get("sell_splits", 0)
            if sell_splits > 0:
                # 분할매도: 1/N씩
                sell_size = 1.0 / sell_splits
            else:
                sell_size = 1.0

            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=sell_size,
                target_pct=0,
                reason=f"bargain_exit: {ticker} target {cfg['target_pct']}% hit "
                       f"(current: {pnl_pct:.1f}%)",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={
                    "pnl_pct": pnl_pct,
                    "sell_splits": sell_splits,
                    "reinvest": cfg["reinvest"],
                    "split_days": cfg["split_days"],
                },
            )

        return Signal(Action.HOLD, ticker, 0, cfg["target_pct"],
                      f"holding: {ticker} pnl={pnl_pct:.1f}%")

    # ------------------------------------------------------------------
    # 파라미터 검증
    # ------------------------------------------------------------------

    def validate_params(self) -> list[str]:
        errors = []
        for ticker, cfg in self._ticker_params.items():
            if cfg["drop_pct"] >= 0:
                errors.append(f"{ticker}: drop_pct must be negative")
            if cfg["target_pct"] <= 0:
                errors.append(f"{ticker}: target_pct must be positive")
            if cfg["add_drop"] >= 0:
                errors.append(f"{ticker}: add_drop must be negative")
        return errors
