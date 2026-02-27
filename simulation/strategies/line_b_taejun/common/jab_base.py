"""잽모드 전략 공통 베이스.

jab_soxl, jab_bitu, jab_tsll, jab_etq 공통:
- _is_in_window(): KST 시간 윈도우 체크
- check_exit(): target_pct 비교 익절
- _make_exit_signal(): 익절 시그널 생성
- _make_hold_signal(): 보유 HOLD 시그널 생성
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

from .base import Action, BaseStrategy, ExitReason, MarketData, Position, Signal

_KST = ZoneInfo("Asia/Seoul")


class JabBase(BaseStrategy):
    """잽모드 전략 공통 베이스."""

    def _is_in_window(self, market: MarketData) -> bool:
        """KST 시간 윈도우 체크 (오버나이트 세션 지원).

        미국 장중 타임스탬프(ET -05:00/-04:00)를 KST로 변환 후 비교한다.
        세션 구조: 17:30 KST → 자정 → 익일 06:59 KST (미국 장마감)

        Examples
        --------
        9:30 AM ET (winter) = 23:30 KST → passes (>= 17:30)
        1:00 PM ET (winter) = 03:00 KST → passes (h < 7)
        4:00 PM ET (winter) = 06:00 KST → passes (h < 7)
        """
        ts = market.time
        if getattr(ts, "tzinfo", None) is not None:
            ts_kst = ts.astimezone(_KST)
        else:
            ts_kst = ts  # naive → 그대로 사용
        h, m = ts_kst.hour, ts_kst.minute
        start_h, start_m = self.params.get("entry_start_kst", (17, 30))
        # 저녁 세션: start_kst 이후
        if h > start_h or (h == start_h and m >= start_m):
            return True
        # 자정 넘어 오전 구간 (00:00 ~ 06:59 KST = 미국 장 오전~오후)
        if h < 7:
            return True
        return False

    def check_exit(self, market: MarketData, position: Position) -> bool:
        """target_pct 익절 체크."""
        ticker = self.params.get("ticker", "")
        current = market.prices.get(ticker, 0)
        pnl = position.pnl_pct(current)
        if pnl is None:
            return False
        return pnl >= self.params["target_pct"]

    def _make_exit_signal(self, market: MarketData, position: Position,
                          ticker: str) -> Signal:
        """공통 익절/홀드 시그널 생성."""
        current = market.prices.get(ticker, 0)
        if current <= 0:
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         "no price data")

        pnl = position.pnl_pct(current)
        if pnl is None:
            return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                         "no price data")

        if pnl >= self.params["target_pct"]:
            return Signal(
                action=Action.SELL,
                ticker=ticker,
                size=1.0,
                target_pct=0,
                reason=f"{self.name} target hit: {pnl:.2f}% >= {self.params['target_pct']}%",
                exit_reason=ExitReason.TARGET_HIT,
                metadata={"pnl_pct": pnl},
            )

        return Signal(Action.HOLD, ticker, 0, self.params["target_pct"],
                      f"holding {ticker}: pnl={pnl:.2f}%")
