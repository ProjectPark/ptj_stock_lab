"""
서킷 브레이커 시스템 (Circuit Breaker)
==========================================
v5 매매 규칙 1절 — CB-1~CB-6 중앙 관리.

CB-1: VIX 일간 +6% → 7거래일 신규 매수 금지
CB-2: GLD 일간 +3% → 3거래일 신규 매수 금지
CB-3: BTC -5% → 해제 시까지 신규 매수 금지
CB-4: BTC +5% → 추격매수(신규 매수) 금지
CB-5: Polymarket 금리 상승 확률 50%+ → 모든 신규 매수 금지 + 레버리지 3일 추가 대기
CB-6: 종목 +20% 과열 → 레버리지→비레버리지 전환

예외 규칙 (1-4절):
- 급락 역매수(crash_buy): CB-1, CB-2, CB-3 발동 중에도 허용. CB-5는 금지.
- Unix(VIX) 방어모드(vix_gold): CB-1 발동 중에도 허용.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..common.params import CIRCUIT_BREAKER


@dataclass
class CBStatus:
    """서킷 브레이커 상태 스냅샷."""
    cb1_active: bool = False    # VIX 급등
    cb1_remaining: int = 0      # 남은 거래일
    cb2_active: bool = False    # GLD 급등
    cb2_remaining: int = 0
    cb3_active: bool = False    # BTC 급락
    cb4_active: bool = False    # BTC 급등
    cb5_active: bool = False    # 금리 상승 우려
    cb5_lev_cooldown: int = 0   # 레버리지 추가 대기 거래일
    cb6_tickers: dict[str, float] = field(default_factory=dict)  # {ticker: peak_price}

    @property
    def any_active(self) -> bool:
        return any([
            self.cb1_active, self.cb2_active,
            self.cb3_active, self.cb4_active, self.cb5_active,
        ])

    @property
    def buy_blocked(self) -> bool:
        """신규 매수가 전면 차단되는지 (급락 역매수 예외 제외)."""
        return self.cb1_active or self.cb2_active or self.cb3_active or self.cb5_active

    @property
    def only_chase_blocked(self) -> bool:
        """추격매수만 차단 (CB-4)."""
        return self.cb4_active and not self.buy_blocked


class CircuitBreaker:
    """서킷 브레이커 — 상태 추적 및 판정.

    Parameters
    ----------
    params : dict | None
        CIRCUIT_BREAKER 파라미터. None이면 기본값.
    total_capital_usd : float
        총 투자금. CB-6 자금 계산에 사용.
    """

    def __init__(
        self,
        params: dict | None = None,
        total_capital_usd: float = 15_000,
    ):
        self.params = params or CIRCUIT_BREAKER
        self.total_capital = total_capital_usd

        # 내부 상태
        self._cb1_remaining: int = 0        # 남은 거래일 카운트다운
        self._cb2_remaining: int = 0
        self._cb3_active: bool = False
        self._cb4_active: bool = False
        self._cb5_active: bool = False
        self._cb5_lev_cooldown: int = 0     # 레버리지 ETF 추가 대기 거래일
        self._cb6_peaks: dict[str, float] = {}  # {ticker: peak_price_after_trigger}

        self._log: list[dict] = []          # 발동/해제 로그

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    @property
    def status(self) -> CBStatus:
        return CBStatus(
            cb1_active=self._cb1_remaining > 0,
            cb1_remaining=self._cb1_remaining,
            cb2_active=self._cb2_remaining > 0,
            cb2_remaining=self._cb2_remaining,
            cb3_active=self._cb3_active,
            cb4_active=self._cb4_active,
            cb5_active=self._cb5_active,
            cb5_lev_cooldown=self._cb5_lev_cooldown,
            cb6_tickers=dict(self._cb6_peaks),
        )

    def is_leverage_cooldown(self) -> bool:
        """CB-5 해제 후 레버리지 ETF 추가 대기 중인지."""
        return self._cb5_lev_cooldown > 0

    def is_leverage_ticker_blocked(self, ticker: str) -> bool:
        """CB-5 레버리지 쿨다운 중에 해당 종목이 차단되는지."""
        if not self.is_leverage_cooldown():
            return False
        lev_tickers = self.params.get("cb5_leverage_tickers", [])
        return ticker in lev_tickers

    def is_cb6_overheated(self, ticker: str) -> bool:
        """CB-6: 해당 종목이 과열(+20%) 상태인지."""
        return ticker in self._cb6_peaks

    def get_cb6_substitute(self, ticker: str) -> str | None:
        """CB-6 과열 시 대체 종목 반환. None이면 매수 전면 금지."""
        mapping = self.params.get("cb6_mapping", {})
        return mapping.get(ticker)

    # ------------------------------------------------------------------
    # 매수 허용 여부 판정
    # ------------------------------------------------------------------

    def check_buy_allowed(
        self,
        ticker: str,
        strategy_name: str = "",
        is_crash_buy: bool = False,
        is_vix_defense: bool = False,
    ) -> tuple[bool, str]:
        """매수 허용 여부를 반환한다.

        Parameters
        ----------
        ticker : str
            매수 대상 종목.
        strategy_name : str
            전략 이름 (예외 규칙 적용에 사용).
        is_crash_buy : bool
            급락 역매수 여부. CB-1/2/3 예외 허용.
        is_vix_defense : bool
            Unix(VIX) 방어모드 (IAU/GDXU) 여부. CB-1 예외 허용.

        Returns
        -------
        (allowed, reason)
        """
        st = self.status

        # CB-5: 모든 신규 매수 금지 (급락 역매수도 금지)
        if st.cb5_active:
            return False, "CB-5: 금리 상승 우려 — 신규 매수 전면 금지"

        # CB-5 레버리지 쿨다운
        if self.is_leverage_ticker_blocked(ticker):
            lev_tickers = self.params.get("cb5_leverage_tickers", [])
            if ticker in lev_tickers:
                return False, (
                    f"CB-5 쿨다운: 레버리지 ETF {ticker} "
                    f"{self._cb5_lev_cooldown}거래일 추가 대기 중"
                )

        # CB-1: VIX 급등 — 급락 역매수 & 방어모드 예외
        if st.cb1_active:
            if is_crash_buy or is_vix_defense:
                pass  # 허용
            else:
                return False, f"CB-1: VIX 급등 — {st.cb1_remaining}거래일 신규 매수 금지"

        # CB-2: GLD 급등 — 급락 역매수 예외
        if st.cb2_active:
            if is_crash_buy:
                pass  # 허용
            else:
                return False, f"CB-2: GLD 급등 — {st.cb2_remaining}거래일 신규 매수 금지"

        # CB-3: BTC 급락 — 급락 역매수 예외
        if st.cb3_active:
            if is_crash_buy:
                pass  # 허용
            else:
                return False, "CB-3: BTC 급락 — 신규 매수 금지 (BTC 회복 시까지)"

        # CB-4: BTC 급등 — 추격매수만 금지 (기존 포지션 매도는 허용)
        if st.cb4_active and not is_crash_buy:
            return False, "CB-4: BTC 급등 — 추격매수 금지"

        # CB-6: 과열 종목
        if self.is_cb6_overheated(ticker):
            sub = self.get_cb6_substitute(ticker)
            if sub:
                return False, f"CB-6: {ticker} 과열 — {sub}로 전환 필요"
            else:
                return False, f"CB-6: {ticker} 과열 — 추가 매수 전면 금지"

        return True, ""

    # ------------------------------------------------------------------
    # 상태 업데이트 (거래일 경과 시 호출)
    # ------------------------------------------------------------------

    def on_trading_day_start(
        self,
        changes: dict[str, float],
        prices: dict[str, float],
        poly: dict[str, float] | None,
        crypto_changes: dict[str, float] | None,
        time: datetime,
    ) -> list[dict]:
        """거래일 시작 시 서킷 브레이커 조건을 평가하고 상태를 업데이트한다.

        Parameters
        ----------
        changes : dict[str, float]
            당일 등락률. {"VIX": 7.5, "GLD": 3.2, ...}
        prices : dict[str, float]
            현재가.
        poly : dict[str, float] | None
            Polymarket 확률.
        crypto_changes : dict[str, float] | None
            크립토 스팟 24h 변동률. {"BTC": -6.0, ...}
        time : datetime
            현재 시각.

        Returns
        -------
        list[dict]
            이번 사이클에 발동/해제된 이벤트 로그.
        """
        events: list[dict] = []

        # ── CB-1: VIX 급등 (+6% 이상) ────────────────────────────────
        vix_chg = changes.get("VIX", 0.0)
        cb1_min = self.params.get("cb1_vix_min", 6.0)
        if vix_chg >= cb1_min:
            cb1_days = self.params.get("cb1_days", 7)
            self._cb1_remaining = cb1_days
            events.append({
                "time": time, "cb": "CB-1", "action": "triggered",
                "reason": f"VIX {vix_chg:+.1f}% >= {cb1_min}%",
                "ban_days": cb1_days,
            })

        # CB-1 카운트다운 (이미 발동 중)
        elif self._cb1_remaining > 0:
            self._cb1_remaining -= 1
            if self._cb1_remaining == 0:
                events.append({
                    "time": time, "cb": "CB-1", "action": "released",
                    "reason": "7거래일 경과",
                })

        # ── CB-2: GLD 급등 (+3% 이상) ────────────────────────────────
        gld_chg = changes.get("GLD", 0.0)
        cb2_min = self.params.get("cb2_gld_min", 3.0)
        if gld_chg >= cb2_min:
            cb2_days = self.params.get("cb2_days", 3)
            self._cb2_remaining = cb2_days
            events.append({
                "time": time, "cb": "CB-2", "action": "triggered",
                "reason": f"GLD {gld_chg:+.1f}% >= {cb2_min}%",
                "ban_days": cb2_days,
            })
        elif self._cb2_remaining > 0:
            self._cb2_remaining -= 1
            if self._cb2_remaining == 0:
                events.append({
                    "time": time, "cb": "CB-2", "action": "released",
                    "reason": "3거래일 경과",
                })

        # ── CB-3: BTC 급락 (-5% 이상) ────────────────────────────────
        btc_chg = (crypto_changes or {}).get("BTC", 0.0)
        cb3_drop = self.params.get("cb3_btc_drop", -5.0)
        was_cb3 = self._cb3_active
        if btc_chg <= cb3_drop:
            if not was_cb3:
                events.append({
                    "time": time, "cb": "CB-3", "action": "triggered",
                    "reason": f"BTC {btc_chg:+.1f}% <= {cb3_drop}%",
                })
            self._cb3_active = True
        else:
            if was_cb3:
                events.append({
                    "time": time, "cb": "CB-3", "action": "released",
                    "reason": f"BTC {btc_chg:+.1f}% > {cb3_drop}% (회복)",
                })
            self._cb3_active = False

        # ── CB-4: BTC 급등 (+5% 이상) ────────────────────────────────
        cb4_surge = self.params.get("cb4_btc_surge", 5.0)
        was_cb4 = self._cb4_active
        if btc_chg >= cb4_surge:
            if not was_cb4:
                events.append({
                    "time": time, "cb": "CB-4", "action": "triggered",
                    "reason": f"BTC {btc_chg:+.1f}% >= {cb4_surge}%",
                })
            self._cb4_active = True
        else:
            if was_cb4:
                events.append({
                    "time": time, "cb": "CB-4", "action": "released",
                    "reason": f"BTC {btc_chg:+.1f}% < {cb4_surge}%",
                })
            self._cb4_active = False

        # ── CB-5: 금리 상승 확률 50%+ ─────────────────────────────────
        rate_prob = (poly or {}).get("rate_hike", 0.0)
        cb5_min = self.params.get("cb5_rate_hike_prob", 0.50)
        was_cb5 = self._cb5_active
        if rate_prob >= cb5_min:
            if not was_cb5:
                events.append({
                    "time": time, "cb": "CB-5", "action": "triggered",
                    "reason": f"Polymarket 금리 상승 {rate_prob:.0%} >= {cb5_min:.0%}",
                })
            self._cb5_active = True
            self._cb5_lev_cooldown = 0  # 활성 중에는 쿨다운 무의미
        else:
            if was_cb5:
                # CB-5 해제 → 레버리지 ETF 3거래일 추가 대기 시작
                cd = self.params.get("cb5_lev_cooldown_days", 3)
                self._cb5_lev_cooldown = cd
                events.append({
                    "time": time, "cb": "CB-5", "action": "released",
                    "reason": f"금리 상승 확률 {rate_prob:.0%} < {cb5_min:.0%}",
                    "lev_cooldown_days": cd,
                })
            self._cb5_active = False
            # 레버리지 쿨다운 카운트다운
            if self._cb5_lev_cooldown > 0:
                self._cb5_lev_cooldown -= 1
                if self._cb5_lev_cooldown == 0:
                    events.append({
                        "time": time, "cb": "CB-5",
                        "action": "lev_cooldown_released",
                        "reason": "레버리지 ETF 추가 대기 완료",
                    })

        # ── CB-6: 과열 종목 +20% ──────────────────────────────────────
        cb6_surge = self.params.get("cb6_surge_min", 20.0)
        cb6_recovery = self.params.get("cb6_recovery_pct", -10.0)
        mapping = self.params.get("cb6_mapping", {})

        for ticker in list(mapping.keys()):
            chg = changes.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)

            if ticker not in self._cb6_peaks:
                # 신규 발동 체크
                if chg >= cb6_surge:
                    self._cb6_peaks[ticker] = price
                    events.append({
                        "time": time, "cb": "CB-6", "action": "triggered",
                        "ticker": ticker, "reason": f"{ticker} {chg:+.1f}% >= {cb6_surge}%",
                        "substitute": mapping.get(ticker),
                    })
            else:
                # 해제 체크 (고점 대비 -10%)
                peak = self._cb6_peaks[ticker]
                if price > peak:
                    self._cb6_peaks[ticker] = price  # 고점 갱신
                elif peak > 0 and (price - peak) / peak * 100 <= cb6_recovery:
                    del self._cb6_peaks[ticker]
                    events.append({
                        "time": time, "cb": "CB-6", "action": "released",
                        "ticker": ticker,
                        "reason": f"{ticker} 고점 대비 {(price/peak-1)*100:.1f}% 조정",
                    })

        self._log.extend(events)
        return events

    # ------------------------------------------------------------------
    # 로그
    # ------------------------------------------------------------------

    def get_log(self) -> list[dict]:
        return list(self._log)

    def reset_log(self) -> None:
        self._log.clear()

    def summary(self) -> dict:
        """현재 서킷 브레이커 요약."""
        st = self.status
        return {
            "cb1": {"active": st.cb1_active, "remaining_days": st.cb1_remaining},
            "cb2": {"active": st.cb2_active, "remaining_days": st.cb2_remaining},
            "cb3": {"active": st.cb3_active},
            "cb4": {"active": st.cb4_active},
            "cb5": {
                "active": st.cb5_active,
                "lev_cooldown_days": st.cb5_lev_cooldown,
            },
            "cb6": {"overheated_tickers": list(st.cb6_tickers.keys())},
            "any_buy_blocked": st.buy_blocked,
        }
