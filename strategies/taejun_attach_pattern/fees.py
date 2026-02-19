"""
taejun_attach_pattern - 수수료 계산
====================================
KIS 미국주식 수수료 체계를 전략 시그널에 적용.
backtest_common.py의 수수료 상수를 재사용하되, 전략 엔진용 인터페이스 제공.

수수료 구조 (KIS 미국주식):
- 매수: 수수료 0.25% + 환전 스프레드 0.10% = 0.35%
- 매도: 수수료 0.25% + SEC Fee 0.00278% + 환전 스프레드 0.10% = 0.35278%
"""
from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# 수수료 상수 (KIS 미국주식 기본값)
# ============================================================
DEFAULT_COMMISSION_PCT = 0.25       # 매매 수수료 (매수/매도 각각)
DEFAULT_SEC_FEE_PCT = 0.00278       # SEC Fee (매도 시에만)
DEFAULT_FX_SPREAD_PCT = 0.10        # 환전 스프레드 (편도)


@dataclass
class FeeConfig:
    """수수료 설정.

    기본값은 KIS 미국주식 표준 수수료.
    실제 수수료율이 다른 경우 (이벤트 할인 등) 생성자에서 오버라이드.
    """
    commission_pct: float = DEFAULT_COMMISSION_PCT
    sec_fee_pct: float = DEFAULT_SEC_FEE_PCT
    fx_spread_pct: float = DEFAULT_FX_SPREAD_PCT

    @property
    def buy_rate(self) -> float:
        """매수 수수료율 (%)."""
        return self.commission_pct + self.fx_spread_pct

    @property
    def sell_rate(self) -> float:
        """매도 수수료율 (%)."""
        return self.commission_pct + self.sec_fee_pct + self.fx_spread_pct

    @property
    def round_trip_rate(self) -> float:
        """왕복 수수료율 (%)."""
        return self.buy_rate + self.sell_rate


@dataclass
class FeeResult:
    """수수료 계산 결과."""
    gross_amount: float     # 세전 금액
    fee: float              # 수수료
    net_amount: float       # 순 금액 (gross - fee)
    fee_rate_pct: float     # 적용된 수수료율 (%)


class FeeCalculator:
    """수수료 계산기.

    전략 시그널의 매수/매도 금액에 수수료를 적용하고,
    목표 수익률이 수수료 제외 기준인지 확인하는 유틸리티.
    """

    def __init__(self, config: FeeConfig | None = None):
        self.config = config or FeeConfig()

    def calc_buy(self, amount: float) -> FeeResult:
        """매수 수수료 계산.

        Parameters
        ----------
        amount : float
            총 매수 금액 (USD).

        Returns
        -------
        FeeResult
            net_amount = 실제 주식 매수에 사용되는 금액.
        """
        rate = self.config.buy_rate
        fee = amount * rate / 100
        return FeeResult(
            gross_amount=amount,
            fee=fee,
            net_amount=amount - fee,
            fee_rate_pct=rate,
        )

    def calc_sell(self, proceeds: float) -> FeeResult:
        """매도 수수료 계산.

        Parameters
        ----------
        proceeds : float
            매도 대금 (USD).

        Returns
        -------
        FeeResult
            net_amount = 실제 수취 금액.
        """
        rate = self.config.sell_rate
        fee = proceeds * rate / 100
        return FeeResult(
            gross_amount=proceeds,
            fee=fee,
            net_amount=proceeds - fee,
            fee_rate_pct=rate,
        )

    def net_pnl_pct(self, entry_price: float, exit_price: float) -> float:
        """수수료를 반영한 순수익률(%)을 계산한다.

        Parameters
        ----------
        entry_price : float
            매수가.
        exit_price : float
            매도가.

        Returns
        -------
        float
            수수료 제외 순수익률 (%).
        """
        if entry_price <= 0:
            return 0.0

        gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
        return gross_pnl_pct - self.config.round_trip_rate

    def required_gross_pct(self, target_net_pct: float) -> float:
        """목표 순수익률 달성을 위해 필요한 세전 수익률(%).

        전략의 target_pct가 '수수료 제외' 기준이므로,
        실제 매도 판단시에는 이 값 이상인지 확인해야 함.

        Parameters
        ----------
        target_net_pct : float
            목표 순수익률 (%, 수수료 제외).

        Returns
        -------
        float
            필요한 세전 수익률 (%).
        """
        return target_net_pct + self.config.round_trip_rate

    def is_target_met(self, entry_price: float, current_price: float,
                      target_net_pct: float) -> bool:
        """수수료 반영 후 목표 수익률 달성 여부.

        Parameters
        ----------
        entry_price : float
            매수가.
        current_price : float
            현재가.
        target_net_pct : float
            목표 순수익률 (%, 수수료 제외).

        Returns
        -------
        bool
        """
        net = self.net_pnl_pct(entry_price, current_price)
        return net >= target_net_pct

    def summary(self) -> dict:
        """현재 수수료 설정 요약."""
        return {
            "commission_pct": self.config.commission_pct,
            "sec_fee_pct": self.config.sec_fee_pct,
            "fx_spread_pct": self.config.fx_spread_pct,
            "buy_rate_pct": self.config.buy_rate,
            "sell_rate_pct": self.config.sell_rate,
            "round_trip_pct": self.config.round_trip_rate,
        }
