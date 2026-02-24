"""
JUN 매매법 v2 — 진입 신호
===========================
일봉 조건 E-1~E-3 + 분봉 조건 E-4~E-6 + 금지 규칙 F-1~F-9
출처: docs/rules/line_d/jun_trade_2023_v2.md
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .indicators import (
    intraday_day_position,
    intraday_mom_30min,
    mins_from_open,
    vix_regime,
)
from .universe import MSTU_ENTRY_CUTOFF_MINS, is_banned, is_crypto


@dataclass
class EntryResult:
    allowed: bool
    reason: str                    # 통과 or 거부 사유
    entry_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    day_position: Optional[float] = None
    mom_30min: Optional[float] = None
    mins_open: Optional[int] = None


# ── 일봉 진입 필터 ────────────────────────────────────────────────────────────
def check_daily_entry(
    ticker: str,
    price: float,
    rsi: float,
    pct_from_ma20: float,
    drawdown_20d: float,
    return_5d: float,
    btc_rsi: float,
    btc_10d_ret: float,
    btc_reg: int,
    vix: float,
    params,
) -> tuple[bool, str]:
    """
    일봉 기준 진입 가능 여부 반환.
    반환: (allowed: bool, reason: str)
    """
    # F-5, F-6: 매매 금지 종목
    if is_banned(ticker):
        return False, f"F-5/F-6: {ticker} 매매 금지 종목"

    # VIX fear 이상: 신규 매수 금지
    regime = vix_regime(vix)
    if regime in ("fear", "extreme_fear"):
        return False, f"VIX={vix:.1f} fear 구간 — 신규 매수 금지"

    # F-1: RSI < 30 역발상 금지
    if rsi < 30:
        return False, f"F-1: RSI={rsi:.1f} < 30 — 역발상 매수 금지"

    # E-1: 모멘텀 추세 확인
    if rsi < params.rsi_entry_min:
        return False, f"E-1: RSI={rsi:.1f} < {params.rsi_entry_min}"
    if pct_from_ma20 < params.pct_from_ma20_min:
        return False, f"E-1: MA20 대비 {pct_from_ma20:.1f}% — MA20 아래"
    if drawdown_20d < params.drawdown_20d_max:
        return False, f"E-1: 20일 낙폭 {drawdown_20d:.1f}% < {params.drawdown_20d_max}%"

    # E-2: BTC 과열 시 크립토 진입 금지
    if is_crypto(ticker):
        if btc_rsi >= params.btc_rsi_overheat:
            return False, f"E-2: BTC RSI={btc_rsi:.1f} ≥ {params.btc_rsi_overheat} 과열"
        if btc_10d_ret >= params.btc_10d_return_overheat:
            return False, f"E-2: BTC 10d 수익률={btc_10d_ret:.1f}% ≥ {params.btc_10d_return_overheat}%"
        # F-3: BTC 레짐 3(과열) 크립토 신규 진입 금지
        if btc_reg == 3:
            return False, "F-3: BTC 레짐 3(과열) — 크립토 신규 진입 금지"

    # E-3 / F-7: 급등 추격 금지 (v2.1 크립토/비크립토 예외 분리)
    if return_5d >= params.chase_5d_return_max:
        # 비크립토 예외: MA20 대비 충분히 위에 있으면 허용
        if not is_crypto(ticker) and pct_from_ma20 >= params.chase_5d_noncrypto_ma20_min:
            pass  # 예외 통과
        # 크립토 예외: BTC regime >= 2이고 5d < 15%
        elif is_crypto(ticker) and btc_reg >= 2 and return_5d < params.chase_5d_crypto_exception:
            pass  # 예외 통과
        else:
            return False, f"F-7/E-3: 5일 수익률={return_5d:.1f}% ≥ {params.chase_5d_return_max}% 급등 추격 금지"

    # F-4: VIX < 15 극탐욕 — 레버리지 증가 금지 (진입 자체는 허용, 사이징만 축소)
    # (포지션 사이징은 portfolio.py에서 처리)

    return True, "일봉 조건 통과"


# ── 분봉 진입 탐색 ────────────────────────────────────────────────────────────
def find_intraday_entry(
    ticker: str,
    day_bars: pd.DataFrame,
    params,
    is_pyramid: bool = False,
) -> EntryResult:
    """
    당일 1min 봉에서 E-4~E-6 통과하는 첫 번째 진입 시각 탐색.

    day_bars: 당일 1min OHLCV (timestamp 오름차순 정렬, ET timezone)
    is_pyramid: 피라미딩 진입 여부 (day_position 상한 완화)
    반환: EntryResult
    """
    if day_bars.empty:
        return EntryResult(allowed=False, reason="당일 분봉 데이터 없음")

    day_low  = day_bars["close"].min()
    day_high = day_bars["close"].max()

    # E-6 시간대 상한 (F-8: MSTU 13:30 이후 금지)
    time_end = params.entry_max_from_open
    if ticker == "MSTU":
        time_end = min(time_end, MSTU_ENTRY_CUTOFF_MINS)
    # F-10: 피라미딩 시 13:00 ET 이후 금지
    if is_pyramid:
        time_end = min(time_end, params.pyramid_time_cutoff_mins)

    # 당일 위치 상한
    day_pos_limit = (
        params.pyramid_day_pos_max if is_pyramid else params.day_position_max
    )

    for i, row in day_bars.iterrows():
        ts    = row["timestamp"]
        price = float(row["close"])
        mfo   = mins_from_open(ts)

        # E-6: 시간대
        if mfo < params.entry_min_from_open:
            continue
        if mfo > time_end:
            break  # 이후 봉 모두 시간 초과

        # E-4: 당일 위치
        day_pos = intraday_day_position(price, day_low, day_high)
        if day_pos > day_pos_limit:
            continue

        # E-5: 직전 30분 모멘텀
        bars_before = day_bars[day_bars["timestamp"] < ts]
        mom = intraday_mom_30min(bars_before, price)
        if mom is None:
            continue
        if mom < params.mom_30min_min:
            continue
        if mom > params.mom_30min_max:  # F-9: 급등 추격
            continue
        # F-11: 피라미딩 시 비음수 모멘텀 요구
        if is_pyramid and mom < params.pyramid_mom_min:
            continue

        return EntryResult(
            allowed=True,
            reason="분봉 조건 통과 (E-4~E-6)",
            entry_time=ts,
            entry_price=price,
            day_position=round(day_pos, 3),
            mom_30min=round(mom, 3),
            mins_open=mfo,
        )

    return EntryResult(allowed=False, reason="당일 분봉에서 조건 통과 없음")
