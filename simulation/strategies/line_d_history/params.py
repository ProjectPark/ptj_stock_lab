"""
JUN 매매법 v2 — 파라미터 정의
==============================
출처: docs/rules/line_d/jun_trade_2023_v2.md

일봉 조건(E-1~E-3) + 분봉 조건(E-4~E-6) 통합 파라미터.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JunTradeParams:
    # ── E-1: 모멘텀 추세 진입 ─────────────────────────────
    rsi_entry_min: float = 55.0          # RSI(14) 최소
    pct_from_ma20_min: float = 0.0       # MA20 대비 최소 % (0 = MA20 위)
    drawdown_20d_max: float = -15.0      # 20일 고점 대비 최대 낙폭(%)

    # ── E-2: BTC 과열 필터 (크립토 종목) ─────────────────
    btc_rsi_overheat: float = 75.0       # BTC RSI 과열 임계값
    btc_10d_return_overheat: float = 15.0  # BTC 10일 수익률 과열(%)

    # ── E-3: 일봉 급등 추격 금지 ─────────────────────────
    chase_5d_return_max: float = 8.0     # 5일 수익률 최대(%)
    chase_5d_crypto_exception: float = 15.0    # E-3 크립토 예외: BTC regime≥2일 때 상한 완화
    chase_5d_noncrypto_ma20_min: float = 5.0   # E-3 비크립토 예외: MA20 대비 5%↑ 시 면제

    # ── E-4: 당일 위치 (분봉) ─────────────────────────────
    day_position_max: float = 0.4        # 당일 저·고가 대비 상단 40% 이하

    # ── E-5: 직전 30분 모멘텀 (분봉) ─────────────────────
    mom_30min_min: float = -0.5          # 30분 모멘텀 하한(%)
    mom_30min_max: float = 2.0           # 30분 모멘텀 상한(%) — F-9 연동

    # ── E-6: 진입 시간대 (분봉, 장 시작 후 경과분) ──────
    entry_min_from_open: int = 30        # 최소 30분 후 = 10:00 ET
    entry_max_from_open: int = 210       # 최대 210분 후 = 13:00 ET

    # ── 청산 규칙 ─────────────────────────────────────────
    target_pct: float = 15.0            # X-1: 목표 수익률(%)
    ma20_breakdown_pct: float = -15.0   # X-2: 추세 붕괴 기준 (MA20 대비 %)
    stop_loss_pct: float = -20.0        # X-3: 손절 기준(%)
    max_holding_days: int = 45          # X-4: 최대 보유 거래일

    # ── 피라미딩 ──────────────────────────────────────────
    pyramid_gain_pct: float = 5.0       # 직전 매수가 대비 상승 최소(%)
    pyramid_max_adds: int = 2           # 최대 추가 횟수
    pyramid_day_pos_max: float = 0.5    # 피라미딩 허용 당일 위치 상한 (v2.1 F-11)
    pyramid_time_cutoff_mins: int = 210  # F-10: 13:00 ET 이후 피라미딩 금지
    pyramid_mom_min: float = 0.0         # F-11: 피라미딩 시 비음수 모멘텀 요구

    # ── 포지션 사이징 (KRW) ───────────────────────────────
    size_vix_optimal: int = 1_000_000   # VIX 20~25: 100만원
    size_vix_normal: int = 1_000_000    # VIX 15~20: 100만원
    size_vix_overheat: int = 500_000    # VIX < 15:  50만원
    # VIX >= 25: 신규 매수 금지 (0원)

    # ── 기술적 지표 기간 ──────────────────────────────────
    rsi_period: int = 14
    ma_short: int = 20
    ma_long: int = 60
    rolling_high_period: int = 20
