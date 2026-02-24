"""
JUN 매매법 v2 — 종목 유니버스 정의
=====================================
출처: docs/rules/line_d/jun_trade_2023_v2.md  섹션 6, 9(F-5~F-6)
"""
from __future__ import annotations

from typing import Literal

# ── 종목 분류 ──────────────────────────────────────────────────────────────────
CRYPTO_TICKERS: frozenset[str] = frozenset({
    "MSTU", "ROBN", "MSTX", "IREN", "IRE", "BITX", "BITU", "ETHU", "XXRP",
})

TECH_TICKERS: frozenset[str] = frozenset({
    "NVDL", "AMDL", "SOXL", "TQQQ", "TSLL", "PLTR2",
})

# F-5, F-6: 매매 절대 금지 종목
BANNED_TICKERS: frozenset[str] = frozenset({
    "CONL",   # F-5: 실거래 -56.2M원
    "BNKU",   # F-6: 승률 0%
    "GDXU",   # F-6: 승률 50% 이하
})

# 전체 유효 유니버스 (BANNED 제외)
VALID_TICKERS: frozenset[str] = (CRYPTO_TICKERS | TECH_TICKERS) - BANNED_TICKERS

# F-8: MSTU 시간 제한 (13:30 ET = 장 시작 후 240분)
MSTU_ENTRY_CUTOFF_MINS: int = 240

# ── 분류 함수 ──────────────────────────────────────────────────────────────────
TickerClass = Literal["crypto", "tech", "banned", "other"]


def classify(ticker: str) -> TickerClass:
    """종목을 크립토/기술/금지/기타로 분류한다."""
    if ticker in BANNED_TICKERS:
        return "banned"
    if ticker in CRYPTO_TICKERS:
        return "crypto"
    if ticker in TECH_TICKERS:
        return "tech"
    return "other"


def is_crypto(ticker: str) -> bool:
    return ticker in CRYPTO_TICKERS


def is_banned(ticker: str) -> bool:
    return ticker in BANNED_TICKERS
