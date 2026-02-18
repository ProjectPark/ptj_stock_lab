"""
Polymarket 지표 설정
====================
데이터 모델, 지표 Registry, 해석 임계값.
모든 모듈(poly_fetcher, poly_signals)이 이 파일을 참조한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================
# Enums
# ============================================================
class SlugType(Enum):
    """Slug 생성 패턴 유형"""
    DAILY = "daily"              # {month}-{day}
    DAILY_WITH_YEAR = "daily_y"  # {month}-{day}-{year}
    WEEKLY = "weekly"            # {month}-{week_start}-{week_end}
    MONTHLY = "monthly"          # {month}-{year}
    FOMC = "fomc"                # {month} — 탐색 필요


class MarketType(Enum):
    """마켓 응답 파싱 전략"""
    BINARY = "binary"            # Up/Down 단일 마켓
    MULTI_LEVEL = "multi_level"  # Above $X? 다중 마켓
    RANGE = "range"              # $X-$Y 구간 다중 마켓
    REACH_DIP = "reach_dip"      # reach/dip 분리 다중 마켓
    FED = "fed"                  # 4개 독립 질문


class SignalLevel(Enum):
    """종합 신호 수준"""
    STRONG_BUY = "strong_buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    STRONG_WARN = "strong_warn"


# ============================================================
# 데이터 모델
# ============================================================
@dataclass(frozen=True)
class MarketProb:
    """하나의 마켓 결과"""
    label: str           # "Up", "$68,000", "$66K-$68K", "25bp 인하"
    prob: float          # 0.0 ~ 1.0
    value: float | None  # 가격 수준 (파싱된 숫자). Binary면 None


@dataclass
class IndicatorResult:
    """하나의 지표 수집 결과"""
    name: str                            # "btc_up_down"
    title: str                           # "Bitcoin Up or Down on February 17?"
    slug: str                            # 실제 조회에 사용된 slug
    markets: list[MarketProb]            # 파싱된 확률 목록
    volume: float                        # 총 거래량 ($)
    fetched_at: datetime                 # 수집 시각
    error: str | None = None             # 실패 시 에러 메시지


@dataclass
class PolySignal:
    """해석된 신호"""
    name: str                            # "composite", "btc_direction", "fed" 등
    level: SignalLevel                   # 종합 수준
    score: float                         # -1.0 ~ +1.0 정규화 점수
    summary: str                         # 한줄 요약
    detail: dict = field(default_factory=dict)  # 세부 정보


# ============================================================
# API 설정
# ============================================================
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 10  # seconds

# FOMC 탐색 대상 월 (FOMC 회의가 있는 달)
FOMC_MONTHS = [
    "january", "march", "april", "may", "june",
    "july", "september", "october", "november", "december",
]


# ============================================================
# 지표 Registry
# ============================================================
INDICATORS: dict[str, dict] = {
    "btc_up_down": {
        "slug_template": "bitcoin-up-or-down-on-{month}-{day}",
        "slug_type": SlugType.DAILY,
        "market_type": MarketType.BINARY,
        "label": "BTC 당일 방향",
    },
    "btc_above_today": {
        "slug_template": "bitcoin-above-on-{month}-{day}",
        "slug_type": SlugType.DAILY,
        "market_type": MarketType.MULTI_LEVEL,
        "price_regex": r"above \$([0-9,]+)",
        "label": "BTC 당일 가격수준",
    },
    "btc_above_tomorrow": {
        "slug_template": "bitcoin-above-on-{month}-{day}",
        "slug_type": SlugType.DAILY,
        "market_type": MarketType.MULTI_LEVEL,
        "price_regex": r"above \$([0-9,]+)",
        "label": "BTC 익일 가격수준",
        "day_offset": 1,
    },
    "btc_price_today": {
        "slug_template": "bitcoin-price-on-{month}-{day}",
        "slug_type": SlugType.DAILY,
        "market_type": MarketType.RANGE,
        "label": "BTC 당일 예상 구간",
    },
    "btc_weekly": {
        "slug_template": "what-price-will-bitcoin-hit-{month}-{week_start}-{week_end}",
        "slug_type": SlugType.WEEKLY,
        "market_type": MarketType.REACH_DIP,
        "label": "BTC 주간 가격범위",
    },
    "btc_monthly": {
        "slug_template": "what-price-will-bitcoin-hit-in-{month}-{year}",
        "slug_type": SlugType.MONTHLY,
        "market_type": MarketType.REACH_DIP,
        "label": "BTC 월간 가격범위",
    },
    "eth_above_today": {
        "slug_template": "ethereum-above-on-{month}-{day}",
        "slug_type": SlugType.DAILY,
        "market_type": MarketType.MULTI_LEVEL,
        "price_regex": r"above \$([0-9,]+)",
        "label": "ETH 당일 가격수준",
    },
    "eth_monthly": {
        "slug_template": "what-price-will-ethereum-hit-in-{month}-{year}",
        "slug_type": SlugType.MONTHLY,
        "market_type": MarketType.REACH_DIP,
        "label": "ETH 월간 가격범위",
    },
    "ndx_up_down": {
        "slug_template": "ndx-up-or-down-on-{month}-{day}-{year}",
        "slug_type": SlugType.DAILY_WITH_YEAR,
        "market_type": MarketType.BINARY,
        "label": "나스닥 당일 방향",
    },
    "fed_decision": {
        "slug_template": "fed-decision-in-{month}",
        "slug_type": SlugType.FOMC,
        "market_type": MarketType.FED,
        "label": "Fed 금리 결정",
    },
}


# ============================================================
# 시그널 해석 임계값
# ============================================================
# BTC Up/Down
BTC_UP_STRONG_BUY = 0.65    # Up > 65% → 강한 상승 기대
BTC_UP_WEAK_BUY = 0.55      # Up 55~65% → 약한 상승
BTC_UP_WEAK_SELL = 0.45     # Up 45~55% → 불확실
BTC_UP_STRONG_WARN = 0.35   # Up < 35% → 강한 하락 기대

# Fed 신호
FED_STRONG_BUY = 0.5        # 가중치 점수 > 0.5 → 강한 매수
FED_WEAK_BUY = 0.2          # 가중치 점수 > 0.2 → 약한 매수

# NDX
NDX_DOWN_WARN = 0.70        # Down > 70% → 강한 하락 기대

# 종합 점수 → SignalLevel 매핑
COMPOSITE_STRONG_BUY = 0.5
COMPOSITE_WEAK_BUY = 0.15
COMPOSITE_WEAK_SELL = -0.15
COMPOSITE_STRONG_WARN = -0.5
