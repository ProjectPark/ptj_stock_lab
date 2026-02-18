"""
Polymarket 시그널 해석 모듈
===========================
IndicatorResult → PolySignal 변환.
poly_fetcher를 import하지 않는 순수 함수 모듈.
"""
from __future__ import annotations

import logging
from typing import Optional

from .poly_config import (
    BTC_UP_STRONG_BUY,
    BTC_UP_STRONG_WARN,
    BTC_UP_WEAK_BUY,
    BTC_UP_WEAK_SELL,
    COMPOSITE_STRONG_BUY,
    COMPOSITE_STRONG_WARN,
    COMPOSITE_WEAK_BUY,
    COMPOSITE_WEAK_SELL,
    FED_STRONG_BUY,
    FED_WEAK_BUY,
    IndicatorResult,
    MarketProb,
    PolySignal,
    SignalLevel,
)

logger = logging.getLogger(__name__)


# ============================================================
# 유틸리티
# ============================================================

def _is_valid(result: Optional[IndicatorResult]) -> bool:
    """result가 유효한지 확인 (None이 아니고 error도 없는지)."""
    return result is not None and result.error is None


def _score_to_level(score: float) -> SignalLevel:
    """종합 점수를 SignalLevel로 변환."""
    if score >= COMPOSITE_STRONG_BUY:
        return SignalLevel.STRONG_BUY
    if score >= COMPOSITE_WEAK_BUY:
        return SignalLevel.WEAK_BUY
    if score >= COMPOSITE_WEAK_SELL:
        return SignalLevel.NEUTRAL
    if score >= COMPOSITE_STRONG_WARN:
        return SignalLevel.WEAK_SELL
    return SignalLevel.STRONG_WARN


# ============================================================
# 개별 시그널 함수
# ============================================================

def compute_btc_pivot(result: IndicatorResult) -> float | None:
    """
    MULTI_LEVEL 타입에서 50% 분기점을 선형 보간으로 계산.

    markets를 value 기준 오름차순 정렬 후,
    prob이 0.5를 교차하는 두 점 사이에서 선형 보간한다.
    """
    if not _is_valid(result):
        return None

    # value가 있는 마켓만 필터링, value 오름차순 정렬
    priced = [m for m in result.markets if m.value is not None]
    if len(priced) < 2:
        return None

    # value 오름차순, 확률은 내림차순이어야 함 (Above $X 구조)
    priced.sort(key=lambda m: m.value)  # type: ignore[arg-type]

    # 50% 교차점 탐색
    for i in range(len(priced) - 1):
        p_low = priced[i]
        p_high = priced[i + 1]

        # prob이 0.5를 사이에 두는 두 점 찾기
        # "Above $X" 구조에서는 낮은 가격일수록 prob이 높고 높은 가격일수록 낮음
        if (p_low.prob >= 0.5 >= p_high.prob) or (p_high.prob >= 0.5 >= p_low.prob):
            # 두 점의 prob이 같으면 중간값 반환
            prob_diff = p_low.prob - p_high.prob
            if abs(prob_diff) < 1e-9:
                return (p_low.value + p_high.value) / 2  # type: ignore[operator]

            # 선형 보간: value = v_low + (v_high - v_low) * (p_low - 0.5) / (p_low - p_high)
            ratio = (p_low.prob - 0.5) / prob_diff
            pivot = p_low.value + (p_high.value - p_low.value) * ratio  # type: ignore[operator]
            return pivot

    # 교차점을 못 찾은 경우 (모두 50% 이상이거나 이하)
    logger.debug("50%% 교차점을 찾지 못함: %s", result.name)
    return None


def compute_fed_signal(result: IndicatorResult) -> PolySignal | None:
    """
    Fed 가중치 공식:
      fed_signal = (cut_25bp_prob * 2.0) + (cut_50bp_prob * 3.0) - (hike_prob * 2.0)
    """
    if not _is_valid(result):
        return None

    cut_25bp = 0.0
    cut_50bp = 0.0
    hike = 0.0
    no_change = 0.0

    for m in result.markets:
        label = m.label.lower()
        if "50bp" in label and ("인하" in label or "cut" in label):
            cut_50bp = m.prob
        elif "25bp" in label and ("인하" in label or "cut" in label):
            cut_25bp = m.prob
        elif "인상" in label or "hike" in label:
            hike = m.prob
        elif "변경" in label or "no change" in label:
            no_change = m.prob

    score = (cut_25bp * 2.0) + (cut_50bp * 3.0) - (hike * 2.0)

    if score >= FED_STRONG_BUY:
        level = SignalLevel.STRONG_BUY
        desc = "강한 매수 신호"
    elif score >= FED_WEAK_BUY:
        level = SignalLevel.WEAK_BUY
        desc = "약한 매수 신호"
    elif score < 0:
        level = SignalLevel.STRONG_WARN
        desc = "경고 (인상 우세)"
    else:
        level = SignalLevel.NEUTRAL
        desc = "중립"

    summary = f"Fed 25bp 인하 {cut_25bp:.0%} — {desc}"

    return PolySignal(
        name="fed",
        level=level,
        score=min(max(score, -1.0), 1.0),
        summary=summary,
        detail={
            "cut_25bp": cut_25bp,
            "cut_50bp": cut_50bp,
            "hike": hike,
            "no_change": no_change,
            "raw_score": score,
        },
    )


def compute_btc_direction(result: IndicatorResult) -> PolySignal | None:
    """
    BINARY 타입 (btc_up_down)의 Up 확률로 방향 판정.
    """
    if not _is_valid(result):
        return None

    up_prob = 0.0
    for m in result.markets:
        label = m.label.lower()
        if "up" in label or "상승" in label:
            up_prob = m.prob
            break

    if up_prob >= BTC_UP_STRONG_BUY:
        level = SignalLevel.STRONG_BUY
        desc = "강한 상승"
    elif up_prob >= BTC_UP_WEAK_BUY:
        level = SignalLevel.WEAK_BUY
        desc = "약한 상승"
    elif up_prob >= BTC_UP_WEAK_SELL:
        level = SignalLevel.NEUTRAL
        desc = "불확실"
    elif up_prob >= BTC_UP_STRONG_WARN:
        level = SignalLevel.WEAK_SELL
        desc = "하락 기대"
    else:
        level = SignalLevel.STRONG_WARN
        desc = "강한 하락"

    return PolySignal(
        name="btc_direction",
        level=level,
        score=(up_prob - 0.5) * 2.0,
        summary=f"BTC Up {up_prob:.1%} — {desc}",
        detail={"up_prob": up_prob},
    )


def compute_price_consensus(result: IndicatorResult) -> PolySignal | None:
    """
    RANGE 타입 (btc_price_today)에서 확률 최고 구간을 식별.
    상위 2개 구간의 확률 합 > 90%이면 '안정', 아니면 '변동성 확대 예상'.
    """
    if not _is_valid(result):
        return None

    if not result.markets:
        return None

    # 확률 내림차순 정렬
    sorted_markets = sorted(result.markets, key=lambda m: m.prob, reverse=True)
    top = sorted_markets[0]
    top2_sum = sum(m.prob for m in sorted_markets[:2])

    if top2_sum >= 0.9:
        stability = "안정"
        level = SignalLevel.NEUTRAL
    else:
        stability = "변동성 확대 예상"
        level = SignalLevel.WEAK_SELL

    summary = f"예상 구간 {top.label} ({top.prob:.0%}) — {stability}"

    return PolySignal(
        name="price_consensus",
        level=level,
        score=0.0,  # 방향성이 아니라 안정성 지표
        summary=summary,
        detail={
            "top_range": top.label,
            "top_prob": top.prob,
            "top2_sum": top2_sum,
            "stability": stability,
            "all_ranges": [
                {"label": m.label, "prob": m.prob}
                for m in sorted_markets
                if m.prob >= 0.01
            ],
        },
    )


def compute_direction_change(
    today: IndicatorResult,
    tomorrow: IndicatorResult,
) -> PolySignal | None:
    """
    오늘 vs 내일 50% 분기점을 비교하여 방향 전환 감지.
    차이 < $500 → 횡보, 내일 > 오늘 → 상승 전환, 내일 < 오늘 → 하락 전환.
    """
    today_pivot = compute_btc_pivot(today)
    tomorrow_pivot = compute_btc_pivot(tomorrow)

    if today_pivot is None or tomorrow_pivot is None:
        return None

    diff = tomorrow_pivot - today_pivot

    if abs(diff) < 500:
        level = SignalLevel.NEUTRAL
        desc = "횡보"
    elif diff > 0:
        level = SignalLevel.WEAK_BUY
        desc = "상승 전환 기대"
    else:
        level = SignalLevel.WEAK_SELL
        desc = "하락 전환 경고"

    # 정규화: $5000 차이를 ±1.0으로 매핑
    score = max(-1.0, min(1.0, diff / 5000.0))

    return PolySignal(
        name="direction_change",
        level=level,
        score=score,
        summary=f"오늘 ${today_pivot:,.0f} → 내일 ${tomorrow_pivot:,.0f} — {desc}",
        detail={
            "today_pivot": today_pivot,
            "tomorrow_pivot": tomorrow_pivot,
            "diff": diff,
        },
    )


def compute_weekly_bias(result: IndicatorResult) -> PolySignal | None:
    """
    REACH_DIP 타입. reach(상승)와 dip(하락)의 확률 비교.
    label 접두사 "reach"→상승, "dip"→하락.
    """
    if not _is_valid(result):
        return None

    reach_probs: list[MarketProb] = []
    dip_probs: list[MarketProb] = []

    for m in result.markets:
        label = m.label.lower()
        if label.startswith("reach") or label.startswith("상승"):
            reach_probs.append(m)
        elif label.startswith("dip") or label.startswith("하락"):
            dip_probs.append(m)

    if not reach_probs and not dip_probs:
        return None

    # 가장 높은 reach/dip 확률
    max_reach = max((m.prob for m in reach_probs), default=0.0)
    max_dip = max((m.prob for m in dip_probs), default=0.0)

    if max_dip > max_reach:
        level = SignalLevel.WEAK_SELL
        desc = "하방 압력"
    elif max_reach > max_dip:
        level = SignalLevel.WEAK_BUY
        desc = "상방 기대"
    else:
        level = SignalLevel.NEUTRAL
        desc = "균형"

    score = (max_reach - max_dip) * 2.0
    score = max(-1.0, min(1.0, score))

    # 요약에 가장 유의미한 마켓 표시
    reach_str = f"상승 최고 {max_reach:.0%}" if reach_probs else "상승 없음"
    dip_str = f"하락 최고 {max_dip:.0%}" if dip_probs else "하락 없음"

    return PolySignal(
        name="weekly_bias",
        level=level,
        score=score,
        summary=f"{reach_str} vs {dip_str} — {desc}",
        detail={
            "reach": [{"label": m.label, "prob": m.prob, "value": m.value}
                      for m in reach_probs],
            "dip": [{"label": m.label, "prob": m.prob, "value": m.value}
                    for m in dip_probs],
            "max_reach": max_reach,
            "max_dip": max_dip,
        },
    )


# ============================================================
# 종합 시그널
# ============================================================

def compute_composite_signal(
    results: dict[str, IndicatorResult],
) -> PolySignal | None:
    """
    종합 판단:
      score = (btc_up - 0.5) * 2.0
            + fed_weighted * 0.5
            - max(0, ndx_down - 0.5)
    """
    # --- BTC Up/Down ---
    btc_up = 0.5  # 기본값 (중립)
    btc_result = results.get("btc_up_down")
    if _is_valid(btc_result):
        for m in btc_result.markets:  # type: ignore[union-attr]
            label = m.label.lower()
            if "up" in label or "상승" in label:
                btc_up = m.prob
                break

    # --- Fed ---
    fed_weighted = 0.0
    fed_result = results.get("fed_decision")
    if _is_valid(fed_result):
        fed_sig = compute_fed_signal(fed_result)  # type: ignore[arg-type]
        if fed_sig is not None:
            fed_weighted = fed_sig.detail.get("raw_score", 0.0)

    # --- NDX Down ---
    ndx_down = 0.5
    ndx_result = results.get("ndx_up_down")
    if _is_valid(ndx_result):
        for m in ndx_result.markets:  # type: ignore[union-attr]
            label = m.label.lower()
            if "down" in label or "하락" in label:
                ndx_down = m.prob
                break

    # --- BTC 50% 분기점 (참고용) ---
    btc_pivot = None
    above_result = results.get("btc_above_today")
    if _is_valid(above_result):
        btc_pivot = compute_btc_pivot(above_result)  # type: ignore[arg-type]

    # --- 종합 점수 ---
    score = (
        (btc_up - 0.5) * 2.0
        + fed_weighted * 0.5
        - max(0.0, ndx_down - 0.5)
    )

    level = _score_to_level(score)

    level_desc = {
        SignalLevel.STRONG_BUY: "강한 매수",
        SignalLevel.WEAK_BUY: "약한 매수",
        SignalLevel.NEUTRAL: "중립",
        SignalLevel.WEAK_SELL: "약한 매도",
        SignalLevel.STRONG_WARN: "강한 경고",
    }

    summary = f"종합 {score:+.2f} — {level_desc[level]}"

    return PolySignal(
        name="composite",
        level=level,
        score=max(-1.0, min(1.0, score)),
        summary=summary,
        detail={
            "btc_up": btc_up,
            "fed_weighted": fed_weighted,
            "ndx_down": ndx_down,
            "btc_pivot": btc_pivot,
            "raw_score": score,
        },
    )


# ============================================================
# 일괄 생성
# ============================================================

def generate_all_poly_signals(
    results: dict[str, IndicatorResult],
) -> dict[str, PolySignal]:
    """개별 시그널 + 종합 시그널을 한번에 생성."""
    signals: dict[str, PolySignal] = {}

    # BTC direction
    if "btc_up_down" in results:
        sig = compute_btc_direction(results["btc_up_down"])
        if sig is not None:
            signals["btc_direction"] = sig

    # BTC pivot (PolySignal이 아닌 float이지만 dict에 넣기 위해 PolySignal로 감싸기)
    if "btc_above_today" in results:
        pivot = compute_btc_pivot(results["btc_above_today"])
        if pivot is not None:
            signals["btc_pivot"] = PolySignal(
                name="btc_pivot",
                level=SignalLevel.NEUTRAL,
                score=0.0,
                summary=f"BTC 50% 분기점: ${pivot:,.0f}",
                detail={"pivot": pivot},
            )

    # Fed
    if "fed_decision" in results:
        sig = compute_fed_signal(results["fed_decision"])
        if sig is not None:
            signals["fed"] = sig

    # Price consensus
    if "btc_price_today" in results:
        sig = compute_price_consensus(results["btc_price_today"])
        if sig is not None:
            signals["price_consensus"] = sig

    # Direction change (today vs tomorrow)
    if "btc_above_today" in results and "btc_above_tomorrow" in results:
        sig = compute_direction_change(
            results["btc_above_today"],
            results["btc_above_tomorrow"],
        )
        if sig is not None:
            signals["direction_change"] = sig

    # Weekly bias
    if "btc_weekly" in results:
        sig = compute_weekly_bias(results["btc_weekly"])
        if sig is not None:
            signals["weekly_bias"] = sig

    # Composite
    composite = compute_composite_signal(results)
    if composite is not None:
        signals["composite"] = composite

    logger.info("생성된 시그널: %s", list(signals.keys()))
    return signals
