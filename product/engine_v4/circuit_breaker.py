"""
product/engine_v4/circuit_breaker.py
=====================================
v4 서킷브레이커 (CB-1~6) 순수 함수 구현.

backtest_v4.py _update_circuit_breaker_state() 로직을 stateless 순수 함수로 이식.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class CBState:
    """서킷브레이커 상태 (입출력 통합)."""

    # 출력: 매매 차단 여부
    block_all: bool = False         # CB-1/2/3/5 → 전체 차단 (매도도 포함 가능)
    block_new_buys: bool = False    # CB-4(BTC 급등) → 신규 매수만 차단
    active_rules: list[str] = field(default_factory=list)  # 활성 CB 규칙 이름 목록

    # 상태 유지 (호출자가 관리)
    vix_cooldown_days: int = 0      # VIX 쿨다운 잔여 거래일
    gld_cooldown_days: int = 0      # GLD 쿨다운 잔여 거래일
    overheat_state: dict = field(default_factory=dict)  # {ticker: {"active": bool, "peak": float}}


def evaluate_circuit_breaker(
    changes: dict[str, dict],
    poly_probs: dict | None,
    cb_state: CBState,
    params: dict,
) -> CBState:
    """
    CB-1~6 판단 후 새 CBState 반환.

    Args:
        changes: {"VIX": {"change_pct": 4.0}, "GLD": {"change_pct": 3.5},
                  "BITU": {"change_pct": 14.0}, ...}
                  BITU change_pct를 BTC 프록시로 사용 (backtest_v4.py와 동일).
        poly_probs: {"rate_hike": 0.6} — None이면 금리 CB 비활성.
        cb_state: 이전 호출에서 반환된 CBState (쿨다운 카운터 포함).
        params: DEFAULT_PARAMS 또는 오버라이드 dict.

    Returns:
        새 CBState (block_all, block_new_buys, active_rules, 업데이트된 카운터).

    Notes:
        - vix/gld cooldown_days 감소는 advance_cooldown_day() 로 처리.
          여기서는 트리거 발생 시 카운터를 max() 로 갱신만 한다.
        - BITU를 BTC 프록시로 사용 (backtest_v4.py 라인 1664).
    """
    # ── 파라미터 로드 ─────────────────────────────────────────────
    cb_vix_spike_pct: float = params.get("cb_vix_spike_pct", 3.0)
    cb_vix_cooldown_days: int = params.get("cb_vix_cooldown_days", 13)
    cb_gld_spike_pct: float = params.get("cb_gld_spike_pct", 3.0)
    cb_gld_cooldown_days: int = params.get("cb_gld_cooldown_days", 3)
    cb_btc_crash_pct: float = params.get("cb_btc_crash_pct", -6.0)
    cb_btc_surge_pct: float = params.get("cb_btc_surge_pct", 13.5)
    cb_rate_hike_prob_pct: float = params.get("cb_rate_hike_prob_pct", 50.0)
    cb_overheat_pct: float = params.get("cb_overheat_pct", 20.0)
    cb_overheat_recovery_pct: float = params.get("cb_overheat_recovery_pct", -10.0)
    cb_overheat_tickers: list[str] = params.get(
        "cb_overheat_tickers", ["SOXL", "CONL", "IRE", "MSTU"]
    )

    # ── 입력값 추출 ───────────────────────────────────────────────
    vix_pct: float = changes.get("VIX", {}).get("change_pct", 0.0)
    gld_pct: float = changes.get("GLD", {}).get("change_pct", 0.0)
    # BITU를 BTC 프록시로 사용 (backtest_v4.py 라인 1664)
    btc_proxy_pct: float = changes.get("BITU", {}).get("change_pct", 0.0)
    rate_hike_raw = (poly_probs or {}).get("rate_hike", None)
    rate_hike_prob_pct: float = (rate_hike_raw * 100.0) if rate_hike_raw is not None else 0.0

    # ── 상태 복사 (불변성 보장) ───────────────────────────────────
    new_vix_cooldown: int = cb_state.vix_cooldown_days
    new_gld_cooldown: int = cb_state.gld_cooldown_days
    new_overheat_state: dict = copy.deepcopy(cb_state.overheat_state)

    # ── CB-1: VIX 급등 ────────────────────────────────────────────
    if vix_pct >= cb_vix_spike_pct:
        new_vix_cooldown = max(new_vix_cooldown, cb_vix_cooldown_days)

    # ── CB-2: GLD 급등 ────────────────────────────────────────────
    if gld_pct >= cb_gld_spike_pct:
        new_gld_cooldown = max(new_gld_cooldown, cb_gld_cooldown_days)

    # ── CB-6: 과열 종목 상태 업데이트 ─────────────────────────────
    overheated_now: set[str] = set()
    for ticker in cb_overheat_tickers:
        state = new_overheat_state.setdefault(ticker, {"active": False, "peak": 0.0})
        info = changes.get(ticker, {})
        pct = info.get("change_pct")
        price = info.get("close")

        active: bool = bool(state.get("active", False))
        peak: float = float(state.get("peak", 0.0))

        if price is not None and price > 0:
            if active:
                peak = max(peak, float(price))
                if peak > 0:
                    drawdown_pct = (float(price) - peak) / peak * 100.0
                    if drawdown_pct <= cb_overheat_recovery_pct:
                        active = False
                        peak = float(price)
            if (not active) and pct is not None and pct >= cb_overheat_pct:
                active = True
                peak = float(price)

        state["active"] = active
        state["peak"] = peak
        if active:
            overheated_now.add(ticker)

    # ── CB-3: BTC 급락 ────────────────────────────────────────────
    btc_crash_active: bool = btc_proxy_pct <= cb_btc_crash_pct

    # ── CB-4: BTC 급등 ────────────────────────────────────────────
    btc_surge_active: bool = btc_proxy_pct >= cb_btc_surge_pct

    # ── CB-5: 금리 상승 우려 ──────────────────────────────────────
    rate_hike_active: bool = (
        rate_hike_raw is not None
        and rate_hike_raw != 0.5
        and rate_hike_prob_pct >= cb_rate_hike_prob_pct
    )

    # ── 차단 플래그 계산 ──────────────────────────────────────────
    vix_cooldown_active: bool = new_vix_cooldown > 0
    gld_cooldown_active: bool = new_gld_cooldown > 0

    block_all: bool = (
        vix_cooldown_active
        or gld_cooldown_active
        or btc_crash_active
        or rate_hike_active
    )
    block_new_buys: bool = block_all or btc_surge_active

    # ── 활성 CB 규칙 이름 목록 ────────────────────────────────────
    active_rules: list[str] = []
    if vix_cooldown_active:
        active_rules.append("CB-1_VIX_SPIKE")
    if gld_cooldown_active:
        active_rules.append("CB-2_GLD_SPIKE")
    if btc_crash_active:
        active_rules.append("CB-3_BTC_CRASH")
    if btc_surge_active:
        active_rules.append("CB-4_BTC_SURGE")
    if rate_hike_active:
        active_rules.append("CB-5_RATE_HIKE")
    if overheated_now:
        active_rules.append("CB-6_OVERHEAT_ANY")

    return CBState(
        block_all=block_all,
        block_new_buys=block_new_buys,
        active_rules=active_rules,
        vix_cooldown_days=new_vix_cooldown,
        gld_cooldown_days=new_gld_cooldown,
        overheat_state=new_overheat_state,
    )


def advance_cooldown_day(cb_state: CBState) -> CBState:
    """
    거래일이 넘어갈 때 쿨다운 카운터를 1씩 감소시킨다.

    vix_cooldown_days, gld_cooldown_days를 각각 1 감소
    (0 미만으로 내려가지 않음).

    Args:
        cb_state: 현재 CBState.

    Returns:
        vix_cooldown_days / gld_cooldown_days 가 1 감소된 새 CBState.
        block_all / block_new_buys / active_rules 는 변경하지 않음.
        (다음 evaluate_circuit_breaker() 호출 시 갱신됨.)
    """
    new_vix = max(0, cb_state.vix_cooldown_days - 1)
    new_gld = max(0, cb_state.gld_cooldown_days - 1)

    return CBState(
        block_all=cb_state.block_all,
        block_new_buys=cb_state.block_new_buys,
        active_rules=list(cb_state.active_rules),
        vix_cooldown_days=new_vix,
        gld_cooldown_days=new_gld,
        overheat_state=copy.deepcopy(cb_state.overheat_state),
    )
