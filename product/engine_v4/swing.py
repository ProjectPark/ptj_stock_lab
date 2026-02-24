"""
product/engine_v4/swing.py
============================
v4 스윙 매매 진입/매도 판단 순수 함수.
포지션 관리는 호출자 책임.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SwingSignal:
    """스윙 진입 신호."""
    should_enter: bool = False
    trigger_type: str = ""          # "momentum" | "vix" | ""
    targets: list[str] = field(default_factory=list)  # 진입 대상 티커들
    weight_pct: float = 0.0         # 진입 비중 (%)


def check_swing_entry(
    changes: dict[str, dict],
    is_swing_active: bool,
    params: dict,
) -> SwingSignal:
    """
    스윙 진입 여부 판단.

    Args:
        changes: {"VIX": {"change_pct": 4.0}, "SOXL": {"change_pct": 28.0}, ...}
        is_swing_active: 현재 스윙 모드 활성 여부 (활성이면 진입 불가)
        params: DEFAULT_PARAMS

    Returns:
        SwingSignal

    로직 (backtest_v4.py _evaluate_swing_mode):
    - is_swing_active=True → SwingSignal(should_enter=False)
    - VIX change_pct >= cb_vix_spike_pct → trigger_type="vix", targets=[swing_stage2_gld_ticker]
    - swing_eligible_tickers 중 change_pct >= swing_trigger_pct → trigger_type="momentum", targets=[해당 티커들]
    """
    if is_swing_active:
        return SwingSignal(should_enter=False)

    cb_vix_spike_pct: float = params.get("cb_vix_spike_pct", 3.0)
    swing_trigger_pct: float = params.get("swing_trigger_pct", 27.5)
    swing_eligible_tickers: list[str] = params.get("swing_eligible_tickers", [])
    swing_stage2_gld_ticker: str = params.get("swing_stage2_gld_ticker", "GLD")
    swing_vix_stage1_weight_pct: float = params.get("swing_vix_stage1_weight_pct", 80.0)
    swing_stage1_weight_pct: float = params.get("swing_stage1_weight_pct", 90.0)

    vix_pct: float = changes.get("VIX", {}).get("change_pct", 0.0)
    vix_trigger = vix_pct >= cb_vix_spike_pct

    if vix_trigger:
        return SwingSignal(
            should_enter=True,
            trigger_type="vix",
            targets=[swing_stage2_gld_ticker],
            weight_pct=swing_vix_stage1_weight_pct,
        )

    momentum_targets: list[str] = []
    for ticker in swing_eligible_tickers:
        pct = changes.get(ticker, {}).get("change_pct")
        if pct is None:
            continue
        if pct >= swing_trigger_pct:
            momentum_targets.append(ticker)

    if momentum_targets:
        return SwingSignal(
            should_enter=True,
            trigger_type="momentum",
            targets=momentum_targets,
            weight_pct=swing_stage1_weight_pct,
        )

    return SwingSignal(should_enter=False)


@dataclass
class SwingExitSignal:
    """스윙 매도 신호."""
    should_exit: bool = False
    reason: str = ""            # "stage1_atr_stop" | "stage1_drawdown" | "stage1_maturity" | "stage2_stop" | "stage2_maturity" | ...


def check_swing_exit_stage1_momentum(
    swing_peak_price: float,
    initial_entry_price: float,
    entry_atr: float,
    cur_price: float,
    elapsed_days: int,
    params: dict,
) -> SwingExitSignal:
    """
    Stage1 모멘텀 스윙 매도 판단.

    소스: backtest_v4.py _process_swing_sells() Stage1 모멘텀 로직
    """
    swing_stage1_atr_mult: float = params.get("swing_stage1_atr_mult", 2.5)
    swing_stage1_drawdown_pct: float = params.get("swing_stage1_drawdown_pct", -11.0)
    swing_stage1_hold_days: int = params.get("swing_stage1_hold_days", 21)

    atr_stop = initial_entry_price - (swing_stage1_atr_mult * max(entry_atr, 0.01))
    peak = max(swing_peak_price, 1e-9)
    drawdown_pct = (cur_price - peak) / peak * 100.0

    if cur_price <= atr_stop:
        return SwingExitSignal(should_exit=True, reason="stage1_atr_stop")

    if drawdown_pct <= swing_stage1_drawdown_pct:
        return SwingExitSignal(should_exit=True, reason="stage1_drawdown")

    if elapsed_days >= swing_stage1_hold_days:
        return SwingExitSignal(should_exit=True, reason="stage1_maturity")

    return SwingExitSignal(should_exit=False)


def check_swing_exit_stage1_vix(
    initial_entry_price: float,
    cur_price: float,
    elapsed_days: int,
    params: dict,
) -> SwingExitSignal:
    """
    Stage1 VIX 스윙 (GLD 보유) 매도 판단.
    """
    swing_stage2_stop_pct: float = params.get("swing_stage2_stop_pct", -5.0)
    swing_vix_stage1_hold_days: int = params.get("swing_vix_stage1_hold_days", 105)

    if initial_entry_price > 0:
        pnl_pct = (cur_price - initial_entry_price) / initial_entry_price * 100.0
        if pnl_pct <= swing_stage2_stop_pct:
            return SwingExitSignal(should_exit=True, reason="stage2_stop")

    if elapsed_days >= swing_vix_stage1_hold_days:
        return SwingExitSignal(should_exit=True, reason="stage1_maturity")

    return SwingExitSignal(should_exit=False)


def check_swing_exit_stage2_momentum(
    initial_entry_price: float,
    cur_price: float,
    elapsed_days: int,
    params: dict,
) -> SwingExitSignal:
    """Stage2 모멘텀 매도 판단."""
    swing_stage2_stop_pct: float = params.get("swing_stage2_stop_pct", -5.0)
    swing_stage2_hold_days: int = params.get("swing_stage2_hold_days", 105)

    if initial_entry_price > 0:
        pnl_pct = (cur_price - initial_entry_price) / initial_entry_price * 100.0
        if pnl_pct <= swing_stage2_stop_pct:
            return SwingExitSignal(should_exit=True, reason="stage2_stop")

    if elapsed_days >= swing_stage2_hold_days:
        return SwingExitSignal(should_exit=True, reason="stage2_maturity")

    return SwingExitSignal(should_exit=False)


def check_swing_exit_stage2_vix(
    elapsed_days: int,
    params: dict,
) -> SwingExitSignal:
    """Stage2 VIX 쿨다운 완료 판단."""
    swing_vix_stage2_cooldown_days: int = params.get("swing_vix_stage2_cooldown_days", 63)

    if elapsed_days >= swing_vix_stage2_cooldown_days:
        return SwingExitSignal(should_exit=True, reason="stage2_maturity")

    return SwingExitSignal(should_exit=False)
