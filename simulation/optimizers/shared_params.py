"""v3/v4/v5 공유 파라미터 헬퍼.

v2에서 상속된 11개 공유 baseline 파라미터를 한 곳에서 관리한다.
각 버전의 get_baseline_params()에서 params.update(get_shared_baseline_params())로 사용.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config


def get_shared_baseline_params() -> dict:
    """v2에서 상속된 공유 파라미터의 현재 config 값."""
    return {
        "STOP_LOSS_PCT": config.STOP_LOSS_PCT,
        "STOP_LOSS_BULLISH_PCT": config.STOP_LOSS_BULLISH_PCT,
        "COIN_SELL_PROFIT_PCT": config.COIN_SELL_PROFIT_PCT,
        "COIN_SELL_BEARISH_PCT": config.COIN_SELL_BEARISH_PCT,
        "CONL_SELL_PROFIT_PCT": config.CONL_SELL_PROFIT_PCT,
        "CONL_SELL_AVG_PCT": config.CONL_SELL_AVG_PCT,
        "DCA_DROP_PCT": config.DCA_DROP_PCT,
        "MAX_HOLD_HOURS": config.MAX_HOLD_HOURS,
        "TAKE_PROFIT_PCT": config.TAKE_PROFIT_PCT,
        "PAIR_GAP_SELL_THRESHOLD_V2": config.PAIR_GAP_SELL_THRESHOLD_V2,
        "PAIR_SELL_FIRST_PCT": config.PAIR_SELL_FIRST_PCT,
    }
