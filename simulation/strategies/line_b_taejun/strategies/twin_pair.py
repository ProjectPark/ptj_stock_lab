"""
쌍둥이 페어 갭 전략 (Twin-Pair)
================================
Legacy: signals_v2.check_twin_pairs_v2()

v5 변경:
- entry_threshold: 2.2% (4-3절)
- CONL/IRE: 40/60 분할 매도 메타데이터 추가 (4-5절)
  - 40%: 갭 수렴 즉시 매도
  - 60%: +5% 고정 익절
"""
from __future__ import annotations

from ..common.base import Action, BaseStrategy, MarketData, Position, Signal
from ..common.registry import register

# 40/60 분할 매도 적용 종목 (4-5절)
_SPLIT_SELL_TICKERS = frozenset({"CONL", "IRE"})
_SPLIT_TP_PCT = 5.0       # 고정 익절 +5%
_SPLIT_TP_RATIO = 0.60    # 60%는 고정 익절
_SPLIT_GAP_RATIO = 0.40   # 40%는 갭 수렴 즉시 매도

# 코인 페어 후행 종목 (4-2절): 당일 1종목만 선택
_COIN_FOLLOW_TICKERS = frozenset({"MSTU", "IRE"})
_VOLATILITY_GAP = 0.5     # 변동성 차이 임계값 (%)


@register
class TwinPairStrategy(BaseStrategy):
    """쌍둥이 페어 갭 분석 — multi-follow + 매도 기준.

    v5: entry_threshold=2.2%, CONL/IRE 40/60 분할 매도 지원.
    """

    name = "twin_pair"
    version = "1.1"
    description = "선행 종목과 후행 종목의 갭을 분석하여 진입/매도 시그널 생성"

    def __init__(self, params: dict | None = None):
        defaults = {
            "entry_threshold": 2.2,    # v5: 2.2% (v4까지 1.5%)
            "sell_threshold": 0.9,
            "coin_follow_volatility_gap": 0.5,  # v5 4-2절
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def select_coin_follow(
        self, changes: dict, volumes: dict | None = None
    ) -> str:
        """MSTU vs IRE 당일 1종목 선택 (v5 4-2절).

        Parameters
        ----------
        changes : dict
            {ticker: {"change_pct": float, ...}, ...}
        volumes : dict | None
            {ticker: float, ...} 거래량

        Returns
        -------
        str
            선택된 후행 종목 ("MSTU" or "IRE")
        """
        mstu_data = changes.get("MSTU", {})
        ire_data = changes.get("IRE", {})

        mstu_vol = abs(mstu_data.get("change_pct", 0.0))
        ire_vol = abs(ire_data.get("change_pct", 0.0))

        vol_gap = self.params.get("coin_follow_volatility_gap", 0.5)

        # 1. 변동성 차이 >= 0.5%
        if abs(mstu_vol - ire_vol) >= vol_gap:
            return "MSTU" if mstu_vol > ire_vol else "IRE"

        # 2. 거래량 비교
        if volumes:
            mstu_volume = volumes.get("MSTU", 0.0)
            ire_volume = volumes.get("IRE", 0.0)
            if mstu_volume != ire_volume:
                return "MSTU" if mstu_volume > ire_volume else "IRE"

        # 3. 기본: MSTU
        return "MSTU"

    def evaluate(self, changes: dict, pairs: dict, volumes: dict | None = None) -> list[dict]:
        """Legacy 호환 — 기존 check_twin_pairs_vN()과 동일한 list[dict] 반환.

        Parameters
        ----------
        changes : dict
            {ticker: {"change_pct": float, ...}, ...}
        pairs : dict
            {key: {"lead": str, "follow": [str, ...], "label": str}, ...}
        volumes : dict | None
            {ticker: float, ...} 거래량 (v5 4-2절 코인 후행 종목 선택에 사용)

        Returns
        -------
        list[dict]
            각 follow별 시그널 (pair, lead, follow, lead_pct, follow_pct, gap, signal, message)
        """
        entry_threshold = self.params.get("entry_threshold", 1.5)
        sell_threshold = self.params.get("sell_threshold", 0.9)
        results: list[dict] = []

        # v5 4-2절: 코인 후행 종목 당일 선택
        selected_coin_follow = self.select_coin_follow(changes, volumes)

        for _key, pair_cfg in pairs.items():
            lead = pair_cfg["lead"]
            follows = pair_cfg["follow"]
            label = pair_cfg.get("label", _key)

            lead_data = changes.get(lead, {})
            lead_pct = lead_data.get("change_pct", 0.0)

            for follow_ticker in follows:
                # v5 4-2절: 코인 페어에서 선택되지 않은 후행 종목 스킵
                if follow_ticker in _COIN_FOLLOW_TICKERS and follow_ticker != selected_coin_follow:
                    continue

                follow_data = changes.get(follow_ticker, {})
                follow_pct = follow_data.get("change_pct", 0.0)
                gap = lead_pct - follow_pct

                # 40/60 분할 매도 대상 여부
                is_split = follow_ticker in _SPLIT_SELL_TICKERS

                if gap <= sell_threshold and follow_pct > 0:
                    signal = "SELL"
                    if is_split:
                        # CONL/IRE: 40% 갭 수렴 즉시 매도 (4-5절)
                        message = (
                            f"{label} | {follow_ticker} 갭 {gap:+.2f}% ≤ {sell_threshold}% "
                            f"& 양봉 → 갭수렴 매도 {int(_SPLIT_GAP_RATIO*100)}% "
                            f"(+{_SPLIT_TP_PCT}% 고정익절 {int(_SPLIT_TP_RATIO*100)}% 대기)"
                        )
                    else:
                        message = (
                            f"{label} | {follow_ticker} 갭 {gap:+.2f}% ≤ {sell_threshold}% "
                            f"& 양봉 → 80% 즉시 매도"
                        )
                elif gap >= entry_threshold:
                    signal = "ENTRY"
                    message = (
                        f"{label} | {lead} +{lead_pct:.2f}% vs "
                        f"{follow_ticker} +{follow_pct:.2f}% (갭 {gap:+.2f}%) → 진입"
                    )
                else:
                    signal = "HOLD"
                    message = (
                        f"{label} | {lead} {lead_pct:+.2f}% vs "
                        f"{follow_ticker} {follow_pct:+.2f}% (갭 {gap:+.2f}%) → 대기"
                    )

                result_item: dict = {
                    "pair": label,
                    "lead": lead,
                    "follow": follow_ticker,
                    "lead_pct": lead_pct,
                    "follow_pct": follow_pct,
                    "gap": gap,
                    "signal": signal,
                    "message": message,
                }

                # v5 4-2절: 코인 후행 종목 선택 정보
                if follow_ticker in _COIN_FOLLOW_TICKERS:
                    result_item["coin_follow_selected"] = selected_coin_follow

                # 40/60 분할 매도 메타데이터 (CONL/IRE)
                if is_split and signal == "SELL":
                    result_item["split_sell"] = True
                    result_item["gap_ratio"] = _SPLIT_GAP_RATIO     # 0.40
                    result_item["tp_ratio"] = _SPLIT_TP_RATIO        # 0.60
                    result_item["tp_pct"] = _SPLIT_TP_PCT            # +5%
                elif signal == "SELL":
                    result_item["split_sell"] = False
                    result_item["first_sell_ratio"] = 0.80
                    result_item["remaining_ratio"] = 0.30            # 잔여분 30%씩 5분 간격

                results.append(result_item)

        return results

    def check_entry(self, market: MarketData) -> bool:
        return False  # 단일 종목 진입 판단은 evaluate()에서 처리

    def check_exit(self, market: MarketData, position: Position) -> bool:
        return False  # evaluate()에서 SELL 시그널로 처리

    def generate_signal(self, market: MarketData,
                        position: Position | None = None) -> Signal:
        return Signal(Action.SKIP, "", 0, 0, "use evaluate() for twin pair signals")
