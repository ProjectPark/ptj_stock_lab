"""
Legacy 시그널 필터 — OOP 래퍼
==============================
MarketModeFilter, GoldFilter, SidewaysDetector

v5 변경사항:
- MarketModeFilter: 강세장 기준을 btc_up → ndx_up 으로 수정 (2-4절)
- SidewaysDetector: v5 6개 기술지표 경로 추가 (SPY/QQQ OHLCV 기반, 2-2절)
  기존 5-indicator 경로(poly/GLD/gap_fail)는 legacy 백테스트 호환용으로 유지.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class MarketModeFilter:
    """Polymarket 확률 기반 시황 판정 필터.

    Legacy: signals_v2.determine_market_mode()
    v3+: sideways_active 추가
    v5: 강세장 기준 btc_up → ndx_up (2-4절)
    """
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def evaluate(self, poly_probs: dict | None, sideways_active: bool = False) -> str:
        """시황 모드를 반환한다: "sideways" | "bearish" | "bullish" | "normal"

        우선순위: sideways > bearish > bullish > normal

        v5 변경: 강세장 기준 = Polymarket 나스닥(ndx_up) >= 70% (2-4절)
        """
        if sideways_active:
            return "sideways"

        if poly_probs is None:
            return "normal"

        # v5: 강세장 = ndx_up(나스닥) >= 70% (v4까지는 btc_up)
        ndx_up = poly_probs.get("ndx_up", 0.0)
        btc_up = poly_probs.get("btc_up", 0.0)
        eth_up = poly_probs.get("eth_up", 0.0)

        if ndx_up >= 0.70:
            return "bullish"
        if btc_up <= 0.20 and ndx_up <= 0.20 and eth_up <= 0.20:
            return "bearish"

        return "normal"


class GoldFilter:
    """GLD 변동률 기반 매매 허용 필터.

    Legacy: signals_v2.check_gold_signal_v2()
    """
    def evaluate(self, gld_pct: float) -> dict:
        """GLD 변동률 기반 매매 허용 여부를 반환한다.

        Returns: dict with ticker, change_pct, warning, allow_extra_buy, message
        """
        warning = gld_pct > 0
        allow_extra_buy = gld_pct < 0

        if warning:
            message = f"GLD +{gld_pct:.2f}% — 금 양봉, 전체 매매 금지"
        elif allow_extra_buy:
            message = f"GLD {gld_pct:.2f}% — 금 음봉, 현금 추가 매수 허용"
        else:
            message = "GLD 0.00% — 변동 없음, 기존 포지션 유지"

        return {
            "ticker": "GLD",
            "change_pct": gld_pct,
            "warning": warning,
            "allow_extra_buy": allow_extra_buy,
            "message": message,
        }


class SidewaysDetector:
    """횡보장 감지 필터.

    v5 (기본 경로): SPY/QQQ OHLCV 기반 6개 기술적 지표 (2-2절).
    Legacy 경로 (indicators dict 직접 전달): 하위 호환용.

    v5 6개 지표 (SPY·QQQ 각각 평가, 하나라도 3개 이상 충족 → 횡보장):
      1. ATR 감소: 14일 ATR이 20일 평균 대비 20% 이상 감소
      2. 거래량 감소: 20일 평균 대비 30% 이상 감소
      3. 20 EMA 평평: 기울기 절대값 0.1% 이하 (5일 기준)
      4. RSI 박스권: RSI(14) 45~55 범위
      5. BB 폭 축소: BB(20,2) 폭이 60일 최저 하위 20%
      6. 고저점 차이: 당일 고-저 차이 2% 이하
    """
    def __init__(self, params: dict | None = None):
        from ..common.params import SIDEWAYS_DETECTOR
        defaults = dict(SIDEWAYS_DETECTOR)
        if params:
            defaults.update(params)
        self.params = defaults

    # ------------------------------------------------------------------
    # v5 기술지표 계산 (내부)
    # ------------------------------------------------------------------

    def _compute_v5_indicators(self, df: "pd.DataFrame") -> dict[str, bool]:
        """OHLCV DataFrame으로 6개 v5 기술지표를 계산한다.

        Parameters
        ----------
        df : pd.DataFrame
            columns: open, high, low, close, volume (oldest→newest order)

        Returns
        -------
        dict[str, bool]
            각 지표 이름 → 충족 여부
        """
        import pandas as pd

        p = self.params
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # 1. ATR 감소
        atr_period = p.get("atr_period", 14)
        atr_ma_period = p.get("atr_ma_period", 20)
        atr_drop_pct = p.get("atr_drop_pct", 0.20)

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(atr_period).mean()
        atr_ma20 = atr14.rolling(atr_ma_period).mean()
        ind_atr = bool(
            atr_ma20.iloc[-1] > 0
            and atr14.iloc[-1] <= atr_ma20.iloc[-1] * (1 - atr_drop_pct)
        )

        # 2. 거래량 감소
        vol_ma_period = p.get("volume_ma_period", 20)
        vol_drop_pct = p.get("volume_drop_pct", 0.30)
        vol_ma20 = volume.rolling(vol_ma_period).mean()
        ind_vol = bool(
            vol_ma20.iloc[-1] > 0
            and volume.iloc[-1] <= vol_ma20.iloc[-1] * (1 - vol_drop_pct)
        )

        # 3. 20 EMA 평평 (5일 기준 기울기)
        ema_period = p.get("ema_period", 20)
        ema_slope_days = p.get("ema_slope_days", 5)
        ema_slope_pct = p.get("ema_slope_pct", 0.001)  # 0.1%
        ema20 = close.ewm(span=ema_period, adjust=False).mean()
        ind_ema = False
        if len(ema20) >= ema_slope_days + 1:
            base = ema20.iloc[-(ema_slope_days + 1)]
            slope = (ema20.iloc[-1] - base) / base if base > 0 else 0.0
            ind_ema = abs(slope) <= ema_slope_pct

        # 4. RSI 박스권
        rsi_period = p.get("rsi_period", 14)
        rsi_lo = p.get("rsi_lo", 45)
        rsi_hi = p.get("rsi_hi", 55)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        with pd.option_context("mode.use_inf_as_na", True):
            rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - 100 / (1 + rs)
        rsi_val = rsi.iloc[-1]
        ind_rsi = bool(not pd.isna(rsi_val) and rsi_lo <= rsi_val <= rsi_hi)

        # 5. BB 폭 축소 (60일 하위 20%)
        bb_period = p.get("bb_period", 20)
        bb_quantile_period = p.get("bb_quantile_period", 60)
        bb_quantile_pct = p.get("bb_quantile_pct", 0.20)
        bb_mid = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std()
        bb_width = (bb_std * 4) / bb_mid.replace(0, float("nan")) * 100
        bb_q20 = bb_width.rolling(bb_quantile_period).quantile(bb_quantile_pct)
        ind_bb = bool(
            not pd.isna(bb_width.iloc[-1]) and not pd.isna(bb_q20.iloc[-1])
            and bb_width.iloc[-1] <= bb_q20.iloc[-1]
        )

        # 6. 고저점 차이 2% 이하
        hl_max_pct = p.get("hl_max_pct", 2.0)
        hl_pct = (
            (high.iloc[-1] - low.iloc[-1]) / low.iloc[-1] * 100
            if low.iloc[-1] > 0 else 0.0
        )
        ind_hl = hl_pct <= hl_max_pct

        return {
            "atr_decline": ind_atr,
            "volume_decline": ind_vol,
            "ema_flat": ind_ema,
            "rsi_range": ind_rsi,
            "bb_narrow": ind_bb,
            "hl_small": ind_hl,
        }

    # ------------------------------------------------------------------
    # 퍼블릭 API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ohlcv: "dict[str, Any] | None" = None,
        indicators: dict[str, bool] | None = None,
        # legacy params (deprecated, 하위 호환용)
        poly_probs: dict | None = None,
        changes: dict | None = None,
        gap_fail_count: int = 0,
        trigger_fail_count: int = 0,
    ) -> dict:
        """횡보장 여부를 판정한다.

        Parameters
        ----------
        ohlcv : dict | None
            {"SPY": pd.DataFrame, "QQQ": pd.DataFrame} — v5 기본 경로.
            DataFrame columns: open, high, low, close, volume
        indicators : dict[str, bool] | None
            외부에서 직접 지표를 전달하는 경로. 제공 시 ohlcv 무시.
        poly_probs, changes, gap_fail_count, trigger_fail_count
            legacy 5-indicator 경로 (deprecated).

        Returns
        -------
        dict with indicators, count, is_sideways, message
        """
        min_signals = self.params.get("min_signals", 3)

        # ── 경로 1: 외부 indicators dict 직접 전달 ────────────────────
        if indicators is not None:
            total = len(indicators)
            count = sum(1 for v in indicators.values() if v)
            is_sideways = count >= min_signals
            met = [k for k, v in indicators.items() if v]
            if is_sideways:
                message = f"횡보장 감지 ({count}/{total} 충족: {', '.join(met)}) → 현금 100%"
            else:
                message = f"횡보장 아님 ({count}/{total} 충족, 기준 {min_signals}개)"
            return {
                "indicators": indicators,
                "count": count,
                "is_sideways": is_sideways,
                "message": message,
            }

        # ── 경로 2: OHLCV 기반 v5 기술지표 (기본) ────────────────────
        if ohlcv is not None:
            tickers = self.params.get("tickers", ["SPY", "QQQ"])
            all_inds: dict[str, dict[str, bool]] = {}
            sideways_any = False

            for ticker in tickers:
                df = ohlcv.get(ticker)
                if df is None or len(df) < 60:
                    continue
                try:
                    inds = self._compute_v5_indicators(df)
                except Exception:
                    continue
                all_inds[ticker] = inds
                count = sum(1 for v in inds.values() if v)
                if count >= min_signals:
                    sideways_any = True

            # 가장 많이 충족한 종목 기준으로 요약
            if all_inds:
                best_ticker = max(
                    all_inds,
                    key=lambda t: sum(1 for v in all_inds[t].values() if v),
                )
                best_inds = all_inds[best_ticker]
                best_count = sum(1 for v in best_inds.values() if v)
                met = [k for k, v in best_inds.items() if v]
            else:
                best_inds = {}
                best_count = 0
                met = []

            if sideways_any:
                triggered = [t for t in tickers if t in all_inds
                             and sum(1 for v in all_inds[t].values() if v) >= min_signals]
                message = (
                    f"횡보장 감지 ({'/'.join(triggered)} ≥{min_signals}개 충족) → 현금 100%"
                )
            else:
                message = f"횡보장 아님 ({best_count}/{len(best_inds)} 충족, 기준 {min_signals}개)"

            return {
                "indicators": best_inds,
                "per_ticker": all_inds,
                "count": best_count,
                "is_sideways": sideways_any,
                "message": message,
            }

        # ── 경로 3: Legacy — 5개 Poly/GLD 지표 (deprecated) ─────────
        changes = changes or {}
        ind: dict[str, bool] = {}

        poly_low = self.params.get("poly_low", 0.40)
        poly_high = self.params.get("poly_high", 0.60)
        gld_threshold = self.params.get("gld_threshold", 0.3)
        gap_fail_threshold = self.params.get("gap_fail_threshold", 2)
        trigger_fail_threshold = self.params.get("trigger_fail_threshold", 2)
        index_threshold = self.params.get("index_threshold", 0.5)

        if poly_probs is not None:
            btc_up = poly_probs.get("btc_up", 0.5)
            ind["poly_range"] = poly_low <= btc_up <= poly_high
        else:
            ind["poly_range"] = False

        gld_data = changes.get("GLD", {})
        gld_pct = abs(gld_data.get("change_pct", 0.0) if isinstance(gld_data, dict) else gld_data)
        ind["gld_flat"] = gld_pct <= gld_threshold
        ind["gap_fail"] = gap_fail_count >= gap_fail_threshold
        ind["trigger_fail"] = trigger_fail_count >= trigger_fail_threshold

        spy_pct = changes.get("SPY", {})
        qqq_pct = changes.get("QQQ", {})
        spy_val = abs(spy_pct.get("change_pct", 0.0) if isinstance(spy_pct, dict) else spy_pct)
        qqq_val = abs(qqq_pct.get("change_pct", 0.0) if isinstance(qqq_pct, dict) else qqq_pct)
        ind["index_flat"] = spy_val <= index_threshold and qqq_val <= index_threshold

        count = sum(1 for v in ind.values() if v)
        is_sideways = count >= min_signals
        met = [k for k, v in ind.items() if v]

        if is_sideways:
            message = f"횡보장 감지 ({count}/5 충족: {', '.join(met)}) → 현금 100%"
        else:
            message = f"횡보장 아님 ({count}/5 충족, 기준 {min_signals}개)"

        return {
            "indicators": ind,
            "count": count,
            "is_sideways": is_sideways,
            "message": message,
        }
