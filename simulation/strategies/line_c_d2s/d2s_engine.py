"""
D2S 엔진 — 실거래 행동 추출 기반 일봉 전략
============================================
trading_rules_attach_v1.md의 16개 규칙(R1~R16)을 구현.
953건 실거래 D2S(Discretionary-to-Systematic) 분석에서 추출.

2026-02-24: Study A/B/C 반영
  - Study A: R14 그라데이션 (4단계: panic/basic/optimal/super)
  - Study B: 종목별 역발상 차별화 (MSTU/ROBN/CONL/AMDL)
  - Study C: market_score 0차 게이트 (6개 시황 신호 통합)

일봉(daily) 단위로 작동하며, 기술적 지표(RSI, MACD, BB, ATR)를
직접 계산한다 (외부 TA 라이브러리 의존 없음).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .params_d2s import D2S_ENGINE

# ============================================================
# 기술적 지표 계산 (외부 의존 없음)
# ============================================================


def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """RSI(Relative Strength Index)를 계산한다."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal, Histogram을 반환한다."""
    ema_fast = closes.ewm(span=fast, min_periods=fast).mean()
    ema_slow = closes.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(
    closes: pd.Series, period: int = 20, std: int = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저밴드 상단/하단/%B를 반환한다."""
    sma = closes.rolling(period).mean()
    rolling_std = closes.rolling(period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    pct_b = (closes - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, pct_b


def calc_atr(
    highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14,
) -> pd.Series:
    """ATR(Average True Range)를 계산한다."""
    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def calc_relative_volume(volumes: pd.Series, period: int = 20) -> pd.Series:
    """상대 거래량 (현재 / N일 평균)."""
    avg = volumes.rolling(period).mean()
    return volumes / avg.replace(0, np.nan)


# ============================================================
# 데이터 컨테이너
# ============================================================


@dataclass
class DailySnapshot:
    """하루치 시장 데이터 스냅샷."""
    trading_date: date
    # OHLCV (ticker → float)
    opens: dict[str, float] = field(default_factory=dict)
    highs: dict[str, float] = field(default_factory=dict)
    lows: dict[str, float] = field(default_factory=dict)
    closes: dict[str, float] = field(default_factory=dict)
    volumes: dict[str, float] = field(default_factory=dict)
    # 등락률
    changes: dict[str, float] = field(default_factory=dict)
    # 기술적 지표 (ticker → float)
    rsi: dict[str, float] = field(default_factory=dict)
    macd_bullish: dict[str, bool] = field(default_factory=dict)
    bb_pct_b: dict[str, float] = field(default_factory=dict)
    atr: dict[str, float] = field(default_factory=dict)
    atr_quantile: dict[str, float] = field(default_factory=dict)
    rel_volume: dict[str, float] = field(default_factory=dict)
    # 시황
    poly_btc_up: float | None = None
    confidence_signal: bool = False
    spy_streak: int = 0       # SPY 연속 상승일 수
    riskoff_streak: int = 0   # Study A: 연속 리스크오프(GLD↑+SPY↓) 일수
    weekday: int = 0           # 0=Mon, 4=Fri


@dataclass
class D2SPosition:
    """D2S 엔진 포지션."""
    ticker: str
    entry_price: float
    qty: float
    entry_date: date
    dca_count: int = 1
    cost_basis: float = 0.0  # 총 투자 금액


# ============================================================
# D2S 시그널 엔진
# ============================================================


class D2SEngine:
    """D2S 엔진 — 실거래 행동 추출 기반 일봉 전략.

    Parameters
    ----------
    params : dict | None
        D2S_ENGINE 파라미터. None이면 기본값.
    """

    def __init__(self, params: dict | None = None):
        self.p = params or D2S_ENGINE
        self._tickers = self.p["tickers"]
        self._pairs = self.p["twin_pairs"]
        self._weights = self.p["ticker_weights"]

    # ------------------------------------------------------------------
    # 0차 게이트: market_score 계산 (Study C, §1-8)
    # ------------------------------------------------------------------

    def compute_market_score(self, snap: DailySnapshot) -> float:
        """6개 시황 신호를 0~1 통합 점수로 합산한다.

        Returns
        -------
        float
            0.0 (최악) ~ 1.0 (최적)
        """
        gld_pct = snap.changes.get("GLD", 0.0)
        spy_pct = snap.changes.get("SPY", 0.0)

        # gld_score: GLD 등락률 기반 (R1). -2%~+2% 범위 정규화
        gld_raw = float(np.clip(gld_pct / 2.0, -1.0, 1.0))
        gld_score = (gld_raw + 1.0) / 2.0  # 0~1

        # spy_score: SPY 등락률 기반 (R6). -2%~+2% 범위 정규화
        spy_raw = float(np.clip(spy_pct / 2.0, -1.0, 1.0))
        spy_score = (spy_raw + 1.0) / 2.0  # 0~1

        # riskoff_score: GLD↑+SPY↓ 발동 여부 (R14 기반)
        riskoff = gld_pct > 0 and spy_pct < 0
        spy_min_th = self.p.get("riskoff_spy_min_threshold", -1.5)
        riskoff_valid = riskoff and spy_pct >= spy_min_th  # 패닉 구간 제외
        riskoff_score = 0.0
        if riskoff_valid:
            riskoff_score = 0.8  # 기본 발동
            # 최적 구간 추가 가점
            if (gld_pct >= self.p.get("riskoff_gld_optimal_min", 0.5)
                    and spy_pct <= self.p.get("riskoff_spy_optimal_max", -0.5)):
                riskoff_score = 1.0

        # streak_score: SPY 연속 상승이 낮을수록 좋음 (R13 기반)
        streak_max = self.p.get("spy_streak_max", 3)
        streak_score = max(0.0, 1.0 - snap.spy_streak / max(streak_max, 1))

        # vol_score: 평균 상대 거래량 (R16 기반, 유일 유의 p=0.041)
        vols = [v for v in snap.rel_volume.values() if v > 0 and not np.isnan(v)]
        if vols:
            avg_vol = float(np.mean(vols))
            # 1.2~2.0x 구간 최적 → 0~1 정규화
            vol_score = float(np.clip((avg_vol - 0.5) / 2.0, 0.0, 1.0))
        else:
            vol_score = 0.5

        # btc_score: BTC 확률 (R3 대용, BITU 등락률 기반)
        btc_up = snap.poly_btc_up
        btc_min = self.p.get("btc_up_min", 0.40)
        btc_max = self.p.get("btc_up_max", 0.75)
        if btc_up is not None:
            if btc_min <= btc_up <= btc_max:
                btc_score = 1.0  # 적정 구간
            elif btc_up < btc_min:
                btc_score = 0.2  # 너무 낮음
            else:
                btc_score = 0.2  # 너무 높음 (억제 구간)
        else:
            btc_score = 0.5  # 데이터 없으면 중립

        # 가중 합산
        w = self.p.get("market_score_weights", {
            "gld_score": 0.20, "spy_score": 0.15, "riskoff_score": 0.25,
            "streak_score": 0.15, "vol_score": 0.15, "btc_score": 0.10,
        })
        score = (
            w.get("gld_score", 0.20) * gld_score
            + w.get("spy_score", 0.15) * spy_score
            + w.get("riskoff_score", 0.25) * riskoff_score
            + w.get("streak_score", 0.15) * streak_score
            + w.get("vol_score", 0.15) * vol_score
            + w.get("btc_score", 0.10) * btc_score
        )

        # riskoff_streak 연속 특급 보너스 (Study A §1-5)
        if snap.riskoff_streak >= self.p.get("riskoff_consecutive_boost", 3):
            score = min(score + 0.10, 1.0)

        return round(float(score), 4)

    # ------------------------------------------------------------------
    # 1차 게이트: 시황 필터 (R1, R3, R13, R14) — Study A 반영
    # ------------------------------------------------------------------

    def check_market_filter(self, snap: DailySnapshot) -> dict[str, Any]:
        """시황 필터를 평가하고 결과를 반환한다.

        R14(GLD↑+SPY↓)가 발동하면 R1/R3/R13을 오버라이드한다 (최우선).
        SPY < -1.5% 패닉 구간에서는 R14 미발동이나 비중 50% 축소.

        Returns
        -------
        dict
            blocked, reason, riskoff_boost, riskoff_optimal, riskoff_super,
            riskoff_panic, spy_bearish, size_factor
        """
        gld_pct = snap.changes.get("GLD", 0.0)
        spy_pct = snap.changes.get("SPY", 0.0)
        spy_min_th = self.p.get("riskoff_spy_min_threshold", -1.5)

        _base = {
            "blocked": False, "reason": "",
            "riskoff_boost": False, "riskoff_optimal": False,
            "riskoff_super": False, "riskoff_panic": False,
            "spy_bearish": False, "size_factor": 1.0,
        }

        # ── R14 최우선 판단 ─────────────────────────────────────
        riskoff_gld_up_spy_down = (
            self.p.get("riskoff_gld_up_spy_down", True)
            and gld_pct > 0 and spy_pct < 0
        )

        if riskoff_gld_up_spy_down:
            # 패닉 구간: SPY < -1.5% → R14 미발동, 비중 50% 축소
            if spy_pct < spy_min_th:
                return {**_base,
                        "riskoff_panic": True,
                        "size_factor": self.p.get("riskoff_panic_size_factor", 0.5)}

            # R14 발동 → R1/R3/R13 오버라이드
            riskoff_optimal = (
                gld_pct >= self.p.get("riskoff_gld_optimal_min", 0.5)
                and spy_pct <= self.p.get("riskoff_spy_optimal_max", -0.5)
            )
            riskoff_super = (
                snap.riskoff_streak >= self.p.get("riskoff_consecutive_boost", 3)
            )
            return {**_base,
                    "riskoff_boost": True,
                    "riskoff_optimal": riskoff_optimal,
                    "riskoff_super": riskoff_super}

        # ── R14 미발동 시 → R1/R3/R13 순서대로 체크 ────────────

        # R1: GLD 매수 억제
        if gld_pct >= self.p["gld_suppress_threshold"]:
            return {**_base, "blocked": True,
                    "reason": f"R1: GLD {gld_pct:+.2f}% >= {self.p['gld_suppress_threshold']}%"}

        # R3: BTC 확률 억제 (상한)
        if snap.poly_btc_up is not None and snap.poly_btc_up > self.p["btc_up_max"]:
            return {**_base, "blocked": True,
                    "reason": f"R3: BTC_up {snap.poly_btc_up:.2f} > {self.p['btc_up_max']}"}

        # R3: BTC 확률 억제 (하한)
        if snap.poly_btc_up is not None and snap.poly_btc_up < self.p.get("btc_up_min", 0.40):
            return {**_base, "blocked": True,
                    "reason": f"R3: BTC_up {snap.poly_btc_up:.2f} < {self.p.get('btc_up_min', 0.40)}"}

        # confidence_signal 억제
        if self.p["confidence_suppress"] and snap.confidence_signal:
            return {**_base, "blocked": True, "reason": "R: confidence_signal active"}

        # R13: SPY 연속 상승 매수 금지
        if snap.spy_streak >= self.p["spy_streak_max"]:
            return {**_base, "blocked": True,
                    "reason": f"R13: SPY streak {snap.spy_streak} >= {self.p['spy_streak_max']}"}

        # R6: SPY 하락장 역발상
        spy_bearish = spy_pct <= self.p["spy_bearish_threshold"]
        return {**_base, "spy_bearish": spy_bearish}

    # ------------------------------------------------------------------
    # 2차 게이트: 쌍둥이 갭 + OOS 규칙 (R2)
    # ------------------------------------------------------------------

    def check_twin_gaps(self, snap: DailySnapshot) -> list[dict]:
        """쌍둥이 페어 갭을 계산하고 진입 후보를 반환한다."""
        results = []
        for pair_name, pair_cfg in self._pairs.items():
            lead = pair_cfg["lead"]
            follow = pair_cfg["follow"]
            lead_pct = snap.changes.get(lead, 0.0)
            follow_pct = snap.changes.get(follow, 0.0)
            gap = lead_pct - follow_pct

            # OOS Decision Tree 필터
            if pair_name == "bank_CONL":
                if gap > self.p["gap_bank_conl_max"]:
                    continue
                robn_pct = snap.changes.get("ROBN", 0.0)
                if robn_pct > self.p["robn_pct_max"]:
                    continue

            if gap > 0:
                results.append({
                    "pair": pair_name,
                    "follow": follow,
                    "gap": gap,
                    "lead_pct": lead_pct,
                    "follow_pct": follow_pct,
                })
        return results

    # ------------------------------------------------------------------
    # 3차 게이트: 기술적 지표 필터 (R7~R9, R16)
    # ------------------------------------------------------------------

    def check_technical_filter(
        self, ticker: str, snap: DailySnapshot,
    ) -> dict[str, Any]:
        """종목의 기술적 지표를 평가한다.

        Returns
        -------
        dict
            blocked: bool, reason: str, combo_optimal: bool, atr_boost: bool
        """
        rsi = snap.rsi.get(ticker)
        bb = snap.bb_pct_b.get(ticker)
        macd_bull = snap.macd_bullish.get(ticker, False)
        atr_q = snap.atr_quantile.get(ticker, 0.5)
        rel_vol = snap.rel_volume.get(ticker, 1.0)

        # R7: RSI 진입 금지
        if rsi is not None and rsi > self.p["rsi_danger_zone"]:
            return {"blocked": True, "reason": f"R7: RSI {rsi:.1f} > {self.p['rsi_danger_zone']}",
                    "combo_optimal": False, "atr_boost": False}

        # R8: 볼린저 진입 금지
        if bb is not None and bb > self.p["bb_danger_zone"]:
            return {"blocked": True, "reason": f"R8: BB%B {bb:.2f} > {self.p['bb_danger_zone']}",
                    "combo_optimal": False, "atr_boost": False}

        # R9: MACD 콤보 최적 (MACD bullish + RSI 40~60 + Vol 1.2~2.0x)
        rsi_in_range = (rsi is not None
                        and self.p["rsi_entry_min"] <= rsi <= self.p["rsi_entry_max"])
        vol_normal = (self.p["vol_entry_min"] <= rel_vol <= self.p["vol_entry_max"])
        combo_optimal = macd_bull and rsi_in_range and vol_normal

        # R16: ATR Q4 진입 우대
        atr_boost = atr_q >= self.p["atr_high_quantile"]

        return {"blocked": False, "reason": "",
                "combo_optimal": combo_optimal, "atr_boost": atr_boost}

    # ------------------------------------------------------------------
    # 4차 게이트: 캘린더 + 역발상 + 종목별 차별화 (Study B, R15, R6)
    # ------------------------------------------------------------------

    def check_entry_quality(
        self, ticker: str, snap: DailySnapshot,
        market_ctx: dict, tech_ctx: dict,
    ) -> dict[str, Any]:
        """진입 품질 점수를 계산한다.

        Study B: 종목별 역발상 × R14 교차 차별화 반영.
          - MSTU: R14 발동 시 역발상만 허용 (순발상 절대 금지)
          - ROBN: R14 발동 + 순발상 → 우대 (68.8%)
          - CONL: R14 없이 역발상 금지
          - AMDL: 금요일 + 역발상 -1.5% 이하 특화

        Returns
        -------
        dict
            score: float (0~1), size_hint: str ("large"|"small"|"skip"),
            reasons: list[str]
        """
        score = 0.5  # 기저
        reasons: list[str] = []
        ticker_pct = snap.changes.get(ticker, 0.0)
        riskoff = market_ctx.get("riskoff_boost", False)
        is_friday = snap.weekday == 4
        is_contrarian = ticker_pct < self.p["contrarian_entry_threshold"]  # 기본: 0.0

        # ── Study B: 종목별 역발상 × R14 차별화 ─────────────────

        # MSTU: R14 발동 시 역발상만 허용 (순발상 = 절대 금지, 25% 승률)
        if ticker == "MSTU" and riskoff and self.p.get("mstu_riskoff_contrarian_only", True):
            if not is_contrarian:
                return {"score": 0.0, "size_hint": "skip",
                        "reasons": ["MSTU:riskoff+momentum=BLOCKED(25%)"]}
            score += 0.10
            reasons.append("MSTU:riskoff+contrarian")

        # ROBN: R14 발동 + 순발상(모멘텀) → 우대 (68.8%)
        elif ticker == "ROBN" and riskoff and not is_contrarian:
            if self.p.get("robn_riskoff_momentum_boost", True):
                score += 0.20
                reasons.append("ROBN:riskoff+momentum_boost(68.8%)")

        # CONL: R14 없이 역발상 효과 없음 (46.7%) → R14 없으면 역발상 진입 금지
        elif ticker == "CONL":
            if self.p.get("conl_contrarian_require_riskoff", True):
                if not riskoff and is_contrarian:
                    return {"score": 0.0, "size_hint": "skip",
                            "reasons": ["CONL:contrarian_without_riskoff=BLOCKED(46.7%)"]}

        # AMDL: 금요일 + 역발상 -1.5% 이하 특화 (66.7%)
        if ticker == "AMDL" and is_friday:
            amdl_th = self.p.get("amdl_friday_contrarian_threshold", -1.5)
            if ticker_pct <= amdl_th:
                score += 0.15
                reasons.append(f"AMDL:friday+contrarian({ticker_pct:+.1f}%<={amdl_th}%)")

        # ── R14: 리스크오프 강도별 부스트 (Study A) ──────────────
        if market_ctx.get("riskoff_super"):
            score += 0.30
            reasons.append("R14:riskoff_super(100%)")
        elif market_ctx.get("riskoff_optimal"):
            score += 0.20
            reasons.append("R14:riskoff_optimal(72.7%)")
        elif riskoff:
            score += 0.10
            reasons.append("R14:riskoff_basic")

        # R6: SPY 하락 역발상
        if market_ctx.get("spy_bearish"):
            score += 0.10
            reasons.append("R6:spy_bearish")

        # R9: 콤보 최적
        if tech_ctx.get("combo_optimal"):
            score += 0.15
            reasons.append("R9:combo_optimal")

        # R16: ATR 고변동
        if tech_ctx.get("atr_boost"):
            score += 0.05
            reasons.append("R16:atr_boost")

        # R15: 금요일 부스트
        if self.p["friday_boost"] and is_friday:
            score += 0.05
            reasons.append("R15:friday")

        # 역발상 진입 (종목 하락 시 우대)
        if is_contrarian:
            score += 0.10
            reasons.append(f"contrarian:{ticker_pct:+.1f}%")

        # BB 하단 우대
        bb = snap.bb_pct_b.get(ticker)
        if bb is not None and bb <= self.p["bb_entry_max"]:
            score += 0.05
            reasons.append(f"R8:bb_low({bb:.2f})")

        score = min(score, 1.0)
        size_hint = "large" if score >= 0.7 else "small"

        return {"score": score, "size_hint": size_hint, "reasons": reasons}

    # ------------------------------------------------------------------
    # 청산 판단 (R4, R5)
    # ------------------------------------------------------------------

    def check_exit(
        self,
        position: D2SPosition,
        snap: DailySnapshot,
    ) -> dict[str, Any]:
        """청산 조건을 확인한다.

        Returns
        -------
        dict
            should_exit: bool, reason: str
        """
        current_price = snap.closes.get(position.ticker, 0)
        if current_price <= 0 or position.entry_price <= 0:
            return {"should_exit": False, "reason": ""}

        pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        hold_days = (snap.trading_date - position.entry_date).days

        # R4: 이익실현
        if pnl_pct >= self.p["take_profit_pct"]:
            return {"should_exit": True,
                    "reason": f"R4:take_profit pnl={pnl_pct:+.1f}% >= {self.p['take_profit_pct']}%"}

        # 보유 기간 초과
        if hold_days > self.p["optimal_hold_days_max"]:
            return {"should_exit": True,
                    "reason": f"hold_days={hold_days} > {self.p['optimal_hold_days_max']}"}

        return {"should_exit": False, "reason": ""}

    # ------------------------------------------------------------------
    # 종합 일별 시그널 생성
    # ------------------------------------------------------------------

    def generate_daily_signals(
        self,
        snap: DailySnapshot,
        positions: dict[str, D2SPosition],
        daily_buy_counts: dict[str, int],
    ) -> list[dict]:
        """하루치 시그널을 생성한다.

        Parameters
        ----------
        snap : DailySnapshot
        positions : dict[str, D2SPosition]
            현재 보유 포지션.
        daily_buy_counts : dict[str, int]
            당일 종목별 매수 횟수.

        Returns
        -------
        list[dict]
            각 시그널: action, ticker, size, reason, score, market_score
        """
        signals = []

        # === 청산 스캔 ===
        for ticker, pos in list(positions.items()):
            exit_ctx = self.check_exit(pos, snap)
            if exit_ctx["should_exit"]:
                signals.append({
                    "action": "SELL",
                    "ticker": ticker,
                    "size": 1.0,
                    "reason": exit_ctx["reason"],
                    "score": 0,
                    "market_score": 0,
                })

        # === 0차 게이트: market_score (Study C, §13-0) ===
        market_score = self.compute_market_score(snap)
        suppress_th = self.p.get("market_score_suppress", 0.40)
        if market_score < suppress_th:
            return signals  # 당일 진입 전면 억제

        # A/B 등급: 진입 비중 조정에 활용
        entry_a_th = self.p.get("market_score_entry_a", 0.60)
        entry_b_th = self.p.get("market_score_entry_b", 0.55)
        score_grade = "A" if market_score >= entry_a_th else "B"

        # === 1차 게이트: 시황 필터 (R1, R3, R13, R14) ===
        market_ctx = self.check_market_filter(snap)
        market_ctx["market_score"] = market_score
        market_ctx["score_grade"] = score_grade

        if market_ctx["blocked"]:
            return signals  # 매도만 반환

        # 패닉 구간 비중 축소 계수
        size_factor = market_ctx.get("size_factor", 1.0)
        # B등급이면 추가 비중 축소 없음 (A등급과 동일 취급)

        # === 쌍둥이 갭 진입 후보 ===
        gap_candidates = self.check_twin_gaps(snap)

        # === 각 후보 종목에 대해 기술적/품질 필터 ===
        buy_candidates = []
        seen_tickers: set[str] = set()

        # 갭 기반 후보
        for gc in gap_candidates:
            ticker = gc["follow"]
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            # DCA 제한 (R5)
            if daily_buy_counts.get(ticker, 0) >= self.p["dca_max_daily"]:
                continue

            tech_ctx = self.check_technical_filter(ticker, snap)
            if tech_ctx["blocked"]:
                continue

            quality = self.check_entry_quality(ticker, snap, market_ctx, tech_ctx)
            if quality["size_hint"] == "skip":
                continue

            buy_candidates.append({
                "ticker": ticker,
                "source": f"gap:{gc['pair']}({gc['gap']:+.2f}%)",
                "score": quality["score"],
                "size_hint": quality["size_hint"],
                "reasons": quality["reasons"],
            })

        # 단독 역발상 진입 — 종목 하락 + 기술적 유리 + 미스커버 종목
        for ticker in self._tickers:
            if ticker in seen_tickers:
                continue
            if daily_buy_counts.get(ticker, 0) >= self.p["dca_max_daily"]:
                continue
            ticker_pct = snap.changes.get(ticker, 0.0)
            # 역발상: 종목이 하락 중이어야 단독 진입
            if ticker_pct >= self.p["contrarian_entry_threshold"]:
                continue
            tech_ctx = self.check_technical_filter(ticker, snap)
            if tech_ctx["blocked"]:
                continue
            quality = self.check_entry_quality(ticker, snap, market_ctx, tech_ctx)
            if quality["size_hint"] == "skip":
                continue
            # 단독 진입은 콤보 최적 또는 리스크오프일 때만
            if quality["score"] >= 0.65:
                seen_tickers.add(ticker)
                buy_candidates.append({
                    "ticker": ticker,
                    "source": f"contrarian({ticker_pct:+.1f}%)",
                    "score": quality["score"],
                    "size_hint": quality["size_hint"],
                    "reasons": quality["reasons"],
                })

        # 리스크오프 매수 — 갭/단독 후보가 없어도 강제 진입
        if market_ctx.get("riskoff_boost") and not buy_candidates:
            for ticker in self._tickers:
                if ticker in seen_tickers:
                    continue
                if daily_buy_counts.get(ticker, 0) >= self.p["dca_max_daily"]:
                    continue
                tech_ctx = self.check_technical_filter(ticker, snap)
                if tech_ctx["blocked"]:
                    continue
                quality = self.check_entry_quality(ticker, snap, market_ctx, tech_ctx)
                if quality["size_hint"] == "skip":
                    continue
                if quality["score"] >= 0.6:
                    buy_candidates.append({
                        "ticker": ticker,
                        "source": "riskoff_forced",
                        "score": quality["score"],
                        "size_hint": quality["size_hint"],
                        "reasons": quality["reasons"],
                    })

        # 점수 순 정렬, 상위 3개만
        buy_candidates.sort(key=lambda x: x["score"], reverse=True)
        for cand in buy_candidates[:3]:
            base_size = (self.p["buy_size_large"]
                         if cand["size_hint"] == "large"
                         else self.p["buy_size_small"])
            size = round(base_size * size_factor, 4)
            signals.append({
                "action": "BUY",
                "ticker": cand["ticker"],
                "size": size,
                "reason": f"{cand['source']} | {', '.join(cand['reasons'])}",
                "score": cand["score"],
                "market_score": market_score,
            })

        return signals


# ============================================================
# 기술적 지표 전처리기
# ============================================================


class TechnicalPreprocessor:
    """market_daily DataFrame에서 기술적 지표를 미리 계산한다."""

    def __init__(self, params: dict | None = None):
        self.p = params or D2S_ENGINE

    def compute(self, df_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """종목별 기술적 지표를 계산한다.

        Parameters
        ----------
        df_daily : pd.DataFrame
            MultiIndex columns (Ticker, Price), Date index.

        Returns
        -------
        dict[str, pd.DataFrame]
            ticker → DataFrame(date, rsi, macd_bullish, bb_pct_b, atr, atr_quantile, rel_volume, change_pct)
        """
        tickers = sorted(set(c[0] for c in df_daily.columns))
        result = {}

        for ticker in tickers:
            try:
                close = df_daily[(ticker, "Close")]
                high = df_daily[(ticker, "High")]
                low = df_daily[(ticker, "Low")]
                volume = df_daily[(ticker, "Volume")]
            except KeyError:
                continue

            if close.isna().all():
                continue

            rsi = calc_rsi(close, self.p["rsi_period"])
            macd_line, signal_line, _ = calc_macd(
                close, self.p["macd_fast"], self.p["macd_slow"], self.p["macd_signal"],
            )
            _, _, bb_pct_b = calc_bollinger(close, self.p["bb_period"], self.p["bb_std"])
            atr = calc_atr(high, low, close, self.p["atr_period"])
            rel_vol = calc_relative_volume(volume, self.p["vol_avg_period"])

            prev_close = close.shift(1)
            change_pct = ((close - prev_close) / prev_close * 100).fillna(0)

            # ATR quantile (rolling 60일 기준)
            atr_q = atr.rolling(60, min_periods=20).rank(pct=True)

            indicators = pd.DataFrame({
                "close": close,
                "open": df_daily.get((ticker, "Open"), close),
                "high": high,
                "low": low,
                "volume": volume,
                "rsi": rsi,
                "macd_bullish": macd_line > signal_line,
                "bb_pct_b": bb_pct_b,
                "atr": atr,
                "atr_quantile": atr_q,
                "rel_volume": rel_vol,
                "change_pct": change_pct,
            })
            result[ticker] = indicators

        return result
