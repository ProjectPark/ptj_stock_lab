"""
JUN 매매법 v2.1 — 백테스트 엔진
=================================
독립 오케스트레이터: 일봉 필터 → 분봉 진입 탐색 구조.
BacktestBase 미사용 (Line A/B 전용 구조와 호환되지 않음).

데이터 소스:
  - 1분봉: data/market/ohlcv/backtest_1min_3y.parquet
  - BTC 일봉: data/market/daily/crypto_daily.parquet
  - VIX 일봉: data/market/daily/extra_signals_daily.parquet
  - 종목별 일봉: 1분봉에서 일별 집계
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .entry import check_daily_entry, find_intraday_entry
from .exit import check_exit
from .indicators import btc_regime, compute_daily_indicators, compute_rsi
from .params import JunTradeParams
from .portfolio import Portfolio
from .universe import VALID_TICKERS

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class JunTradeEngine:
    """JUN 매매법 v2.1 백테스트 엔진."""

    def __init__(
        self,
        params: JunTradeParams | None = None,
        start_date: date = date(2023, 10, 1),
        end_date: date = date(2026, 2, 17),
        fx_usd_krw: float = 1350.0,
    ):
        self.params = params or JunTradeParams()
        self.start_date = start_date
        self.end_date = end_date
        self.fx_usd_krw = fx_usd_krw

        # 데이터 (run() 에서 로드)
        self._min1: pd.DataFrame | None = None       # 전체 1min 데이터
        self._min1_index: dict | None = None          # {(date_str, symbol): DataFrame}
        self._daily: dict[str, pd.DataFrame] = {}     # ticker → 지표 포함 일봉
        self._btc_daily: pd.DataFrame | None = None   # BTC 일봉 + 지표
        self._vix_daily: dict[str, float] = {}        # date_str → VIX close

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    def _load_data(self) -> None:
        """모든 데이터 소스를 로드하고 전처리한다."""
        print("[engine] 데이터 로드 중...")

        # 1) 1분봉
        path_1min = DATA_DIR / "market" / "ohlcv" / "backtest_1min_3y.parquet"
        self._min1 = pd.read_parquet(path_1min)
        # 날짜 문자열 컬럼 확인/생성
        if "date" not in self._min1.columns:
            self._min1["date"] = self._min1["timestamp"].dt.strftime("%Y-%m-%d")
        # preindex: (date_str, symbol) → 해당 봉 슬라이스
        self._min1_index = {}
        for (d, s), grp in self._min1.groupby(["date", "symbol"]):
            self._min1_index[(d, s)] = grp.sort_values("timestamp").reset_index(drop=True)
        print(f"  1min: {len(self._min1):,} bars, {len(self._min1_index):,} day-symbol groups")

        # 2) 종목별 일봉 — 1분봉에서 일별 집계
        print("[engine] 일봉 집계 중...")
        for ticker in VALID_TICKERS:
            ticker_bars = self._min1[self._min1["symbol"] == ticker]
            if ticker_bars.empty:
                continue
            daily_agg = (
                ticker_bars.groupby("date")
                .agg(
                    open=("open", "first"),
                    high=("high", "max"),
                    low=("low", "min"),
                    close=("close", "last"),
                    volume=("volume", "sum"),
                )
            )
            daily_agg.index = pd.to_datetime(daily_agg.index)
            daily_agg = daily_agg.sort_index()
            self._daily[ticker] = compute_daily_indicators(daily_agg, self.params)
        print(f"  일봉: {len(self._daily)} tickers")

        # 3) BTC 일봉
        path_btc = DATA_DIR / "market" / "daily" / "crypto_daily.parquet"
        btc_raw = pd.read_parquet(path_btc)
        btc_raw = btc_raw[btc_raw["symbol"] == "BTC"].copy()
        btc_raw["date_dt"] = pd.to_datetime(btc_raw["date"])
        btc_raw = btc_raw.set_index("date_dt").sort_index()
        btc_raw = btc_raw.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        self._btc_daily = compute_daily_indicators(btc_raw, self.params)
        # BTC 10d return + regime (pre-compute)
        c = self._btc_daily["close"]
        self._btc_daily["return_10d_btc"] = (c / c.shift(10) - 1) * 100
        self._btc_daily["btc_regime"] = self._btc_daily.apply(
            lambda r: btc_regime(
                r["close"], r["ma20"], r["ma60"], r["rsi"], r.get("return_10d_btc", 0)
            )
            if not (math.isnan(r["rsi"]) if isinstance(r["rsi"], float) else False)
            else 0,
            axis=1,
        )
        print(f"  BTC 일봉: {len(self._btc_daily)} rows")

        # 4) VIX 일봉
        path_vix = DATA_DIR / "market" / "daily" / "extra_signals_daily.parquet"
        vix_raw = pd.read_parquet(path_vix)
        vix_raw = vix_raw[vix_raw["ticker"] == "VIX"].copy()
        for _, row in vix_raw.iterrows():
            d_str = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
            self._vix_daily[d_str] = float(row["Close"])
        print(f"  VIX 일봉: {len(self._vix_daily)} days")

    # ── 헬퍼: 데이터 조회 ─────────────────────────────────────────────────────
    def _get_1min_bars(self, ticker: str, trading_day: date) -> pd.DataFrame:
        """해당 날짜의 1min 봉 반환."""
        key = (trading_day.strftime("%Y-%m-%d"), ticker)
        return self._min1_index.get(key, pd.DataFrame())

    def _get_daily(self, ticker: str, trading_day: date) -> pd.Series | None:
        """해당 날짜의 일봉 지표 반환."""
        df = self._daily.get(ticker)
        if df is None:
            return None
        ts = pd.Timestamp(trading_day)
        if ts in df.index:
            return df.loc[ts]
        return None

    def _get_btc(self, trading_day: date) -> dict:
        """BTC 지표 딕셔너리 반환."""
        ts = pd.Timestamp(trading_day)
        if self._btc_daily is not None and ts in self._btc_daily.index:
            row = self._btc_daily.loc[ts]
            return {
                "btc_rsi": _safe(row.get("rsi")),
                "btc_10d_ret": _safe(row.get("return_10d_btc")),
                "btc_reg": int(row.get("btc_regime", 0)),
            }
        return {"btc_rsi": 50.0, "btc_10d_ret": 0.0, "btc_reg": 1}

    def _get_vix(self, trading_day: date) -> float:
        """VIX 종가 반환. 데이터 없으면 최근 값 사용."""
        d_str = trading_day.strftime("%Y-%m-%d")
        if d_str in self._vix_daily:
            return self._vix_daily[d_str]
        # 가장 가까운 이전 거래일 VIX
        for delta in range(1, 10):
            prev = (pd.Timestamp(trading_day) - pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
            if prev in self._vix_daily:
                return self._vix_daily[prev]
        return 20.0  # 기본값

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    def run(self, verbose: bool = True) -> dict:
        """백테스트 실행. 결과 딕셔너리 반환."""
        self._load_data()

        portfolio = Portfolio(self.params, fx_usd_krw=self.fx_usd_krw)

        # 거래일 목록: 1분봉 데이터에 존재하는 날짜 중 기간 내
        all_dates = sorted({
            pd.Timestamp(d).date()
            for d in self._min1["date"].unique()
        })
        bt_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]

        if verbose:
            print(f"\n[engine] 백테스트 시작: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)} 거래일)")
            print(f"  유니버스: {sorted(VALID_TICKERS)}")
            print()

        for day_idx, trading_day in enumerate(bt_dates):
            vix_close = self._get_vix(trading_day)
            btc_data = self._get_btc(trading_day)

            # 당일 종가 맵 (스냅샷용)
            close_prices: dict[str, float] = {}
            for ticker in VALID_TICKERS:
                d = self._get_daily(ticker, trading_day)
                if d is not None:
                    close_prices[ticker] = float(d["close"])

            # ── 기존 포지션 청산 체크 ──
            tickers_to_close: list[tuple[str, float, str]] = []
            for ticker, pos in list(portfolio.positions.items()):
                d = self._get_daily(ticker, trading_day)
                if d is None:
                    continue
                close_px = float(d["close"])
                ma20 = float(d["ma20"]) if not _isnan(d.get("ma20")) else 0.0
                holding = pos.holding_days(trading_day)
                exit_result = check_exit(pos.avg_price_usd, close_px, ma20, holding, self.params)
                if exit_result.should_exit:
                    tickers_to_close.append((ticker, close_px, exit_result.reason))

            for ticker, close_px, reason in tickers_to_close:
                trade = portfolio.close_position(ticker, close_px, trading_day, reason)
                if verbose and trade:
                    print(
                        f"  [{trading_day}] CLOSE {ticker}: "
                        f"{trade.exit_reason} pnl={trade.pnl_pct:+.1f}% "
                        f"(${trade.pnl_usd:+.2f})"
                    )

            # ── 기존 포지션 피라미딩 체크 ──
            for ticker, pos in list(portfolio.positions.items()):
                d = self._get_daily(ticker, trading_day)
                if d is None:
                    continue
                close_px = float(d["close"])
                rsi = float(d["rsi"]) if not _isnan(d.get("rsi")) else 50.0

                if pos.can_pyramid(close_px, rsi, self.params):
                    day_bars = self._get_1min_bars(ticker, trading_day)
                    if day_bars.empty:
                        continue
                    intraday = find_intraday_entry(ticker, day_bars, self.params, is_pyramid=True)
                    if intraday.allowed:
                        ok = portfolio.add_to_position(ticker, intraday.entry_price, vix_close, intraday.entry_time)
                        if verbose and ok:
                            print(
                                f"  [{trading_day}] PYRAMID {ticker}: "
                                f"${intraday.entry_price:.2f} add#{pos.add_count} "
                                f"@{intraday.entry_time}"
                            )

            # ── 신규 진입 탐색 ──
            for ticker in sorted(VALID_TICKERS - set(portfolio.positions.keys())):
                d = self._get_daily(ticker, trading_day)
                if d is None:
                    continue

                close_px = float(d["close"])
                rsi = float(d["rsi"]) if not _isnan(d.get("rsi")) else 0.0
                pct_ma20 = float(d["pct_from_ma20"]) if not _isnan(d.get("pct_from_ma20")) else -99.0
                dd20 = float(d["drawdown_20d"]) if not _isnan(d.get("drawdown_20d")) else -99.0
                ret5 = float(d["return_5d"]) if not _isnan(d.get("return_5d")) else 0.0

                allowed, reason = check_daily_entry(
                    ticker, close_px, rsi, pct_ma20, dd20, ret5,
                    btc_data["btc_rsi"], btc_data["btc_10d_ret"], btc_data["btc_reg"],
                    vix_close, self.params,
                )
                if not allowed:
                    continue

                day_bars = self._get_1min_bars(ticker, trading_day)
                if day_bars.empty:
                    continue
                intraday = find_intraday_entry(ticker, day_bars, self.params, is_pyramid=False)
                if intraday.allowed:
                    ok = portfolio.open_position(ticker, intraday.entry_price, vix_close, trading_day, intraday.entry_time)
                    if verbose and ok:
                        print(
                            f"  [{trading_day}] OPEN {ticker}: "
                            f"${intraday.entry_price:.2f} "
                            f"@{intraday.entry_time}"
                        )

            # ── 일별 스냅샷 ──
            portfolio.snapshot(trading_day, close_prices)

        # ── 미청산 포지션 강제 정리 (기간 종료) ──
        if portfolio.positions:
            last_day = bt_dates[-1]
            for ticker in list(portfolio.positions.keys()):
                d = self._get_daily(ticker, last_day)
                px = float(d["close"]) if d is not None else portfolio.positions[ticker].avg_price_usd
                trade = portfolio.close_position(ticker, px, last_day, "END_OF_PERIOD")
                if verbose and trade:
                    print(
                        f"  [{last_day}] FORCE_CLOSE {ticker}: "
                        f"pnl={trade.pnl_pct:+.1f}% (${trade.pnl_usd:+.2f})"
                    )

        summary = portfolio.summary()
        if verbose:
            print(f"\n{'='*60}")
            print(f"[engine] 백테스트 완료")
            print(f"  거래 수: {summary.get('trades', 0)}")
            print(f"  승률: {summary.get('win_rate', 0):.1f}%")
            print(f"  평균 손익: {summary.get('avg_pnl_pct', 0):.2f}%")
            print(f"  총 손익(USD): ${summary.get('total_pnl_usd', 0):.2f}")
            print(f"  청산 사유: {summary.get('exit_reasons', {})}")
            print(f"{'='*60}")

        return {
            "summary": summary,
            "closed_trades": portfolio.closed_trades,
            "daily_snapshots": portfolio.daily_snapshots,
            "equity_curve": [
                (snap["date"], snap["total_value_usd"])
                for snap in portfolio.daily_snapshots
            ],
        }


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def _safe(val, default: float = 0.0) -> float:
    """NaN-safe float 변환."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _isnan(val) -> bool:
    """NaN 체크 (None 포함)."""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return True
