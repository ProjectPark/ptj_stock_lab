#!/usr/bin/env python3
"""
PTJ 매매법 v2 - 1분봉 백테스트 시뮬레이션 (USD 기준)
=====================================================
v2 매매 규칙 전체를 적용한 1년간 시뮬레이션.
모든 금액은 USD 기준. 환전(FX) 없음.

Data:
  - data/backtest_1min_v2.parquet  (1-min bars, US/Eastern tz)
  - polymarket/history/*.json      (daily probabilities)

Dependencies:
  - config.py         : v2 parameters (TICKERS_V2, TWIN_PAIRS_V2, ...)
  - signals_v2.py     : pure signal functions
  - backtest_common.py: data loading, fee calculations, metrics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
import signals_v2
import backtest_common

# ============================================================
# Constants
# ============================================================
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
MARKET_MINUTES_PER_DAY = 390  # 9:30 ~ 16:00 = 6.5 hours
KST_TZ = "Asia/Seoul"


# ============================================================
# Data Models
# ============================================================
@dataclass
class Position:
    """개별 포지션 상태."""
    ticker: str
    entries: list  # DCA tracking: [{time, price, qty, usd}, ...]
    total_qty: float
    avg_price: float
    total_invested_usd: float
    dca_count: int  # max 7 (DCA_MAX_COUNT from config)
    initial_entry_price: float  # DCA reference price
    first_entry_time: datetime  # for time-based stop loss
    # Staged sell state
    staged_sell_active: bool = False
    staged_sell_start: datetime | None = None
    staged_sell_remaining: float = 0.0
    staged_sell_step: int = 0
    # Carry state (bullish mode next-day hold)
    carry: bool = False
    # Signal type that opened this position
    signal_type: str = ""
    # CONL-specific: track if opened via conditional CONL
    is_conl_conditional: bool = False
    # COIN-specific: track if opened via conditional COIN
    is_coin_conditional: bool = False


@dataclass
class TradeV2:
    """매매 기록."""
    date: date
    ticker: str
    side: str  # BUY / SELL
    price: float
    qty: float
    amount_usd: float
    pnl_usd: float
    pnl_pct: float
    signal_type: str
    exit_reason: str = ""
    dca_level: int = 0  # which DCA level (0=initial, 1-7=DCA)
    fees_usd: float = 0.0
    net_pnl_usd: float = 0.0
    entry_time: object = None
    exit_time: object = None


# ============================================================
# Backtest Engine v2
# ============================================================
class BacktestEngineV2:
    def __init__(
        self,
        initial_capital_usd: float = config.TOTAL_CAPITAL_USD,
        start_date: date = date(2025, 2, 18),
        end_date: date = date(2026, 2, 17),
        use_fees: bool = True,
    ):
        self.initial_capital_usd = initial_capital_usd
        self.cash_usd = initial_capital_usd
        self.start_date = start_date
        self.end_date = end_date
        self.use_fees = use_fees
        self._verbose = False  # set by run()

        # State
        self.positions: dict[str, Position] = {}
        self.trades: list[TradeV2] = []
        self.equity_curve: list[tuple[date, float]] = []
        self._sold_today: set[str] = set()
        self._last_buy_time: dict[str, datetime] = {}

        # Fee accumulators (USD)
        self.total_buy_fees_usd: float = 0.0
        self.total_sell_fees_usd: float = 0.0

        # Statistics
        self.total_trading_days: int = 0
        self.skipped_gold_bars: int = 0  # bars where gold blocked buys

    # ----------------------------------------------------------
    # Conditional time check (KST 17:30)
    # ----------------------------------------------------------
    def _is_conditional_allowed(self, ts) -> bool:
        """조건부 매매 허용 시각인지 확인한다 (한국시간 17:30 이후).

        US 장중 시간은 KST 22:30~06:00 (자정을 넘김).
        허용: 17:30~23:59, 00:00~06:59 (장중 전체)
        차단: 07:00~17:29 (장 마감 후 ~ 오후)
        """
        h, m = config.CONDITIONAL_START_KST
        try:
            kst = ts.tz_convert(KST_TZ)
            kst_time = (kst.hour, kst.minute)
            return kst_time >= (h, m) or kst_time < (7, 0)
        except Exception:
            return True  # 변환 실패 시 허용

    # ----------------------------------------------------------
    # Net profit calculation (for sell decisions)
    # ----------------------------------------------------------
    def _calc_net_profit_pct(self, pos: Position, cur_price: float) -> float:
        """Calculate net profit % after fees."""
        gross_usd = pos.total_qty * cur_price
        if self.use_fees:
            sell_fee = backtest_common.calc_sell_fee(gross_usd)
        else:
            sell_fee = 0.0
        net_usd = gross_usd - sell_fee
        if pos.total_invested_usd <= 0:
            return 0.0
        return (net_usd - pos.total_invested_usd) / pos.total_invested_usd * 100

    # ----------------------------------------------------------
    # Buy eligibility check
    # ----------------------------------------------------------
    def _can_buy(self, ticker: str, ts: datetime) -> bool:
        """Check if we can open a new position on this ticker."""
        if ticker in self.positions:
            return False  # already holding (DCA handled separately)
        if ticker in self._sold_today:
            return False  # no re-buy same day (rule 8.5)
        last = self._last_buy_time.get(ticker)
        if last and (ts - last).total_seconds() < 300:
            return False  # 5-min cooldown (rule 8.4)
        return True

    # ----------------------------------------------------------
    # Market time elapsed (minutes)
    # ----------------------------------------------------------
    @staticmethod
    def _market_minutes_elapsed(entry_time: datetime, current_time: datetime) -> float:
        """Calculate elapsed market minutes between two timestamps."""
        if current_time <= entry_time:
            return 0.0

        entry_date = entry_time.date()
        current_date = current_time.date()

        total_minutes = 0.0

        d = entry_date
        while d <= current_date:
            day_open = datetime.combine(d, MARKET_OPEN)
            day_close = datetime.combine(d, MARKET_CLOSE)

            if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is not None:
                day_open = day_open.replace(tzinfo=entry_time.tzinfo)
                day_close = day_close.replace(tzinfo=entry_time.tzinfo)

            if d == entry_date:
                effective_start = max(entry_time, day_open)
            else:
                effective_start = day_open

            if d == current_date:
                effective_end = min(current_time, day_close)
            else:
                effective_end = day_close

            if effective_end > effective_start:
                total_minutes += (effective_end - effective_start).total_seconds() / 60

            d += timedelta(days=1)

        return total_minutes

    # ----------------------------------------------------------
    # Coin follow selection (MSTU vs IRE)
    # ----------------------------------------------------------
    def _select_coin_follow(self, sym_bars_day: dict[str, list[dict]], trading_date=None) -> str:
        """Select MSTU or IRE for today based on volatility, then volume."""
        candidates = ["MSTU", "IRE"]
        stats: dict[str, dict] = {}

        for ticker in candidates:
            bars = sym_bars_day.get(ticker, [])[:30]
            if not bars:
                continue
            open_price = bars[0]["open"]
            if open_price <= 0:
                continue
            vol = (max(b["high"] for b in bars) - min(b["low"] for b in bars)) / open_price * 100
            volume = sum(b["volume"] for b in bars)
            stats[ticker] = {"volatility": vol, "volume": volume}

        if len(stats) == 0:
            if self._verbose:
                d = trading_date if trading_date else "?"
                print(f"    [WARN] MSTU/IRE 데이터 없음 ({d}), 기본 MSTU 선택")
            return "MSTU"
        if len(stats) == 1:
            return list(stats.keys())[0]

        mstu_vol = stats["MSTU"]["volatility"]
        ire_vol = stats["IRE"]["volatility"]
        diff = abs(mstu_vol - ire_vol)

        if diff >= config.COIN_FOLLOW_VOLATILITY_GAP:
            return "MSTU" if mstu_vol > ire_vol else "IRE"
        else:
            return "MSTU" if stats["MSTU"]["volume"] >= stats["IRE"]["volume"] else "IRE"

    # ----------------------------------------------------------
    # Buy execution
    # ----------------------------------------------------------
    def _buy(
        self,
        ticker: str,
        price: float,
        ts: datetime,
        signal_type: str,
        amount_usd: float,
        is_conl_conditional: bool = False,
        is_coin_conditional: bool = False,
    ) -> None:
        """Execute a buy order (USD)."""
        if price <= 0 or amount_usd <= 0:
            return
        if amount_usd > self.cash_usd:
            amount_usd = self.cash_usd
        if amount_usd < 7:  # minimum ~$7
            return

        # Fee deduction on buy
        if self.use_fees:
            buy_fee = backtest_common.calc_buy_fee(amount_usd)
            self.total_buy_fees_usd += buy_fee
        else:
            buy_fee = 0.0

        net_amount_usd = amount_usd - buy_fee
        qty = net_amount_usd / price

        self.cash_usd -= amount_usd

        # Create or update position
        entry = {"time": ts, "price": price, "qty": qty, "usd": amount_usd}

        if ticker in self.positions:
            # DCA add
            pos = self.positions[ticker]
            pos.entries.append(entry)
            pos.total_qty += qty
            pos.total_invested_usd += amount_usd
            pos.avg_price = pos.total_invested_usd / pos.total_qty
            pos.dca_count += 1
            dca_level = pos.dca_count
        else:
            pos = Position(
                ticker=ticker,
                entries=[entry],
                total_qty=qty,
                avg_price=price,
                total_invested_usd=amount_usd,
                dca_count=0,
                initial_entry_price=price,
                first_entry_time=ts,
                signal_type=signal_type,
                is_conl_conditional=is_conl_conditional,
                is_coin_conditional=is_coin_conditional,
            )
            self.positions[ticker] = pos
            dca_level = 0

        self._last_buy_time[ticker] = ts

        # Record trade
        self.trades.append(TradeV2(
            date=ts.date() if isinstance(ts, (datetime, pd.Timestamp)) else ts,
            ticker=ticker,
            side="BUY",
            price=price,
            qty=qty,
            amount_usd=amount_usd,
            pnl_usd=0.0,
            pnl_pct=0.0,
            signal_type=signal_type,
            dca_level=dca_level,
            fees_usd=buy_fee,
            entry_time=ts,
        ))

    # ----------------------------------------------------------
    # Sell execution (full or partial)
    # ----------------------------------------------------------
    def _sell(
        self,
        ticker: str,
        qty: float,
        price: float,
        ts: datetime,
        exit_reason: str,
    ) -> None:
        """Execute a sell order for given quantity (USD)."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        sell_qty = min(qty, pos.total_qty)
        if sell_qty <= 0:
            return

        # Calculate proceeds
        gross_usd = sell_qty * price

        if self.use_fees:
            sell_fee = backtest_common.calc_sell_fee(gross_usd)
            self.total_sell_fees_usd += sell_fee
        else:
            sell_fee = 0.0

        net_proceeds_usd = gross_usd - sell_fee

        # PnL: proportional cost basis
        fraction = sell_qty / pos.total_qty
        cost_usd = pos.total_invested_usd * fraction
        pnl_usd = net_proceeds_usd - cost_usd
        pnl_pct = (pnl_usd / cost_usd * 100) if cost_usd > 0 else 0.0

        self.cash_usd += net_proceeds_usd

        # Update position
        pos.total_qty -= sell_qty
        pos.total_invested_usd -= cost_usd

        # Record trade
        self.trades.append(TradeV2(
            date=ts.date() if isinstance(ts, (datetime, pd.Timestamp)) else ts,
            ticker=ticker,
            side="SELL",
            price=price,
            qty=sell_qty,
            amount_usd=net_proceeds_usd,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct, 2),
            signal_type=pos.signal_type,
            exit_reason=exit_reason,
            fees_usd=sell_fee,
            net_pnl_usd=round(pnl_usd, 2),
            entry_time=pos.first_entry_time,
            exit_time=ts,
        ))

        self._sold_today.add(ticker)

        # Remove position if fully closed
        if pos.total_qty < 1e-9:
            del self.positions[ticker]

    def _sell_all(
        self,
        ticker: str,
        price: float,
        ts: datetime,
        exit_reason: str,
    ) -> None:
        """Sell entire position for a ticker."""
        if ticker not in self.positions:
            return
        self._sell(ticker, self.positions[ticker].total_qty, price, ts, exit_reason)

    def _partial_sell(
        self,
        ticker: str,
        qty: float,
        price: float,
        ts: datetime,
        exit_reason: str,
    ) -> None:
        """Sell a partial quantity."""
        self._sell(ticker, qty, price, ts, exit_reason)

    # ----------------------------------------------------------
    # Staged sell processing
    # ----------------------------------------------------------
    def _process_staged_sell(self, pos: Position, price: float, ts: datetime) -> None:
        """Process staged sell steps (30% of remaining every 5 min)."""
        if not pos.staged_sell_active:
            return
        if pos.staged_sell_start is None:
            return

        elapsed = (ts - pos.staged_sell_start).total_seconds() / 60
        expected_steps = int(elapsed / config.PAIR_SELL_INTERVAL_MIN)

        while pos.staged_sell_step < expected_steps and pos.staged_sell_remaining > 0:
            sell_qty = pos.staged_sell_remaining * config.PAIR_SELL_REMAINING_PCT
            if sell_qty < 0.001:  # minimum threshold
                sell_qty = pos.staged_sell_remaining
            self._partial_sell(pos.ticker, sell_qty, price, ts, "staged_sell")
            pos.staged_sell_remaining -= sell_qty
            pos.staged_sell_step += 1

        if pos.staged_sell_remaining <= 0:
            pos.staged_sell_active = False

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    def run(self, verbose: bool = True) -> "BacktestEngineV2":
        """Execute the backtest."""
        self._verbose = verbose
        # ---- 1. Load data ----
        print("[1/4] 데이터 준비")
        df = backtest_common.load_parquet(config.DATA_DIR / "backtest_1min_v2.parquet")
        poly = backtest_common.load_polymarket_daily(config.PROJECT_ROOT / "polymarket" / "history")

        # Pre-index DataFrame for O(1) bar lookup
        ts_prices, sym_bars, day_timestamps = backtest_common.preindex_dataframe(df)

        # ---- 2. Prepare trading dates ----
        all_dates = sorted(day_timestamps.keys())
        bt_dates = [d for d in all_dates if self.start_date <= d <= self.end_date]
        warmup_dates = [d for d in all_dates if d < self.start_date]

        # Warmup: compute prev_close
        prev_close: dict[str, float] = {}
        for d in warmup_dates:
            if d not in sym_bars:
                continue
            for ticker, bars in sym_bars[d].items():
                if bars:
                    prev_close[ticker] = bars[-1]["close"]

        self.total_trading_days = len(bt_dates)
        if verbose:
            print(f"\n[2/4] 시뮬레이션 실행")
            print(f"  백테스트 기간: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}일)")
            print(f"  초기 자금: ${self.initial_capital_usd:,.2f}\n")

        # ---- 3. Main day loop ----
        for day_idx, trading_date in enumerate(bt_dates):
            if trading_date not in ts_prices:
                continue

            day_sym = sym_bars[trading_date]
            day_ts = day_timestamps[trading_date]

            # (a) Polymarket probs -> market mode
            poly_probs = poly.get(trading_date, None)
            market_mode = signals_v2.determine_market_mode(poly_probs)

            # Determine stop loss threshold for this day
            if market_mode == "bullish":
                stop_loss_threshold = config.STOP_LOSS_BULLISH_PCT
            else:
                stop_loss_threshold = config.STOP_LOSS_PCT

            # (b) Handle carry positions from previous day
            self._handle_carry_positions(day_sym, day_ts, market_mode, prev_close)

            # (c) Clear daily state
            self._sold_today.clear()

            # (d) Select coin follow for today
            coin_follow = self._select_coin_follow(day_sym, trading_date)

            # Build today's pairs dict with selected follow
            today_pairs = {}
            for pair_key, pair_cfg in config.TWIN_PAIRS_V2.items():
                if pair_key == "coin":
                    today_pairs[pair_key] = {
                        "lead": pair_cfg["lead"],
                        "follow": [coin_follow],
                        "label": f"코인 (BTC -> {coin_follow})",
                    }
                else:
                    today_pairs[pair_key] = pair_cfg

            # (e) Process each 1-min bar (O(1) dict lookup per bar)
            for ts in day_ts:
                # O(1) dict lookup replaces DataFrame filter + iterrows
                cur_prices = ts_prices[trading_date].get(ts, {})

                # Changes from prev_close
                changes = backtest_common.calc_changes(cur_prices, prev_close)

                # Generate signals
                sigs = signals_v2.generate_all_signals_v2(
                    changes,
                    poly_probs=poly_probs,
                    pairs=today_pairs,
                    coin_trigger_pct=config.COIN_TRIGGER_PCT,
                    coin_sell_profit_pct=config.COIN_SELL_PROFIT_PCT,
                    coin_sell_bearish_pct=config.COIN_SELL_BEARISH_PCT,
                    conl_trigger_pct=config.CONL_TRIGGER_PCT,
                    conl_sell_profit_pct=config.CONL_SELL_PROFIT_PCT,
                    conl_sell_avg_pct=config.CONL_SELL_AVG_PCT,
                )

                gold_warning = sigs["gold"]["warning"]

                # ========== SELL PRIORITY ==========
                self._process_sells(
                    cur_prices, changes, sigs, ts, market_mode,
                    stop_loss_threshold,
                )

                # ========== BUY ==========
                self._process_buys(
                    cur_prices, changes, sigs, ts, market_mode,
                    gold_warning, today_pairs,
                )

            # (f) End of day processing
            last_prices = {sym: bars[-1]["close"] for sym, bars in day_sym.items() if bars}
            last_ts = day_ts[-1] if day_ts else None

            if market_mode == "bullish":
                # Mark remaining positions as carry (hold to next day)
                for ticker, pos in self.positions.items():
                    if not pos.carry:
                        pos.carry = True
            else:
                # Close all remaining positions (normal/bearish mode)
                for ticker in list(self.positions.keys()):
                    price = last_prices.get(ticker)
                    if price is not None:
                        self._sell_all(ticker, price, last_ts, "eod_close")

            # Update prev_close
            prev_close.update(last_prices)

            # Record equity
            equity = self._calc_total_equity(cur_prices if last_prices else prev_close)
            self.equity_curve.append((trading_date, equity))

            # Progress output
            if verbose and ((day_idx + 1) % 50 == 0 or day_idx == len(bt_dates) - 1):
                print(
                    f"  [{day_idx+1:>3}/{len(bt_dates)}] {trading_date}  "
                    f"자산: ${equity:,.2f}  "
                    f"현금: ${self.cash_usd:,.2f}  "
                    f"포지션: {len(self.positions)}개  "
                    f"모드: {market_mode}"
                )

        if verbose:
            print("\n  백테스트 완료!")
        return self

    # ----------------------------------------------------------
    # Carry position handling (bullish mode next-day)
    # ----------------------------------------------------------
    def _handle_carry_positions(
        self, sym_bars_day: dict[str, list[dict]], day_ts: list,
        market_mode: str,
        prev_close: dict[str, float] | None = None,
    ) -> None:
        """Handle carry positions from previous day."""
        if not any(pos.carry for pos in self.positions.values()):
            return

        # Get first available prices of the day
        first_prices: dict[str, float] = {}
        for ticker in list(self.positions.keys()):
            bars = sym_bars_day.get(ticker, [])
            if bars:
                first_prices[ticker] = bars[0]["open"]

        first_ts = day_ts[0] if day_ts else None

        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.carry:
                continue

            price = first_prices.get(ticker)
            if price is None:
                continue

            if market_mode != "bullish":
                pos.carry = False
                continue

            # Check if +2% net profit -> sell immediately
            net_pnl_pct = self._calc_net_profit_pct(pos, price)
            if net_pnl_pct >= config.TAKE_PROFIT_PCT:
                self._sell_all(ticker, price, first_ts, "carry_sell")
                continue

            # Check if 양전 AND has net profit -> sell
            prev_price = (prev_close or {}).get(ticker)
            is_positive_from_prev = False
            if prev_price and prev_price > 0:
                is_positive_from_prev = price > prev_price

            if is_positive_from_prev and net_pnl_pct > 0:
                self._sell_all(ticker, price, first_ts, "carry_sell")
            else:
                pos.carry = False

    # ----------------------------------------------------------
    # SELL processing
    # ----------------------------------------------------------
    def _process_sells(
        self,
        cur_prices: dict[str, float],
        changes: dict[str, dict],
        sigs: dict,
        ts: datetime,
        market_mode: str,
        stop_loss_threshold: float,
    ) -> None:
        """Process all sell signals in priority order."""
        # 1. Stop loss check
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue

            change_data = changes.get(ticker, {})
            change_pct = change_data.get("change_pct", 0.0)

            if change_pct <= stop_loss_threshold:
                self._sell_all(ticker, price, ts, "stop_loss")

        # 2. Time stop: holding > 5 hours (market hours)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue

            elapsed_min = self._market_minutes_elapsed(pos.first_entry_time, ts)
            time_limit_min = config.MAX_HOLD_HOURS * 60

            if elapsed_min >= time_limit_min:
                if market_mode == "bullish":
                    net_pnl_pct = self._calc_net_profit_pct(pos, price)
                    if net_pnl_pct >= config.TAKE_PROFIT_PCT:
                        self._sell_all(ticker, price, ts, "time_stop")
                else:
                    self._sell_all(ticker, price, ts, "time_stop")

        # 3. CONL sell (조건부: KST 17:30 이후만)
        if self._is_conditional_allowed(ts):
            for ticker in list(self.positions.keys()):
                pos = self.positions.get(ticker)
                if pos is None or not pos.is_conl_conditional:
                    continue
                price = cur_prices.get(ticker)
                if price is None:
                    continue

                net_pnl_pct = self._calc_net_profit_pct(pos, price)

                if net_pnl_pct >= config.CONL_SELL_PROFIT_PCT:
                    self._sell_all(ticker, price, ts, "conl_profit")
                    continue

                conl_sig = sigs.get("conditional_conl", {})
                if conl_sig.get("sell_on_avg_drop", False):
                    self._sell_all(ticker, price, ts, "conl_avg_drop")

        # 4. COIN sell (조건부: KST 17:30 이후만)
        if self._is_conditional_allowed(ts):
            for ticker in list(self.positions.keys()):
                pos = self.positions.get(ticker)
                if pos is None or not pos.is_coin_conditional:
                    continue
                price = cur_prices.get(ticker)
                if price is None:
                    continue

                net_pnl_pct = self._calc_net_profit_pct(pos, price)

                if market_mode == "bearish":
                    if net_pnl_pct >= config.COIN_SELL_BEARISH_PCT:
                        self._sell_all(ticker, price, ts, "coin_bearish")
                else:
                    if net_pnl_pct >= config.COIN_SELL_PROFIT_PCT:
                        self._sell_all(ticker, price, ts, "coin_profit")

        # 5. Staged sell (ongoing)
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.staged_sell_active:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            self._process_staged_sell(pos, price, ts)

        # 6. Twin SELL signal: gap <= 0.9% -> staged sell
        for pair_sig in sigs.get("twin_pairs", []):
            follow = pair_sig["follow"]
            signal = pair_sig["signal"]

            if signal != "SELL":
                continue
            if follow not in self.positions:
                continue

            pos = self.positions[follow]
            if pos.staged_sell_active:
                continue

            price = cur_prices.get(follow)
            if price is None:
                continue

            # Start staged sell: sell 80% immediately, activate staged for rest
            first_sell_qty = pos.total_qty * config.PAIR_SELL_FIRST_PCT
            remaining_qty = pos.total_qty - first_sell_qty

            self._partial_sell(follow, first_sell_qty, price, ts, "twin_converge")

            if follow in self.positions:
                pos = self.positions[follow]
                pos.staged_sell_active = True
                pos.staged_sell_start = ts
                pos.staged_sell_remaining = pos.total_qty
                pos.staged_sell_step = 0

    # ----------------------------------------------------------
    # BUY processing
    # ----------------------------------------------------------
    def _process_buys(
        self,
        cur_prices: dict[str, float],
        changes: dict[str, dict],
        sigs: dict,
        ts: datetime,
        market_mode: str,
        gold_warning: bool,
        today_pairs: dict,
    ) -> None:
        """Process all buy signals."""
        # 7. GLD positive -> block non-conditional buys
        gld_blocks_non_conditional = gold_warning
        if gld_blocks_non_conditional:
            self.skipped_gold_bars += 1

        # 8. Twin ENTRY: gap >= 1.5% (blocked by GLD)
        if not gld_blocks_non_conditional:
            for pair_sig in sigs.get("twin_pairs", []):
                follow = pair_sig["follow"]
                signal = pair_sig["signal"]

                if signal != "ENTRY":
                    continue
                if not self._can_buy(follow, ts):
                    continue
                price = cur_prices.get(follow)
                if price is None:
                    continue

                self._buy(
                    follow, price, ts, "twin",
                    amount_usd=config.INITIAL_BUY_USD,
                )

        # 9. DCA check for existing positions
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None:
                continue

            if gld_blocks_non_conditional:
                if not pos.is_conl_conditional and not pos.is_coin_conditional:
                    continue

            if ticker == "BRKU":
                continue

            price = cur_prices.get(ticker)
            if price is None:
                continue

            if pos.dca_count >= config.DCA_MAX_COUNT:
                continue
            if pos.total_invested_usd + config.DCA_BUY_USD > config.MAX_PER_STOCK_USD:
                continue

            required_drop_pct = config.DCA_DROP_PCT * (pos.dca_count + 1)
            actual_drop_pct = (price - pos.initial_entry_price) / pos.initial_entry_price * 100

            if actual_drop_pct <= required_drop_pct:
                last = self._last_buy_time.get(ticker)
                if last and (ts - last).total_seconds() < 300:
                    continue

                self._buy(
                    ticker, price, ts, pos.signal_type,
                    amount_usd=config.DCA_BUY_USD,
                    is_conl_conditional=pos.is_conl_conditional,
                    is_coin_conditional=pos.is_coin_conditional,
                )

        # 10. CONL buy (조건부: KST 17:30 이후만)
        if self._is_conditional_allowed(ts):
            conl_sig = sigs.get("conditional_conl", {})
            if conl_sig.get("buy_signal", False):
                if self._can_buy("CONL", ts):
                    price = cur_prices.get("CONL")
                    if price is not None:
                        self._buy(
                            "CONL", price, ts, "conditional_conl",
                            amount_usd=config.INITIAL_BUY_USD,
                            is_conl_conditional=True,
                        )

        # 11. COIN buy (조건부: KST 17:30 이후만)
        if self._is_conditional_allowed(ts):
            coin_sig = sigs.get("conditional_coin", {})
            if coin_sig.get("buy_signal", False):
                target = coin_sig.get("target", config.CONDITIONAL_TARGET_V2)
                if self._can_buy(target, ts):
                    price = cur_prices.get(target)
                    if price is not None:
                        self._buy(
                            target, price, ts, "conditional_coin",
                            amount_usd=config.INITIAL_BUY_USD,
                            is_coin_conditional=True,
                        )

        # 12. Bearish: if mode == "bearish" and BRKU not held -> buy BRKU at 10%
        if market_mode == "bearish":
            bearish_sig = sigs.get("bearish", {})
            if bearish_sig.get("buy_brku", False):
                if self._can_buy("BRKU", ts):
                    price = cur_prices.get("BRKU")
                    if price is not None:
                        brku_amount = self.initial_capital_usd * config.BRKU_WEIGHT_PCT / 100
                        self._buy(
                            "BRKU", price, ts, "bearish",
                            amount_usd=brku_amount,
                        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _calc_total_equity(self, cur_prices: dict[str, float]) -> float:
        """Calculate total equity = cash + position values (USD)."""
        equity = self.cash_usd
        for ticker, pos in self.positions.items():
            price = cur_prices.get(ticker)
            if price is not None:
                equity += pos.total_qty * price
            else:
                equity += pos.total_invested_usd
        return equity

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------
    def print_report(self) -> None:
        """Print console summary report."""
        if not self.equity_curve:
            print("  데이터 없음")
            return

        final = self.equity_curve[-1][1]
        total_ret = (final - self.initial_capital_usd) / self.initial_capital_usd * 100

        sells = [t for t in self.trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl_usd > 0]
        losses = [t for t in sells if t.pnl_usd < 0]
        breakeven = [t for t in sells if t.pnl_usd == 0]
        total_pnl = sum(t.pnl_usd for t in sells)

        mdd = backtest_common.calc_mdd(self.equity_curve)
        sharpe = backtest_common.calc_sharpe(self.equity_curve)
        total_fees = self.total_buy_fees_usd + self.total_sell_fees_usd

        # Exit reason breakdown
        exit_stats: dict[str, dict] = {}
        for t in sells:
            key = t.exit_reason or "unknown"
            if key not in exit_stats:
                exit_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            exit_stats[key]["count"] += 1
            exit_stats[key]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                exit_stats[key]["wins"] += 1

        # Signal-type P&L
        sig_stats: dict[str, dict] = {}
        for t in sells:
            key = t.signal_type
            if key not in sig_stats:
                sig_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            sig_stats[key]["count"] += 1
            sig_stats[key]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                sig_stats[key]["wins"] += 1

        # DCA statistics
        buys = [t for t in self.trades if t.side == "BUY"]
        dca_buys = [t for t in buys if t.dca_level > 0]
        initial_buys = [t for t in buys if t.dca_level == 0]

        print()
        print("=" * 65)
        print("    PTJ 매매법 v2 백테스트 리포트 (USD)")
        print("=" * 65)
        print(f"  기간         : {self.equity_curve[0][0]} ~ {self.equity_curve[-1][0]}")
        print(f"  거래일       : {self.total_trading_days}일")
        print(f"  초기 자금    : ${self.initial_capital_usd:>12,.2f}")
        print(f"  최종 자산    : ${final:>12,.2f}")
        print(f"  총 수익률    : {total_ret:>+.2f}%")
        print(f"  총 손익      : ${total_pnl:>+12,.2f}")
        print(f"  최대 낙폭    : -{mdd:.2f}%")
        print(f"  Sharpe Ratio : {sharpe:.4f}")
        print("-" * 65)
        print(f"  총 매도 횟수 : {len(sells)}")
        print(f"  승 / 패      : {len(wins)}W / {len(losses)}L / {len(breakeven)}E")
        win_rate = len(wins) / len(sells) * 100 if sells else 0
        print(f"  승률         : {win_rate:.1f}%")
        avg_win = np.mean([t.pnl_usd for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_usd for t in losses]) if losses else 0
        print(f"  평균 수익    : ${avg_win:>+12,.2f}")
        print(f"  평균 손실    : ${avg_loss:>+12,.2f}")
        print("-" * 65)

        # Fee breakdown
        print("  [수수료]")
        print(f"    매수 수수료 : ${self.total_buy_fees_usd:>10,.2f}")
        print(f"    매도 수수료 : ${self.total_sell_fees_usd:>10,.2f}")
        print(f"    총 수수료   : ${total_fees:>10,.2f}")
        print("-" * 65)

        # DCA statistics
        print("  [DCA 통계]")
        print(f"    초기 진입   : {len(initial_buys)}회")
        print(f"    물타기      : {len(dca_buys)}회")
        if dca_buys:
            max_dca = max(t.dca_level for t in dca_buys)
            total_dca_usd = sum(t.amount_usd for t in dca_buys)
            print(f"    최대 DCA 단계: {max_dca}")
            print(f"    DCA 총 투입 : ${total_dca_usd:>10,.2f}")
        print("-" * 65)

        # Exit reason breakdown
        if exit_stats:
            print("  [매도 사유별 성과]")
            for key in sorted(exit_stats.keys()):
                s = exit_stats[key]
                wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                print(
                    f"    {key:18s}  {s['count']:>4d}회  "
                    f"P&L ${s['pnl']:>+10,.2f}  승률 {wr:>5.1f}%"
                )
        print("-" * 65)

        # Signal-type P&L
        if sig_stats:
            print("  [시그널별 성과]")
            for key in sorted(sig_stats.keys()):
                s = sig_stats[key]
                wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
                print(
                    f"    {key:18s}  {s['count']:>4d}회  "
                    f"P&L ${s['pnl']:>+10,.2f}  승률 {wr:>5.1f}%"
                )
        print("=" * 65)

    # ----------------------------------------------------------
    # Save trade log CSV
    # ----------------------------------------------------------
    def save_trade_log(self) -> Path | None:
        """Save all trades to CSV."""
        if not self.trades:
            return None

        rows = []
        for t in self.trades:
            rows.append({
                "date": t.date,
                "ticker": t.ticker,
                "side": t.side,
                "price": round(t.price, 4),
                "qty": round(t.qty, 6),
                "amount_usd": round(t.amount_usd, 2),
                "pnl_usd": round(t.pnl_usd, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "net_pnl_usd": round(t.net_pnl_usd, 2),
                "signal_type": t.signal_type,
                "exit_reason": t.exit_reason,
                "dca_level": t.dca_level,
                "fees_usd": round(t.fees_usd, 2),
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            })

        out_df = pd.DataFrame(rows)
        out_path = config.DATA_DIR / "backtest_v2_trades.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  거래 로그: {out_path}")
        return out_path


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("  PTJ 매매법 v2 - 1분봉 백테스트 (USD)")
    print()

    engine = BacktestEngineV2()
    engine.run()

    print("\n[3/4] 리포트 생성")
    engine.print_report()

    print("\n[4/4] 파일 저장")
    engine.save_trade_log()
    print()


if __name__ == "__main__":
    main()
