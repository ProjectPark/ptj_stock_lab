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
  - backtest_base.py  : BacktestBase ABC
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from simulation.strategies import signals_v2
from simulation.backtests import backtest_common
from simulation.backtests.backtest_base import BacktestBase

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
class BacktestEngineV2(BacktestBase):
    def __init__(
        self,
        initial_capital_usd: float = config.TOTAL_CAPITAL_USD,
        start_date: date = date(2025, 2, 18),
        end_date: date = date(2026, 2, 17),
        use_fees: bool = True,
        params=None,
        signal_engine=None,
    ):
        # Backward compat: build params from config if not provided
        if params is None:
            from simulation.strategies.params import v2_params_from_config
            params = v2_params_from_config()

        # OOP signal engine: 기본값으로 CompositeSignalEngine 사용
        if signal_engine is None:
            from simulation.strategies.line_b_taejun.composite_signal_engine import CompositeSignalEngine
            signal_engine = CompositeSignalEngine.from_base_params(params)

        super().__init__(
            params=params,
            start_date=start_date,
            end_date=end_date,
            use_fees=use_fees,
            signal_engine=signal_engine,
        )

    # Backward-compat aliases (v2 uses *_usd naming)
    @property
    def initial_capital_usd(self):
        return self.initial_capital

    @property
    def cash_usd(self):
        return self.cash

    @cash_usd.setter
    def cash_usd(self, value):
        self.cash = value

    @property
    def total_buy_fees_usd(self):
        return self.total_buy_fees

    @total_buy_fees_usd.setter
    def total_buy_fees_usd(self, value):
        self.total_buy_fees = value

    @property
    def total_sell_fees_usd(self):
        return self.total_sell_fees

    @total_sell_fees_usd.setter
    def total_sell_fees_usd(self, value):
        self.total_sell_fees = value

    # ============================================================
    # BacktestBase abstract method implementations
    # ============================================================

    def _version_label(self) -> str:
        return "v2 USD"

    def _init_version_state(self) -> None:
        """v2는 추가 상태 없음."""
        pass

    def _load_extra_data(self, df: pd.DataFrame, poly: dict) -> None:
        """v2는 추가 데이터 없음."""
        pass

    def _warmup_extra(self, warmup_dates: list, sym_bars: dict, prev_close: dict) -> None:
        """v2는 추가 warmup 없음."""
        pass

    def _on_day_start(
        self,
        trading_date: date,
        day_idx: int,
        day_sym: dict,
        day_ts: list,
        poly_probs: dict | None,
        prev_close: dict[str, float],
    ) -> dict:
        """일 시작 시 처리."""
        market_mode = self._signal_engine.market_mode_filter.evaluate(poly_probs)

        if market_mode == "bullish":
            stop_loss_threshold = config.STOP_LOSS_BULLISH_PCT
        else:
            stop_loss_threshold = config.STOP_LOSS_PCT

        # Handle carry positions
        self._handle_carry_positions(day_sym, day_ts, market_mode, prev_close)

        # Clear daily state
        self._sold_today.clear()

        # Select coin follow
        coin_follow = self._select_coin_follow(day_sym, trading_date)
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

        return {
            "market_mode": market_mode,
            "stop_loss_threshold": stop_loss_threshold,
            "poly_probs": poly_probs,
            "today_pairs": today_pairs,
        }

    def _on_bar(
        self,
        ts: datetime,
        cur_prices: dict[str, float],
        changes: dict[str, dict],
        day_ctx: dict,
    ) -> None:
        """1분봉 바 단위 처리."""
        market_mode = day_ctx["market_mode"]
        stop_loss_threshold = day_ctx["stop_loss_threshold"]
        poly_probs = day_ctx["poly_probs"]
        today_pairs = day_ctx["today_pairs"]

        # Generate signals (OOP CompositeSignalEngine)
        sigs = self._signal_engine.generate_all_signals(
            changes,
            poly_probs=poly_probs,
            pairs=today_pairs,
        )

        gold_warning = sigs["gold"]["warning"]

        # SELL
        self._process_sells(
            cur_prices, changes, sigs, ts, market_mode,
            stop_loss_threshold,
        )

        # BUY
        self._process_buys(
            cur_prices, changes, sigs, ts, market_mode,
            gold_warning, today_pairs,
        )

    def _on_day_end(
        self,
        trading_date: date,
        day_idx: int,
        last_prices: dict[str, float],
        last_ts: datetime | None,
        prev_close: dict[str, float],
        day_ctx: dict,
    ) -> None:
        """일 종료 시 처리."""
        market_mode = day_ctx["market_mode"]

        if market_mode == "bullish":
            for ticker, pos in self.positions.items():
                if not pos.carry:
                    pos.carry = True
        else:
            for ticker in list(self.positions.keys()):
                price = last_prices.get(ticker)
                if price is not None:
                    self._sell_all(ticker, price, last_ts, "eod_close")

    def _snapshot_equity(self, cur_prices: dict[str, float]) -> float:
        """v2 총 자산 (USD, no FX)."""
        equity = self.cash
        for ticker, pos in self.positions.items():
            price = cur_prices.get(ticker)
            if price is not None:
                equity += pos.total_qty * price
            else:
                equity += pos.total_invested_usd
        return equity

    def _print_run_header(self) -> None:
        print(f"  초기 자금: ${self.initial_capital:,.2f}\n")

    def _print_day_progress(
        self, day_idx: int, total_days: int, trading_date: date,
        equity: float, day_ctx: dict,
    ) -> None:
        market_mode = day_ctx.get("market_mode", "?")
        print(
            f"  [{day_idx+1:>3}/{total_days}] {trading_date}  "
            f"자산: ${equity:,.2f}  "
            f"현금: ${self.cash:,.2f}  "
            f"포지션: {len(self.positions)}개  "
            f"모드: {market_mode}"
        )

    # ----------------------------------------------------------
    # Conditional time check (KST 17:30)
    # ----------------------------------------------------------
    def _is_conditional_allowed(self, ts) -> bool:
        h, m = config.CONDITIONAL_START_KST
        try:
            kst = ts.tz_convert(KST_TZ)
            kst_time = (kst.hour, kst.minute)
            return kst_time >= (h, m) or kst_time < (7, 0)
        except Exception:
            return True

    # ----------------------------------------------------------
    # Net profit calculation
    # ----------------------------------------------------------
    def _calc_net_profit_pct(self, pos: Position, cur_price: float) -> float:
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
    # Buy eligibility
    # ----------------------------------------------------------
    def _can_buy(self, ticker: str, ts: datetime) -> bool:
        if ticker in self.positions:
            return False
        if ticker in self._sold_today:
            return False
        last = self._last_buy_time.get(ticker)
        if last and (ts - last).total_seconds() < 300:
            return False
        return True

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
        if price <= 0 or amount_usd <= 0:
            return
        if amount_usd > self.cash:
            amount_usd = self.cash
        if amount_usd < 7:
            return

        if self.use_fees:
            buy_fee = backtest_common.calc_buy_fee(amount_usd)
            self.total_buy_fees += buy_fee
        else:
            buy_fee = 0.0

        net_amount_usd = amount_usd - buy_fee
        qty = net_amount_usd / price

        self.cash -= amount_usd

        entry = {"time": ts, "price": price, "qty": qty, "usd": amount_usd}

        if ticker in self.positions:
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
    # Sell execution
    # ----------------------------------------------------------
    def _sell(
        self,
        ticker: str,
        qty: float,
        price: float,
        ts: datetime,
        exit_reason: str,
    ) -> None:
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        sell_qty = min(qty, pos.total_qty)
        if sell_qty <= 0:
            return

        gross_usd = sell_qty * price

        if self.use_fees:
            sell_fee = backtest_common.calc_sell_fee(gross_usd)
            self.total_sell_fees += sell_fee
        else:
            sell_fee = 0.0

        net_proceeds_usd = gross_usd - sell_fee

        fraction = sell_qty / pos.total_qty
        cost_usd = pos.total_invested_usd * fraction
        pnl_usd = net_proceeds_usd - cost_usd
        pnl_pct = (pnl_usd / cost_usd * 100) if cost_usd > 0 else 0.0

        self.cash += net_proceeds_usd

        pos.total_qty -= sell_qty
        pos.total_invested_usd -= cost_usd

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

        if pos.total_qty < 1e-9:
            del self.positions[ticker]

    def _sell_all(self, ticker: str, price: float, ts: datetime, exit_reason: str) -> None:
        if ticker not in self.positions:
            return
        self._sell(ticker, self.positions[ticker].total_qty, price, ts, exit_reason)

    def _partial_sell(self, ticker: str, qty: float, price: float, ts: datetime, exit_reason: str) -> None:
        self._sell(ticker, qty, price, ts, exit_reason)

    # ----------------------------------------------------------
    # Staged sell
    # ----------------------------------------------------------
    def _process_staged_sell(self, pos: Position, price: float, ts: datetime) -> None:
        if not pos.staged_sell_active:
            return
        if pos.staged_sell_start is None:
            return

        elapsed = (ts - pos.staged_sell_start).total_seconds() / 60
        expected_steps = int(elapsed / config.PAIR_SELL_INTERVAL_MIN)

        while pos.staged_sell_step < expected_steps and pos.staged_sell_remaining > 0:
            sell_qty = pos.staged_sell_remaining * config.PAIR_SELL_REMAINING_PCT
            if sell_qty < 0.001:
                sell_qty = pos.staged_sell_remaining
            self._partial_sell(pos.ticker, sell_qty, price, ts, "staged_sell")
            pos.staged_sell_remaining -= sell_qty
            pos.staged_sell_step += 1

        if pos.staged_sell_remaining <= 0:
            pos.staged_sell_active = False

    # ----------------------------------------------------------
    # Carry position handling
    # ----------------------------------------------------------
    def _handle_carry_positions(
        self, sym_bars_day: dict[str, list[dict]], day_ts: list,
        market_mode: str,
        prev_close: dict[str, float] | None = None,
    ) -> None:
        if not any(pos.carry for pos in self.positions.values()):
            return

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

            net_pnl_pct = self._calc_net_profit_pct(pos, price)
            if net_pnl_pct >= config.TAKE_PROFIT_PCT:
                self._sell_all(ticker, price, first_ts, "carry_sell")
                continue

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
        # 1. Stop loss
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

        # 2. Time stop
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

        # 3. CONL sell (KST 17:30 이후만)
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

        # 4. COIN sell (KST 17:30 이후만)
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

        # 5. Staged sell
        for ticker in list(self.positions.keys()):
            pos = self.positions.get(ticker)
            if pos is None or not pos.staged_sell_active:
                continue
            price = cur_prices.get(ticker)
            if price is None:
                continue
            self._process_staged_sell(pos, price, ts)

        # 6. Twin SELL
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

            first_sell_qty = pos.total_qty * config.PAIR_SELL_FIRST_PCT
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
        gld_blocks_non_conditional = gold_warning
        if gld_blocks_non_conditional:
            self.skipped_gold_bars += 1

        # Twin ENTRY
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

        # DCA
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

        # CONL buy (KST 17:30 이후만)
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

        # COIN buy (KST 17:30 이후만)
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

        # Bearish
        if market_mode == "bearish":
            bearish_sig = sigs.get("bearish", {})
            if bearish_sig.get("buy_brku", False):
                if self._can_buy("BRKU", ts):
                    price = cur_prices.get("BRKU")
                    if price is not None:
                        brku_amount = self.initial_capital * config.BRKU_WEIGHT_PCT / 100
                        self._buy(
                            "BRKU", price, ts, "bearish",
                            amount_usd=brku_amount,
                        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _calc_total_equity(self, cur_prices: dict[str, float]) -> float:
        return self._snapshot_equity(cur_prices)

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------
    def print_report(self) -> None:
        if not self.equity_curve:
            print("  데이터 없음")
            return

        final = self.equity_curve[-1][1]
        total_ret = (final - self.initial_capital) / self.initial_capital * 100

        sells = [t for t in self.trades if t.side == "SELL"]
        wins = [t for t in sells if t.pnl_usd > 0]
        losses = [t for t in sells if t.pnl_usd < 0]
        breakeven = [t for t in sells if t.pnl_usd == 0]
        total_pnl = sum(t.pnl_usd for t in sells)

        mdd = self.calc_mdd()
        sharpe = self.calc_sharpe()
        total_fees = self.total_buy_fees + self.total_sell_fees

        exit_stats: dict[str, dict] = {}
        for t in sells:
            key = t.exit_reason or "unknown"
            if key not in exit_stats:
                exit_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            exit_stats[key]["count"] += 1
            exit_stats[key]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                exit_stats[key]["wins"] += 1

        sig_stats: dict[str, dict] = {}
        for t in sells:
            key = t.signal_type
            if key not in sig_stats:
                sig_stats[key] = {"count": 0, "pnl": 0.0, "wins": 0}
            sig_stats[key]["count"] += 1
            sig_stats[key]["pnl"] += t.pnl_usd
            if t.pnl_usd > 0:
                sig_stats[key]["wins"] += 1

        buys = [t for t in self.trades if t.side == "BUY"]
        dca_buys = [t for t in buys if t.dca_level > 0]
        initial_buys = [t for t in buys if t.dca_level == 0]

        print()
        print("=" * 65)
        print("    PTJ 매매법 v2 백테스트 리포트 (USD)")
        print("=" * 65)
        print(f"  기간         : {self.equity_curve[0][0]} ~ {self.equity_curve[-1][0]}")
        print(f"  거래일       : {self.total_trading_days}일")
        print(f"  초기 자금    : ${self.initial_capital:>12,.2f}")
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

        print("  [수수료]")
        print(f"    매수 수수료 : ${self.total_buy_fees:>10,.2f}")
        print(f"    매도 수수료 : ${self.total_sell_fees:>10,.2f}")
        print(f"    총 수수료   : ${total_fees:>10,.2f}")
        print("-" * 65)

        print("  [DCA 통계]")
        print(f"    초기 진입   : {len(initial_buys)}회")
        print(f"    물타기      : {len(dca_buys)}회")
        if dca_buys:
            max_dca = max(t.dca_level for t in dca_buys)
            total_dca_usd = sum(t.amount_usd for t in dca_buys)
            print(f"    최대 DCA 단계: {max_dca}")
            print(f"    DCA 총 투입 : ${total_dca_usd:>10,.2f}")
        print("-" * 65)

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
        out_path = config.RESULTS_DIR / "backtests" / "backtest_v2_trades.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  거래 로그: {out_path}")
        return out_path


# ============================================================
# Convenience wrapper
# ============================================================
def run_backtest_v2(
    params=None,
    start_date: date = date(2025, 2, 18),
    end_date: date = date(2026, 2, 17),
    use_fees: bool = True,
    verbose: bool = True,
) -> BacktestEngineV2:
    """v2 백테스트를 실행하고 엔진을 반환한다."""
    engine = BacktestEngineV2(
        start_date=start_date,
        end_date=end_date,
        use_fees=use_fees,
        params=params,
    )
    engine.run(verbose=verbose)
    return engine


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
