"""BacktestEngine — walk-forward simulation with factor dispatch + fill simulation.

Mirrors nt-base's main.py bar dispatch loop, but runs offline over
historical data. No NautilusTrader dependency — fills are simulated
at bar close price with taker fee deduction.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from base.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Taker fee: 0.04% per round-trip (0.02% per side)
TAKER_FEE = 0.0004


@dataclass
class SimPosition:
    """Simulated position state for one strategy slot."""
    slot_id: str
    is_long: bool = True
    entry_price: float = 0.0
    quantity: float = 0.0
    entry_bar_ts: int = 0
    bars_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = float("inf")

    def reset(self):
        self.entry_price = 0.0
        self.quantity = 0.0
        self.entry_bar_ts = 0
        self.bars_held = 0
        self.highest_price = 0.0
        self.lowest_price = float("inf")


@dataclass
class BacktestResult:
    strategy_id: str
    metrics: dict
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    """Walk-forward backtest over historical bar data.

    Feeds bars to registered strategies, simulates fills, tracks equity.
    """

    def __init__(self, registry, factor_cache: dict,
                 initial_equity: float = 1000.0,
                 taker_fee: float = TAKER_FEE):
        self._registry = registry
        self._factor_cache = factor_cache
        self._initial_equity = initial_equity
        self._taker_fee = taker_fee

        # Per-slot state
        self._positions: dict[str, SimPosition] = {}
        self._equity: dict[str, float] = {}
        self._last_trade_time: dict[str, float] = {}
        # Track the confidence value separately per slot
        self._slot_confidence: dict[str, float] = {}

        # Trade history per slot
        self._trades: dict[str, list[dict]] = {}
        self._equity_curves: dict[str, list[float]] = {}

    def run(self, df_bars: pd.DataFrame, df_btc: pd.DataFrame | None = None,
            df_delta: pd.DataFrame | None = None) -> list[BacktestResult]:
        """Run backtest over all bars.

        Args:
            df_bars: SOL 1m bars with DatetimeIndex, columns: open,high,low,close,volume
            df_btc: BTC 1m bars with close column (optional, for BTC shock detection)
            df_delta: Tick delta 1m buckets with delta column (optional, for CVD)

        Returns:
            list of BacktestResult, one per strategy slot
        """
        slots = self._registry.all_slots()
        if not slots:
            logger.warning("No strategies registered for backtest")
            return []

        # Initialize state for each slot
        for slot in slots:
            sid = slot.strategy_id
            self._positions[sid] = SimPosition(slot_id=sid)
            self._equity[sid] = self._initial_equity
            self._last_trade_time[sid] = 0.0
            self._slot_confidence[sid] = 0.0
            self._trades[sid] = []
            self._equity_curves[sid] = [self._initial_equity]

        n_bars = len(df_bars)
        logger.info(f"Backtest starting: {n_bars} bars, {len(slots)} strategies, "
                     f"equity={self._initial_equity}")

        # Rolling buffers for ATR / BTC shock
        highs_buf: deque[float] = deque(maxlen=30)
        lows_buf: deque[float] = deque(maxlen=30)
        btc_closes: deque[float] = deque(maxlen=5)

        for i, (ts, bar) in enumerate(df_bars.iterrows()):
            close = float(bar["close"])
            high = float(bar.get("high", close))
            low = float(bar.get("low", close))
            ts_ns = int(ts.timestamp() * 1_000_000_000)

            highs_buf.append(high)
            lows_buf.append(low)

            # BTC price
            btc_close = close
            if df_btc is not None and ts in df_btc.index:
                btc_close = float(df_btc.loc[ts, "close"])
            elif "btc_close" in bar:
                try:
                    v = float(bar["btc_close"])
                    if not pd.isna(v):
                        btc_close = v
                except (ValueError, TypeError):
                    pass
            btc_closes.append(btc_close)

            # Tick delta for this bar
            delta_buy = 0.0
            delta_sell = 0.0
            if df_delta is not None and ts in df_delta.index:
                d = float(df_delta.loc[ts, "delta"])
                if d > 0:
                    delta_buy = d
                else:
                    delta_sell = abs(d)

            # Process each strategy slot
            for slot in slots:
                sid = slot.strategy_id

                # Build bar_data dict (same format as nt-base main.py)
                bar_data = {
                    "close": close,
                    "high": high,
                    "low": low,
                    "ts_ns": ts_ns,
                    "btc_close": btc_close,
                    "delta_buy": delta_buy,
                    "delta_sell": delta_sell,
                    "factors": {},  # populated below
                }

                # Push factor values into the strategy adapter
                # (V2SignalAdapter.on_bar will push factors before calling AlphaSignal.on_bar)
                for fname in slot.subscriptions[0].factors if slot.subscriptions else []:
                    key = (fname, ts_ns)
                    if key in self._factor_cache:
                        bar_data["factors"][fname] = self._factor_cache[key]

                pos = self._positions[sid]

                # Check pending tick exit first
                pending_reason = None
                adapter = slot.strategy
                if hasattr(adapter, '_signal') and hasattr(adapter._signal, '_pending_tick_exit'):
                    pending_reason = adapter._signal._pending_tick_exit

                if pending_reason:
                    # Close position via tick exit
                    if pos.quantity > 0:
                        pnl = self._close_position(sid, pos, close, ts_ns, pending_reason)
                        self._trades[sid].append(pnl)
                        self._equity[sid] += pnl["realized_pnl"]
                    pos.reset()
                    # Clear pending tick exit
                    if hasattr(adapter, '_signal'):
                        adapter._signal._pending_tick_exit = None
                    self._equity_curves[sid].append(self._equity[sid])
                    continue

                # Call strategy on_bar
                signal = slot.strategy.on_bar(bar_data)

                # Track confidence for PnL tracking
                if hasattr(adapter, '_signal') and hasattr(adapter._signal, '_signal'):
                    conf = adapter._signal._signal.confidence
                    self._slot_confidence[sid] = conf

                # Increment bars_held for open positions
                if pos.quantity > 0:
                    pos.bars_held += 1
                    if pos.is_long:
                        pos.highest_price = max(pos.highest_price, close)
                    else:
                        pos.lowest_price = min(pos.lowest_price, close)

                if signal is None:
                    self._equity_curves[sid].append(self._equity[sid])
                    continue

                if signal.direction != 0 and pos.quantity == 0:
                    # --- ENTRY ---
                    equity = self._equity[sid]
                    notional = equity * slot.position_size_pct * slot.leverage
                    qty = notional / close
                    fee = notional * self._taker_fee

                    pos.is_long = (signal.direction > 0)
                    pos.entry_price = close
                    pos.quantity = qty
                    pos.entry_bar_ts = ts_ns
                    pos.bars_held = 0
                    pos.highest_price = close
                    pos.lowest_price = close

                    # Activate tick exit manager if available
                    if hasattr(adapter, '_signal') and hasattr(adapter._signal, '_tick_exits'):
                        adapter._signal._tick_exits.open_position(close, pos.is_long, symbol="SOLUSDT-PERP")

                    # Deduct fee from equity immediately
                    self._equity[sid] -= fee

                    self._last_trade_time[sid] = time.time()
                    logger.debug(f"ENTRY {sid} {'LONG' if pos.is_long else 'SHORT'} "
                                 f"px={close:.4f} qty={qty:.4f} reason={signal.reason}")

                elif signal.direction == 0 and pos.quantity > 0 and signal.reason not in ("", "hold"):
                    # --- EXIT ---
                    pnl = self._close_position(sid, pos, close, ts_ns, signal.reason)
                    self._trades[sid].append(pnl)
                    self._equity[sid] += pnl["realized_pnl"]
                    pos.reset()

                    # Deactivate tick exit manager
                    if hasattr(adapter, '_signal') and hasattr(adapter._signal, '_tick_exits'):
                        adapter._signal._tick_exits.close_position()

                    logger.debug(f"EXIT {sid} px={close:.4f} pnl={pnl['realized_pnl']:.4f} "
                                 f"reason={signal.reason}")

                # Track equity curve (even when holding)
                self._equity_curves[sid].append(self._equity[sid])

            # Progress
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"  bar {i+1}/{n_bars} ({ts})")

        # Close any remaining open positions
        last_close = float(df_bars.iloc[-1]["close"])
        last_ts = int(df_bars.index[-1].timestamp() * 1_000_000_000)
        for slot in slots:
            sid = slot.strategy_id
            pos = self._positions[sid]
            if pos.quantity > 0:
                pnl = self._close_position(sid, pos, last_close, last_ts, "end_of_period")
                self._trades[sid].append(pnl)
                self._equity[sid] += pnl["realized_pnl"]
                pos.reset()
                self._equity_curves[sid].append(self._equity[sid])

        # Build results
        hours = n_bars / 60.0  # 1m bars
        results = []
        for slot in slots:
            sid = slot.strategy_id
            trades = self._trades.get(sid, [])
            eq_curve = self._equity_curves.get(sid, [])
            metrics = compute_metrics(trades, eq_curve, self._initial_equity, int(hours))
            results.append(BacktestResult(
                strategy_id=sid,
                metrics=metrics,
                trades=trades,
                equity_curve=eq_curve,
            ))
            logger.info(f"Result {sid}: pnl={metrics['total_pnl']:.2f} "
                        f"({metrics['pnl_pct']:.2f}%) trades={metrics['n_trades']} "
                        f"sharpe={metrics['sharpe']:.2f} dd={metrics['max_dd_pct']:.1f}%")

        return results

    def _close_position(self, sid: str, pos: SimPosition,
                        exit_price: float, exit_ts: int,
                        reason: str) -> dict:
        """Calculate realized PnL for closing a position."""
        if pos.is_long:
            raw_return = (exit_price - pos.entry_price) / pos.entry_price
        else:
            raw_return = (pos.entry_price - exit_price) / pos.entry_price

        notional = pos.quantity * exit_price
        gross_pnl = raw_return * notional
        fee = notional * self._taker_fee
        pnl = gross_pnl - fee

        return {
            "entry_ts": pos.entry_bar_ts,
            "exit_ts": exit_ts,
            "side": "LONG" if pos.is_long else "SHORT",
            "entry": round(pos.entry_price, 4),
            "exit": round(exit_price, 4),
            "quantity": round(pos.quantity, 4),
            "bars_held": pos.bars_held,
            "realized_pnl": round(pnl, 4),
            "gross_pnl": round(gross_pnl, 4),
            "fee": round(fee, 4),
            "return_pct": round(raw_return * 100, 4),
            "exit_reason": reason,
            "confidence": round(self._slot_confidence.get(sid, 0.0), 4),
        }
