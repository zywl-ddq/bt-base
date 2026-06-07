"""
Module:    base/v2_adapter
Purpose:   Protocol adapter bridging nt-base SignalStrategy to trading-v2 AlphaSignal.
           Translates bar_data dict interface to positional on_bar() call.

Class: V2SignalAdapter
  __init__(alpha_signal, strategy_id, symbol, timeframe)
      alpha_signal: AlphaSignal   鈥?trading-v2 signal generator
      strategy_id: str            鈥?unique strategy identifier
      symbol: str                 鈥?trading pair
      timeframe: str              鈥?bar timeframe for subscription

  Implements nt-base SignalStrategy protocol:
      strategy_id    -> returns the configured ID
      subscriptions  -> BarSubscription list from factor_names
      on_bar(dict)   -> extracts close/high/low/ts_ns/factors,
                        pushes factors via set_factor_value(),
                        calls alpha_signal.on_bar(close, high, ...)
      on_shutdown()  -> delegates to alpha_signal
      get_diagnostics() -> delegates to alpha_signal

Protocol Translation:
  nt-base bar_data dict keys:
      "close", "high", "low"  -> direct float conversion
      "ts_ns"                 -> passes through
      "factors"               -> dict of {name: value}, pushed before on_bar
      "btc_close" (optional)  -> defaults to close if missing
      "delta_buy"/"delta_sell" -> defaults to 0.0 if missing

  trading-v2 AlphaSignal.on_bar(close, high, low,
                                 delta_buy_vol, delta_sell_vol,
                                 btc_close, ts_ns) -> StrategySignal

Design Decision:
  The adapter owns factor value pushing. This ensures nt-base's factor
  computation is decoupled from AlphaSignal's factor consumption.
  AlphaSignal.on_bar() sees factor values from bar_data['factors']
  as if they were pushed by a backtest or live factor loop.

Author:    nt-base system
Version:   1.0.0
"""
from __future__ import annotations
"""V2SignalAdapter 鈥?wraps trading-v2 AlphaSignal in nt-base's SignalStrategy.

Bridges the protocol gap:
  nt-base  : on_bar(bar_data: dict) -> StrategySignal | None
  trading-v2: on_bar(close, high, low, delta_buy, delta_sell, btc_close, ts_ns)

Strategy code (AlphaSignal) remains pure 鈥?no DB, no NT, no I/O.
"""
from base.signal_protocol import StrategySignal, BarSubscription


class V2SignalAdapter:
    """Adapts trading-v2 AlphaSignal to nt-base's SignalStrategy protocol."""

    def __init__(self, alpha_signal, strategy_id: str,
                 symbol: str, timeframe: str = "1m"):
        self._signal = alpha_signal
        self._id = strategy_id
        self._symbol = symbol
        self._tf = timeframe
        self._subs = [
            BarSubscription(
                symbol=symbol,
                timeframe=timeframe,
                factors=alpha_signal.factor_names,
            )
        ]

    # 鈹€鈹€ SignalStrategy protocol 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @property
    def strategy_id(self) -> str:
        return self._id

    @property
    def subscriptions(self) -> list[BarSubscription]:
        return self._subs

    def on_bar(self, bar_data: dict) -> StrategySignal | None:
        """Translate nt-base bar dict 鈫?trading-v2 on_bar call."""
        close = float(bar_data.get("close", 0))
        high = float(bar_data.get("high", close))
        low = float(bar_data.get("low", close))
        ts_ns = bar_data.get("ts_ns", 0)

        # Push factor values to AlphaSignal
        factors = bar_data.get("factors", {})
        for fname, value in factors.items():
            self._signal.set_factor_value(fname, ts_ns, float(value))

        # btc_close: if nt-base doesn't provide it, use SOL close as proxy
        btc_close = float(bar_data.get("btc_close", close))

        # delta vol: nt-base doesn't track tick delta on 1m bars
        delta_buy = float(bar_data.get("delta_buy", 0.0))
        delta_sell = float(bar_data.get("delta_sell", 0.0))

        result = self._signal.on_bar(
            close=close, high=high, low=low,
            delta_buy_vol=delta_buy, delta_sell_vol=delta_sell,
            btc_close=btc_close, ts_ns=ts_ns,
        )

        if result is None:
            return None
        # Both projects use @dataclass StrategySignal(direction, reason) - compatible
        return StrategySignal(direction=result.direction, reason=result.reason)

    def on_tick(self, price, size, is_buyer, ts_ns, symbol=""):
        if hasattr(self._signal, "on_tick"):
            self._signal.on_tick(price, size, is_buyer, ts_ns, symbol)

    def on_shutdown(self) -> None:
        if hasattr(self._signal, 'on_shutdown'):
            self._signal.on_shutdown()

    def get_diagnostics(self) -> dict:
        return self._signal.get_diagnostics()

    # 鈹€鈹€ Access 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

    @property
    def inner(self):
        return self._signal
