"""Strategy Signal Protocol — the interface between strategies and trading base.

Any strategy that implements this protocol can run on the TradingBase.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class SignalKind:
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class StrategySignal:
    """Output from a strategy's decision logic."""
    direction: int    # 1=LONG, -1=SHORT, 0=HOLD/FLAT
    reason: str = ""  # for logging/audit

    @property
    def is_entry(self) -> bool:
        return self.direction != 0

    @property
    def is_exit(self) -> bool:
        return self.direction == 0


class SignalStrategy(Protocol):
    """Protocol that all trading strategies must implement.

    The strategy receives market data and current position state,
    and returns a Signal. It does NOT interact with NT or the exchange.
    """

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    def factor_names(self) -> list[str]:
        """Factor names this strategy depends on (for pre-computation)."""
        ...

    def set_factor_value(self, name: str, ts_ns: int, value: float) -> None:
        """Receive a factor value update."""
        ...

    def on_bar(self, close: float, high: float, low: float,
               delta_buy_vol: float, delta_sell_vol: float,
               btc_close: float, ts_ns: int) -> StrategySignal:
        """Called on each new bar. Returns trading signal."""
        ...

    def on_tick(self, price: float, size: float,
                is_buyer: bool, ts_ns: int) -> None:
        """Called on each trade tick (optional, for CVD tracking)."""
        ...

    def get_diagnostics(self) -> dict:
        """Return current state for monitoring."""
        ...
