"""
Module:    base/signal_protocol
Purpose:   Protocol definitions for the strategy execution framework.
           Defines the SignalStrategy protocol, StrategySignal dataclass,
           and BarSubscription data structure 鈥?the contract between
           strategy implementations and the trading base.

Interface:
  SignalStrategy (Protocol)  鈥?strategy must implement:
      strategy_id: str         unique identifier
      subscriptions: list      bar subscriptions (symbol, timeframe, factors)
      on_bar(dict) -> Signal   process bar, return trading signal
      on_shutdown()            cleanup
      get_diagnostics() -> dict monitoring data

  StrategySignal (dataclass) 鈥?direction (1/-1/0) + reason (str)
  BarSubscription (dataclass) 鈥?symbol, timeframe, factors list

Design Decision:
  Strategies are PURE signal generators. They do NOT hold state about
  positions, orders, or account balances. That belongs to the base layer.
  This separation enables independent testing and factor reuse.

Author:    nt-base system
Version:   1.0.0
"""
from __future__ import annotations
"""SignalStrategy protocol — interface between strategies and trading base."""
from dataclasses import dataclass
from typing import Protocol


class SignalKind:
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class StrategySignal:
    direction: int    # 1=LONG, -1=SHORT, 0=FLAT
    reason: str = ""


@dataclass
class BarSubscription:
    symbol: str
    timeframe: str
    factors: list[str]

    def __hash__(self):
        return hash((self.symbol, self.timeframe))


class SignalStrategy(Protocol):
    @property
    def strategy_id(self) -> str: ...
    @property
    def subscriptions(self) -> list[BarSubscription]: ...
    def on_bar(self, bar_data: dict) -> StrategySignal | None: ...
    def on_shutdown(self) -> None: ...
    def get_diagnostics(self) -> dict: ...
