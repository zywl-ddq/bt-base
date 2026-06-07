"""
Module:    base/slot
Purpose:   Runtime state container for a registered trading strategy.
           StrategySlot holds configuration, position state, risk limits,
           and Telegram notification config for one strategy instance.

Data Class: StrategySlot
  Configuration:
    strategy_id: str            unique instance identifier
    strategy: SignalStrategy    the signal generator implementation
    subscriptions: list[BarSubscription]  bar data subscriptions
    stop_pct: float             hard stop-loss percentage (e.g. 0.03 = 3%)
    take_pct: float             take-profit percentage
    max_hold_sec: int           maximum position hold duration
    max_daily_loss_pct: float   daily loss circuit breaker
    cooldown_sec: float         minimum interval between trades
    leverage: int               position leverage multiplier
    position_size_pct: float    fraction of equity per position
    symbol: str                 trading pair (e.g. "SOLUSDT-PERP")
    telegram_bot_token: str     per-strategy Telegram bot token
    telegram_chat_id: str       per-strategy Telegram chat ID

  Runtime State:
    has_position: bool          currently in a position
    entry_price: float          position entry price
    entry_side: str             "LONG" or "SHORT"
    entry_time: float           epoch timestamp of entry
    last_trade_time: float      epoch timestamp of last trade
    daily_pnl: float            cumulative daily PnL
    daily_start_equity: float   equity at start of day
    tripped: bool               circuit breaker triggered

  Computed:
    held_sec: float             seconds since position opened

Invariant:
  has_position == False  =>  entry_price == 0.0 and entry_side == ""
  tripped == True        =>  no new trades will be executed

Author:    nt-base system
Version:   1.1.0
"""
from __future__ import annotations
"""StrategySlot 鈥?runtime state for a registered strategy."""
from dataclasses import dataclass, field
import time
from base.signal_protocol import SignalStrategy, BarSubscription


@dataclass
class StrategySlot:
    strategy_id: str
    strategy: SignalStrategy
    subscriptions: list[BarSubscription] = field(default_factory=list)
    stop_pct: float = 0.03
    take_pct: float = 0.06
    max_hold_sec: int = 3600
    max_daily_loss_pct: float = 0.05
    cooldown_sec: float = 60.0
    leverage: int = 2
    position_size_pct: float = 0.20
    symbol: str = ""

    # Telegram notification (per-strategy bot)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Runtime state
    has_position: bool = False
    entry_price: float = 0.0
    entry_side: str = ""
    entry_time: float = 0.0
    last_trade_time: float = 0.0
    daily_pnl: float = 0.0
    daily_start_equity: float = 0.0
    tripped: bool = False

    @property
    def held_sec(self) -> float:
        if not self.has_position:
            return 0.0
        return time.time() - self.entry_time

    def reset_position(self):
        self.has_position = False
        self.entry_price = 0.0
        self.entry_side = ""
        self.entry_time = 0.0

    def open_position(self, side: str, price: float):
        self.has_position = True
        self.entry_side = side
        self.entry_price = price
        self.entry_time = time.time()
