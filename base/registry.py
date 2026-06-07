"""
Module:    base/registry
Purpose:   Central strategy registry managing StrategySlot lifecycle.
           Maintains slot index and factor-subscription index for efficient
           bar dispatch lookups.

Interface: StrategyRegistry
  register(slot)              鈥?add strategy, update factor index
  unregister(strategy_id)     鈥?remove strategy, clean factor index
  get_slots(symbol, tf)       鈥?find slots subscribed to a specific bar type
  all_slots()                 鈥?all registered slots
  get_active_slots()          鈥?slots currently in a position
  active_factors()            鈥?set of factor names needed by all strategies
  summary()                   鈥?diagnostic snapshot

Performance:
  get_slots() is called on every 1m bar 鈥?O(n) where n = registered strategies.
  Factor index avoids computing unused factors.

Thread Safety:
  All mutations (register/unregister) happen in the async main thread.
  Read operations are safe for concurrent access from the bar dispatch loop.

Author:    nt-base system
Version:   1.0.0
"""
from __future__ import annotations
"""StrategyRegistry — manages strategy slots and factor subscriptions."""
from base.slot import StrategySlot


class StrategyRegistry:
    def __init__(self):
        self._slots: dict[str, StrategySlot] = {}
        self._factor_index: dict[str, set[str]] = {}

    def register(self, slot: StrategySlot) -> None:
        if slot.strategy_id in self._slots:
            raise ValueError(f"Strategy {slot.strategy_id} already registered")
        self._slots[slot.strategy_id] = slot
        self._update_factor_index(slot, add=True)

    def unregister(self, strategy_id: str) -> None:
        slot = self._slots.pop(strategy_id, None)
        if slot:
            self._update_factor_index(slot, add=False)

    def _update_factor_index(self, slot: StrategySlot, add: bool):
        for sub in slot.subscriptions:
            for fname in sub.factors:
                if add:
                    self._factor_index.setdefault(fname, set()).add(slot.strategy_id)
                else:
                    s = self._factor_index.get(fname)
                    if s:
                        s.discard(slot.strategy_id)
                        if not s:
                            del self._factor_index[fname]

    def get_slots(self, symbol: str, timeframe: str) -> list[StrategySlot]:
        result = []
        for slot in self._slots.values():
            for sub in slot.subscriptions:
                if sub.symbol == symbol and sub.timeframe == timeframe:
                    result.append(slot)
                    break
        return result

    def all_slots(self) -> list[StrategySlot]:
        return list(self._slots.values())

    def get_active_slots(self) -> list[StrategySlot]:
        return [s for s in self._slots.values() if s.has_position and not s.tripped]

    def active_factors(self) -> set[str]:
        return set(self._factor_index.keys())

    def get_slot(self, strategy_id: str) -> StrategySlot | None:
        return self._slots.get(strategy_id)

    @property
    def count(self) -> int:
        return len(self._slots)

    def summary(self) -> dict:
        return {
            "total": len(self._slots),
            "active": len(self.get_active_slots()),
            "factors": len(self._factor_index),
            "strategies": list(self._slots.keys()),
        }
