"""Backtest strategy loader — loads strategy config from DB, creates instances.

Unlike nt-base's RegistrationManager (which polls every 5s for hot-reload),
this is a one-shot loader for backtest runs.
"""
import json
import logging
import asyncpg

from base.slot import StrategySlot
from base.v2_adapter import V2SignalAdapter
from strategy.alpha_signal_v3 import AlphaSignal

logger = logging.getLogger(__name__)


class BacktestStrategyLoader:
    """Load a strategy from strategy_instances table for backtesting."""

    def __init__(self, registry, pool: asyncpg.Pool,
                 symbol: str = "SOLUSDT-PERP", timeframe: str = "1m"):
        self._registry = registry
        self._pool = pool
        self._symbol = symbol
        self._timeframe = timeframe

    async def load(self, instance_id: str) -> StrategySlot | None:
        """Load and activate a single strategy from DB.

        Returns the StrategySlot if successful, None if strategy not found.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM strategy_instances WHERE instance_id = $1",
                instance_id,
            )

        if row is None:
            logger.error(f"Strategy not found: {instance_id}")
            return None

        params = row["params"] if isinstance(row["params"], dict) else json.loads(row["params"] or "{}")
        token = row["telegram_bot_token"] or ""
        chat_id = row["telegram_chat_id"] or ""

        logger.info(f"Loading strategy: {instance_id} params={json.dumps(params, default=str)[:200]}")

        try:
            adaptive = params.get("adaptive", {})
            alpha = AlphaSignal(
                gate_factor=params.get("gate_factor", "trend_regime"),
                factor_1=params.get("factor_1", "cvd_divergence"),
                direction_1=params.get("direction_1", -1),
                weight_1=params.get("weight_1", 1.0),
                factor_2=params.get("factor_2", "residual_momentum"),
                direction_2=params.get("direction_2", 1),
                weight_2=params.get("weight_2", 0.5),
                factor_3=params.get("factor_3", "channel_breakout"),
                direction_3=params.get("direction_3", 1),
                weight_3=params.get("weight_3", 1.0),
                signal_threshold=params.get("signal_threshold", 0.28),
                atr_period=params.get("atr_period", 30),
                btc_shock_long=params.get("btc_shock_long", 0.0085),
                btc_shock_short=params.get("btc_shock_short", 0.0075),
                time_limit_long=params.get("time_limit_long", 40),
                time_limit_short=params.get("time_limit_short", 18),
                max_hold_minutes=params.get("max_hold_minutes", 40),
                breakeven_atr_mult=params.get("breakeven_atr_mult", 1.4),
                trail_trigger_atr=params.get("trail_trigger_atr", 2.0),
                trail_stop_atr=params.get("trail_stop_atr", 1.0),
                adaptive=adaptive,
            )

            adapter = V2SignalAdapter(alpha, instance_id, self._symbol, self._timeframe)
            slot = StrategySlot(
                strategy_id=instance_id,
                strategy=adapter,
                subscriptions=adapter.subscriptions,
                stop_pct=params.get("stop_pct", 0.03),
                take_pct=params.get("take_pct", 0.06),
                max_hold_sec=params.get("max_hold_sec", 3600),
                cooldown_sec=params.get("cooldown_sec", 60.0),
                leverage=params.get("leverage", 2),
                position_size_pct=params.get("position_size_pct", 0.20),
                symbol=self._symbol,
                telegram_bot_token=token,
                telegram_chat_id=chat_id,
            )

            self._registry.register(slot)
            logger.info(f"Strategy loaded: {instance_id} factors={alpha.factor_names}")
            return slot

        except Exception as e:
            logger.error(f"Failed to load strategy {instance_id}: {e}", exc_info=True)
            return None

    async def load_all_active(self) -> list[StrategySlot]:
        """Load all strategies with status='active' from DB."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT instance_id FROM strategy_instances WHERE status = 'active'"
            )

        slots = []
        for row in rows:
            slot = await self.load(row["instance_id"])
            if slot:
                slots.append(slot)

        logger.info(f"Loaded {len(slots)} active strategies")
        return slots
