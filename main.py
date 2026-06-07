"""bt-base — backtest base entrypoint.

Loads strategies from TimescaleDB (same strategy_instances table as nt-base),
runs walk-forward backtest over historical bar data, outputs JSON report.

Usage:
    python main.py --strategy AlphaV2-005 --hours 168
    python main.py --strategy AlphaV2-005 --hours 24 --output result.json
    python main.py --all-active --hours 168
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/root/trading-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from shared.env import cfg, assert_required
from shared.log import setup_logging
from shared.db import get_pool, close_pool
from base.registry import StrategyRegistry
from base.registration import BacktestStrategyLoader
from base.backtest_engine import BacktestEngine
from factor.compute import compute_factor_history

logger = setup_logging("bt_base")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ─────────────────────────────────────────────

async def load_bars(pool, symbol: str, hours: int) -> pd.DataFrame:
    """Load 1m bars from TimescaleDB."""
    rows = await pool.fetch(
        """SELECT ts, open, high, low, close, volume
           FROM bars WHERE symbol=$1 AND timeframe='1m'
           AND ts >= NOW() - INTERVAL '1 hour' * $2
           ORDER BY ts""",
        symbol, hours,
    )
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    logger.info(f"Loaded {len(df)} bars for {symbol} ({hours}h)")
    return df


async def load_tick_delta(pool, symbol: str, hours: int) -> pd.DataFrame | None:
    """Load 1m aggregated tick delta for CVD factor."""
    rows = await pool.fetch(
        """SELECT time_bucket('1 minute', ts_event) AS ts,
                  SUM(CASE WHEN aggressor='BUY' THEN size ELSE 0 END) AS buy_vol,
                  SUM(CASE WHEN aggressor='SELL' THEN size ELSE 0 END) AS sell_vol
           FROM ticks WHERE symbol=$1
           AND ts_event >= NOW() - INTERVAL '1 hour' * $2
           GROUP BY 1 ORDER BY ts""",
        symbol, hours,
    )
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        df["delta"] = (df["buy_vol"] - df["sell_vol"]).astype(float)
        logger.info(f"Loaded {len(df)} tick delta buckets")
        return df
    return None


async def load_btc_bars(pool, hours: int) -> pd.DataFrame | None:
    """Load BTC 1m bars for BTC shock detection and residual_momentum."""
    rows = await pool.fetch(
        """SELECT ts, close FROM bars
           WHERE symbol='BTCUSDT-PERP' AND timeframe='1m'
           AND ts >= NOW() - INTERVAL '1 hour' * $1
           ORDER BY ts""",
        hours,
    )
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        df["close"] = df["close"].astype(float)
        logger.info(f"Loaded {len(df)} BTC bars")
        return df
    return None


# ── Factor pre-computation ───────────────────────────────────

def precompute_factors(factor_names: list[str], df_bars: pd.DataFrame,
                       df_btc_5m: pd.DataFrame | None = None) -> dict:
    """Batch compute all factor values over the full bar history.

    Returns:
        dict[(factor_name, ts_ns)] → value
    """
    factor_series = {}

    for fname in factor_names:
        # residual_momentum needs 5m bars + BTC data
        if fname == "residual_momentum":
            if df_btc_5m is not None and len(df_btc_5m) > 0:
                df_sol_5m = df_bars.resample("5min").last().dropna()
                common = df_sol_5m.index.intersection(df_btc_5m.index)
                if len(common) > 30:
                    df_5m = df_sol_5m.loc[common].copy()
                    df_5m["btc_close"] = df_btc_5m["close"].loc[common]
                    try:
                        series = compute_factor_history(fname, df_5m)
                        factor_series[fname] = series
                        logger.info(f"  {fname}: {len(series.dropna())} values")
                    except Exception as e:
                        logger.warning(f"  {fname} failed: {e}")
                        factor_series[fname] = pd.Series(0.0, index=df_bars.index)
                else:
                    factor_series[fname] = pd.Series(0.0, index=df_bars.index)
            else:
                factor_series[fname] = pd.Series(0.0, index=df_bars.index)
        else:
            try:
                series = compute_factor_history(fname, df_bars)
                if isinstance(series, dict):
                    for sub_name, sub_series in series.items():
                        factor_series[sub_name] = sub_series
                        logger.info(f"  {sub_name}: {len(sub_series.dropna())} values")
                else:
                    factor_series[fname] = series
                    logger.info(f"  {fname}: {len(series.dropna())} values")
            except Exception as e:
                logger.warning(f"  {fname} failed: {e}")
                factor_series[fname] = pd.Series(0.0, index=df_bars.index)

    # Build cache: (factor_name, ts_ns) → value
    cache = {}
    for fname, series in factor_series.items():
        for ts, val in series.dropna().items():
            cache[(fname, int(ts.timestamp() * 1_000_000_000))] = float(val)

    logger.info(f"Factor cache built: {len(cache)} entries")
    return cache


# ── Main ─────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="bt-base — backtest base")
    parser.add_argument("--strategy", type=str, default="AlphaV2-005",
                        help="Strategy instance_id to backtest")
    parser.add_argument("--all-active", action="store_true",
                        help="Backtest all active strategies")
    parser.add_argument("--hours", type=int, default=168,
                        help="Backtest window in hours (default: 168 = 7 days)")
    parser.add_argument("--symbol", type=str, default="SOLUSDT-PERP",
                        help="Trading symbol")
    parser.add_argument("--output", type=str, default="",
                        help="Output file path (default: auto-generated)")
    parser.add_argument("--initial-equity", type=float, default=1000.0,
                        help="Starting equity (default: 1000)")
    args = parser.parse_args()

    assert_required()

    pool = await get_pool()
    try:
        # 1. Load strategy from DB
        registry = StrategyRegistry()
        loader = BacktestStrategyLoader(registry, pool, symbol=args.symbol)

        if args.all_active:
            slots = await loader.load_all_active()
        else:
            slot = await loader.load(args.strategy)
            slots = [slot] if slot else []

        if not slots:
            logger.error("No strategies loaded — aborting")
            return

        # Collect all factor names needed
        factor_names = set()
        for slot in slots:
            for sub in slot.subscriptions:
                factor_names.update(sub.factors)
        factor_names = list(factor_names)
        logger.info(f"Factors needed: {factor_names}")

        # 2. Load historical data
        logger.info(f"Loading {args.hours}h of data...")
        df_bars = await load_bars(pool, args.symbol, args.hours)
        if len(df_bars) < 60:
            logger.error(f"Not enough bars: {len(df_bars)} < 60")
            return

        logger.info(f"  Bars: {df_bars.index[0]} → {df_bars.index[-1]} ({len(df_bars)} bars)")

        df_delta = await load_tick_delta(pool, args.symbol, args.hours)
        if df_delta is not None and len(df_delta) > 0:
            # CRITICAL: DB time_bucket labels by period START,
            # nt-base bar_buffer labels by bar CLOSE (1 min later).
            # Shift delta index forward by 1 min so bar[09:47] gets delta from period 09:46-09:47.
            df_delta.index = df_delta.index + pd.Timedelta(minutes=1)

            df_bars["delta"] = df_delta["delta"].reindex(df_bars.index).fillna(0.0)
            df_bars["taker_buy_volume"] = df_delta["buy_vol"].reindex(df_bars.index).fillna(0.0)
            df_bars["taker_sell_volume"] = df_delta["sell_vol"].reindex(df_bars.index).fillna(0.0)
            # Use tick-accumulated volume (matches nt-base bar_buffer exactly)
            # NOT the exchange bar volume from the bars table
            df_bars["volume"] = df_bars["taker_buy_volume"] + df_bars["taker_sell_volume"]
        else:
            df_bars["delta"] = 0.0
            df_bars["taker_buy_volume"] = df_bars["volume"] * 0.5
            df_bars["taker_sell_volume"] = df_bars["volume"] * 0.5

        df_btc = await load_btc_bars(pool, args.hours)

        # Prepare BTC 5m for residual_momentum
        df_btc_5m = None
        if df_btc is not None and len(df_btc) > 0:
            df_btc_5m = df_btc.resample("5min").last().dropna()
            # Merge btc_close into df_bars (ffill to match nt-base's behavior)
            df_bars["btc_close"] = df_btc["close"].reindex(df_bars.index)
            df_bars["btc_close"] = df_bars["btc_close"].ffill().fillna(df_bars["close"])

        # 3. Pre-compute factors
        logger.info("Pre-computing factors...")
        factor_cache = precompute_factors(factor_names, df_bars, df_btc_5m)

        # 4. Run backtest
        logger.info(f"Running backtest: {args.hours}h window, "
                     f"equity={args.initial_equity}")
        engine = BacktestEngine(registry, factor_cache,
                                initial_equity=args.initial_equity)
        results = engine.run(df_bars, df_btc, df_delta)

        # 5. Output
        for result in results:
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"bt_{result.strategy_id}_{ts_str}.json"

            report = {
                "strategy_id": result.strategy_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "hours": args.hours,
                    "symbol": args.symbol,
                    "initial_equity": args.initial_equity,
                    "taker_fee": engine._taker_fee,
                },
                "metrics": result.metrics,
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Report saved: {output_path}")

            # Print summary
            m = result.metrics
            print(f"\n{'='*60}")
            print(f"  Strategy: {result.strategy_id}")
            print(f"  Window: {args.hours}h  |  Bars: {len(df_bars)}")
            print(f"  Initial Equity: {args.initial_equity:.0f}")
            print(f"{'='*60}")
            print(f"  Trades:      {m['n_trades']}")
            print(f"  PnL:         {m['total_pnl']:.4f} ({m['pnl_pct']:.2f}%)")
            print(f"  Win Rate:    {m['win_rate']:.1f}%")
            print(f"  Sharpe:      {m['sharpe']:.4f}")
            print(f"  Calmar:      {m['calmar']:.4f}")
            print(f"  Max DD:      {m['max_dd']:.4f} ({m['max_dd_pct']:.2f}%)")
            print(f"  Avg Win:     {m['avg_win']:.4f}")
            print(f"  Avg Loss:    {m['avg_loss']:.4f}")
            print(f"  Best/Worst:  {m['best_trade']:.4f} / {m['worst_trade']:.4f}")
            print(f"  Avg Bars:    {m['avg_bars_held']:.1f}")
            if m.get("exit_reasons"):
                print(f"  Exit reasons:")
                for r, c in sorted(m["exit_reasons"].items(), key=lambda x: -x[1]):
                    print(f"    {r}: {c}")
            print(f"{'='*60}")

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
