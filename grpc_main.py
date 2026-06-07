"""bt-base gRPC server entrypoint — replays historical bars via gRPC stream.

Loads bars from TimescaleDB, pre-computes factors, waits for strategy
to connect via gRPC, then replays bars one-by-one. Intercepts signals
and simulates fills at bar close.

Usage:
    python grpc_main.py --strategy AlphaV2-005 --hours 168
    python grpc_main.py --strategy AlphaV2-005 --hours 24 --port 50052
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Path setup: bt-base must be before trading-v2 so 'base' resolves to bt-base
sys.path.insert(0, "/root/trading-v2")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # /root/bt-base (pos 0, searched first)

import grpc
import numpy as np
import pandas as pd
import trading_base_pb2 as pb
import trading_base_pb2_grpc as pb_grpc
from base.factor_engine import FactorEngine
from base.metrics import compute_metrics
from shared.env import cfg
from shared.log import setup_logging
from shared.db import get_pool, close_pool

logger = setup_logging("bt_grpc")

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BacktestServicer(pb_grpc.TradingBaseServicer):
    """gRPC servicer for backtest mode.

    Same interface as nt-base, but bars come from DB history,
    signals are simulated (no real execution).
    """

    def __init__(self):
        self._factor_engine = FactorEngine()
        self._config: pb.StrategyConfig | None = None
        self._strategy_id: str = ""
        self._bar_queue: asyncio.Queue = asyncio.Queue()
        self._signals: list[tuple[pb.Signal, pb.Bar]] = []
        self._ready_event = asyncio.Event()
        self._replay_done = False

    async def Register(self, request: pb.StrategyConfig, context) -> pb.RegisterAck:
        sid = request.strategy_id
        if self._config is not None:
            return pb.RegisterAck(ok=False, error="only one strategy per backtest")
        for fd in request.factors:
            try:
                self._factor_engine.register(fd.name, fd.code, dict(fd.params) if fd.params else None)
            except SyntaxError as e:
                return pb.RegisterAck(ok=False, error=f"Factor '{fd.name}': {e}")
        self._config = request
        self._strategy_id = sid
        logger.info(f"Backtest Register: {sid} factors={self._factor_engine.registered_names()}")
        return pb.RegisterAck(ok=True)

    async def Unregister(self, request, context) -> pb.UnregisterAck:
        return pb.UnregisterAck(ok=True)

    async def SubscribeBars(self, request: pb.BarRequest, context):
        logger.info(f"Backtest SubscribeBars: {request.symbol}")
        self._ready_event.set()
        while not self._replay_done:
            try:
                bar = await asyncio.wait_for(self._bar_queue.get(), timeout=0.5)
                yield bar
            except asyncio.TimeoutError:
                if self._replay_done:
                    break
            except asyncio.CancelledError:
                break

    async def SubmitSignal(self, request: pb.Signal, context) -> pb.SignalAck:
        self._signals.append((request, None))  # Bar will be filled in later
        logger.info(f"Backtest Signal: dir={pb.Signal.Direction.Name(request.direction)} "
                     f"reason={request.reason}")
        return pb.SignalAck(accepted=True)

    async def GetState(self, request, context) -> pb.StateResponse:
        return pb.StateResponse()

    async def ClosePosition(self, request, context) -> pb.CloseAck:
        return pb.CloseAck(ok=True)

    def build_bar(self, symbol: str, ts_ns: int, o: float, h: float, l: float, c: float,
                  vol: float, delta: float, buy_vol: float, sell_vol: float,
                  btc: float, df: pd.DataFrame) -> pb.Bar:
        factors = self._factor_engine.execute_all(df)
        return pb.Bar(
            symbol=symbol, ts_ns=ts_ns, open=o, high=h, low=l, close=c,
            volume=vol, delta=delta, taker_buy_vol=buy_vol, taker_sell_vol=sell_vol,
            btc_close=btc, factors=factors,
        )


async def load_data(pool, symbol: str, hours: int):
    """Load bars + tick delta + BTC from TimescaleDB."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    # Bars
    rows = await pool.fetch(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol=$1 AND timeframe='1m' AND ts >= $2 ORDER BY ts",
        symbol, cutoff,
    )
    df = pd.DataFrame([dict(r) for r in rows])
    df["ts"] = pd.to_datetime(df["ts"]); df = df.set_index("ts")
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)

    # Tick delta (size-based, time-aligned)
    d_rows = await pool.fetch(
        "SELECT time_bucket('1 minute', ts_event) AS ts, "
        "SUM(CASE WHEN aggressor='BUY' THEN size ELSE 0 END) AS buy_vol, "
        "SUM(CASE WHEN aggressor='SELL' THEN size ELSE 0 END) AS sell_vol "
        "FROM ticks WHERE symbol=$1 AND ts_event >= $2 "
        "GROUP BY 1 ORDER BY ts",
        symbol, cutoff,
    )
    if d_rows:
        df_d = pd.DataFrame([dict(r) for r in d_rows])
        df_d["ts"] = pd.to_datetime(df_d["ts"]); df_d = df_d.set_index("ts")
        df_d.index = df_d.index + pd.Timedelta(minutes=1)  # align with bar close
        df["delta"] = (df_d["buy_vol"] - df_d["sell_vol"]).reindex(df.index).fillna(0.0)
        df["taker_buy_volume"] = df_d["buy_vol"].reindex(df.index).fillna(0.0)
        df["taker_sell_volume"] = df_d["sell_vol"].reindex(df.index).fillna(0.0)
        df["volume"] = df["taker_buy_volume"] + df["taker_sell_volume"]
    else:
        df["delta"] = 0.0
        df["taker_buy_volume"] = 0.0
        df["taker_sell_volume"] = 0.0

    # BTC bars
    btc_rows = await pool.fetch(
        "SELECT ts, close FROM bars WHERE symbol='BTCUSDT-PERP' AND timeframe='1m' "
        "AND ts >= $1 ORDER BY ts",
        cutoff,
    )
    if btc_rows:
        df_btc = pd.DataFrame([dict(r) for r in btc_rows])
        df_btc["ts"] = pd.to_datetime(df_btc["ts"]); df_btc = df_btc.set_index("ts")
        df["btc_close"] = df_btc["close"].reindex(df.index).ffill().fillna(df["close"])

    logger.info(f"Data loaded: {len(df)} bars, {symbol} ({hours}h)")
    return df


async def run_backtest(servicer: BacktestServicer, df_bars: pd.DataFrame,
                        initial_equity: float = 1000.0, taker_fee: float = 0.0004):
    """Replay bars through gRPC, simulate fills."""
    symbol = "SOLUSDT-PERP"
    trades = []
    equity = initial_equity
    equity_curve = [initial_equity]

    # Position state
    in_position = False
    is_long = True
    entry_price = 0.0
    entry_ts = 0
    bars_held = 0
    pos_highest = 0.0
    pos_lowest = float("inf")
    stop_price = None
    take_price = None
    trailing_trigger = None
    trailing_stop = None

    n = len(df_bars)
    logger.info(f"Replaying {n} bars...")

    for i, (ts, bar) in enumerate(df_bars.iterrows()):
        ts_ns = int(ts.timestamp() * 1_000_000_000)
        close = float(bar["close"])
        high = float(bar.get("high", close))
        low = float(bar.get("low", close))
        vol = float(bar.get("volume", 0))
        delta = float(bar.get("delta", 0))
        buy_vol = float(bar.get("taker_buy_volume", 0))
        sell_vol = float(bar.get("taker_sell_volume", 0))
        btc = float(bar.get("btc_close", close))

        # Build and push Bar
        pb_bar = servicer.build_bar(
            symbol=symbol, ts_ns=ts_ns, o=float(bar["open"]),
            h=high, l=low, c=close, vol=vol, delta=delta,
            buy_vol=buy_vol, sell_vol=sell_vol, btc=btc, df=df_bars,
        )

        # Check stop/take conditions
        if in_position:
            bars_held += 1
            if is_long:
                pos_highest = max(pos_highest, high)  # use bar high for trailing
                # Check conditions
                if stop_price and low <= stop_price:
                    exit_px = stop_price
                    exit_reason = f"Hard SL at {stop_price:.4f}"
                elif take_price and high >= take_price:
                    exit_px = take_price
                    exit_reason = f"Take Profit at {take_price:.4f}"
                elif trailing_stop and low <= trailing_stop:
                    exit_px = trailing_stop
                    exit_reason = f"Trailing at {trailing_stop:.4f}"
                else:
                    exit_px = None
                    exit_reason = None
            else:
                pos_lowest = min(pos_lowest, low)
                if stop_price and high >= stop_price:
                    exit_px = stop_price
                    exit_reason = f"Hard SL at {stop_price:.4f}"
                elif take_price and low <= take_price:
                    exit_px = take_price
                    exit_reason = f"Take Profit at {take_price:.4f}"
                elif trailing_stop and high >= trailing_stop:
                    exit_px = trailing_stop
                    exit_reason = f"Trailing at {trailing_stop:.4f}"
                else:
                    exit_px = None
                    exit_reason = None

            if exit_px is not None:
                raw_return = (exit_px - entry_price) / entry_price if is_long else (entry_price - exit_px) / entry_price
                notional = (equity * 0.2 * 3)  # position_size_pct * leverage
                fee = notional * taker_fee * 2
                pnl = raw_return * notional - fee
                trades.append({
                    "entry_ts": entry_ts, "exit_ts": ts_ns,
                    "side": "LONG" if is_long else "SHORT",
                    "entry": entry_price, "exit": exit_px,
                    "bars_held": bars_held,
                    "realized_pnl": round(pnl, 4),
                    "return_pct": round(raw_return * 100, 4),
                    "exit_reason": exit_reason,
                })
                equity += pnl
                in_position = False
                stop_price = take_price = trailing_trigger = trailing_stop = None

        # Push bar to strategy (blocks until strategy processes it or timeout)
        try:
            servicer._bar_queue.put_nowait(pb_bar)
        except asyncio.QueueFull:
            pass

        # Process any pending signals from strategy
        await asyncio.sleep(0)  # yield to let gRPC handler process

        # Check if strategy submitted a signal for this bar
        new_signals = servicer._signals[:]
        servicer._signals.clear()

        for sig, _ in new_signals:
            if sig.direction == pb.Signal.FLAT and in_position:
                # Strategy wants to exit
                raw_return = (close - entry_price) / entry_price if is_long else (entry_price - close) / entry_price
                notional = (equity * 0.2 * 3)
                fee = notional * taker_fee * 2
                pnl = raw_return * notional - fee
                trades.append({
                    "entry_ts": entry_ts, "exit_ts": ts_ns,
                    "side": "LONG" if is_long else "SHORT",
                    "entry": entry_price, "exit": close,
                    "bars_held": bars_held,
                    "realized_pnl": round(pnl, 4),
                    "return_pct": round(raw_return * 100, 4),
                    "exit_reason": sig.reason,
                })
                equity += pnl
                in_position = False
                stop_price = take_price = None

            elif sig.direction != pb.Signal.FLAT and not in_position:
                # Entry
                in_position = True
                is_long = (sig.direction == pb.Signal.LONG)
                entry_price = close
                entry_ts = ts_ns
                bars_held = 0
                pos_highest = close
                pos_lowest = close

                # Apply exit conditions from signal
                if sig.HasField("exit"):
                    ec = sig.exit
                    if ec.HasField("hard_stop_price"):
                        stop_price = ec.hard_stop_price
                    if ec.HasField("take_profit_price"):
                        take_price = ec.take_profit_price

        equity_curve.append(equity)

        if (i + 1) % 100 == 0:
            logger.info(f"  replay {i+1}/{n} bars, equity={equity:.2f}")

    # Close any open position at end
    if in_position:
        last_close = float(df_bars.iloc[-1]["close"])
        raw_return = (last_close - entry_price) / entry_price if is_long else (entry_price - last_close) / entry_price
        notional = (equity * 0.2 * 3)
        fee = notional * taker_fee * 2
        pnl = raw_return * notional - fee
        trades.append({
            "entry_ts": entry_ts, "exit_ts": int(df_bars.index[-1].timestamp() * 1e9),
            "side": "LONG" if is_long else "SHORT",
            "entry": entry_price, "exit": last_close,
            "bars_held": bars_held,
            "realized_pnl": round(pnl, 4),
            "return_pct": round(raw_return * 100, 4),
            "exit_reason": "end_of_period",
        })
        equity += pnl

    servicer._replay_done = True
    # Wait for strategy to finish processing (drain any remaining signals)
    await asyncio.sleep(2)
    return trades, equity_curve


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="AlphaV2-005")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--symbol", type=str, default="SOLUSDT-PERP")
    parser.add_argument("--port", type=int, default=50052, help="gRPC TCP port")
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    args = parser.parse_args()

    pool = await get_pool()

    try:
        # 1. Load data
        logger.info(f"Loading {args.hours}h data...")
        df = await load_data(pool, args.symbol, args.hours)
        if len(df) < 60:
            logger.error(f"Not enough bars: {len(df)}")
            return

        # 2. Start gRPC server
        servicer = BacktestServicer()
        server = grpc.aio.server()
        pb_grpc.add_TradingBaseServicer_to_server(servicer, server)
        server.add_insecure_port(f"0.0.0.0:{args.port}")
        await server.start()
        logger.info(f"bt-base gRPC listening on :{args.port}")

        # 3. Wait for strategy to connect and subscribe
        logger.info("Waiting for strategy to connect...")
        try:
            await asyncio.wait_for(servicer._ready_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for strategy to connect")
            await server.stop(grace=1.0)
            return
        logger.info("Strategy connected, starting replay...")

        trades, equity_curve = await run_backtest(servicer, df, args.initial_equity)

        # 4. Output
        metrics = compute_metrics(trades, equity_curve, args.initial_equity, args.hours)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"bt_grpc_{args.strategy}_{ts_str}.json"

        report = {
            "strategy_id": args.strategy,
            "mode": "grpc",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {"hours": args.hours, "symbol": args.symbol, "initial_equity": args.initial_equity},
            "metrics": metrics,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report: {output_path}")
        m = metrics
        print(f"\n{'='*60}")
        print(f"  Strategy: {args.strategy} (gRPC backtest)")
        print(f"  Trades: {m['n_trades']}  PnL: {m['total_pnl']:.4f} ({m['pnl_pct']:.2f}%)")
        print(f"  Win: {m['win_rate']:.1f}%  Sharpe: {m['sharpe']:.2f}  DD: {m['max_dd_pct']:.2f}%")
        print(f"{'='*60}")

        await server.stop(grace=2.0)

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
