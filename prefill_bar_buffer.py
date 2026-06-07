"""Prefill bar buffer from tick data on nt-base startup."""
import asyncio
import asyncpg
from collections import deque
from datetime import datetime, timezone


async def prefill_bar_buffer(pool: asyncpg.Pool, n_bars: int = 300):
    """Aggregate last N 1m bars from ticks table, return deque ready for dispatch."""

    sol_query = """
    SELECT
        date_trunc('minute', ts_event) AS ts,
        FIRST(price, ts_event) AS open,
        MAX(price) AS high,
        MIN(price) AS low,
        LAST(price, ts_event) AS close,
        SUM(size) AS volume,
        SUM(CASE WHEN aggressor = 'BUY' THEN size ELSE 0 END) AS taker_buy_volume,
        SUM(CASE WHEN aggressor = 'SELL' THEN size ELSE 0 END) AS taker_sell_volume
    FROM ticks
    WHERE symbol = 'SOLUSDT-PERP'
    GROUP BY date_trunc('minute', ts_event)
    ORDER BY ts DESC
    LIMIT $1
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sol_query, n_bars)

    if not rows:
        print("prefill: no tick data found, starting with empty buffer")
        return deque(maxlen=n_bars), 0.0

    rows = list(reversed(rows))  # chronological order

    # Fetch BTC close prices for same time range
    btc_query = """
    SELECT
        date_trunc('minute', ts_event) AS ts,
        LAST(price, ts_event) AS close
    FROM ticks
    WHERE symbol = 'BTCUSDT-PERP'
      AND ts_event >= $1 AND ts_event < $2
    GROUP BY date_trunc('minute', ts_event)
    ORDER BY ts
    """
    btc_map = {}
    try:
        async with pool.acquire() as conn:
            ts_start = rows[0]['ts']
            ts_end = rows[-1]['ts']
            btc_rows = await conn.fetch(btc_query, ts_start, ts_end)
            btc_map = {r['ts']: float(r['close']) for r in btc_rows}
    except Exception:
        pass

    buf = deque(maxlen=n_bars)
    latest_btc = 0.0
    for r in rows:
        ts = r['ts']
        buy_vol = float(r['taker_buy_volume'] or 0)
        sell_vol = float(r['taker_sell_volume'] or 0)
        delta = buy_vol - sell_vol
        btc_close = btc_map.get(ts)
        if btc_close and btc_close > 0:
            latest_btc = btc_close
        # Forward-fill: use last known BTC to avoid NaN gaps
        btc_for_bar = btc_close if (btc_close and btc_close > 0) else (latest_btc if latest_btc > 0 else None)

        # Store as tz-naive for compatibility with live bar ts_event
        ts_naive = ts.replace(tzinfo=None)
        buf.append({
            "ts": ts_naive,
            "open": float(r['open'] or r['close'] or 0),
            "high": float(r['high'] or r['close'] or 0),
            "low": float(r['low'] or r['close'] or 0),
            "close": float(r['close'] or 0),
            "volume": float(r['volume'] or 0),
            "delta": delta,
            "taker_buy_volume": buy_vol,
            "taker_sell_volume": sell_vol,
            "btc_close": btc_for_bar,
        })

    btc_matched = sum(1 for b in buf if b['btc_close'] is not None)
    print(f"prefill: loaded {len(buf)} SOL bars, {btc_matched} with BTC match")
    return buf, latest_btc


async def main():
    pool = await asyncpg.create_pool(
        user='nautilus_admin',
        password='timescaledb_A2026cccvvv',
        database='trading_data',
        host='127.0.0.1'
    )

    import time
    t0 = time.time()
    buf, btc = await prefill_bar_buffer(pool, 300)
    elapsed = time.time() - t0

    if buf:
        print(f"First: {buf[0]['ts']} close={buf[0]['close']:.2f} btc={buf[0].get('btc_close')}")
        print(f"Last:  {buf[-1]['ts']} close={buf[-1]['close']:.2f} btc={buf[-1].get('btc_close')}")
        avg_delta = sum(b['delta'] for b in buf) / len(buf)
        print(f"Avg delta: {avg_delta:.2f}, BTC latest: {btc:.2f}")
        print(f"Query time: {elapsed:.2f}s")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
