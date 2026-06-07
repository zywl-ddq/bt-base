"""
Module:    shared/db
Purpose:   Database connection pool management for TimescaleDB.
           Provides singleton pool lifecycle (create/close) shared across all modules.
Interface: get_pool() -> asyncpg.Pool
           close_pool() -> None
Dependencies: asyncpg, shared.env
Author:    nt-base system
Version:   1.0.0
Security:  Pool credentials sourced from environment (cfg.timescale.dsn), never hardcoded.
"""
from __future__ import annotations
"""Shared asyncpg connection pool. Single import point for all components."""


import asyncio
import logging
from typing import Optional

import asyncpg

from shared.env import cfg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None
_lock = asyncio.Lock()


async def get_pool(min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """Lazy singleton pool, safe for concurrent first-callers."""
    global _pool
    if _pool is not None:
        return _pool
    async with _lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=cfg.timescale.dsn,
                min_size=min_size,
                max_size=max_size,
            )
            logger.info(
                "asyncpg pool ready: %s:%s/%s",
                cfg.timescale.host,
                cfg.timescale.port,
                cfg.timescale.database,
            )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("asyncpg pool closed")


async def healthcheck() -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        v = await conn.fetchval("SELECT 1")
    return v == 1
