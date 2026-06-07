"""
Module:    shared/env
Purpose:   Centralized environment configuration loader.
           Reads .env file once at import time, exposes typed AppCfg dataclass.
           All other modules MUST import cfg from here 鈥?never call os.getenv directly.

Interface: cfg: AppCfg          鈥?typed configuration singleton
           assert_required()    鈥?raises RuntimeError if critical secrets missing

Configuration Sections:
  BinanceCfg    鈥?exchange API credentials
  TelegramCfg   鈥?bot token and admin chat
  TimescaleCfg  鈥?database host/port/user/password, exposes .dsn property
  LLMCfg        鈥?LLM provider/model/API key
  AppCfg        鈥?top-level: mode, symbols, risk params, sub-configs

Security:
  Secrets sourced from environment only (os.environ).
  .env file parsed defensively (strips inline comments, respects existing env vars).
  assert_required() must be called at every entrypoint before trading begins.

Author:    nt-base system
Version:   1.0.0
"""
from __future__ import annotations
"""Centralized environment loader.

Reads /root/nt-base/.env once and exposes a typed `cfg` object.
All other modules MUST import `from shared.env import cfg` instead of
calling os.getenv directly.
"""


import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_env_file(path: Path) -> None:
    """Minimal .env loader (no external dep). Existing os.environ wins."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.lstrip()
        # strip inline comments unless quoted; we only support unquoted values here
        if not (v.startswith('"') or v.startswith("'")):
            # Any '#' starts a comment when value is unquoted
            hash_idx = v.find("#")
            if hash_idx >= 0:
                v = v[:hash_idx]
        v = v.strip().strip('"').strip("'")
        if not k:
            continue
        # Defensive: if systemd's EnvironmentFile parser left an inline comment in
        # the value (it doesn't strip ' #...'), override with our cleaned value.
        existing = os.environ.get(k)
        if existing is None or "#" in existing:
            os.environ[k] = v


_load_env_file(ENV_FILE)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _getint(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _getfloat(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class BinanceCfg:
    api_key: str = field(default_factory=lambda: _get("BINANCE_API_KEY"))
    api_secret: str = field(default_factory=lambda: _get("BINANCE_API_SECRET"))


@dataclass(frozen=True)
class TelegramCfg:
    bot_token: str = field(default_factory=lambda: _get("TELEGRAM_BOT_TOKEN"))
    admin_chat_id: int = field(default_factory=lambda: _getint("TELEGRAM_ADMIN_CHAT_ID", 0))


@dataclass(frozen=True)
class TimescaleCfg:
    host: str = field(default_factory=lambda: _get("TS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _getint("TS_PORT", 5432))
    user: str = field(default_factory=lambda: _get("TS_USER", "nautilus_admin"))
    password: str = field(default_factory=lambda: _get("TS_PASSWORD"))
    database: str = field(default_factory=lambda: _get("TS_DB", "trading_data"))

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class LLMCfg:
    provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "zhipu"))
    model: str = field(default_factory=lambda: _get("LLM_MODEL", "glm-5.1"))
    api_key: str = field(default_factory=lambda: _get("ZHIPU_API_KEY"))
    base_url: str = field(
        default_factory=lambda: _get(
            "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        )
    )
    monthly_budget_usd: float = field(
        default_factory=lambda: _getfloat("LLM_MONTHLY_BUDGET_USD", 50.0)
    )


@dataclass(frozen=True)
class AppCfg:
    mode: Literal["sandbox", "live"] = field(
        default_factory=lambda: _get("MODE", "sandbox")  # type: ignore[arg-type]
    )
    trading_mode: Literal["live", "sandbox", "data_only"] = field(
        default_factory=lambda: _get("TRADING_MODE", "live")  # type: ignore[arg-type]
    )
    sandbox_initial_usdt: float = field(
        default_factory=lambda: _getfloat("SANDBOX_INITIAL_USDT", 1000.0)
    )
    live_initial_usdt: float = field(
        default_factory=lambda: _getfloat("LIVE_INITIAL_USDT", 1000.0)
    )

    primary_symbol: str = field(
        default_factory=lambda: _get("PRIMARY_SYMBOL", "SOLUSDT")
    )
    data_symbols: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip() for s in _get("DATA_SYMBOLS", "").split(",") if s.strip()
        )
    )
    taker_fee: float = field(default_factory=lambda: _getfloat("TAKER_FEE", 0.0005))
    max_daily_symbols: int = field(
        default_factory=lambda: _getint("MAX_DAILY_SYMBOLS", 2)
    )
    scanner_top_n: int = field(default_factory=lambda: _getint("SCANNER_TOP_N", 5))
    scanner_utc_hour: int = field(default_factory=lambda: _getint("SCANNER_UTC_HOUR", 12))
    scanner_utc_minute: int = field(
        default_factory=lambda: _getint("SCANNER_UTC_MINUTE", 0)
    )
    infer_interval_sec: int = field(
        default_factory=lambda: _getint("INFER_INTERVAL_SEC", 5)
    )
    check_interval_sec: int = field(
        default_factory=lambda: _getint("CHECK_INTERVAL_SEC", 1)
    )

    rdagent_loop_hours: int = field(
        default_factory=lambda: _getint("RDAGENT_LOOP_HOURS", 6)
    )
    hard_floor_equity_ratio: float = field(
        default_factory=lambda: _getfloat("HARD_FLOOR_EQUITY_RATIO", 0.5)
    )

    nautilus_py: str = field(
        default_factory=lambda: _get(
            "NAUTILUS_PY", "/root/miniconda3/envs/nautilus/bin/python"
        )
    )
    rdagent_py: str = field(
        default_factory=lambda: _get(
            "RDAGENT_PY", "/root/miniconda3/envs/rdagent/bin/python"
        )
    )

    binance: BinanceCfg = field(default_factory=BinanceCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    timescale: TimescaleCfg = field(default_factory=TimescaleCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)


cfg = AppCfg()


def assert_required() -> None:
    """Raise if any critical secret is missing. Call from entrypoints.

    bt-base version: only checks TimescaleDB credentials.
    Binance/Telegram are not required for backtesting.
    """
    missing: list[str] = []
    if not cfg.timescale.password:
        missing.append("TS_PASSWORD")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")