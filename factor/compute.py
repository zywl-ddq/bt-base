"""
Module:    factor/compute
Purpose:   Shared factor computation engine. Both backtest and live environments
           call the SAME factor code on the SAME 1m bar data.

Updates v1.1:
  - compute_factor_history: supports multi-column factor output (DataFrame).
    When a factor returns a DataFrame, each column becomes a separate factor:
      trend_regime -> {trend_regime: direction_val, trend_confidence: conf_val}
  - Backward compatible: factors returning a single Series work as before.
"""
import logging
import inspect
import numpy as np
import scipy
import pandas as pd

logger = logging.getLogger(__name__)

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "range": range, "round": round,
    "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    "isinstance": isinstance, "ValueError": ValueError,
    "TypeError": TypeError, "ZeroDivisionError": ZeroDivisionError,
}

_FACTOR_NAMESPACE_BASE = {"scipy": scipy, "pd": pd, "np": np, "__builtins__": __builtins__}


def _load_factor_code(code: str) -> str:
    from pathlib import Path
    for factors_dir in (
        Path("/root/trading-v2/factors"),
        Path("/root/cogalpha-crypto/output/factors"),
    ):
        candidate = factors_dir / f"{code}.py"
        if candidate.exists():
            return candidate.read_text()
    return code


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    import decimal
    _DECIMAL = decimal.Decimal
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in (np.dtype("int64"), np.dtype("int32")):
            df[col] = df[col].astype(np.float64)
        elif dtype == np.dtype("object"):
            sample = df[col].dropna()
            if len(sample) and isinstance(sample.iloc[0], _DECIMAL):
                df[col] = df[col].astype(np.float64)
    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    for col in ["bid_ask_spread_avg", "order_book_imbalance"]:
        if col not in df: df[col] = 0.0
    if "bid_depth_weighted" not in df: df["bid_depth_weighted"] = close * 0.999
    if "ask_depth_weighted" not in df: df["ask_depth_weighted"] = close * 1.001
    for col in ["funding_rate", "funding_settle_countdown", "open_interest",
                 "oi_delta_5m", "buy_liquidation_value", "sell_liquidation_value",
                 "liquidation_imbalance", "large_trade_ratio"]:
        if col not in df: df[col] = 0.0
    if "long_short_ratio" not in df: df["long_short_ratio"] = 1.0
    if "volume" not in df: df["volume"] = 0.0
    if "taker_buy_volume" not in df: df["taker_buy_volume"] = df["volume"] * 0.5
    if "taker_sell_volume" not in df: df["taker_sell_volume"] = df["volume"] * 0.5
    if "delta" not in df: df["delta"] = df["taker_buy_volume"] - df["taker_sell_volume"]
    if "buy_sell_pressure_ratio" not in df: df["buy_sell_pressure_ratio"] = 1.0
    if "avg_trade_size" not in df: df["avg_trade_size"] = df["volume"] / (df.get("trade_count", 1) + 1)
    if "realized_vol_5m" not in df: df["realized_vol_5m"] = close.pct_change().rolling(5).std().fillna(0)
    if "realized_vol_1h" not in df: df["realized_vol_1h"] = close.pct_change().rolling(60).std().fillna(0)
    if "trend_strength_1h" not in df:
        df["trend_strength_1h"] = close.diff(60).fillna(0) / (close.shift(60).fillna(close.iloc[0]) + 1e-9)
    if "btc_close" not in df: df["btc_close"] = np.nan
    return df


def _execute_factor(code: str, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
    df_enriched = _add_derived_columns(_coerce_numeric(df.copy()))
    namespace = {**_FACTOR_NAMESPACE_BASE, "df": df_enriched}
    exec(code, namespace)

    # Pattern 1: callable functions named factor_*
    for name, obj in namespace.items():
        if callable(obj) and name.startswith("factor_"):
            try:
                sig = inspect.signature(obj)
                if "timescale" in sig.parameters:
                    result = obj(df=namespace["df"], timescale="1min")
                else:
                    result = obj(df=namespace["df"])
            except ValueError:
                result = obj(df=namespace["df"])

            if isinstance(result, pd.DataFrame):
                return result
            if isinstance(result, pd.Series):
                return pd.to_numeric(result, errors="coerce")

    # Pattern 2: already-computed Series/DataFrame named factor_*
    for name, obj in namespace.items():
        if name.startswith("factor_"):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                return obj

    raise ValueError("No factor_* Series/DataFrame found in factor code namespace")


def compute_factor_history(factor_code: str, df_1min: pd.DataFrame):
    """Compute factor over full 1min history.

    Returns:
        - pd.Series for single-column factors (backward compatible)
        - dict[str, pd.Series] for multi-column factors (DataFrame output)
    """
    code = _load_factor_code(factor_code)
    result = _execute_factor(code, df_1min)

    if isinstance(result, pd.DataFrame):
        out = {}
        for col in result.columns:
            s = result[col].copy()
            s = s.reindex(df_1min.index)
            out[col] = s.dropna()
        return out

    result = result.reindex(df_1min.index)
    return result.dropna()


def compute_factor_incremental(factor_code: str, df_full: pd.DataFrame) -> float | dict | None:
    code = _load_factor_code(factor_code)
    try:
        result = _execute_factor(code, df_full)
        if isinstance(result, pd.DataFrame):
            out = {}
            for col in result.columns:
                s = result[col].dropna()
                out[col] = float(s.iloc[-1]) if len(s) > 0 else 0.0
            return out
        if result.empty:
            return None
        return float(result.iloc[-1])
    except Exception as e:
        logger.error(f"factor_incremental failed: {e}", exc_info=True)
        return None
