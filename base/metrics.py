"""Performance metrics for backtest results."""
import numpy as np


def compute_metrics(trades: list[dict], equity_curve: list[float],
                    starting_equity: float = 1000.0,
                    hours: int = 168) -> dict:
    """Compute comprehensive backtest performance metrics.

    Args:
        trades: list of trade dicts with keys: realized_pnl, exit_reason, bars_held, side, entry, exit
        equity_curve: list of equity values at each bar
        starting_equity: initial account equity
        hours: backtest window length in hours

    Returns:
        dict of metrics
    """
    if not trades:
        return {
            "n_trades": 0, "total_pnl": 0, "pnl_pct": 0,
            "sharpe": 0, "calmar": 0, "max_dd": 0, "max_dd_pct": 0,
            "win_rate": 0, "avg_win": 0, "avg_loss": 0,
            "profit_factor": 0, "best_trade": 0, "worst_trade": 0,
            "avg_bars_held": 0, "exit_reasons": {},
            "equity_curve": equity_curve,
            "starting_equity": starting_equity,
            "trades": [],
        }

    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    pnls = [t.get("realized_pnl", 0) for t in trades]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    n = len(pnls)

    # Drawdown
    if equity_curve and len(equity_curve) > 1:
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = float(np.min(dd))
        max_dd_pct = float(np.min(dd) / starting_equity * 100) if starting_equity > 0 else 0
    else:
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = float(np.min(cum - peak)) if len(cum) > 0 else 0.0
        max_dd_pct = max_dd / starting_equity * 100 if starting_equity > 0 else 0

    # Sharpe (annualized, assuming 1m bars)
    avg_pnl = np.mean(pnls) if pnls else 0
    std_pnl = np.std(pnls) if n > 1 else 1e-9
    # 365 * 24 * 60 = 525600 minutes per year, trades are per-bar
    sharpe = (avg_pnl / std_pnl) * np.sqrt(365 * 24) if std_pnl > 0 else 0

    # Calmar
    calmar = total_pnl / abs(max_dd) if abs(max_dd) > 0 else 0

    # Win/loss stats
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0

    # Profit factor
    gross_profit = sum(win_pnls) if win_pnls else 0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Average bars held
    avg_bars = np.mean([t.get("bars_held", 0) for t in trades]) if trades else 0

    # Annualized return
    hours_traded = hours
    pnl_pct = (total_pnl / starting_equity) * 100 if starting_equity > 0 else 0
    annualized_return = pnl_pct * (365 * 24 / hours_traded) if hours_traded > 0 else 0

    return {
        "n_trades": n,
        "total_pnl": round(total_pnl, 4),
        "pnl_pct": round(pnl_pct, 4),
        "annualized_return_pct": round(annualized_return, 2),
        "sharpe": round(sharpe, 4),
        "calmar": round(calmar, 4),
        "max_dd": round(max_dd, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "win_rate": round(wins / n * 100, 2) if n > 0 else 0,
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 4),
        "best_trade": round(max(pnls), 4),
        "worst_trade": round(min(pnls), 4),
        "avg_bars_held": round(avg_bars, 1),
        "exit_reasons": exit_reasons,
        "equity_curve": [round(float(e), 4) for e in equity_curve],
        "starting_equity": starting_equity,
        "trades": trades,
    }
