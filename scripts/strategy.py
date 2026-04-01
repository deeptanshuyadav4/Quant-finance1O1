"""
Moving Average Crossover Strategy + Backtester
-----------------------------------------------
Implements a simple trading strategy:
  - BUY  when the short-term SMA crosses ABOVE the long-term SMA (golden cross)
  - SELL when the short-term SMA crosses BELOW the long-term SMA (death cross)

Then backtests the strategy against a buy-and-hold benchmark and saves a
performance chart to images/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "AAPL_stock_data.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df


def generate_signals(df: pd.DataFrame,
                     short_window: int = 20,
                     long_window: int = 50) -> pd.DataFrame:
    """
    Generate buy/sell signals based on SMA crossover.

    Signal column:
        1  = hold long position (SMA_short > SMA_long)
        0  = out of market     (SMA_short <= SMA_long)

    Position column shows the daily change in signal (1 = buy, -1 = sell).
    """
    df = df.copy()
    df[f"SMA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"SMA_{long_window}"] = df["Close"].rolling(window=long_window).mean()

    # Signal: 1 when short SMA is above long SMA, else 0
    df["Signal"] = 0
    df.loc[df[f"SMA_{short_window}"] > df[f"SMA_{long_window}"], "Signal"] = 1

    # Position: difference in signal day-over-day (detects crossovers)
    df["Position"] = df["Signal"].diff()

    return df


def backtest(df: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.DataFrame:
    """
    Simulate the moving average crossover strategy.

    Returns a DataFrame with:
        - Holdings  : value of stock held
        - Cash      : uninvested cash
        - Total     : total portfolio value
        - Returns   : daily portfolio return
    """
    df = df.copy()

    # Number of shares bought/sold on each signal
    # We buy as many whole shares as we can afford at the time of the signal
    portfolio = pd.DataFrame(index=df.index)
    portfolio["Close"] = df["Close"]
    portfolio["Signal"] = df["Signal"]
    portfolio["Position"] = df["Position"]

    shares_held = 0
    cash = initial_capital
    holdings_list = []
    cash_list = []

    for _, row in portfolio.iterrows():
        if row["Position"] == 1:          # BUY signal – go long
            shares_held = int(cash // row["Close"])
            cash -= shares_held * row["Close"]
        elif row["Position"] == -1:       # SELL signal – exit position
            cash += shares_held * row["Close"]
            shares_held = 0

        holdings_list.append(shares_held * row["Close"])
        cash_list.append(cash)

    portfolio["Holdings"] = holdings_list
    portfolio["Cash"] = cash_list
    portfolio["Total"] = portfolio["Holdings"] + portfolio["Cash"]
    portfolio["Returns"] = portfolio["Total"].pct_change()

    return portfolio


def calculate_performance(portfolio: pd.DataFrame,
                           df: pd.DataFrame,
                           initial_capital: float = 10_000.0) -> dict:
    """Calculate strategy and buy-and-hold performance metrics."""
    # Strategy metrics
    strat_total_return = (portfolio["Total"].iloc[-1] - initial_capital) / initial_capital * 100
    strat_returns = portfolio["Returns"].dropna()
    strat_sharpe = (strat_returns.mean() / strat_returns.std() * np.sqrt(252)
                    if strat_returns.std() > 0 else 0.0)

    rolling_max = portfolio["Total"].cummax()
    drawdown = (portfolio["Total"] - rolling_max) / rolling_max * 100
    strat_max_dd = drawdown.min()

    # Buy-and-hold metrics
    bh_shares = int(initial_capital // df["Close"].iloc[0])
    bh_final = bh_shares * df["Close"].iloc[-1] + (initial_capital - bh_shares * df["Close"].iloc[0])
    bh_total_return = (bh_final - initial_capital) / initial_capital * 100

    return {
        "strategy_total_return_pct": round(strat_total_return, 2),
        "strategy_sharpe_ratio": round(strat_sharpe, 3),
        "strategy_max_drawdown_pct": round(strat_max_dd, 2),
        "buy_hold_total_return_pct": round(bh_total_return, 2),
        "final_portfolio_value": round(portfolio["Total"].iloc[-1], 2),
    }


def plot_strategy(df: pd.DataFrame,
                  portfolio: pd.DataFrame,
                  short_window: int = 20,
                  long_window: int = 50,
                  initial_capital: float = 10_000.0,
                  ticker: str = "AAPL") -> str:
    """Save a two-panel chart: (1) price + signals, (2) equity curves."""
    short_col = f"SMA_{short_window}"
    long_col = f"SMA_{long_window}"

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ── Panel 1: Price + SMAs + signals ───────────────────────────────────
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], label="Close Price", linewidth=1, color="#1f77b4")
    ax1.plot(df.index, df[short_col], label=short_col, linewidth=1.3,
             color="orange", linestyle="--")
    ax1.plot(df.index, df[long_col], label=long_col, linewidth=1.3,
             color="red", linestyle="--")

    buys = df[df["Position"] == 1]
    sells = df[df["Position"] == -1]
    ax1.scatter(buys.index, buys["Close"], marker="^", color="green",
                s=80, label="Buy Signal", zorder=5)
    ax1.scatter(sells.index, sells["Close"], marker="v", color="red",
                s=80, label="Sell Signal", zorder=5)

    ax1.set_title(f"{ticker} – Moving Average Crossover Strategy", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Equity curves ─────────────────────────────────────────────
    ax2 = axes[1]
    bh_shares = int(initial_capital // df["Close"].iloc[0])
    bh_cash_left = initial_capital - bh_shares * df["Close"].iloc[0]
    bh_value = bh_shares * df["Close"] + bh_cash_left

    ax2.plot(portfolio.index, portfolio["Total"], label="Strategy Portfolio",
             linewidth=1.5, color="green")
    ax2.plot(df.index, bh_value, label="Buy & Hold", linewidth=1.5,
             color="#1f77b4", linestyle="--")
    ax2.axhline(initial_capital, color="gray", linestyle=":", linewidth=0.8)

    ax2.set_title("Portfolio Value Comparison", fontsize=12)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Portfolio Value (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    fig.tight_layout()

    filepath = os.path.join(IMAGES_DIR, f"{ticker}_strategy_backtest.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filepath}")
    return filepath


def print_performance(metrics: dict) -> None:
    print(f"\n{'='*50}")
    print("  Backtest Performance Report")
    print(f"{'='*50}")
    print(f"  Strategy Total Return : {metrics['strategy_total_return_pct']:>8.2f}%")
    print(f"  Buy & Hold Return     : {metrics['buy_hold_total_return_pct']:>8.2f}%")
    print(f"  Strategy Sharpe Ratio : {metrics['strategy_sharpe_ratio']:>8.3f}")
    print(f"  Max Drawdown          : {metrics['strategy_max_drawdown_pct']:>8.2f}%")
    print(f"  Final Portfolio Value : ${metrics['final_portfolio_value']:>10,.2f}")
    print(f"{'='*50}\n")


def main() -> None:
    initial_capital = 10_000.0

    print("Loading data …")
    df = load_data(DATA_PATH)

    print("Generating signals …")
    df = generate_signals(df, short_window=20, long_window=50)

    print("Running backtest …")
    portfolio = backtest(df, initial_capital=initial_capital)

    metrics = calculate_performance(portfolio, df, initial_capital=initial_capital)
    print_performance(metrics)

    print("Generating strategy chart …")
    plot_strategy(df, portfolio, short_window=20, long_window=50, initial_capital=initial_capital)

    print("Backtest complete.")


if __name__ == "__main__":
    main()
