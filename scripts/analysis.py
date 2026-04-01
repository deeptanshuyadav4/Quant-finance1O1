"""
Stock Data Analysis Script
--------------------------
Loads stock market data, calculates moving averages and daily returns,
and generates visualizations saved to the images/ directory.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "AAPL_stock_data.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """Load stock CSV and parse the Date column as a datetime index."""
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df


def calculate_moving_averages(df: pd.DataFrame,
                               short_window: int = 20,
                               long_window: int = 50) -> pd.DataFrame:
    """Add short-term and long-term simple moving averages to the DataFrame."""
    df = df.copy()
    df[f"SMA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
    df[f"SMA_{long_window}"] = df["Close"].rolling(window=long_window).mean()
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily percentage returns and cumulative returns."""
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Cumulative_Return"] = (1 + df["Close"].pct_change()).cumprod() - 1
    return df


def plot_price_with_moving_averages(df: pd.DataFrame,
                                    short_window: int = 20,
                                    long_window: int = 50,
                                    ticker: str = "AAPL") -> str:
    """Plot closing price with short and long moving averages."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["Close"], label="Close Price", linewidth=1.2, color="#1f77b4")
    ax.plot(df.index, df[f"SMA_{short_window}"],
            label=f"{short_window}-Day SMA", linewidth=1.5, color="orange", linestyle="--")
    ax.plot(df.index, df[f"SMA_{long_window}"],
            label=f"{long_window}-Day SMA", linewidth=1.5, color="red", linestyle="--")

    ax.set_title(f"{ticker} – Close Price with Moving Averages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(IMAGES_DIR, f"{ticker}_price_moving_averages.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filepath}")
    return filepath


def plot_daily_returns(df: pd.DataFrame, ticker: str = "AAPL") -> str:
    """Plot the distribution of daily percentage returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series
    axes[0].plot(df.index, df["Daily_Return"], linewidth=0.8, color="#1f77b4")
    axes[0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0].set_title(f"{ticker} – Daily Returns (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Return (%)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[0].grid(True, alpha=0.3)

    # Histogram
    returns_clean = df["Daily_Return"].dropna()
    axes[1].hist(returns_clean, bins=40, color="#1f77b4", edgecolor="white", alpha=0.8)
    axes[1].axvline(returns_clean.mean(), color="red", linestyle="--",
                    label=f"Mean: {returns_clean.mean():.2f}%")
    axes[1].set_title(f"{ticker} – Return Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Daily Return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    filepath = os.path.join(IMAGES_DIR, f"{ticker}_daily_returns.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filepath}")
    return filepath


def print_summary(df: pd.DataFrame, ticker: str = "AAPL") -> None:
    """Print a concise statistical summary of the stock data."""
    print(f"\n{'='*50}")
    print(f"  {ticker} Stock Data Summary")
    print(f"{'='*50}")
    print(f"  Period       : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Trading Days : {len(df)}")
    print(f"  Start Price  : ${df['Close'].iloc[0]:.2f}")
    print(f"  End Price    : ${df['Close'].iloc[-1]:.2f}")
    print(f"  Min Close    : ${df['Close'].min():.2f}")
    print(f"  Max Close    : ${df['Close'].max():.2f}")
    returns = df["Daily_Return"].dropna()
    print(f"  Avg Daily Ret: {returns.mean():.4f}%")
    print(f"  Volatility   : {returns.std():.4f}%  (daily std dev)")
    total = df["Cumulative_Return"].iloc[-1] * 100
    print(f"  Total Return : {total:.2f}%")
    print(f"{'='*50}\n")


def main() -> None:
    print("Loading data …")
    df = load_data(DATA_PATH)

    print("Calculating moving averages …")
    df = calculate_moving_averages(df, short_window=20, long_window=50)

    print("Calculating returns …")
    df = calculate_returns(df)

    print_summary(df)

    print("Generating plots …")
    plot_price_with_moving_averages(df)
    plot_daily_returns(df)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
