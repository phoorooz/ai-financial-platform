from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("ml/data/raw")


def download_stock_data(
    symbol: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download historical market data."""

    print(f"Downloading {symbol} data...")

    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )

    if data.empty:
        raise ValueError(f"No data found for {symbol}")

    # yfinance may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Standardize column names.
    data.columns = [column.lower() for column in data.columns]

    return data


def save_data(data: pd.DataFrame, symbol: str) -> Path:
    """Save downloaded data as CSV."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol.lower()}_historical.csv"
    output_path = RAW_DATA_DIR / filename

    data.to_csv(output_path, index=False)

    print(f"\nData saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    symbol = "BTC-USD"

    df = download_stock_data(
        symbol=symbol,
        start="2020-01-01",
        end="2026-01-01",
    )

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:", df.shape)

    save_data(df, symbol)