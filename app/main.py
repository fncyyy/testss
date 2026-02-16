import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start_date: str, end_date: str, interval: str = "5m") -> pd.DataFrame:
    """
    Fetches historical data for a given ticker.

    Args:
        ticker: The symbol (e.g., 'NQ=F', 'ES=F').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        interval: Data interval (e.g., '1m', '5m', '15m', '1h', '1d').

    Returns:
        pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date} at {interval} interval...")
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    if data.empty:
        # Try fetching without start/end if specific dates fail, or just raise error
        # yfinance sometimes fails with specific dates for intraday if they are too far back
        # Let's just return empty or raise
        print(f"Warning: No data found for {ticker}.")
        return pd.DataFrame()

    # Flatten multi-level columns if present (yfinance recently changed this behavior)
    if isinstance(data.columns, pd.MultiIndex):
        # We only want the Price level (or just the column names if simple)
        # Usually checking if 'Ticker' is in level 1
        if data.columns.nlevels > 1:
             data.columns = data.columns.get_level_values(0)

    # Standardize column names
    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # Keep only required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in required_cols if c in data.columns]
    data = data[available_cols]

    # Drop rows with NaN
    data = data.dropna()

    return data

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamples the dataframe to a higher timeframe.

    Args:
        df: The source DataFrame (must have DatetimeIndex).
        timeframe: Target timeframe (e.g., '1h', '4h').

    Returns:
        pd.DataFrame: Resampled DataFrame.
    """
    if df.empty:
        return df

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)

    # Convert timeframe format if needed (e.g. '1m' to '1min')
    # Pandas usually accepts '1h', '4h', '1min'
    # Streamlit/User might pass '1h'. Pandas 'h' is deprecated for 'H'? No, 'h' is usually fine in newer pandas.
    # But let's be safe.

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Handle missing columns if volume is not present (unlikely)
    if 'volume' not in df.columns:
        agg_dict.pop('volume')

    try:
        resampled = df.resample(timeframe).agg(agg_dict)
        resampled = resampled.dropna()
        return resampled
    except Exception as e:
        print(f"Error resampling data: {e}")
        return pd.DataFrame()
