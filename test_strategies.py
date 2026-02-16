from app.data_loader import fetch_data, resample_data
import pandas as pd
import datetime

def test_fetch_and_resample():
    # Use a recent date for intraday data as yfinance limits historical data for small intervals
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')

    ticker = "ES=F"
    print(f"Testing fetch_data for {ticker} from {start_date} to {end_date}...")

    # Test 5m fetch
    df_5m = fetch_data(ticker, start_date=start_date, end_date=end_date, interval="5m")

    if df_5m.empty:
        print("Test Failed: Fetch returned empty DataFrame.")
        # Try a longer timeframe just in case market is closed or something
        print("Retrying with daily data to confirm API works...")
        df_1d = fetch_data(ticker, start_date=start_date, end_date=end_date, interval="1d")
        if df_1d.empty:
            print("Test Failed: Even daily data is empty. Check API/Ticker.")
            return
        else:
            print("Daily data fetch successful.")
            df_5m = df_1d # proceed with daily for structure check
    else:
        print(f"Fetch Successful. Shape: {df_5m.shape}")
        print(df_5m.head())

    # Verify columns
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(col in df_5m.columns for col in expected_cols):
        print("Column Verification Passed.")
    else:
        print(f"Column Verification Failed. Found: {df_5m.columns}")

    # Test Resample
    print("\nTesting resample_data to 1h...")
    df_1h = resample_data(df_5m, '1h')

    if df_1h.empty:
        print("Resample returned empty DataFrame.")
    else:
        print(f"Resample Successful. Shape: {df_1h.shape}")
        print(df_1h.head())
        # Check if index is correct
        print(f"Index freq: {df_1h.index.freqstr}")

if __name__ == "__main__":
    test_fetch_and_resample()
