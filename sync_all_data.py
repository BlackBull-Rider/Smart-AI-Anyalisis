import pandas as pd
from datetime import datetime, timedelta
from backend.repository.stock_repository import repository
from backend.providers.yahoo_provider import YahooProvider

# Initialize
provider = YahooProvider()
symbols = [row['symbol'] for row in repository.get_active_symbols()]

print(f"Starting incremental sync for {len(symbols)} symbols...")

for symbol in symbols:
    # Incremental Logic
    last_date = repository.get_last_history_date(symbol)
    
    if last_date:
        # Convert string date to datetime
        start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # First time sync (last 30 days)
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if we already have the latest data
    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        continue

    print(f"Syncing {symbol} | From: {start_date} To: {end_date}")
    
    try:
        df = provider.get_history(symbol, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            repository.save_history(df)
            print(f"✓ Saved {len(df)} rows for {symbol}")
        else:
            print(f"⚠ No data for {symbol}")
    except Exception as e:
        print(f"✗ Failed {symbol}: {e}")

print("--- Sync Complete ---")
