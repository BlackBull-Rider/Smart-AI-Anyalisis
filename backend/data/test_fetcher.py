from backend.data.data_fetcher import (
    fetch_ohlcv,
    fetch_fundamental,
    fetch_ipo,
    fetch_stock_master,
)

symbol = "RELIANCE"

print("=" * 60)
print("OHLCV")
print(fetch_ohlcv(symbol, 5))

print("=" * 60)
print("FUNDAMENTAL")
print(fetch_fundamental(symbol))

print("=" * 60)
print("IPO")
try:
    print(fetch_ipo(symbol))
except Exception as e:
    print(e)

print("=" * 60)
print("STOCK MASTER")
print(fetch_stock_master(symbol))
