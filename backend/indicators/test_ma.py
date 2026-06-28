from backend.data.data_fetcher import fetch_ohlcv
from backend.indicators.moving_average import *

symbol = "RELIANCE"

df = fetch_ohlcv(symbol, limit=500)

print("=" * 70)
print(f"SYMBOL : {symbol}")
print(f"ROWS   : {len(df)}")
print("=" * 70)

tests = {
    "SMA20": sma(df, length=20),
    "EMA20": ema(df, length=20),
    "SMMA20": smma(df, length=20),
    "WMA20": wma(df, length=20),
    "VWMA20": vwma(df, length=20),
    "LSMA20": lsma(df, length=20),

    "DEMA20": dema(df, length=20),
    "TEMA20": tema(df, length=20),
    "TRIMA20": trima(df, length=20),
    "HMA20": hma(df, length=20),
    "ZLEMA20": zlema(df, length=20),

    "KAMA10": kama(df, length=10),
    "ALMA9": alma(df, length=9),
    "T3_5": t3(df, length=5),
    "MCGINLEY14": mcginley_dynamic(df, length=14),
    "VIDYA9": vidya(df, length=9),
}

print()

for name, series in tests.items():
    print("=" * 70)
    print(name)

    last = series.dropna()

    if last.empty:
        print("Result : ALL NaN")
        continue

    print(last.tail())

print("=" * 70)
print("ALL MOVING AVERAGES FINISHED")
print("=" * 70)
