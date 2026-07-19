import yfinance as yf
from backend.indicators.indicator_engine import build_features

df = yf.download("RELIANCE.NS", period="6mo", auto_adjust=False)

if df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

df.columns = (
    df.columns.astype(str)
              .str.strip()
              .str.lower()
              .str.replace(" ", "_")
)

print(df.columns.tolist())

result = build_features(df)

print(result.tail())
