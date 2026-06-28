import inspect
import traceback
import pandas as pd

from backend.data.data_fetcher import fetch_ohlcv

from backend.indicators import (
    statistics,
    candle,
    support_resistance,
    smart_money,
)

MODULES = [
    statistics,
    candle,
    support_resistance,
    smart_money,
]

symbol = "RELIANCE"
df = fetch_ohlcv(symbol, limit=500)

for module in MODULES:
    print("\n" + "=" * 80)
    print("MODULE:", module.__name__)
    print("=" * 80)

    for name, func in inspect.getmembers(module, inspect.isfunction):

        # private function skip
        if name.startswith("_"):
            continue

        try:
            sig = inspect.signature(func)

            kwargs = {}

            for p in sig.parameters.values():

                if p.name == "data":
                    kwargs["data"] = df

                elif p.name == "source":
                    kwargs["source"] = "close"

                elif p.default is not inspect._empty:
                    kwargs[p.name] = p.default

            result = func(**kwargs)

            print(f"\n{name}  ✅")

            if isinstance(result, pd.Series):
                print(result.tail())

            elif isinstance(result, pd.DataFrame):
                print(result.tail())

            else:
                print(result)

        except Exception as e:
            print(f"\n{name}  ❌")
            print(type(e).__name__, e)
