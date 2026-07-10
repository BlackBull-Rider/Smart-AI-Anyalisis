import inspect
import pandas as pd

from backend.indicators.data_fetcher import fetch_ohlcv
from backend.indicators.core import (
    moving_average,
    momentum,
    volume,
    volatility,
    candle,
    pattern,
    support_resistance,
    statistics,
    smart_money,
)

df = fetch_ohlcv("RELIANCE")

modules = [
    moving_average,
    momentum,
    volume,
    volatility,
    candle,
    pattern,
    support_resistance,
    statistics,
    smart_money,
]

print("=" * 100)
print("GREEN BULL RIDER V6 - INDICATOR TEST")
print("=" * 100)

passed = 0
failed = 0

for module in modules:
    print(f"\n### {module.__name__}")
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            sig = inspect.signature(fn)
            args = []

            for p in sig.parameters.values():
                pname = p.name.lower()

                if pname in ("df", "data", "ohlcv"):
                    args.append(df)
                elif pname in ("benchmark", "bench"):
                    args.append(df)
                elif p.default != inspect._empty:
                    pass
                else:
                    raise Exception(f"Needs arg: {p.name}")

            out = fn(*args)

            if isinstance(out, (pd.Series, pd.DataFrame, dict, float, int, bool)):
                print(f"PASS  {name}")
                passed += 1
            else:
                print(f"PASS  {name} ({type(out).__name__})")
                passed += 1

        except Exception as e:
            print(f"FAIL  {name} --> {e}")
            failed += 1

print("\n" + "=" * 100)
print(f"PASS : {passed}")
print(f"FAIL : {failed}")
print(f"TOTAL: {passed + failed}")
print("=" * 100)
