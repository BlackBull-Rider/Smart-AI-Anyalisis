import inspect
import importlib
import backend.indicators.candle as candle
import backend.indicators.momentum as mom
import backend.indicators.moving_average as ma
import backend.indicators.smart_money as smc
import backend.indicators.statistics as stat
import backend.indicators.support_resistance as sr
import backend.indicators.volatility as vol
import backend.indicators.volume as vol_u

def count_in_module(module):
    # সব ফাংশন এক্সট্রাক্ট করা
    functions = [f[0] for f in inspect.getmembers(module, inspect.isfunction)]
    # কিছু হেল্পার ফাংশন বাদ দিয়ে শুধু ইন্ডিকেটর গুনা (ঐচ্ছিক)
    return functions

modules = {
    "candle": candle,
    "momentum": mom,
    "moving_average": ma,
    "smart_money": smc,
    "statistics": stat,
    "support_resistance": sr,
    "volatility": vol,
    "volume": vol_u
}

print("--- INDICATOR COUNT REPORT ---")
total = 0
for name, mod in modules.items():
    funcs = count_in_module(mod)
    print(f"{name}: {len(funcs)} functions")
    total += len(funcs)

print(f"\nমোট ফাংশন সংখ্যা: {total}")
