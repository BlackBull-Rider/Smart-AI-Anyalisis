import inspect
import importlib
import sys
import logging

# তোর ৮টি ইন্ডিকেটর ফাইল
INDICATOR_MODULES = [
    "backend.indicators.candle",
    "backend.indicators.momentum",
    "backend.indicators.moving_average",
    "backend.indicators.smart_money",
    "backend.indicators.statistics",
    "backend.indicators.support_resistance",
    "backend.indicators.volatility",
    "backend.indicators.volume"
]

class DynamicRegistry:
    def __init__(self):
        self._registry = {}
        self._load_all()

    def _load_all(self):
        print("--- DISCOVERING INDICATORS ---")
        for module_path in INDICATOR_MODULES:
            try:
                mod = importlib.import_module(module_path)
                # সব ফাংশন খুঁজে বের করা
                funcs = inspect.getmembers(mod, inspect.isfunction)
                for name, func in funcs:
                    # ফাংশনের ফুল পাথ আইডি হিসেবে সেভ করা
                    key = f"{module_path.split('.')[-1]}.{name}"
                    self._registry[key] = func
                print(f"Loaded {len(funcs)} from {module_path}")
            except Exception as e:
                print(f"Failed to load {module_path}: {e}")
        print(f"Total Indicators Registered: {len(self._registry)}")

    def get(self, indicator_key):
        return self._registry.get(indicator_key)

# সিঙ্গেলটন ইনস্ট্যান্স (সারা সিস্টেমে একটাই থাকবে)
registry = DynamicRegistry()
