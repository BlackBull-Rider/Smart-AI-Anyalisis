import pandas as pd
from backend.data.data_fetcher import fetch_ohlcv
from backend.registry.feature_engine import build_features
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer
from backend.analyzers.pattern_analyzer import PatternAnalyzer

TEST_STOCKS = [
    "ADANIPOWER",
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASHOKLEY",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJCON",
    "BAJAJFINSV",
    "BAJFINANCE",
    "BALAMINES",
    "BANDHANBNK",
    "BANKBARODA",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CANBK",
    "CENTRALBK",
    "CHOLAFIN",
    "CIPLA",
    "COALINDIA",
    "CPSEETF",
    "DCBBANK",
    "DIVISLAB",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "FEDERALBNK",
    "GAIL"
]

def run():
    total = 0
    success = 0
    failed = 0

    for symbol in TEST_STOCKS:
        total += 1

        print("\n" + "=" * 80)
        print(f"PIPELINE : {symbol}")
        print("=" * 80)

        try:
            print("Fetching data...")

            # ১. ডাটাবেসের সাথে নাম ম্যাচ করার জন্য .NS রিমুভ করা হলো
            clean_symbol = symbol.replace(".NS", "")

            # ২. yfinance বাদ দিয়ে লোকাল ডাটাবেস থেকে ফেচ করা হলো
            df = fetch_ohlcv(clean_symbol, limit=300)

            if df.empty:
                raise Exception("No Data")

            # ৩. লোকাল ডিবিতে MultiIndex থাকে না, তাই শুধু কলামের নাম লোয়ারকেস (lowercase) করা হলো
            df.columns = [str(c).lower() for c in df.columns]

            print("Building Features...")
            df = build_features(df)

            print("Running Trend Analyzer...")
            trend = TrendAnalyzer().analyze(df)

            if "trend" in trend:
                trend_state = trend["trend"].get(
                    "state",
                    trend["trend"].get("regime", "Unknown")
                )
            else:
                trend_state = trend["direction"].get("status", "Unknown")

            print("Trend :", trend_state)

            print("Running Momentum Analyzer...")
            momentum = MomentumAnalyzer().analyze(df)

            if "momentum_strength" in momentum:
                momentum_state = momentum["momentum_strength"].get("status")
            else:
                momentum_state = momentum["strength"].get(
                    "state",
                    momentum["strength"].get("status")
                )

            print("Momentum :", momentum_state)

            print("Running Volatility Analyzer...")
            volatility = VolatilityAnalyzer().analyze(df)

            print(
                "Volatility :",
                volatility["state"]["regime"]
            )

            print("Running Volume Analyzer...")
            volume = VolumeAnalyzer().analyze(df)

            print(
                "Volume :",
                volume["smart_volume"]["dominance"]
            )
            print("Running Candle Analyzer...")
            candle = CandleAnalyzer().analyze(df)

            if "candle_psychology" in candle:
                candle_state = candle["candle_psychology"].get("status", "Unknown")
            else:
                candle_state = "Unknown"

            print("Candle :", candle_state)

            print("Running Pattern Analyzer...")
            pattern = PatternAnalyzer().analyze(df)

            if "advanced_pattern_metrics" in pattern:
                primary_pattern = pattern["advanced_pattern_metrics"].get("primary_pattern_id", "NONE")
                if primary_pattern == "NONE":
                    pattern_state = "No Pattern Detected"
                else:
                    pattern_state = f"{primary_pattern} Detected"
            else:
                pattern_state = "Unknown"

            print("Pattern :", pattern_state)

            print("STATUS : PASS")
            success += 1

        except Exception as e:
            failed += 1
            print("STATUS : FAILED")
            print(type(e).__name__)
            print(e)

    print("\n")
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    print(f"Total Stocks : {total}")
    print(f"Passed       : {success}")
    print(f"Failed       : {failed}")

    print("=" * 80)


if __name__ == "__main__":
    run()
