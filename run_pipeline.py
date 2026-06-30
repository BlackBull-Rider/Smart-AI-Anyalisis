import yfinance as yf
import pandas as pd

from backend.registry.feature_engine import build_features

from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer

TEST_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "TATAMOTORS.NS",
    "MARUTI.NS",
    "M&M.NS",
    "BHARTIARTL.NS",
    "BAJFINANCE.NS",
    "SUNPHARMA.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "ADANIPORTS.NS",
    "NTPC.NS"
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

            df = yf.download(
                symbol,
                period="300d",
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                raise Exception("No Data")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

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
