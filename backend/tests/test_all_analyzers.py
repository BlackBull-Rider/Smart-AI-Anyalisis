# backend/tests/test_all_analyzers.py

import json
import traceback
from pathlib import Path

import pandas as pd

from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.smart_money_analyzer import SmartMoneyAnalyzer
from backend.analyzers.institutional_analyzer import InstitutionalAnalyzer


CSV_FILE = "data/processed/RELIANCE.csv"


def print_header(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def run_analyzer(name, analyzer, df):

    print_header(name)

    try:

        result = analyzer.analyze(df)

        print(json.dumps(result, indent=4, default=str))

        print(f"\n{name} : SUCCESS")

        return True

    except Exception as e:

        print(f"\n{name} : FAILED")

        print(e)

        traceback.print_exc()

        return False


def main():

    df = pd.read_csv(CSV_FILE)

    analyzers = [

        ("TREND ANALYZER", TrendAnalyzer()),

        ("MOMENTUM ANALYZER", MomentumAnalyzer()),

        ("VOLUME ANALYZER", VolumeAnalyzer()),

        ("VOLATILITY ANALYZER", VolatilityAnalyzer()),

        ("SMART MONEY ANALYZER", SmartMoneyAnalyzer()),

        ("INSTITUTIONAL ANALYZER", InstitutionalAnalyzer()),

    ]

    passed = 0

    failed = 0

    for name, analyzer in analyzers:

        ok = run_analyzer(name, analyzer, df)

        if ok:
            passed += 1
        else:
            failed += 1

    print_header("FINAL RESULT")

    print("Total Analyzer :", len(analyzers))
    print("Passed         :", passed)
    print("Failed         :", failed)

    if failed == 0:
        print("\nALL ANALYZERS PASSED")
    else:
        print("\nSOME ANALYZERS FAILED")


if __name__ == "__main__":
    main()
