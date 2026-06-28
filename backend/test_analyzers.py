import json
import traceback
import pandas as pd

from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer


CSV_FILE = "data/processed/RELIANCE.csv"


def print_header(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def run(name, analyzer, df):

    print_header(name)

    try:
        result = analyzer.analyze(df)

        print(json.dumps(result, indent=4, default=str))

        print("\n✅ SUCCESS")

    except Exception as e:

        print("\n❌ FAILED")

        traceback.print_exc()

        print(e)


def main():

    df = pd.read_csv(CSV_FILE)

    run(
        "TREND ANALYZER",
        TrendAnalyzer(),
        df
    )

    run(
        "MOMENTUM ANALYZER",
        MomentumAnalyzer(),
        df
    )


if __name__ == "__main__":
    main()
