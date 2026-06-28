import json
import time
import traceback

from backend.indicators.data_fetcher import fetch_ohlcv
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer


SYMBOL = "RELIANCE"


def run_test(name, analyzer, df):
    start = time.perf_counter()

    try:
        result = analyzer.analyze(df)

        elapsed = time.perf_counter() - start

        return {
            "name": name,
            "status": "PASS",
            "execution_time": round(elapsed, 4),
            "result_type": type(result).__name__,
            "keys": list(result.keys()) if isinstance(result, dict) else [],
        }

    except Exception as e:

        elapsed = time.perf_counter() - start

        return {
            "name": name,
            "status": "FAIL",
            "execution_time": round(elapsed, 4),
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():

    print("=" * 80)
    print("GREEN BULL RIDER ANALYZER TEST")
    print("=" * 80)

    df = fetch_ohlcv(SYMBOL, limit=500)

    print(f"Rows : {len(df)}")
    print(f"Columns : {list(df.columns)}")
    print()

    analyzers = [
        ("Trend Analyzer", TrendAnalyzer()),
        ("Momentum Analyzer", MomentumAnalyzer()),
    ]

    report = []

    for name, analyzer in analyzers:

        print(f"Running {name}...")

        result = run_test(name, analyzer, df)

        report.append(result)

        print(result["status"])

        if result["status"] == "FAIL":
            print(result["error"])

        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(r["status"] == "PASS" for r in report)
    failed = sum(r["status"] == "FAIL" for r in report)

    print(f"PASS : {passed}")
    print(f"FAIL : {failed}")

    with open("analyzer_test_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print()
    print("Report saved -> analyzer_test_report.json")


if __name__ == "__main__":
    main()
