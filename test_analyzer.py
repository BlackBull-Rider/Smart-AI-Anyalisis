#!/usr/bin/env python3

import argparse
import json
import sqlite3
import importlib
import pandas as pd

from backend.config.settings import settings

def get_historical_features(db, symbol=None, limit=300):
    """
    ডেটাবেস থেকে ৩০০ দিনের হিস্টোরিক্যাল ডেটা এনে Pandas DataFrame রিটার্ন করে।
    """
    query = """
        SELECT *
        FROM feature_history
    """
    params = ()

    if symbol:
        query += " WHERE symbol = ?"
        params = (symbol,)

    # লেটেস্ট ডেটা আগে আনার জন্য DESC, কিন্তু টাইমলাইন সোজা রাখতে পরে ASC করা হবে
    query += f"""
        ORDER BY date DESC
        LIMIT {limit}
    """

    # SQL query থেকে সরাসরি Pandas DataFrame তৈরি
    df = pd.read_sql_query(query, db, params=params)

    if df.empty:
        raise RuntimeError(
            "No rows found in feature_history"
            + (f" for symbol={symbol!r}" if symbol else "")
        )

    # DataFrame-এর ডেটা পুরোনো থেকে নতুন (Ascending) অর্ডারে সাজানো
    df = df.sort_values(by="date").reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Run a backend analyzer against the feature_history."
    )

    parser.add_argument(
        "--analyzer",
        required=True,
        help="Analyzer function name, e.g. trend_analyzer or candle_analyzer",
    )

    parser.add_argument(
        "--symbol",
        help="Optional symbol filter, e.g. RELIANCE",
    )

    args = parser.parse_args()

    # Dynamically import the analyzer
    try:
        module = importlib.import_module(f"backend.analyzers.{args.analyzer}")
        analyzer_func = getattr(module, args.analyzer, None)
    except ModuleNotFoundError:
        raise SystemExit(f"Error: Module backend.analyzers.{args.analyzer} not found.")

    if analyzer_func is None or not callable(analyzer_func):
        raise SystemExit(
            f"Error: Callable function '{args.analyzer}' not found inside backend/analyzers/{args.analyzer}.py"
        )

    db = sqlite3.connect(settings.database_path)
    db.row_factory = sqlite3.Row

    try:
        # ৩০০ দিনের DataFrame আনা হলো
        df = get_historical_features(db, args.symbol, limit=300)

        # লেটেস্ট দিনের ডেটা (Dict) - অন্যান্য রেগুলার অ্যানালাইজারের জন্য
        latest_data = df.iloc[-1].to_dict()
        
        display_symbol = latest_data.get('symbol', args.symbol or 'UNKNOWN')
        display_date = latest_data.get('date', latest_data.get('timestamp', 'UNKNOWN'))

        # অ্যানালাইজারে features (Dict) এবং context (DataFrame) পাঠানো হচ্ছে
        # ট্রেন্ড ইঞ্জিন চাইলে context["history_df"] থেকে ৩০০ দিনের ডেটা পড়তে পারবে
        context = {
            "history_df": df
        }
        
        result = analyzer_func(latest_data, context=context)

        # ------------------------------------------------------------
        # Meta Info Header
        # ------------------------------------------------------------
        print("=" * 60)
        print(f"🎯 SYMBOL   : {display_symbol}")
        print(f"📅 DATE     : {display_date}")
        print(f"⚙️  ANALYZER : {args.analyzer}")
        print(f"📊 DATA ROWS: {len(df)} days loaded")
        print("=" * 60)

        # ------------------------------------------------------------
        # JSON Output
        # ------------------------------------------------------------
        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ))

    finally:
        db.close()

if __name__ == "__main__":
    main()
