#!/usr/bin/env python3

import argparse
import json
import sqlite3
import pandas as pd
import numpy as np

from backend.config.settings import settings

# Import L1 Analyzers
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.pattern_analyzer import PatternAnalyzer
from backend.analyzers.smc_analyzer import SmartMoneyAnalyzer
from backend.analyzers.sr_analyzer import SupportResistanceAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer

# Import L2 Master Engine
from backend.engines.l2_master_engine import L2MasterEngine

def get_historical_features(db, symbol=None, limit=300):
    query = "SELECT * FROM feature_history"
    params = ()
    if symbol:
        query += " WHERE symbol = ?"
        params = (symbol,)
    query += f" ORDER BY date DESC LIMIT {limit}"

    df = pd.read_sql_query(query, db, params=params)
    if df.empty:
        raise RuntimeError(f"No rows found in feature_history for symbol={symbol}")

    return df.sort_values(by="date").reset_index(drop=True)

def generate_feature_blocks(df: pd.DataFrame) -> dict:
    """
    Maps the full DataFrame to the expected block categories for each analyzer.
    The V6 engines are smart enough to filter missing columns natively.
    """
    return {
        "trend_analyzer": {"adaptive": df, "baseline": df, "regression_fit": df, "regression_slope": df, "regime": df, "momentum": df},
        "volatility_analyzer": {"atr_block": df, "band_block": df, "dispersion_block": df, "osc_block": df, "noise_block": df},
        "momentum_analyzer": {"oscillator_block": df, "kinematic_block": df, "directional_block": df},
        "pattern_analyzer": {"compression_metrics": df, "kinetic_patterns": df, "accumulation_patterns": df, "distribution_patterns": df, "geometry_boundaries": df, "meta_metrics": df},
        "smart_money_analyzer": {"structure_block": df, "liquidity_block": df, "zone_block": df, "pricing_block": df, "scoring_block": df},
        "sr_analyzer": {"meta_metrics": df, "dynamic_zones": df, "cluster_metrics": df, "levels": df},
        "volume_analyzer": {"relative_metrics": df, "flow_metrics": df, "kinetic_metrics": df, "vwap_metrics": df, "institutional_metrics": df},
        "candle_analyzer": {"pressure_block": df, "volatility_block": df, "rejection_block": df, "gap_block": df}
    }

def main():
    parser = argparse.ArgumentParser(description="Run V6 Full Institutional Architecture against Live DB.")
    parser.add_argument("--symbol", default="RELIANCE", help="Symbol to test, e.g. RELIANCE")
    parser.add_argument("--limit", type=int, default=300, help="Days of historical data to fetch")
    args = parser.parse_args()

    print(f"🔄 Initializing Green Bull V6 Pipeline for {args.symbol}...")

    # Initialize all engines
    l1_engines = {
        "trend_analyzer": TrendAnalyzer(),
        "volatility_analyzer": VolatilityAnalyzer(),
        "momentum_analyzer": MomentumAnalyzer(),
        "pattern_analyzer": PatternAnalyzer(),
        "smart_money_analyzer": SmartMoneyAnalyzer(),
        "sr_analyzer": SupportResistanceAnalyzer(),
        "volume_analyzer": VolumeAnalyzer(),
        "candle_analyzer": CandleAnalyzer()
    }
    master_engine = L2MasterEngine()

    # Connect to DB and fetch data
    db = sqlite3.connect(settings.database_path)
    try:
        df = get_historical_features(db, args.symbol, limit=args.limit)
        latest_row = df.iloc[-1].to_dict()
        current_price = float(latest_row.get("close", 0.0))
        display_date = latest_row.get('date', latest_row.get('timestamp', 'UNKNOWN'))
        
        print(f"✅ Loaded {len(df)} rows from Database. Current Price: {current_price}")
        
        # Route Data Blocks
        feature_blocks = generate_feature_blocks(df)

        # Execute L1 Analyzers
        print("\n⚙️ Executing L1 Analyzers...")
        l1_contracts = {}
        for analyzer_name, engine in l1_engines.items():
            try:
                blocks = feature_blocks.get(analyzer_name, {})
                result = engine.analyze(blocks)
                l1_contracts[analyzer_name] = result
                print(f"  [✔] {analyzer_name} completed.")
            except Exception as e:
                print(f"  [❌] {analyzer_name} FAILED: {str(e)}")
                l1_contracts[analyzer_name] = {}

        # Execute L2 Master Engine
        print("\n🧠 Executing L2 Master Engine (Setup & Execution)...")
        final_decision = master_engine.evaluate_market(l1_contracts, current_price)

        # Display Final Combined Result
        print("\n" + "=" * 60)
        print(f"🎯 SYMBOL   : {args.symbol}")
        print(f"📅 DATE     : {display_date}")
        print(f"📊 DATA ROWS: {len(df)} days processed")
        print("=" * 60)
        
        # We wrap L1 and L2 results together for full visibility
        full_output = {
            "L2_MASTER_DECISION": final_decision,
            "L1_ANALYZER_CONTRACTS": l1_contracts
        }
        
        print(json.dumps(full_output, indent=2, ensure_ascii=False, allow_nan=False))

    except Exception as e:
        print(f"\n❌ Pipeline Execution Failed: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
