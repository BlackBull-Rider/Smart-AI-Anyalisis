#!/usr/bin/env python3

import sqlite3
import pandas as pd
import argparse
from tqdm import tqdm

from backend.config.settings import settings
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.pattern_analyzer import PatternAnalyzer
from backend.analyzers.smc_analyzer import SmartMoneyAnalyzer
from backend.analyzers.sr_analyzer import SupportResistanceAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer
from backend.engines.l2_master_engine import L2MasterEngine

def get_historical_features(db, symbol, limit):
    query = f"SELECT * FROM feature_history WHERE symbol = ? ORDER BY date DESC LIMIT {limit}"
    df = pd.read_sql_query(query, db, params=(symbol,))
    return df.sort_values(by="date").reset_index(drop=True)

def generate_blocks(df):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--scan_days", type=int, default=100, help="Number of days to scan backward")
    args = parser.parse_args()

    db = sqlite3.connect(settings.database_path)
    # Fetch 300 days so the rolling windows (like 200 SMA) have enough data to calculate
    df = get_historical_features(db, args.symbol, limit=300)
    db.close()

    if len(df) < args.scan_days + 50:
        print("Not enough data to scan.")
        return

    l1_engines = {
        "trend_analyzer": TrendAnalyzer(), "volatility_analyzer": VolatilityAnalyzer(),
        "momentum_analyzer": MomentumAnalyzer(), "pattern_analyzer": PatternAnalyzer(),
        "smart_money_analyzer": SmartMoneyAnalyzer(), "sr_analyzer": SupportResistanceAnalyzer(),
        "volume_analyzer": VolumeAnalyzer(), "candle_analyzer": CandleAnalyzer()
    }
    master_engine = L2MasterEngine()

    print(f"\n🔍 Scanning {args.symbol} for Institutional Setups over the last {args.scan_days} days...\n")
    
    found_setups = 0
    
    # Loop through the historical data day by day
    start_idx = len(df) - args.scan_days
    for i in tqdm(range(start_idx, len(df)), desc="Scanning"):
        # Slice data up to the current day in the loop
        current_slice = df.iloc[:i+1].copy()
        current_date = current_slice.iloc[-1].get('date', 'Unknown Date')
        current_price = float(current_slice.iloc[-1].get('close', 0.0))
        
        blocks = generate_blocks(current_slice)
        l1_contracts = {name: engine.analyze(blocks.get(name, {})) for name, engine in l1_engines.items()}
        
        decision = master_engine.evaluate_market(l1_contracts, current_price)
        
        setup = decision.get("setup", {})
        execution = decision.get("execution", {})
        
        # If a setup is found, print it
        if setup.get("setup_name") != "none":
            found_setups += 1
            print(f"\n" + "="*50)
            print(f"🚨 SETUP FOUND: {current_date}")
            print(f"==================================================")
            print(f"🔹 Setup Type : {setup['setup_name'].upper()}")
            print(f"🔹 Direction  : {setup['direction'].upper()} (Readiness: {setup['readiness']})")
            print(f"🔹 Confluence : {setup['confluence_score']} / 5")
            print(f"🔹 Execution  : {execution['status'].upper()}")
            if execution['status'] in ['execute_now', 'pending_close']:
                print(f"   ↳ Entry: {execution['entry_price_ref']}")
                print(f"   ↳ RRR  : {execution['validated_rrr']} ({execution['rrr_profile']})")
            else:
                print(f"   ↳ Reason: {execution.get('status')}")
            print("="*50)

    print(f"\n✅ Scan Complete! Found {found_setups} institutional setups in {args.scan_days} days.")

if __name__ == "__main__":
    main()
