#!/usr/bin/env python3

import argparse
import json
import sqlite3
import importlib
import pandas as pd
import numpy as np

from backend.config.settings import settings

# V6 Class Mapping
ANALYZER_CLASSES = {
    "trend_analyzer": "TrendAnalyzer",
    "volatility_analyzer": "VolatilityAnalyzer",
    "momentum_analyzer": "MomentumAnalyzer",
    "pattern_analyzer": "PatternAnalyzer",
    "smc_analyzer": "SmartMoneyAnalyzer",
    "sr_analyzer": "SupportResistanceAnalyzer",
    "volume_analyzer": "VolumeAnalyzer",
    "candle_analyzer": "CandleAnalyzer"
}

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

def route_feature_blocks(df: pd.DataFrame, analyzer_name: str) -> dict:
    """
    Routes flat database columns into the specific grouped DataFrames 
    expected by the V6 Institutional Analyzers.
    """
    blocks = {}
    cols = set(df.columns)

    if analyzer_name == "trend_analyzer":
        blocks["adaptive"] = df[[c for c in cols if c in ['kama_10', 'alma_9', 'dema_20', 'tema_20', 'trima_20', 'wma_20', 'hma_20', 'mcginley_14', 'smma_20', 't3_5', 'vidya_9', 'vwma_20', 'zlema_20', 'lsma_25']]]
        blocks["baseline"] = df[[c for c in cols if c in ['sma_20', 'ema_20', 'sma_50', 'ema_50', 'sma_100', 'ema_100', 'sma_200', 'ema_200']]]
        blocks["regression_fit"] = df[[c for c in cols if c in ['regression_r2', 'linear_regression_r2', 'r_squared', 'adjusted_r_squared']]]
        blocks["regression_slope"] = df[[c for c in cols if c in ['regression_slope', 'linear_regression_slope', 'rolling_slope', 'slope_percentage', 'trend_angle']]]
        blocks["regime"] = df[[c for c in cols if c in ['supertrend', 'supertrend_trend']]]
        blocks["momentum"] = df[[c for c in cols if c in ['adx_14', 'plus_di', 'minus_di', 'dx']]]
    
    elif analyzer_name == "volatility_analyzer":
        blocks["atr_block"] = df[[c for c in cols if 'atr' in c or 'tr' == c]]
        blocks["band_block"] = df[[c for c in cols if 'bb_' in c]]
        blocks["dispersion_block"] = df[[c for c in cols if c in ['std_20', 'var_20', 'standard_error', 'hv_21', 'parkinson_vol', 'garman_klass', 'rogers_satchell', 'yang_zhang']]]
        blocks["osc_block"] = df[[c for c in cols if c in ['chaikin_vol', 'ulcer_index', 'volatility_ratio', 'expansion_index', 'volatility_osc']]]
        blocks["noise_block"] = df[[c for c in cols if c in ['choppiness', 'vhf']]]

    elif analyzer_name == "momentum_analyzer":
        blocks["oscillator_block"] = df[[c for c in cols if c in ['rsi_14', 'stoch_k', 'stoch_d', 'stoch_rsi_k', 'stoch_rsi_d', 'williams_r', 'ultimate_osc', 'cci_20']]]
        blocks["kinematic_block"] = df[[c for c in cols if c in ['macd', 'macd_signal', 'macd_hist', 'ppo', 'ppo_signal', 'ppo_hist', 'roc_12', 'mom_10', 'trix_18', 'dpo_20', 'rsi_slope']]]
        blocks["directional_block"] = df[[c for c in cols if c in ['adx_14', 'plus_di', 'minus_di', 'dx']]]

    else:
        # Generic routing for Pattern, SMC, S/R, Volume, Candle
        # (Passes the whole DataFrame dynamically for analyzers that auto-filter internally)
        blocks = {
            "meta_metrics": df, "dynamic_zones": df, "cluster_metrics": df, "levels": df,
            "compression_metrics": df, "kinetic_patterns": df, "accumulation_patterns": df, 
            "distribution_patterns": df, "structure_block": df, "liquidity_block": df, 
            "zone_block": df, "pricing_block": df, "scoring_block": df,
            "relative_metrics": df, "flow_metrics": df, "kinetic_metrics": df, 
            "institutional_metrics": df, "pressure_block": df, "volatility_block": df, 
            "rejection_block": df, "gap_block": df
        }
        
    return blocks

def main():
    parser = argparse.ArgumentParser(description="Run V6 Institutional Analyzer against Live DB.")
    parser.add_argument("--analyzer", required=True, help="e.g. trend_analyzer")
    parser.add_argument("--symbol", help="e.g. RELIANCE")
    args = parser.parse_args()

    # Get V6 Class Name
    class_name = ANALYZER_CLASSES.get(args.analyzer)
    if not class_name:
        raise SystemExit(f"Error: Analyzer '{args.analyzer}' is not registered in V6 architecture.")

    # Dynamically import the class
    try:
        module = importlib.import_module(f"backend.analyzers.{args.analyzer}")
        AnalyzerClass = getattr(module, class_name)
        analyzer_instance = AnalyzerClass()
    except Exception as e:
        raise SystemExit(f"Error loading {class_name} from backend.analyzers.{args.analyzer}: {e}")

    db = sqlite3.connect(settings.database_path)
    db.row_factory = sqlite3.Row

    try:
        df = get_historical_features(db, args.symbol, limit=300)
        latest_row = df.iloc[-1].to_dict()

        display_symbol = latest_row.get('symbol', args.symbol or 'UNKNOWN')
        display_date = latest_row.get('date', latest_row.get('timestamp', 'UNKNOWN'))

        # Route exact blocks
        feature_blocks = route_feature_blocks(df, args.analyzer)

        # Execute V6 Core Engine
        result = analyzer_instance.analyze(feature_blocks)

        print("=" * 60)
        print(f"🎯 SYMBOL   : {display_symbol}")
        print(f"📅 DATE     : {display_date}")
        print(f"⚙️  ENGINE   : {class_name} (V6 L1)")
        print(f"📊 DATA ROWS: {len(df)} days processed")
        print("=" * 60)

        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))

    finally:
        db.close()

if __name__ == "__main__":
    main()
