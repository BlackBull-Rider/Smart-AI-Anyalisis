import pandas as pd
import numpy as np
import json
import logging
from datetime import date, datetime

# =====================================================================
# ১. Importing Repository and DB Connection
# =====================================================================
from backend.repository.stock_repository import repository
from backend.db.connection import db

# =====================================================================
# ২. Importing All 12 Analyzers
# =====================================================================
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.fundamental_analyzer import FundamentalAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.support_resistance_analyzer import SupportResistanceAnalyzer
from backend.analyzers.smart_money_analyzer import SmartMoneyAnalyzer
from backend.analyzers.pattern_analyzer import PatternAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.market_regime_analyzer import MarketRegimeAnalyzer
from backend.analyzers.ipo_analyzer import IPOAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer
from backend.analyzers.institutional_analyzer import InstitutionalAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):  # <--- FIX: Added Boolean support
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, date, datetime)):
            return obj.isoformat()
        if pd.isna(obj):
            return 0.0
        return super(NpEncoder, self).default(obj)

def sanitize_none(data):
    if isinstance(data, list):
        return [sanitize_none(x) for x in data]
    elif isinstance(data, dict):
        return {k: (0.0 if v is None else sanitize_none(v)) for k, v in data.items()}
    return data

def pad_shareholding(data):
    if not data:
        return data
    required_keys = ['fii_holding', 'dii_holding', 'promoter_holding', 'public_holding', 'retail_holding', 'mutual_fund_holding']
    for row in data:
        for key in required_keys:
            if row.get(key) is None:
                row[key] = 0.0
    return data

def strict_analyzer_test(symbol="RELIANCE"):
    logging.info(f"Starting STRICT Repository Mapping Test for {symbol}...")

    logging.info("Fetching 'feature_history'...")
    rows = db.fetchall("SELECT * FROM feature_history WHERE symbol=?", (symbol,))
    if not rows:
        logging.error(f"❌ No feature_history found for {symbol}! Please run L1 engine first.")
        return
        
    df = pd.DataFrame([dict(row) for row in rows])
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    aliases = {
        'rsi': ['rsi_14', 'rsi_'],
        'atr': ['atr_14', 'atr_'],
        'macd_line': ['macd', 'macd_line'],
        'macd_histogram': ['macd_hist', 'macdh'],
        'adx': ['adx_14', 'adx_'],
        'roc': ['roc_12', 'roc_14', 'roc_'],
        'momentum': ['mom_10', 'mom_14', 'mom_', 'momentum'],
        'linreg_slope': ['regression_slope', 'linear_regression_slope', 'linreg_slope'],
        'linreg_r2': ['regression_r2', 'linear_regression_r2', 'linreg_r2'],
        'bos': ['bos', 'bos_up', 'sm_bos'],
        'choch': ['choch', 'choch_up', 'sm_choch'],
        'delivery_volume': ['delivery', 'delivery_volume', 'delivery_qty']
    }
    
    for target, possible in aliases.items():
        if target not in df.columns:
            for col in df.columns:
                if any(p in col for p in possible):
                    df[target] = df[col]
                    break
                    
    required_cols = ['rsi', 'atr', 'macd_line', 'macd_histogram', 'adx', 'roc', 'momentum', 'linreg_slope', 'linreg_r2', 'bos', 'choch', 'delivery_volume']
    for req in required_cols:
        if req not in df.columns:
            df[req] = 0.0

    # <--- FIX: Warning fix for Pandas Future Downcasting
    pd.set_option('future.no_silent_downcasting', True)
    df = df.fillna(0.0).infer_objects(copy=False)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True, drop=False)

    logging.info("Fetching auxiliary tables via StockRepository...")
    
    fundamental_data = sanitize_none(repository.get_fundamental_snapshot(symbol) or repository.get_fundamental(symbol))
    financial_data = sanitize_none(repository.get_financials(symbol))
    earnings_data = sanitize_none(repository.get_earnings(symbol))
    profile_data = sanitize_none(repository.get_company_profile(symbol))
    shareholding_data = pad_shareholding(sanitize_none(repository.get_shareholding(symbol)))
    corporate_actions = sanitize_none(repository.get_corporate_actions(symbol))
    ipo_data = sanitize_none(repository.get_ipo_data(symbol))
    macro_data = sanitize_none(repository.get_latest_macro_environment())

    analyzers = {
        "trend": TrendAnalyzer(),
        "momentum": MomentumAnalyzer(),
        "volatility": VolatilityAnalyzer(),
        "volume": VolumeAnalyzer(),
        "pattern": PatternAnalyzer(),
        "support_resistance": SupportResistanceAnalyzer(),
        "smart_money": SmartMoneyAnalyzer(),
        "market_regime": MarketRegimeAnalyzer(),
        "candle": CandleAnalyzer(),
        "fundamental": FundamentalAnalyzer(),
        "institutional": InstitutionalAnalyzer(),
        "ipo": IPOAnalyzer()
    }

    results = {}

    logging.info("Injecting mapped data into Analyzers...")

    for name, analyzer in analyzers.items():
        try:
            if name == "fundamental":
                res = analyzer.analyze(df, fundamental=fundamental_data, financials=financial_data, profile=profile_data, earnings=earnings_data)
            elif name == "institutional":
                res = analyzer.analyze(df, shareholding=shareholding_data)
            elif name == "ipo":
                res = analyzer.analyze(df, corporate_actions=corporate_actions, ipo_data=ipo_data)
            elif name == "market_regime":
                try:
                    res = analyzer.analyze(df, macro_data=macro_data)
                except TypeError:
                    res = analyzer.analyze(df)
            else:
                res = analyzer.analyze(df)

            results[name] = res
            logging.info(f"✅ [{name.upper()}] Executed Successfully.")
            
        except Exception as e:
            logging.error(f"❌ [{name.upper()}] FAILED: {str(e)}")
            results[name] = {"error": str(e), "status": "FAILED"}

    output_file = f"{symbol}_Analyzer_Output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        
    logging.info(f"🎉 PERFECT RUN! All mapped JSON data saved to: {output_file}")

if __name__ == "__main__":
    strict_analyzer_test("RELIANCE")
