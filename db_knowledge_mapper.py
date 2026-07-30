import pandas as pd
from backend.db.connection import db
from backend.analyzers.trend_analyzer import TrendAnalyzer
from backend.analyzers.momentum_analyzer import MomentumAnalyzer
from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
from backend.analyzers.volume_analyzer import VolumeAnalyzer
from backend.analyzers.pattern_analyzer import PatternAnalyzer
from backend.analyzers.support_resistance_analyzer import SupportResistanceAnalyzer
from backend.analyzers.smart_money_analyzer import SmartMoneyAnalyzer
from backend.analyzers.market_regime_analyzer import MarketRegimeAnalyzer
from backend.analyzers.fundamental_analyzer import FundamentalAnalyzer
from backend.analyzers.ipo_analyzer import IPOAnalyzer
from backend.analyzers.candle_analyzer import CandleAnalyzer
from backend.analyzers.institutional_analyzer import InstitutionalAnalyzer

# 1. Instantiate all engines to get their EXACT requirements
engines = {
    "Trend": TrendAnalyzer(), "Momentum": MomentumAnalyzer(), "Volatility": VolatilityAnalyzer(),
    "Volume": VolumeAnalyzer(), "Pattern": PatternAnalyzer(), "SupportResistance": SupportResistanceAnalyzer(),
    "SmartMoney": SmartMoneyAnalyzer(), "MarketRegime": MarketRegimeAnalyzer(), "Fundamental": FundamentalAnalyzer(),
    "IPO": IPOAnalyzer(), "Candle": CandleAnalyzer(), "Institutional": InstitutionalAnalyzer()
}

required_features = set()
engine_requirements = {}

for name, engine in engines.items():
    if hasattr(engine, 'EXPECTED_SCHEMA'):
        schema = {str(f).lower().strip() for f in engine.EXPECTED_SCHEMA}
        required_features.update(schema)
        engine_requirements[name] = schema

# 2. Extract EXACT schema from the Database
db_columns = set()
table_details = {}

# Get all tables
try:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [dict(r).get('name', list(dict(r).values())[0]) for r in rows]
except:
    rows = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [dict(r).get('table_name', list(dict(r).values())[0]) for r in rows]

tables = [t for t in tables if not t.startswith('sqlite_')]

# Read all columns from all tables
for table in tables:
    try:
        col_info = db.fetchall(f"PRAGMA table_info({table});")
        cols = {dict(c)['name'].lower() for c in col_info}
        db_columns.update(cols)
        table_details[table] = cols
    except:
        pass

# 3. Map Aliases (Connecting DB names to Engine names)
ALIAS_MAP = {
    'rsi': ['rsi_14', 'rsi'], 'atr': ['atr_14', 'atr', 'atrp_14'],
    'macd_line': ['macd', 'macd_line'], 'macd_signal': ['macd_signal', 'macds'],
    'macd_histogram': ['macd_hist', 'macdh', 'macd_histogram'], 'adx': ['adx_14', 'adx'],
    'roc': ['roc_14', 'roc'], 'momentum': ['mom_14', 'momentum'],
    'linreg_slope': ['linreg_slope_14', 'linreg_slope', 'slope'], 'linreg_r2': ['linreg_r2_14', 'linreg_r2', 'r2'],
    'vwap': ['vwap'], 'bbw': ['bbw_20_2.0', 'bbw'], 'chop': ['chop_14', 'chop'],
    'ei': ['ei_14', 'ei'], 'sqz': ['sqz_20', 'sqz'], 'mfi': ['mfi_14', 'mfi'],
    'cmf': ['cmf_20', 'cmf'], 'adl': ['accdist', 'adl'], 'vix_proxy': ['vix', 'vix_proxy'],
    'advance_decline_line': ['advance_decline', 'adl_line']
}

# Create a virtual set of available columns applying the aliases
virtual_db_columns = set(db_columns)
for target, aliases in ALIAS_MAP.items():
    if any(a in db_columns for a in aliases):
        virtual_db_columns.add(target)

# 4. Find exactly what is missing
found_features = required_features.intersection(virtual_db_columns)
missing_features = required_features - virtual_db_columns

print(f"\n{'='*80}")
print(f" 🧠 DATABASE KNOWLEDGE ACQUISITION (THE 'GOBOR MATHA' CLEANUP)")
print(f"{'='*80}")
print(f"📊 Total Tables Found: {len(tables)}")
print(f"📊 Total Unique Columns in DB: {len(db_columns)}")
print(f"🎯 Total Features Required by 12 Analyzers: {len(required_features)}")
print("-" * 80)
print(f"✅ FOUND IN DB: {len(found_features)} features")
print(f"❌ MISSING IN DB: {len(missing_features)} features")
print("-" * 80)

if missing_features:
    print("\n🚨 EXACT MISSING FEATURES LIST (Layer-1 needs to generate these):")
    for f in sorted(missing_features):
        needed_by = [eng for eng, req in engine_requirements.items() if f in req]
        print(f"   -> {f} (Needed by: {', '.join(needed_by)})")
else:
    print("\n✅ AMAZING! DB contains 100% of the required features!")
print(f"\n{'='*80}\n")
