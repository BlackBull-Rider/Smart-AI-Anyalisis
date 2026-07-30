import inspect
from backend.db.connection import db

# 1. Importing the 12 analyzers exactly as requested
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

engines = {
    "Trend": TrendAnalyzer(), "Momentum": MomentumAnalyzer(), "Volatility": VolatilityAnalyzer(),
    "Volume": VolumeAnalyzer(), "Pattern": PatternAnalyzer(), "SupportResistance": SupportResistanceAnalyzer(),
    "SmartMoney": SmartMoneyAnalyzer(), "MarketRegime": MarketRegimeAnalyzer(), "Fundamental": FundamentalAnalyzer(),
    "IPO": IPOAnalyzer(), "Candle": CandleAnalyzer(), "Institutional": InstitutionalAnalyzer()
}

all_found_features = set()

# 2. Extracting from every attribute of the Class/Instance
for name, engine in engines.items():
    # Look at all attributes of the class instance
    for attr_name in dir(engine):
        if attr_name.startswith('_'): continue # Skip internal python stuff
        try:
            val = getattr(engine, attr_name)
            # If it's a list, tuple, or set, check its contents
            if isinstance(val, (list, tuple, set)):
                for item in val:
                    if isinstance(item, str) and len(item) > 1 and item not in ['date', 'symbol', 'timestamp']:
                        all_found_features.add(item.lower().strip())
            # If it's a string, maybe it's a single feature
            elif isinstance(val, str) and len(val) > 2 and val not in ['date', 'symbol', 'timestamp']:
                # Heuristic: if it looks like a feature name
                if any(x in val for x in ['_pct', '_ratio', 'rsi', 'macd', 'ema', 'sma', 'obv']):
                    all_found_features.add(val.lower().strip())
        except:
            continue

# 3. Get DB Schema
db_cols = set()
try:
    # Get all tables
    tables = [t[0] for t in db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")]
except:
    tables = [t[0] for t in db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")]

for table in tables:
    try:
        cols = [c[1].lower() for c in db.fetchall(f"PRAGMA table_info({table});")]
        db_cols.update(cols)
    except: continue

# 4. Final Report
print(f"Total features found via exact attribute inspection: {len(all_found_features)}")
missing = all_found_features - db_cols

print(f"\nMissing from DB: {len(missing)}")
for f in sorted(list(missing)):
    print(f" -> {f}")

# Optional: List all found to cross-check with your 272 count
print(f"\nAll Found Features:")
for f in sorted(list(all_found_features)):
    print(f"  {f}")
