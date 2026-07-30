import inspect
import re
import pandas as pd
from backend.db.connection import db
from backend.repository.stock_repository import repository

# ১২টা ইঞ্জিন ইমপোর্ট
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

def get_engine_requirements():
    engines = {
        "Trend": TrendAnalyzer(), "Momentum": MomentumAnalyzer(), "Volatility": VolatilityAnalyzer(),
        "Volume": VolumeAnalyzer(), "Pattern": PatternAnalyzer(), "SupportResistance": SupportResistanceAnalyzer(),
        "SmartMoney": SmartMoneyAnalyzer(), "MarketRegime": MarketRegimeAnalyzer(), "Fundamental": FundamentalAnalyzer(),
        "IPO": IPOAnalyzer(), "Candle": CandleAnalyzer(), "Institutional": InstitutionalAnalyzer()
    }
    
    requirements = {}
    for name, engine in engines.items():
        source = inspect.getsource(engine.__class__)
        #Regex দিয়ে কোড থেকে ফিচার নাম তোলা
        matches = re.findall(r"(?:df|fundamental|financials|profile|corporate_actions|shareholding|earnings|kwargs)\[['\"]([^'\"]+)['\"]\]", source)
        get_matches = re.findall(r"(?:df|fundamental|financials|profile|corporate_actions|shareholding|earnings|kwargs)\.get\(['\"]([^'\"]+)['\"]\)", source)
        
        reqs = {m.lower().strip() for m in (matches + get_matches) if m not in ['date', 'symbol', 'timestamp']}
        requirements[name] = reqs
    return requirements

def get_all_db_columns():
    all_cols = set()
    try:
        tables = [t[0] for t in db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")]
    except:
        tables = [t[0] for t in db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")]
    
    for table in tables:
        if table.startswith('sqlite_'): continue
        try:
            cols = [c[1].lower() for c in db.fetchall(f"PRAGMA table_info({table});")]
            all_cols.update(cols)
        except: continue
    return all_cols

# MAIN AUDIT
requirements = get_engine_requirements()
db_cols = get_all_db_columns()

print(f"\n{'='*80}")
print(f" 🚨 AUDIT REPORT: ENGINE REQUIREMENTS VS DATABASE SCHEMA")
print(f"{'='*80}\n")

total_missing = 0
for engine_name, reqs in requirements.items():
    missing = [r for r in reqs if r not in db_cols]
    if missing:
        print(f"❌ {engine_name} ENGINE:")
        print(f"   Missing: {', '.join(sorted(missing))}")
        total_missing += len(missing)
    else:
        print(f"✅ {engine_name} ENGINE: All data found.")

print(f"\n{'='*80}")
print(f"TOTAL MISSING FEATURES: {total_missing}")
print(f"{'='*80}")
