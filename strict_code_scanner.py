import inspect
import re
from backend.db.connection import db

# 1. Importing the 12 files directly
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

required_features = set()

# 2. Extracting features ONLY from raw source code (Ignoring metadata like EXPECTED_SCHEMA)
for name, engine in engines.items():
    try:
        source = inspect.getsource(engine.__class__)
        # Regex to catch df['x'], df.get('x'), dict['x'], dict.get('x')
        df_matches = re.findall(r"(?:df|fundamental|financials|profile|corporate_actions|shareholding|earnings|kwargs)\[['\"]([^'\"]+)['\"]\]", source)
        df_get_matches = re.findall(r"(?:df|fundamental|financials|profile|corporate_actions|shareholding|earnings|kwargs)\.get\(['\"]([^'\"]+)['\"]\)", source)
        
        combined = set(df_matches + df_get_matches)
        combined = {f.lower().strip() for f in combined if len(f) > 1 and f not in ['date', 'symbol', 'timestamp']}
        required_features.update(combined)
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

# 3. Getting DB Schema
db_columns = set()
try:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [dict(r).get('name', list(dict(r).values())[0]) for r in rows]
except:
    rows = db.fetchall("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = [dict(r).get('table_name', list(dict(r).values())[0]) for r in rows]

for table in [t for t in tables if not t.startswith('sqlite_')]:
    try:
        col_info = db.fetchall(f"PRAGMA table_info({table});")
        db_columns.update({dict(c)['name'].lower() for c in col_info})
    except: pass

# 4. Alias Mapping (The bridge between Engine request and DB name)
ALIAS_MAP = {
    'rsi': ['rsi_14', 'rsi'], 'atr': ['atr_14', 'atr', 'atrp_14'],
    'macd_line': ['macd', 'macd_line'], 'macd_signal': ['macd_signal', 'macds'],
    'macd_histogram': ['macd_hist', 'macdh', 'macd_histogram'], 'adx': ['adx_14', 'adx'],
    'roc': ['roc_14', 'roc'], 'momentum': ['mom_14', 'momentum'],
    'linreg_slope': ['linreg_slope_14', 'linreg_slope', 'slope'], 'linreg_r2': ['linreg_r2_14', 'linreg_r2', 'r2'],
    'vwap': ['vwap'], 'bbw': ['bbw_20_2.0', 'bbw'], 'chop': ['chop_14', 'chop'],
    'ei': ['ei_14', 'ei'], 'sqz': ['sqz_20', 'sqz'], 'mfi': ['mfi_14', 'mfi'],
    'cmf': ['cmf_20', 'cmf'], 'adl': ['accdist', 'adl'], 'vix_proxy': ['vix', 'vix_proxy'],
    'advance_decline_line': ['advance_decline', 'adl_line'],
    'sales': ['total_revenue', 'revenue', 'operating_revenue'],
    'capex': ['capital_expenditure', 'capex_yoy'],
    'receivables': ['accounts_receivable', 'net_receivables'],
    'net_profit': ['net_income', 'pat'],
    'ipo_size': ['issue_size', 'total_issue'],
    'total_equity': ['shareholder_equity', 'equity']
}

virtual_db = set(db_columns)
for target, aliases in ALIAS_MAP.items():
    if any(a in db_columns for a in aliases):
        virtual_db.add(target)

missing = required_features - virtual_db

print(f"Total features extracted from code: {len(required_features)}")
print(f"Total features found in DB (after alias map): {len(required_features.intersection(virtual_db))}")
print(f"Missing features: {len(missing)}")
for f in sorted(list(missing)):
    print(f" -> {f}")
