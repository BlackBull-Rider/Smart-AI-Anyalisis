import pandas as pd
from backend.db.connection import db
from backend.analyzers.analyzer_engine import analyzer_engine
import warnings
warnings.simplefilter(action='ignore')

symbol = 'RELIANCE'

print(f"\n{'='*80}")
print(f" 🚨 LAYER-2 STRICT AUDIT: NO PLACEHOLDERS, NO DUMMIES FOR [{symbol}]")
print(f"{'='*80}")

def fetch_table_data(query, is_list=False):
    try:
        res = db.fetchall(query)
        if res: return [dict(r) for r in res] if is_list else dict(res[0])
    except Exception: pass
    return [] if is_list else {}

df_rows = fetch_table_data(f"SELECT * FROM feature_history WHERE symbol='{symbol}' ORDER BY date ASC LIMIT 200", is_list=True)
df = pd.DataFrame(df_rows) if df_rows else pd.DataFrame()
fundamental = fetch_table_data(f"SELECT * FROM fundamental_data WHERE symbol='{symbol}'")
financials = fetch_table_data(f"SELECT * FROM financial_data WHERE symbol='{symbol}' ORDER BY fiscal_year DESC LIMIT 1")
profile = fetch_table_data(f"SELECT * FROM company_profile WHERE symbol='{symbol}'")
ipo = fetch_table_data(f"SELECT * FROM ipo_data WHERE symbol='{symbol}'")
shareholding = fetch_table_data(f"SELECT * FROM shareholding_data WHERE symbol='{symbol}' ORDER BY quarter DESC LIMIT 4", is_list=True)
macro = fetch_table_data("SELECT * FROM macro_environment ORDER BY date DESC LIMIT 1")

input_data = {
    "df": df, "fundamental": fundamental, "financials": financials,
    "profile": profile, "ipo": ipo, "macro": macro, "shareholding": shareholding
}

original_analyzers = {}

for name, engine in analyzer_engine.engines.items():
    original_analyzers[name] = engine.analyze
    
    def make_wrapper(eng_name, real_analyze_method):
        def wrapper(analyzer_df, **kwargs):
            print(f"\n{'='*60}")
            print(f" 📥 AUDIT REPORT FOR: {eng_name.upper()} ENGINE")
            print(f"{'='*60}")
            
            last_row = analyzer_df.iloc[-1].to_dict() if not analyzer_df.empty else {}
            
            # 🔴 STRICT AUDIT: ইঞ্জিন অনুযায়ী আমরা চেক করছি যে রিয়াল ডেটাফ্রেমে ডেটা আছে কি না।
            # এখানে কোনো ফালতু 0.0 বা ডামি বসানো নেই। 
            audit_columns = []
            if eng_name == "fundamental":
                audit_columns = ['pe_ratio', 'pe', 'pb_ratio', 'debt_to_equity', 'eps', 'revenue_growth_yoy', 'total_revenue', 'market_cap']
            elif eng_name == "ipo":
                audit_columns = ['ipo_size', 'issue_size', 'listing_price', 'gmp', 'subscription_qib']
            elif eng_name == "trend":
                audit_columns = ['macd', 'macd_signal', 'adx_14', 'adx', 'sma_200', 'ema_50']
            elif eng_name == "momentum":
                audit_columns = ['rsi_14', 'rsi', 'stoch_k', 'cci_20', 'roc_14', 'roc']
            elif eng_name == "volume":
                audit_columns = ['volume', 'rvol_20', 'volume_ratio', 'mfi_14', 'adl']
            elif eng_name == "volatility":
                audit_columns = ['atr_14', 'bb_width', 'bbw', 'bb_squeeze', 'chop_14']
            elif eng_name == "candle":
                audit_columns = ['body_percent', 'body_pct', 'bullish_candle', 'doji']
            elif eng_name == "institutional":
                audit_columns = ['fii_holding', 'dii_holding', 'promoter_holding']
            else:
                audit_columns = ['close', 'volume', 'vwap']

            print("📊 EXACT VALUES INSIDE THE DATAFRAME MEMORY (RAW DATABSE VALUES):")
            
            for col in audit_columns:
                col = col.lower().strip()
                if col in last_row:
                    val = last_row[col]
                    if pd.isna(val):
                        print(f"   ⚠️ {col:<20} : NaN (Null in Database)")
                    elif isinstance(val, float):
                        print(f"   ✅ {col:<20} : {val:.4f}")
                    else:
                        print(f"   ✅ {col:<20} : {val}")
                else:
                    print(f"   ❌ {col:<20} : MISSING (Not Found in DataFrame)")

            return real_analyze_method(analyzer_df, **kwargs)
        return wrapper
        
    engine.analyze = make_wrapper(name, engine.analyze)

analyzer_engine.run(symbol, input_data)

for name, engine in analyzer_engine.engines.items():
    engine.analyze = original_analyzers[name]

print("\n✅ STRICT AUDIT COMPLETE!")
