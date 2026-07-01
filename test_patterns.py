import yfinance as yf
import pandas as pd
from backend.registry.feature_engine import build_features
from backend.analyzers.pattern_analyzer import PatternAnalyzer
import warnings
warnings.filterwarnings("ignore")

# Nifty 50 Top Stocks
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "HUL.NS", "SBI.NS", "ITC.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", 
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "TATASTEEL.NS", "NTPC.NS",
    "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS", "COALINDIA.NS", "ONGC.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "WIPRO.NS", "HCLTECH.NS", "SBILIFE.NS",
    "GRASIM.NS", "ADANIPORTS.NS", "TECHM.NS", "BAJAJFINSV.NS", "HINDALCO.NS",
    "INDUSINDBK.NS", "EICHERMOT.NS", "DRREDDY.NS", "CIPLA.NS", "TATACONSUM.NS",
    "APOLLOHOSP.NS", "DIVISLAB.NS", "UPL.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS",
    "HEROMOTOCO.NS", "LTIM.NS", "NESTLEIND.NS", "SHREECEM.NS", "TATAELXSI.NS"
]

def run_50_stock_test():
    analyzer = PatternAnalyzer()
    total_patterns_found = 0
    
    print("🚀 Starting Nifty 50 Pattern Backtest (Last 1 Year Data)...\n")
    
    for symbol in NIFTY_50:
        try:
            df = yf.download(symbol, period="1y", progress=False, auto_adjust=False)
            if df.empty: continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            
            df = build_features(df)
            
            # Check last 30 days of the 1-year data for active patterns
            found_for_stock = 0
            for i in range(len(df)-30, len(df)):
                window_df = df.iloc[:i+1]
                result = analyzer.analyze(window_df)
                
                primary_id = result.get("advanced_pattern_metrics", {}).get("primary_pattern_id", "NONE")
                
                if primary_id != "NONE":
                    date_str = window_df.index[-1].strftime('%Y-%m-%d')
                    conf = result["advanced_pattern_metrics"].get("system_confidence", 0.0)
                    print(f"✅ [{symbol}] {date_str} -> {primary_id} Detected! (Conf: {conf:.1f}%)")
                    found_for_stock += 1
                    total_patterns_found += 1
                    
            if found_for_stock == 0:
                print(f"❌ [{symbol}] No patterns detected in the last 30 days.")
                
        except Exception as e:
            print(f"⚠️ Error on {symbol}: {e}")

    print("\n" + "="*50)
    print(f"🎯 Total Patterns Detected Across 50 Stocks: {total_patterns_found}")
    print("="*50)

if __name__ == "__main__":
    run_50_stock_test()
