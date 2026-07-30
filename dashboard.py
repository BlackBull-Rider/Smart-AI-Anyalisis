import streamlit as st
import sqlite3
import pandas as pd
import json
import os
import sys

# Backend ফোল্ডারটাকে ইমপোর্ট করার জন্য সিস্টেম পাথ সেট করা
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.config.settings import settings

st.set_page_config(page_title="Green Bull Rider V6", layout="wide")
st.title("🚀 Green Bull Rider V6 - AI Intelligence Dashboard")

# Settings থেকে ডায়নামিক ডেটাবেস পাথ নেওয়া
db_path = str(settings.database_path)

try:
    conn = sqlite3.connect(db_path)
    # Load completed AI Decisions
    df = pd.read_sql_query("SELECT symbol, timestamp, payload FROM decision_history", conn)
    
    if not df.empty:
        st.success(f"✅ Loaded {len(df)} Analyzed Stocks from Database")
        
        # Parse JSON payload to extract columns
        parsed_data = []
        for _, row in df.iterrows():
            sym = row['symbol']
            try:
                payload = json.loads(row['payload'])
                
                # Extracting Decisions
                regime = payload.get("market_regime", {}).get("decision", {}).get("market_permission", "UNKNOWN")
                entry = payload.get("entry", {}).get("decision", {}).get("entry_priority", "UNKNOWN")
                
                # Extracting Deep Health Scores
                health = payload.get("market_regime", {}).get("decision", {}).get("market_health", {})
                trend_score = round(health.get("trend_health", 0), 2)
                mom_score = round(health.get("momentum_health", 0), 2)
                risk_score = round(health.get("risk_health", 0), 2)
                
                parsed_data.append({
                    "Symbol": sym, 
                    "Signal": entry, 
                    "Regime": regime,
                    "Trend Health": trend_score,
                    "Momentum Health": mom_score,
                    "Risk Level": risk_score
                })
            except Exception as e:
                pass
                
        display_df = pd.DataFrame(parsed_data)
        
        # Streamlit Magic DataFrame (Can be sorted and searched dynamically)
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("⚠️ No AI decisions found in database yet. Run the pipeline first.")
    
    conn.close()
except Exception as e:
    st.error(f"Error loading database from {db_path}: {e}")
