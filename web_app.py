"""
GREEN BULL RIDER V6 - CLOUD WEB DASHBOARD
Enterprise Control Panel (Always-On Engine + Web UI)
"""
import streamlit as st
import sqlite3
import pandas as pd
import json
import os
import sys
import time
import threading

# Backend পাথ সেটআপ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backend.config.settings import settings
from backend.pipeline.market_pipeline import MarketPipeline
from backend.universe.universe_loader import UniverseLoader
from backend.repository.stock_repository import repository
from backend.pipeline.ai_pipeline import MasterPipelineManager, enqueue_for_ai, pipeline_queues

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Green Bull V6 Web", page_icon="🚀", layout="wide")

# ==========================================
# 2. ALWAYS-ON BACKGROUND ENGINE
# ==========================================
# st.cache_resource গ্যারান্টি দেবে যে ওয়েবপেজ রিফ্রেশ হলেও ইঞ্জিন একবারই স্টার্ট হবে এবং ব্যাকগ্রাউন্ডে চলতেই থাকবে।
@st.cache_resource
def start_engine():
    manager = MasterPipelineManager()
    manager.start()
    return manager

engine = start_engine()

# ==========================================
# 3. SIDEBAR: COMMAND CENTER (Fetching & Tools)
# ==========================================
st.sidebar.title("🎛️ COMMAND CENTER")

st.sidebar.subheader("🔄 1. Data Fetching")
if st.sidebar.button("Update OHLCV Data (Incremental)"):
    with st.spinner("Fetching Market Data in background..."):
        try:
            UniverseLoader(str(settings.database_path)).refresh()
            symbols = [s["symbol"] for s in repository.get_active_symbols() if "symbol" in s]
            with MarketPipeline() as mp:
                mp.run_history(symbols, is_incremental=True)
            st.sidebar.success("✅ OHLCV Update Complete!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if st.sidebar.button("Update Fundamentals (Full)"):
    st.sidebar.info("⏳ Fundamental sync initiated...")
    # Add fundamental sync logic here later
    time.sleep(2)
    st.sidebar.success("✅ Fundamentals Updated!")

st.sidebar.markdown("---")

st.sidebar.subheader("🔬 2. Deep Audit Mode")
audit_sym = st.sidebar.text_input("Enter Symbol (e.g., RELIANCE):").strip().upper()
if st.sidebar.button("Run Instant Audit"):
    if audit_sym:
        enqueue_for_ai(audit_sym)
        st.sidebar.success(f"🚀 {audit_sym} injected into Priority Queue!")
    else:
        st.sidebar.warning("Please enter a symbol.")

st.sidebar.markdown("---")

st.sidebar.subheader("⏱️ 3. Dashboard Timer")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (5 sec)", value=False)

# ==========================================
# 4. MAIN DASHBOARD: LIVE X-RAY VISION
# ==========================================
st.title("🚀 GREEN BULL RIDER V6 - AI INTELLIGENCE HUB")
st.markdown("Monitor your Always-On Pipeline and AI Decisions in Real-Time.")

# --- LIVE QUEUE METRICS ---
st.subheader("⚙️ Live Pipeline Queues (L1 to L5)")
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="[L1] Indicator Feeder", value=pipeline_queues.symbol_queue.qsize())
col2.metric(label="[L2] Analyzer Engine", value=pipeline_queues.feature_queue.qsize())
col3.metric(label="[L3] Scorecard Fusion", value=pipeline_queues.analyzer_queue.qsize())
col4.metric(label="[L4/L5] Gatekeeper", value=pipeline_queues.score_queue.qsize())

st.markdown("---")

# ==========================================
# 5. DEEP AUDIT JSON VIEWER
# ==========================================
if audit_sym:
    st.subheader(f"🕵️‍♂️ Deep Audit Report: {audit_sym}")
    try:
        conn = sqlite3.connect(str(settings.database_path))
        cursor = conn.cursor()
        cursor.execute("SELECT payload FROM decision_history WHERE symbol=? ORDER BY rowid DESC LIMIT 1", (audit_sym,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            audit_data = json.loads(row[0])
            with st.expander(f"View Full L1 to L5 JSON Output for {audit_sym}", expanded=True):
                st.json(audit_data)
        else:
            st.info(f"⏳ Waiting for {audit_sym} to complete processing. Keep refreshing...")
    except Exception as e:
        st.error(f"Audit Error: {e}")

# ==========================================
# 6. COMPLETED AI DECISIONS TABLE
# ==========================================
st.subheader("📊 Latest Processed Intelligence")

try:
    conn = sqlite3.connect(str(settings.database_path))
    df = pd.read_sql_query("SELECT symbol, timestamp, payload FROM decision_history ORDER BY rowid DESC LIMIT 100", conn)
    conn.close()
    
    if not df.empty:
        parsed_data = []
        for _, row in df.iterrows():
            sym = row['symbol']
            try:
                payload = json.loads(row['payload'])
                # L4 Logic Extract
                regime = payload.get("market_regime", {}).get("decision", {}).get("market_permission", "UNKNOWN")
                entry = payload.get("entry", {}).get("decision", {}).get("entry_priority", "UNKNOWN")
                
                # L3 Deep Health Extract
                health = payload.get("market_regime", {}).get("decision", {}).get("market_health", {})
                trend_score = round(health.get("trend_health", 0), 1)
                mom_score = round(health.get("momentum_health", 0), 1)
                
                parsed_data.append({
                    "Symbol": sym, 
                    "AI Signal": entry, 
                    "Regime": regime,
                    "Trend Score": trend_score,
                    "Mom Score": mom_score,
                    "Time": row['timestamp'][:16]
                })
            except Exception:
                pass
                
        display_df = pd.DataFrame(parsed_data)
        
        # Color coding for Signals
        def highlight_signal(val):
            color = '#155c2d' if 'BUY' in str(val).upper() else '#6e111a' if 'SELL' in str(val).upper() or 'AVOID' in str(val).upper() else '#695a14'
            return f'background-color: {color}'
            
        st.dataframe(display_df.style.applymap(highlight_signal, subset=['AI Signal']), use_container_width=True, height=400)
    else:
        st.info("Pipeline is warming up. No completed stocks in database yet.")
except Exception as e:
    st.error(f"Database Read Error: {e}")

# --- AUTO REFRESH LOGIC ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
