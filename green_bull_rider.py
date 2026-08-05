"""
GREEN BULL RIDER V6 - UNIVERSAL SUPERVISOR ENGINE
"""
import time
import sqlite3
import logging
import sys
import os
import warnings
import threading
import json
from datetime import datetime
from collections import deque

REAL_STDOUT = sys.stdout
sys.stdout = open(os.devnull, 'w')

from backend.config.settings import settings
from backend.pipeline.market_pipeline import MarketPipeline
from backend.universe.universe_loader import UniverseLoader
from backend.repository.stock_repository import repository
from backend.pipeline.ai_pipeline import MasterPipelineManager, enqueue_for_ai, pipeline_queues

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

system_logs = deque([""] * 10, maxlen=10)
def log_event(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    system_logs.appendleft(f"[{ts}] {level}: {msg}")

def safe_serialize(obj):
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    elif isinstance(obj, dict): return {str(k): safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [safe_serialize(i) for i in obj]
    else: return str(obj)

def check_control():
    if os.path.exists("control_cmd.json"):
        try:
            with open("control_cmd.json", "r") as f:
                cmd = json.load(f).get("action")
            os.remove("control_cmd.json")
            return cmd
        except: pass
    return None

while True:
    engine_state = "WAITING"
    with open("live_state.json", "w") as f:
        json.dump({"status": "WAITING", "logs": list(system_logs)}, f)
        
    if os.path.exists("ui_command.json"): os.remove("ui_command.json")
    while not os.path.exists("ui_command.json"): time.sleep(0.5)
        
    with open("ui_command.json", "r") as f:
        setup_cmd = json.load(f)
    os.remove("ui_command.json")

    ans_ohlcv = setup_cmd.get("sync_ohlcv", False)
    ans_fund = setup_cmd.get("sync_fund", False)
    audit_sym = setup_cmd.get("audit_sym", "").strip().upper()

    sync_aborted = False
    if ans_ohlcv or ans_fund:
        log_event("Universe & Data Sync Started...", "SYSTEM")
        UniverseLoader(str(settings.database_path)).refresh()
        active_syms_db = [s["symbol"] for s in repository.get_active_symbols() if "symbol" in s]
        
        if ans_ohlcv:
            with MarketPipeline() as mp:
                def sync_progress(current, total, msg):
                    if check_control() == "stop": raise InterruptedError("STOPPED")
                    safe_msg = msg[:35] + "..." if len(msg) > 35 else msg
                    with open("live_state.json", "w") as f:
                        json.dump({"status": "SYNCING", "sync_msg": f"[{current}/{total}] {safe_msg}"}, f)
                try:
                    mp.run_history(active_syms_db, is_incremental=True, progress_callback=sync_progress)
                except InterruptedError:
                    log_event("Data sync aborted by user.", "WARN")
                    sync_aborted = True

    if sync_aborted: continue 

    try:
        conn = sqlite3.connect(settings.database_path, timeout=30)
        db_symbols = [row[0] for row in conn.cursor().execute("SELECT DISTINCT symbol FROM historical_data").fetchall()]
        conn.close()
    except: db_symbols = []

    if audit_sym and audit_sym in db_symbols: db_symbols = [audit_sym]

    manager = MasterPipelineManager(audit_symbol=audit_sym)
    manager.start()
    total_stocks = len(db_symbols)
    pushed_symbols = []
    feeder_done = False
    engine_state = "RUNNING"
    log_event(f"Pipeline started for {total_stocks} symbols.", "SUCCESS")

    def feed_symbols():
        global feeder_done, engine_state
        for sym in db_symbols:
            while engine_state == "PAUSED": time.sleep(0.5)
            if engine_state == "STOPPING": break
            while not enqueue_for_ai(sym):
                if engine_state == "STOPPING": break
                time.sleep(0.1)
            if engine_state == "STOPPING": break
            pushed_symbols.append(sym)
        feeder_done = True
        
    threading.Thread(target=feed_symbols, daemon=True).start()

    while engine_state in ["RUNNING", "PAUSED"]:
        time.sleep(0.5) 
        ctrl = check_control()
        if ctrl == "pause": engine_state = "PAUSED"; log_event("Pipeline Paused", "WARN")
        elif ctrl == "resume": engine_state = "RUNNING"; log_event("Pipeline Resumed", "SUCCESS")
        elif ctrl == "stop": engine_state = "STOPPING"

        if engine_state == "STOPPING":
            log_event("Pipeline stopped & reset by user.", "ERROR")
            break

        comp_list = list(pipeline_queues.completed_stocks)
        active_syms = [s for s in pushed_symbols if s not in {s.get("symbol") for s in comp_list if isinstance(s, dict)}][:10]

        web_data = {
            "status": engine_state,
            "progress": {"current": len(comp_list), "total": total_stocks},
            "queues": {
                "q1": pipeline_queues.symbol_queue.qsize(),
                "q2": pipeline_queues.feature_queue.qsize(),
                "q3": pipeline_queues.analyzer_queue.qsize(),
                "q4": pipeline_queues.score_queue.qsize()
            },
            "active_symbols": active_syms,
            "completed": safe_serialize(comp_list[-50:]), # 🚀 DUMPING THE ENTIRE DICTIONARY!
            "logs": list(system_logs)
        }
        
        with open("live_state_tmp.json", "w") as f: json.dump(web_data, f)
        os.replace("live_state_tmp.json", "live_state.json")

        if feeder_done and web_data["queues"]["q1"] == 0 and web_data["queues"]["q2"] == 0 and web_data["queues"]["q3"] == 0 and web_data["queues"]["q4"] == 0 and not active_syms:
            time.sleep(2)
            log_event("Mission Accomplished! All done.", "SUCCESS")
            break

    manager.stop()
    manager.join()
    time.sleep(1)
