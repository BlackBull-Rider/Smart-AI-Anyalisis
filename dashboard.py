"""
GREEN BULL RIDER V6 - MASTER EXECUTION ENGINE (INTEGRATED LIVE UI)
"""

import time
import sqlite3
import logging
import sys
import os
import warnings
import threading
from datetime import datetime
from collections import deque
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich import box

# --- GHOST PRINT SILENCER ---
REAL_STDOUT = sys.stdout
sys.stdout = open(os.devnull, 'w')

def ui_print(text=""):
    REAL_STDOUT.write(str(text) + "\n")
def ui_write(text=""):
    REAL_STDOUT.write(str(text))
    REAL_STDOUT.flush()
def ui_input(prompt: str) -> str:
    REAL_STDOUT.write(prompt)
    REAL_STDOUT.flush()
    return sys.stdin.readline().strip()

from backend.config.settings import settings
from backend.pipeline.market_pipeline import MarketPipeline
from backend.universe.universe_loader import UniverseLoader
from backend.repository.stock_repository import repository
from backend.pipeline.ai_pipeline import MasterPipelineManager, enqueue_for_ai, pipeline_queues

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

system_logs = deque([""] * 3, maxlen=3)
def log_event(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    system_logs.appendleft(f" [{ts}] {level:<7}: {msg}")

# ==========================================
# WIZARD: INTERACTIVE PRE-FLIGHT CHECK
# ==========================================
ui_print("🚀 GREEN BULL RIDER V6 | INTERACTIVE SETUP WIZARD")
UniverseLoader(str(settings.database_path)).refresh()
active_syms_db = [s["symbol"] for s in repository.get_active_symbols() if "symbol" in s]

current_date = datetime.now().strftime("%d %B")
ans_ohlcv = ui_input(f" Update OHLCV up to {current_date}? (y/n): ").lower()
if ans_ohlcv == 'y':
    with MarketPipeline() as mp:
        mp.run_history(active_syms_db, is_incremental=True)
ui_input("Update Fundamental Data? (y/n): ")
audit_sym = ui_input("Enter symbol for DEEP AUDIT or Enter to continue: ").strip().upper()

# ==========================================
# START PIPELINE
# ==========================================
manager = MasterPipelineManager(audit_symbol=audit_sym)
manager.start()

total_stocks = len(active_syms_db) if not audit_sym else 1
pushed_symbols = []
feeder_done = False

def feed_symbols():
    global feeder_done
    syms = [audit_sym] if audit_sym else active_syms_db
    for sym in syms:
        while not enqueue_for_ai(sym): time.sleep(0.1)
        pushed_symbols.append(sym)
    feeder_done = True
threading.Thread(target=feed_symbols, daemon=True).start()

def get_stage_str(sym):
    def in_q(q):
        with q.mutex:
            for item in q.queue:
                if isinstance(item, dict) and item.get("symbol") == sym: return True
                if getattr(item, "symbol", None) == sym: return True
                if item == sym: return True
        return False
    if in_q(pipeline_queues.symbol_queue):   return "1⏳ 2  3  4/5"
    if in_q(pipeline_queues.feature_queue):  return "1✓ 2⏳ 3  4/5"
    if in_q(pipeline_queues.analyzer_queue): return "1✓ 2✓ 3⏳ 4/5"
    if in_q(pipeline_queues.score_queue):    return "1✓ 2✓ 3✓ 4⏳"
    return "⚙️ WORKER"

# ==========================================
# UI RENDER ENGINE (RICH LIVE)
# ==========================================
console = Console()

def generate_layout(comp_list, active_syms, q1, q2, q3, q4):
    table = Table(box=box.SIMPLE, expand=True)
    table.add_column("SYM", style="cyan")
    table.add_column("STAGE", style="yellow")
    table.add_column("SIGNAL", style="green")
    
    for sym in active_syms:
        table.add_row(sym, get_stage_str(sym), "PROCESSING")
    
    for item in comp_list[-4:]:
        if isinstance(item, dict):
            sym = item.get("symbol", "UNK")
            rec = item.get("final_recommendation", "N/A")
            table.add_row(sym, "1✓2✓3✓4✓5✓", rec)
            
    log_text = "\n".join([l for l in system_logs if l])
    return Layout(Panel(table, title="LIVE ANALYSIS", subtitle=f"Q1:{q1} Q2:{q2} Q3:{q3} Q4:{q4} | LOGS: {log_text}"))

try:
    with Live(console=console, screen=True, refresh_per_second=4) as live:
        while True:
            comp_list = list(pipeline_queues.completed_stocks)
            finished_syms = {s.get("symbol") for s in comp_list if isinstance(s, dict)}
            active_syms = [s for s in pushed_symbols if s not in finished_syms][:6]

            live.update(generate_layout(
                comp_list, active_syms, 
                pipeline_queues.symbol_queue.qsize(),
                pipeline_queues.feature_queue.qsize(),
                pipeline_queues.analyzer_queue.qsize(),
                pipeline_queues.score_queue.qsize()
            ))
            
            if feeder_done and not active_syms and len(comp_list) >= len(pushed_symbols):
                time.sleep(1)
                break
            time.sleep(0.2)

finally:
    sys.stdout = REAL_STDOUT
    print("\n✅ MISSION ACCOMPLISHED!")
    manager.stop()
    manager.join()
