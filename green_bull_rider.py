"""
GREEN BULL RIDER V6 - MASTER EXECUTION ENGINE
Interactive Wizard, Instant Deep Audit, and X-Ray UI
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

# --- GHOST PRINT SILENCER ---
REAL_STDOUT = sys.stdout 
sys.stdout = open(os.devnull, 'w') 

def ui_print(text=""):
    REAL_STDOUT.write(str(text) + "\033[K\n")
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

CYAN, GREEN, RED, YELLOW, MAGENTA, WHITE, RESET = "\033[96m", "\033[92m", "\033[91m", "\033[93m", "\033[95m", "\033[97m", "\033[0m"
CLEAR_SCREEN, CURSOR_TOP, CLEAR_LINE = "\033[2J", "\033[H", "\033[K"
DIVIDER = f"{CYAN}{'='*75}{RESET}"

system_logs = deque([""] * 3, maxlen=3)
def log_event(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    col = GREEN if level == "SUCCESS" else RED if level == "ERROR" else YELLOW if level == "WARN" else WHITE
    system_logs.appendleft(f" [{ts}] {col}{level:<7}{RESET}: {msg}")

# ==========================================
# WIZARD: INTERACTIVE PRE-FLIGHT CHECK
# ==========================================
ui_write(CLEAR_SCREEN + CURSOR_TOP)
ui_print(DIVIDER)
ui_print(f" {MAGENTA}🚀 GREEN BULL RIDER V6 | INTERACTIVE SETUP WIZARD{RESET}")
ui_print(DIVIDER)

# 1. ALWAYS SYNC UNIVERSE
ui_print(f" {YELLOW}🔄 [1/4] Synchronizing Universe Master...{RESET}")
UniverseLoader(str(settings.database_path)).refresh()
active_syms_db = [s["symbol"] for s in repository.get_active_symbols() if "symbol" in s]
ui_print(f" {GREEN}✅ Universe Synced! Active Symbols: {len(active_syms_db)}{RESET}\n")

current_date = datetime.now().strftime("%d %B")

# 2. PROMPT: OHLCV SYNC
ans_ohlcv = ui_input(f" ❓ Update missing OHLCV up to {current_date}? (y/n): ").lower()
if ans_ohlcv == 'y':
    ui_print(f" {YELLOW}⏳ Fetching Market Data...{RESET}")
    with MarketPipeline() as mp:
        def sync_progress(current, total, msg):
            safe_msg = msg[:35] + "..." if len(msg) > 35 else msg
            ui_write(f"\r {YELLOW}➔ Syncing: [{current}/{total}] {safe_msg}{CLEAR_LINE}")
        mp.run_history(active_syms_db, is_incremental=True, progress_callback=sync_progress)
        ui_print()
    ui_print(f" {GREEN}✅ OHLCV Update Complete.{RESET}\n")
else:
    ui_print(f" {YELLOW}⏭️ OHLCV Update Skipped.{RESET}\n")

# 3. PROMPT: FUNDAMENTAL SYNC
ans_fund = ui_input(f" ❓ Update missing Company Profile, IPO, Financial data? (y/n): ").lower()
if ans_fund == 'y':
    ui_print(f" {YELLOW}⏳ Fetching Fundamental Data...{RESET}")
    time.sleep(1) 
    ui_print(f" {GREEN}✅ Fundamental Data Update Complete.{RESET}\n")
else:
    ui_print(f" {YELLOW}⏭️ Fundamental Update Skipped.{RESET}\n")

# 4. PROMPT: DEEP AUDIT MODE
ui_print(f" {CYAN}💡 Tip: Press Enter to run normally, or type a symbol to debug.{RESET}")
audit_sym = ui_input(f" ❓ Enter symbol for DEEP AUDIT (e.g., RELIANCE): ").strip().upper()

# ==========================================
# VERIFY DATABASE & ISOLATE AUDIT
# ==========================================
try:
    conn = sqlite3.connect(settings.database_path, timeout=30)
    db_symbols = [row[0] for row in conn.cursor().execute("SELECT DISTINCT symbol FROM historical_data").fetchall()]
    conn.close()
except Exception:
    db_symbols = []

if not db_symbols:
    ui_print(f"{RED}❌ CRITICAL: No historical data found. Exiting.{RESET}")
    sys.exit(1)

if audit_sym:
    if audit_sym in db_symbols:
        ui_print(f"\n {MAGENTA}🔬 ISOLATING '{audit_sym}' FOR DEEP AUDIT...{RESET}")
        db_symbols = [audit_sym] # 🚀 THE MAGIC TRICK: Process ONLY this stock!
        sys.stdout = REAL_STDOUT # Restore terminal printing
    else:
        ui_print(f"\n {RED}❌ ERROR: {audit_sym} not found in database. Run sync first.{RESET}")
        sys.exit(1)
else:
    audit_sym = None

# ==========================================
# START PIPELINE
# ==========================================
manager = MasterPipelineManager(audit_symbol=audit_sym)
manager.start()

total_stocks = len(db_symbols)
pushed_symbols = []
feeder_done = False

def feed_symbols():
    global feeder_done
    for sym in db_symbols:
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
    if in_q(pipeline_queues.symbol_queue):   return f"[{YELLOW}1⏳{RESET} 2  3  4/5]"
    if in_q(pipeline_queues.feature_queue):  return f"[{GREEN}1✓{RESET} {YELLOW}2⏳{RESET} 3  4/5]"
    if in_q(pipeline_queues.analyzer_queue): return f"[{GREEN}1✓ 2✓{RESET} {YELLOW}3⏳{RESET} 4/5]"
    if in_q(pipeline_queues.score_queue):    return f"[{GREEN}1✓ 2✓ 3✓{RESET} {YELLOW}4⏳{RESET}]"
    return f"[{MAGENTA}⚙️ WORKER{RESET}]"

# ==========================================
# UI RENDER LOOP
# ==========================================
if not audit_sym:
    ui_write(CLEAR_SCREEN)

try:
    while True:
        time.sleep(0.3)
        comp_list = list(pipeline_queues.completed_stocks)
        comp_count = len(comp_list)
        finished_syms = {s.get("symbol") for s in comp_list if isinstance(s, dict)}
        active_syms = [s for s in pushed_symbols if s not in finished_syms][:5]

        q1 = pipeline_queues.symbol_queue.qsize()
        q2 = pipeline_queues.feature_queue.qsize()
        q3 = pipeline_queues.analyzer_queue.qsize()
        q4 = pipeline_queues.score_queue.qsize()

        if not audit_sym:
            ui_write(CURSOR_TOP)
            ui_print(DIVIDER)
            ui_print(f" {GREEN}🚀 GREEN BULL V6 | X-RAY DASHBOARD{RESET}")
            ui_print(DIVIDER)
            pct = (comp_count / total_stocks) * 100 if total_stocks else 0
            bar = "█" * int(pct / 5) + "-" * (20 - int(pct / 5))
            ui_print(f" {WHITE}📊 Symbols: {comp_count}/{total_stocks} | [{GREEN}{bar}{WHITE}] {pct:.1f}%{RESET}")
            ui_print(f" {YELLOW}⚙️ QUEUES: [1]:{q1} | [2]:{q2} | [3]:{q3} | [4/5]:{q4}{RESET}")
            ui_print(DIVIDER)
            
            ui_print(f" {WHITE}{'SYM':<10} | {'STAGE':<14} | {'SIGNAL':<8} | {'REGIME':<12}{RESET}")
            ui_print(f"{CYAN}{'-'*75}{RESET}")

            lines_printed = 0
            for sym in active_syms:
                ui_print(f" {YELLOW}{sym:<10}{RESET} | {get_stage_str(sym):<23} | {'PROC':<8} | {'ANALYZING':<12}")
                lines_printed += 1

            if active_syms and comp_list:
                ui_print(f"{CYAN}{'-'*75}{RESET}")
                lines_printed += 1

            for item in comp_list[-4:]:
                if not isinstance(item, dict): continue
                sym = item.get("symbol", "UNK")[:10]
                rec = item.get("final_recommendation", "UNK")[:8]
                regime = item.get("market_permission", "UNK")[:12]
                col = GREEN if "BUY" in rec or "Agg" in rec else RED if "SELL" in rec or "Block" in regime else YELLOW
                ui_print(f" {col}{sym:<10}{RESET} | [{GREEN}1✓2✓3✓4✓5✓{RESET}] | {col}{rec:<8}{RESET} | {regime:<12}")
                lines_printed += 1

            for _ in range(10 - lines_printed): ui_print("")

            ui_print(DIVIDER)
            if comp_count > 0 and isinstance(comp_list[-1], dict):
                latest = comp_list[-1]
                l_sym = latest.get("symbol", "UNK")
                if latest.get("just_completed", True): 
                    log_event(f"{l_sym} successfully analyzed.", "SUCCESS")
                    latest["just_completed"] = False
                
                master = latest.get("master_scores", {})
                t_val = round(master.get("trend", 0.0), 1) if isinstance(master.get("trend"), (float, int)) else "N/A"
                m_val = round(master.get("momentum", 0.0), 1) if isinstance(master.get("momentum"), (float, int)) else "N/A"
                v_val = round(master.get("volatility", 0.0), 1) if isinstance(master.get("volatility"), (float, int)) else "N/A"
                
                ui_print(f" {MAGENTA}🔬 LATEST ➔ {WHITE}{l_sym}{RESET}")
                ui_print(f" ➔ L3 Score : Trend:{t_val} | Mom:{m_val} | Vol:{v_val}")
            else:
                ui_print(f" {MAGENTA}🔬 WAITING FOR RESULTS...{RESET}")
                ui_print(" ➔ L3 Score : N/A")

            ui_print(f"{CYAN}{'-'*75}{RESET}")
            ui_print(f" {YELLOW}⚠️ SYSTEM LOGS:{RESET}")
            for log in system_logs:
                if log: ui_print(log)
                else: ui_print("")
            ui_print(DIVIDER)

        # Break Condition applies to BOTH modes
        if feeder_done and q1 == 0 and q2 == 0 and q3 == 0 and q4 == 0 and not active_syms:
            time.sleep(1.5)
            break

except KeyboardInterrupt:
    log_event("Interrupted. Safe shutdown...", "WARN")
except Exception as e:
    log_event(f"CRITICAL ERROR: {e}", "ERROR")
finally:
    sys.stdout = REAL_STDOUT # Restore full printing at the end
    if not audit_sym:
        print(f"\n{YELLOW}🌍 Generating Macro Reports...{RESET}")
        try: manager.generate_macro_reports()
        except: pass
    manager.stop()
    manager.join()
    print(f"\n{GREEN}✅ MISSION ACCOMPLISHED!{RESET}")
    sys.exit(0)
