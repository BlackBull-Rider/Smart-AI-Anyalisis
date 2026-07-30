"""
GREEN BULL RIDER V6
Module: backend/pipeline/ai_pipeline.py
Enterprise AI Pipeline - Fluid Load Balancing & Deep Audit Ready
"""
from __future__ import annotations
import logging
import queue
import threading
import sqlite3
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from backend.config.settings import settings
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine
from backend.analyzers.analyzer_engine import analyzer_engine
from backend.engines.scoring_orchestrator import ScoringOrchestrator
from backend.decision.decision_orchestrator import DecisionOrchestrator

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FeaturePayload:
    symbol: str
    timestamp: str
    df: pd.DataFrame

@dataclass(frozen=True)
class AnalyzerPayload:
    symbol: str
    timestamp: str
    l2_raw_data: Dict[str, Any]

@dataclass(frozen=True)
class ScorePayload:
    symbol: str
    timestamp: str
    scorecard: Dict[str, Any]

class AIPipelineQueues:
    def __init__(self) -> None:
        maxsize = settings.performance.batch_size if hasattr(settings, 'performance') else 200
        self.symbol_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=maxsize)
        self.feature_queue: queue.Queue[Optional[FeaturePayload]] = queue.Queue(maxsize=maxsize)
        self.analyzer_queue: queue.Queue[Optional[AnalyzerPayload]] = queue.Queue(maxsize=maxsize)
        self.score_queue: queue.Queue[Optional[ScorePayload]] = queue.Queue(maxsize=maxsize)
        self.db_write_queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize * 2)
        self.completed_stocks = []
        self.audit_symbol = None # 🔍 DEEP AUDIT TRIGGER

pipeline_queues = AIPipelineQueues()

def enqueue_for_ai(symbol: str) -> bool:
    try:
        pipeline_queues.symbol_queue.put(symbol, timeout=2.0)
        return True
    except queue.Full:
        return False

def safe_put(q: queue.Queue, item: Any, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.5)
            return True
        except queue.Full:
            continue
    return False

# ==============================================================================
# FLUID WORKERS: Dynamic Load Balancing Architecture
# ==============================================================================
class FluidWorker(threading.Thread):
    """ 
    Smart Worker that dynamically shifts layers if a bottleneck is detected. 
    It checks queues in reverse order (L4 -> L3 -> L2 -> L1) to clear blockages first.
    """
    def __init__(self, primary_role: int, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self.primary_role = primary_role
        self._stop_event = threading.Event()
        
        # Initialize Engines locally for the thread
        self.scoring_orch = ScoringOrchestrator()
        self.decision_orch = DecisionOrchestrator()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            # 🚀 DYNAMIC LOAD BALANCING LOGIC 🚀
            # Always try to clear downstream bottlenecks first to prevent pipeline jamming
            
            # Check L4 (Score -> Decision)
            if self.primary_role == 4 or pipeline_queues.score_queue.qsize() > 5:
                if self._process_l4(): continue
                
            # Check L3 (Analyzer -> Score)
            if self.primary_role == 3 or pipeline_queues.analyzer_queue.qsize() > 10:
                if self._process_l3(): continue
                
            # Check L2 (Feature -> Analyzer)
            if self.primary_role == 2 or pipeline_queues.feature_queue.qsize() > 10:
                if self._process_l2(): continue
                
            # Check L1 (Symbol -> Feature) - Primary Feeder
            if self.primary_role == 1:
                if self._process_l1(): continue
                
            # If all queues are empty, sleep briefly to prevent CPU thrashing
            time.sleep(0.1)

    # --- LAYER 1 PROCESSING ---
    def _process_l1(self) -> bool:
        try:
            symbol = pipeline_queues.symbol_queue.get_nowait()
            if symbol is None:
                pipeline_queues.symbol_queue.task_done()
                self._stop_event.set()
                return True
                
            conn = sqlite3.connect(str(settings.database_path), timeout=10.0)
            query = f"SELECT * FROM (SELECT date, open, high, low, close, volume FROM historical_data WHERE symbol='{symbol}' ORDER BY date DESC LIMIT 350) sub ORDER BY date ASC"
            raw_df = pd.read_sql_query(query, conn, parse_dates=["date"])
            conn.close()

            if not raw_df.empty:
                df = indicator_engine.run(symbol, raw_df)
                if df is not None and not df.empty:
                    df.fillna(method='ffill', inplace=True)
                    latest_date = df.index[-1]
                    timestamp_str = latest_date.isoformat() if hasattr(latest_date, 'isoformat') else str(latest_date)
                    payload = FeaturePayload(symbol=symbol, timestamp=timestamp_str, df=df)
                    
                    safe_put(pipeline_queues.feature_queue, payload, self._stop_event)
                    safe_put(pipeline_queues.db_write_queue, ("FEATURE", payload), self._stop_event)
            
            pipeline_queues.symbol_queue.task_done()
            return True
        except queue.Empty:
            return False
        except Exception:
            if 'symbol' in locals(): pipeline_queues.symbol_queue.task_done()
            return True

    # --- LAYER 2 PROCESSING ---
    def _process_l2(self) -> bool:
        try:
            payload = pipeline_queues.feature_queue.get_nowait()
            if payload is None:
                pipeline_queues.feature_queue.task_done()
                self._stop_event.set()
                return True
                
            l2_raw_data = analyzer_engine.run({"symbol": payload.symbol, "in_memory_df": payload.df})
            if l2_raw_data:
                analyzer_payload = AnalyzerPayload(symbol=payload.symbol, timestamp=payload.timestamp, l2_raw_data=l2_raw_data)
                safe_put(pipeline_queues.analyzer_queue, analyzer_payload, self._stop_event)
                safe_put(pipeline_queues.db_write_queue, ("ANALYZER", analyzer_payload), self._stop_event)
                
            pipeline_queues.feature_queue.task_done()
            return True
        except queue.Empty:
            return False
        except Exception:
            if 'payload' in locals(): pipeline_queues.feature_queue.task_done()
            return True

    # --- LAYER 3 PROCESSING ---
    def _process_l3(self) -> bool:
        try:
            payload = pipeline_queues.analyzer_queue.get_nowait()
            if payload is None:
                pipeline_queues.analyzer_queue.task_done()
                self._stop_event.set()
                return True
                
            scorecard = self.scoring_orch.calculate(payload.l2_raw_data)
            score_payload = ScorePayload(symbol=payload.symbol, timestamp=payload.timestamp, scorecard=scorecard)
            safe_put(pipeline_queues.score_queue, score_payload, self._stop_event)
            safe_put(pipeline_queues.db_write_queue, ("SCORE", score_payload), self._stop_event)
            
            pipeline_queues.analyzer_queue.task_done()
            return True
        except queue.Empty:
            return False
        except Exception:
            if 'payload' in locals(): pipeline_queues.analyzer_queue.task_done()
            return True

    # --- LAYER 4/5 PROCESSING & DEEP AUDIT ---
    def _process_l4(self) -> bool:
        try:
            payload = pipeline_queues.score_queue.get_nowait()
            if payload is None:
                pipeline_queues.score_queue.task_done()
                self._stop_event.set()
                return True
                
            master_decision = self.decision_orch.generate_master_decision(payload.scorecard)
            
            # 🔍 DEEP AUDIT MODE 🔍 (Only prints if the symbol matches)
            if pipeline_queues.audit_symbol and pipeline_queues.audit_symbol.upper() == payload.symbol.upper():
                print(f"\n{'-'*80}\n🕵️‍♂️ DEEP AUDIT X-RAY: {payload.symbol}\n{'-'*80}")
                print("🔵 [LAYER 3 SCORECARD]:")
                print(json.dumps(payload.scorecard, indent=2, default=str))
                print("\n🟣 [LAYER 4 MASTER DECISION]:")
                print(json.dumps(master_decision, indent=2, default=str))
                print(f"{'-'*80}\n")
            
            from backend.decision.ai_orchestrator import AIOrchestrator
            if not hasattr(self, 'l5_orchestrator'):
                self.l5_orchestrator = AIOrchestrator()

            actual_l4_data = master_decision.get("decisions", master_decision) if isinstance(master_decision, dict) else {}
            l5_result = self.l5_orchestrator.generate_executive_summary(actual_l4_data)

            stock_data = master_decision.copy() if isinstance(master_decision, dict) else {}
            stock_data["symbol"] = payload.symbol
            stock_data["just_completed"] = True # For UI Logging
            
            # 🛠️ THE FIX: Inject Layer 3 scores for the UI Dashboard
            stock_data["master_scores"] = payload.scorecard.get("master_scores", {})
            
            # 🛠️ THE FIX: Map L4 Entry Priority to UI's Signal Column
            entry_dec = master_decision.get("decisions", {}).get("entry", {}).get("decision", {})
            stock_data["final_recommendation"] = entry_dec.get("entry_priority", "UNKNOWN").upper()
            
            if isinstance(l5_result, dict):
                stock_data.update(l5_result.get("l5_payload", {}))

            pipeline_queues.completed_stocks.append(stock_data)
            safe_put(pipeline_queues.db_write_queue, ("DECISION", (payload.symbol, payload.timestamp, master_decision)), self._stop_event)
            safe_put(pipeline_queues.db_write_queue, ("AI_HISTORY", (payload.symbol, payload.timestamp, l5_result)), self._stop_event)
            
            pipeline_queues.score_queue.task_done()
            return True
        except queue.Empty:
            return False
        except Exception:
            if 'payload' in locals(): pipeline_queues.score_queue.task_done()
            return True

# ==============================================================================
# ASYNC DATABASE WRITER (Unchanged, operates independently)
# ==============================================================================
class DedicatedDBWriter(threading.Thread):
    def __init__(self, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            conn = sqlite3.connect(str(settings.database_path), timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("CREATE TABLE IF NOT EXISTS decision_history (symbol TEXT, timestamp TEXT, payload TEXT)")
            conn.execute("CREATE TABLE IF NOT EXISTS ai_history (symbol TEXT, timestamp TEXT, narrative TEXT, payload TEXT)")
            conn.commit()
            conn.close()
        except Exception:
            pass

        while not self._stop_event.is_set():
            try:
                task = pipeline_queues.db_write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                pipeline_queues.db_write_queue.task_done()
                break

            try:
                task_type, payload = task[0], task[1]
                if task_type == "FEATURE":
                    repository.save_features(payload.symbol, payload.df)
                elif task_type == "ANALYZER":
                    repository.save_analyzer_history(payload.symbol, payload.timestamp, payload.l2_raw_data)
                elif task_type == "SCORE":
                    repository.save_scoring_history(payload.symbol, payload.timestamp, payload.scorecard)
                elif task_type == "DECISION":
                    conn = sqlite3.connect(str(settings.database_path), timeout=10.0)
                    conn.execute("DELETE FROM decision_history WHERE symbol=?", (payload[0],))
                    conn.execute("INSERT INTO decision_history (symbol, timestamp, payload) VALUES (?, ?, ?)", (payload[0], payload[1], json.dumps(payload[2])))
                    conn.commit()
                    conn.close()
                elif task_type == "AI_HISTORY":
                    conn = sqlite3.connect(str(settings.database_path), timeout=10.0)
                    narrative = payload[2].get("executive_narrative", "N/A") if isinstance(payload[2], dict) else "N/A"
                    conn.execute("DELETE FROM ai_history WHERE symbol=?", (payload[0],))
                    conn.execute("INSERT INTO ai_history (symbol, timestamp, narrative, payload) VALUES (?, ?, ?, ?)", (payload[0], payload[1], narrative, json.dumps(payload[2])))
                    conn.commit()
                    conn.close()
            except Exception:
                pass
            finally:
                pipeline_queues.db_write_queue.task_done()

# ==============================================================================
# MASTER PIPELINE MANAGER
# ==============================================================================
class MasterPipelineManager:
    def __init__(self, audit_symbol: str = None) -> None:
        self.logger = logging.getLogger('PipelineManager')
        self.workers: List[threading.Thread] = []
        if audit_symbol:
            pipeline_queues.audit_symbol = audit_symbol

    def start(self) -> None:
        # Initial Allocation: L1:4, L2:1, L3:1, L4:1 (Fluid roles handle dynamic scaling)
        for _ in range(4): self.workers.append(FluidWorker(primary_role=1))
        self.workers.append(FluidWorker(primary_role=2))
        self.workers.append(FluidWorker(primary_role=3))
        self.workers.append(FluidWorker(primary_role=4))
        
        self.workers.append(DedicatedDBWriter())

        for worker in self.workers:
            worker.start()

    def stop(self) -> None:
        for worker in self.workers:
            if hasattr(worker, "stop"): worker.stop()

        # Unblock queues
        for _ in range(8): pipeline_queues.symbol_queue.put(None)
        for _ in range(8): pipeline_queues.feature_queue.put(None)
        for _ in range(8): pipeline_queues.analyzer_queue.put(None)
        for _ in range(8): pipeline_queues.score_queue.put(None)
        pipeline_queues.db_write_queue.put(None)

    def join(self) -> None:
        for worker in self.workers:
            worker.join()

    def generate_macro_reports(self):
        from backend.decision.ai_orchestrator import AIOrchestrator
        orchestrator = AIOrchestrator()
        all_stocks = pipeline_queues.completed_stocks
        if not all_stocks: return

        result = orchestrator.generate_macro_intelligence(all_stocks, portfolio_data=[], cash_pct=100.0)
        if result["status"] == "SUCCESS":
            ranked = result["ranked_universe"]
            top_stock = ranked[0]["_tie_breakers"]["symbol"] if ranked and "_tie_breakers" in ranked[0] else "N/A"
            self.logger.info(f"🏆 Universe Ranked! Top Stock: {top_stock}")
