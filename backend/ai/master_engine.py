import logging
import sqlite3
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
from contextlib import contextmanager

# 1. TOP-LEVEL IMPORTS
from backend.analyzers import (trend_analyzer, momentum_analyzer, volume_analyzer, 
                               volatility_analyzer, smart_money_analyzer, institutional_analyzer)
from backend.engines import (compounder_engine, swing_engine, long_term_engine)
from backend.decision import (entry_engine, stoploss_engine, target_engine, risk_engine, 
                             reward_engine, exit_engine, allocation_engine, position_size_engine, 
                             confidence_engine, conviction_engine, recommendation_engine)

# 2. MASTER ORCHESTRATOR
class GreenBullOrchestrator:
    def __init__(self, db_path: str = "audit_trail.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("GBR_Production_Stable")
        self.weights = {
            "Trend": 0.20, "SmartMoney": 0.20, "Institutional": 0.20,
            "Risk": 0.15, "Volume": 0.10, "Momentum": 0.10, "Volatility": 0.05
        }
        self._init_db()

    def _init_db(self):
        with self._db_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS audit (
                timestamp TEXT, symbol TEXT, master_score REAL, rec TEXT, 
                conf REAL, conviction REAL, risk_reward REAL, version TEXT, execution_time REAL)""")

    @contextmanager
    def _db_connection(self):
        conn = sqlite3.connect(self.db_path)
        try: yield conn
        finally: conn.commit(); conn.close()

    def _validate_production_data(self, df) -> bool:
        """Strict Production-Grade Validation"""
        if df.empty or len(df) < 200: return False # Need min candles for EMA200
        if not all(col in df.columns for col in ['open','high','low','close','volume']): return False
        if df.isnull().values.any() or (df < 0).any().any(): return False # Negative Volume/Price check
        if df.index.duplicated().any(): return False
        return True

    def _run_engine_stable(self, func, ctx) -> Dict[str, Any]:
        """Strict Execution Contract"""
        start = time.perf_counter()
        try:
            result = func(ctx)
            if not isinstance(result, dict) or 'score' not in result:
                raise ValueError(f"Engine {func.__name__} returned invalid contract format")
            return {"status": "SUCCESS", "data": result, "duration": time.perf_counter() - start}
        except Exception as e:
            self.logger.error(f"Engine {func.__name__} Critical Failure: {e}")
            return {"status": "FAILED", "data": {"score": 0}, "duration": 0}

    def generate_final_master_report(self, ctx) -> Dict[str, Any]:
        wall_clock_start = time.perf_counter()
        
        # 1. Validation
        if not self._validate_production_data(ctx.df):
            return {"STATUS": "ERROR", "REASON": "INVALID_PRODUCTION_DATA"}

        # 2. Engine Registry (Complete Integration)
        registry = {
            "Trend": trend_analyzer.analyze_trend,
            "SmartMoney": smart_money_analyzer.analyze_smart_money,
            "Institutional": institutional_analyzer.analyze_institutional,
            "Volume": volume_analyzer.analyze_volume,
            "Momentum": momentum_analyzer.analyze_momentum,
            "Volatility": volatility_analyzer.analyze_volatility,
            "Swing": swing_engine.calculate_swing_score
        }

        # 3. Parallel Execution
        results = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._run_engine_stable, func, ctx): name for name, func in registry.items()}
            for f in as_completed(futures):
                results[futures[f]] = f.result()

        # 4. Weighted Consensus
        master_score = sum(results[k]['data']['score'] * self.weights.get(k, 0) for k in results if results[k]['status'] == "SUCCESS")

        # 5. Decision Flow (Safe Dependent Execution)
        entry = entry_engine.get_entry_strategy(ctx)
        
        if entry.get("action") == "EXECUTE_ENTRY":
            sl = stoploss_engine.calculate_stoploss_levels(ctx, entry.get('price'))
            target = target_engine.calculate_targets(ctx, entry.get('price'))
            risk = risk_engine.calculate_risk_metrics(ctx, entry.get('price'), sl.get('price'))
            pos = position_size_engine.calculate_position_size(ctx, sl.get('price'))
            conf = confidence_engine.calculate_confidence_score(master_score, results)
            conv = conviction_engine.calculate_conviction_score(conf)
            rec = recommendation_engine.get_recommendation(master_score, conf, conv)
        else:
            # Safe Fallback
            entry, sl, target, risk, pos, conf, conv, rec = {}, {}, {}, {}, {}, 0, 0, {"action": "WAIT"}

        # 6. Final Report (Production Ready)
        report = {
            "timestamp": datetime.now().isoformat(),
            "master_score": master_score,
            "recommendation": rec,
            "execution": {"entry": entry, "sl": sl, "targets": target, "pos": pos},
            "audit_trail": {name: res['status'] for name, res in results.items()},
            "performance": {"wall_clock": time.perf_counter() - wall_clock_start, "details": {name: res['duration'] for name, res in results.items()}}
        }

        # 7. Persistent Audit
        with self._db_connection() as conn:
            conn.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?,?,?)", 
                         (report['timestamp'], ctx.symbol, master_score, rec.get('action', 'NONE'), 
                          float(conf), float(conv), float(risk.get('rr_ratio', 0)), "v101", report['performance']['wall_clock']))
        
        return report
