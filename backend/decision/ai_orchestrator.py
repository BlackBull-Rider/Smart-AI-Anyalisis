"""
GREEN BULL RIDER V6
Layer-5: Master AI Orchestrator
Module: backend/decision/ai_orchestrator.py
"""

import time
import logging
from typing import Dict, Any, List

# --- Single-Stock Engines ---
from backend.ai.master_engine import MasterEngine
from backend.ai.recommendation_engine import RecommendationEngine
from backend.ai.watchlist_engine import WatchlistEngine
from backend.ai.explanation_engine import ExplanationEngine

# --- Multi-Stock / Macro Engines ---
from backend.ai.ranking_engine import RankingEngine
from backend.ai.portfolio_engine import PortfolioEngine
from backend.ai.dashboard_engine import DashboardEngine

logger = logging.getLogger(__name__)

class AIOrchestrator:
    """
    Master AI Orchestrator for Layer-5.
    Handles both Single-Stock Streaming (Master, Rec, Watchlist, Exp) 
    and Multi-Stock Batch Processing (Ranking, Portfolio, Dashboard).
    """
    def __init__(self):
        # Initialize all 7 Layer-5 Engines
        self.master_engine = MasterEngine()
        self.rec_engine = RecommendationEngine()
        self.watchlist_engine = WatchlistEngine()
        self.explanation_engine = ExplanationEngine()
        
        self.ranking_engine = RankingEngine()
        self.portfolio_engine = PortfolioEngine()
        self.dashboard_engine = DashboardEngine()

    def generate_executive_summary(self, l4_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the 4 Single-Stock Engines for streaming data."""
        start_time = time.perf_counter()
        
        try:
            # 1. Canonical Master Fusion
            # 🛠️ FIXED: Unpacking the Layer-4 Decisions Box!
            actual_l4_data = l4_decisions.get("decisions", l4_decisions)
            
            master_res = self.master_engine.evaluate(actual_l4_data)
            aggregated_data = {**actual_l4_data, **master_res.get("decision", {})}

            # 2. Final Investment Recommendation
            rec_res = self.rec_engine.evaluate(aggregated_data)
            aggregated_data.update(rec_res.get("decision", {}))

            # 3. Opportunity Watchlist Generation
            wl_res = self.watchlist_engine.evaluate(aggregated_data)
            aggregated_data.update(wl_res.get("decision", {}))

            # 4. Institutional Explanation (AI Narrative)
            exp_res = self.explanation_engine.evaluate(
                master=master_res.get("decision"),
                ranking=None,      
                recommendation=rec_res.get("decision"),
                watchlist=wl_res.get("decision"),
                portfolio=None,    
                dashboard=None     
            )

            execution_time = round((time.perf_counter() - start_time) * 1000, 2)
            
            master_status = master_res.get("decision", {}).get("master_status", "UNKNOWN")
            recommendation = rec_res.get("decision", {}).get("recommendation", "UNKNOWN")
            narrative = exp_res.get("decision", {}).get("ai_narrative", "Explanation pending.")

            return {
                "status": "SUCCESS",
                "execution_time_ms": execution_time,
                "master_status": master_status,
                "final_recommendation": recommendation,
                "executive_narrative": narrative,
                "l5_payload": {
                    "master": master_res,
                    "recommendation": rec_res,
                    "watchlist": wl_res,
                    "explanation": exp_res
                }
            }

        except Exception as e:
            logger.error(f"AI Orchestrator (Single-Stock) failed: {e}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}

    def generate_macro_intelligence(self, all_stocks: List[Dict[str, Any]], portfolio_data: List[Dict[str, Any]], cash_pct: float) -> Dict[str, Any]:
        """Runs the 3 Multi-Stock / Macro Engines after batch processing is complete."""
        try:
            # 1. Ranking Engine
            ranked_universe = self.ranking_engine.evaluate_ranking(all_stocks)
            
            # 2. Portfolio Engine
            watchlist_data = [s for s in ranked_universe if s.get("decision", {}).get("watchlist_status") in ["ACTIVE", "READY"]]
            portfolio_res = self.portfolio_engine.evaluate_portfolio(portfolio_data, watchlist_data, cash_pct)
            
            # 3. Dashboard Engine
            dashboard_res = self.dashboard_engine.evaluate_dashboard(
                market=ranked_universe[0] if ranked_universe else None, # Proxy for global market regime
                portfolio=portfolio_res.get("decision"),
                recommendations=ranked_universe,
                watchlist=watchlist_data
            )
            
            return {
                "status": "SUCCESS",
                "ranked_universe": ranked_universe,
                "portfolio_intelligence": portfolio_res,
                "executive_dashboard": dashboard_res
            }
            
        except Exception as e:
            logger.error(f"AI Orchestrator (Macro Intelligence) failed: {e}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}
