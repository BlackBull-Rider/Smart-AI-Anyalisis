"""
GREEN BULL RIDER V6
Pipeline: Orchestrator
Module: orchestrator.py

Institutional Pipeline Orchestrator. (V3.0.0 - Enterprise Edition)
Responsible for executing the complete Layer-4 and Layer-5 decision flow.
"""

import logging
from typing import Any, Dict, List

# Layer-5 Engine Imports
from backend.ai.master_engine import evaluate_master
from backend.ai.ranking_engine import RankingEngine
from backend.ai.recommendation_engine import evaluate_recommendation
from backend.ai.watchlist_engine import evaluate_watchlist
from backend.ai.portfolio_engine import evaluate_portfolio
from backend.ai.dashboard_engine import evaluate_dashboard
from backend.ai.explanation_engine import evaluate_explanation

logger = logging.getLogger(__name__)

class InstitutionalPipeline:
    """
    Orchestrates the entire Layer-5 evaluation process for a universe of stocks
    and the macro portfolio.
    """

    def __init__(self):
        self.ranking_engine = RankingEngine()
        
    def execute_market_scan(
        self, 
        universe_l4_data: List[Dict[str, Any]], 
        current_portfolio: List[Dict[str, Any]], 
        available_cash_pct: float,
        market_macro_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline across the entire stock universe.
        
        Args:
            universe_l4_data: List of Layer-4 output dictionaries for each stock.
            current_portfolio: List of currently held stocks with their L4/L5 data.
            available_cash_pct: Macro cash reserve percentage.
            market_macro_data: Market regime and systemic risk data.
            
        Returns:
            A comprehensive institutional dashboard payload containing all AI decisions.
        """
        logger.info("Initiating Layer-5 Pipeline Execution...")

        # ---------------------------------------------------------
        # STEP 1: MASTER FUSION (Per Stock)
        # ---------------------------------------------------------
        logger.info("Running Master Engine Fusion...")
        master_outputs = []
        for stock_l4 in universe_l4_data:
            master_decision = evaluate_master(stock_l4)
            
            # Merge L4 data with Master output so downstream engines have full context
            fused_stock = {**stock_l4, **master_decision.get("decision", {})}
            master_outputs.append(fused_stock)

        # ---------------------------------------------------------
        # STEP 2: RELATIVE RANKING (Universe Level)
        # ---------------------------------------------------------
        logger.info("Running Ranking Engine...")
        # Ranking Engine takes the whole list, sorts it, and injects percentiles/ranks
        ranked_universe = self.ranking_engine.evaluate_ranking(master_outputs)

        # ---------------------------------------------------------
        # STEP 3 & 4: RECOMMENDATION & WATCHLIST (Per Stock)
        # ---------------------------------------------------------
        logger.info("Running Recommendation and Watchlist Engines...")
        final_processed_universe = []
        watchlist_candidates = []
        recommendations_list = []

        for ranked_stock in ranked_universe:
            # 3. Recommendation
            rec_decision = evaluate_recommendation(ranked_stock)
            rec_payload = rec_decision.get("decision", {})
            stock_with_rec = {**ranked_stock, **rec_payload}
            recommendations_list.append(stock_with_rec)

            # 4. Watchlist
            wl_decision = evaluate_watchlist(stock_with_rec)
            wl_payload = wl_decision.get("decision", {})
            stock_with_wl = {**stock_with_rec, **wl_payload}
            
            # If it's a valid watchlist item, store it for the Portfolio/Dashboard engines
            if wl_payload.get("watchlist_status") in ["ACTIVE", "READY", "MONITOR"]:
                watchlist_candidates.append(stock_with_wl)

            # Generate Explanation for individual stock
            explanation = evaluate_explanation(
                master=ranked_stock,
                ranking=ranked_stock, # Rank is inside ranked_stock
                recommendation=rec_payload,
                watchlist=wl_payload,
                portfolio=None,
                dashboard=None
            )
            stock_with_wl["explanation"] = explanation.get("decision", {})
            
            final_processed_universe.append(stock_with_wl)

        # ---------------------------------------------------------
        # STEP 5: MACRO PORTFOLIO INTELLIGENCE
        # ---------------------------------------------------------
        logger.info("Running Portfolio Engine...")
        portfolio_decision = evaluate_portfolio(
            portfolio=current_portfolio,
            watchlist=watchlist_candidates,
            cash_pct=available_cash_pct
        )
        portfolio_payload = portfolio_decision.get("decision", {})

        # ---------------------------------------------------------
        # STEP 6: EXECUTIVE DASHBOARD
        # ---------------------------------------------------------
        logger.info("Running Dashboard Engine...")
        dashboard_decision = evaluate_dashboard(
            market=market_macro_data,
            portfolio=portfolio_payload,
            recommendations=recommendations_list,
            watchlist=watchlist_candidates
        )
        dashboard_payload = dashboard_decision.get("decision", {})

        # ---------------------------------------------------------
        # STEP 7: MACRO EXPLANATION
        # ---------------------------------------------------------
        logger.info("Generating Macro Explanations...")
        macro_explanation = evaluate_explanation(
            master=None,
            ranking=None,
            recommendation=None,
            watchlist=None,
            portfolio=portfolio_payload,
            dashboard=dashboard_payload
        )

        logger.info("Pipeline Execution Complete.")

        # Return the ultimate systemic state
        return {
            "dashboard": dashboard_payload,
            "portfolio": portfolio_payload,
            "macro_explanation": macro_explanation.get("decision", {}),
            "processed_universe": final_processed_universe
        }

# Example Usage:
# orchestrator = InstitutionalPipeline()
# result = orchestrator.execute_market_scan(l4_data_list, current_portfolio, 25.0, market_data)
