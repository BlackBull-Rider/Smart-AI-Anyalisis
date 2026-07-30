"""
GREEN BULL RIDER V6
Layer-4: Master Decision Orchestrator
Module: backend/decision/decision_orchestrator.py
"""

import time
import logging
from typing import Dict, Any

# Importing all Layer-4 Engines
from backend.decision.market_regime_engine import MarketRegimeEngine
from backend.decision.risk_engine import RiskEngine
from backend.decision.reward_engine import RewardEngine
from backend.decision.entry_engine import EntryEngine
from backend.decision.stoploss_engine import StopLossEngine
from backend.decision.target_engine import TargetEngine
from backend.decision.exit_engine import ExitEngine
from backend.decision.allocation_engine import AllocationEngine
from backend.decision.position_size_engine import PositionSizeEngine
from backend.decision.holding_engine import HoldingEngine
from backend.decision.confidence_engine import ConfidenceEngine
from backend.decision.conviction_engine import ConvictionEngine

logger = logging.getLogger(__name__)

class DecisionOrchestrator:
    """
    Master Controller for Layer-4.
    Executes all institutional decision engines in a strict cascading sequence.
    """
    def __init__(self):
        self.regime_engine = MarketRegimeEngine()
        self.risk_engine = RiskEngine()
        self.reward_engine = RewardEngine()
        self.entry_engine = EntryEngine()
        self.stoploss_engine = StopLossEngine()
        self.target_engine = TargetEngine()
        self.exit_engine = ExitEngine()
        self.allocation_engine = AllocationEngine()
        self.position_size_engine = PositionSizeEngine()
        self.holding_engine = HoldingEngine()
        self.confidence_engine = ConfidenceEngine()
        self.conviction_engine = ConvictionEngine()

    def generate_master_decision(self, l3_scorecard: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        # Cumulative payload that grows as each engine runs
        master_payload = {"layer3_data": l3_scorecard}
        decisions = {}

        try:
            # 1. GATEKEEPER (Permissions)
            regime_out = self.regime_engine.evaluate(l3_scorecard)
            decisions["market_regime"] = regime_out
            master_payload.update(regime_out.get("decision", {}))

            # 2. FOUNDATION (Risk & Reward)
            risk_out = self.risk_engine.evaluate(master_payload)
            reward_out = self.reward_engine.evaluate(master_payload)
            decisions["risk"] = risk_out
            decisions["reward"] = reward_out
            master_payload.update(risk_out.get("decision", {}))
            master_payload.update(reward_out.get("decision", {}))

            # 3. EXECUTION STRUCTURE (Entry, StopLoss, Target, Exit)
            entry_out = self.entry_engine.evaluate(master_payload)
            master_payload.update(entry_out.get("decision", {}))

            stoploss_out = self.stoploss_engine.evaluate(master_payload)
            target_out = self.target_engine.evaluate(master_payload)
            exit_out = self.exit_engine.evaluate(master_payload)

            decisions["entry"] = entry_out
            decisions["stoploss"] = stoploss_out
            decisions["target"] = target_out
            decisions["exit"] = exit_out

            master_payload.update(stoploss_out.get("decision", {}))
            master_payload.update(target_out.get("decision", {}))
            master_payload.update(exit_out.get("decision", {}))

            # 4. CAPITAL & TIME (Allocation, Position Size, Holding)
            allocation_out = self.allocation_engine.evaluate(master_payload)
            master_payload.update(allocation_out.get("decision", {}))

            position_out = self.position_size_engine.evaluate(master_payload)
            holding_out = self.holding_engine.evaluate(master_payload)

            decisions["allocation"] = allocation_out
            decisions["position_size"] = position_out
            decisions["holding"] = holding_out

            master_payload.update(position_out.get("decision", {}))
            master_payload.update(holding_out.get("decision", {}))

            # 5. VALIDATION (Confidence & Conviction)
            # These run last to evaluate the quality of all generated structures
            confidence_out = self.confidence_engine.evaluate(master_payload)
            master_payload.update(confidence_out.get("decision", {}))

            conviction_out = self.conviction_engine.evaluate(master_payload)
            decisions["confidence"] = confidence_out
            decisions["conviction"] = conviction_out

            # Assemble Final Layer-4 Master Payload
            execution_time = round((time.perf_counter() - start_time) * 1000, 2)
            
            return {
                "status": "SUCCESS",
                "execution_time_ms": execution_time,
                "market_permission": regime_out.get("decision", {}).get("market_permission", "UNKNOWN"),
                "conviction_level": conviction_out.get("decision", {}).get("conviction_level", "UNKNOWN"),
                "decisions": decisions
            }

        except Exception as e:
            logger.error(f"Decision Orchestrator failed: {e}", exc_info=True)
            return {"status": "FAILED", "error": str(e), "decisions": decisions}
