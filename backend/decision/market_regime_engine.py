"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: market_regime_engine.py

Institutional Market Regime Gatekeeper Engine.
Inherits from BaseDecisionEngine. This is the master gatekeeper for the entire 
Layer-4 Decision pipeline. It ingests Layer-3 Scoring Engine outputs (Trend, Momentum, 
Smart Money, etc.) and determines macro market permissions. 

It DOES NOT generate Entry, Exit, Target, or Stop Loss. It solely evaluates 
whether the market environment is safe for capital deployment and restricts/allows 
subsequent decision engines accordingly.
"""

import time
from typing import Any
from dataclasses import dataclass, asdict

from backend.decision.base_decision_engine import (
    BaseDecisionEngine,
    DecisionConfig,
    DecisionEvidence,
    DecisionStatus,
    DecisionTrace
)


# =====================================================================
# ENGINE PROFILE (Configurable & Swappable)
# =====================================================================
def get_market_regime_profile() -> DecisionConfig:
    """Returns the primary DecisionConfig profile for the Gatekeeper Engine."""
    return DecisionConfig(
        profile_name="Institutional_Regime_Filter",
        version="2.1.0",
        stage="Layer-4: Decision Gatekeeper",
        schema_version="1.0",
        api_version="v6",
        decision_method="Multi-Dimensional Health & Risk Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={},  # Layer-4 uses adaptive logic thresholds, not strict score weighting
        thresholds={
            "conflict_penalty": 15.0,
            "strong_bull_threshold": 85.0,
            "bull_threshold": 65.0,
            "weak_bull_threshold": 55.0,
            "sideways_threshold": 45.0,
            "weak_bear_threshold": 35.0,
            "bear_threshold": 20.0
        }
    )


# =====================================================================
# MARKET REGIME DECISION ENGINE (Inherits BaseDecisionEngine)
# =====================================================================
class MarketRegimeEngine(BaseDecisionEngine):
    """
    Institutional Market Regime & Permission Engine.
    Acts as the strict gatekeeper for all downstream trading decisions.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_market_regime_profile())
        # Layer-4 specific state
        self._restrictions: list[str] = []

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_market_regime(engine_outputs)

    def evaluate_market_regime(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main decision execution pipeline.
        
        Args:
            engine_outputs: The dictionary payload compiled from all Layer-3 Scoring Engines.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary containing market permissions.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(engine_outputs, start_time, "REGIME")

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (agnostic to Layer-3 schema evolutions)
            flat_data = self._flatten_dict(engine_outputs)
            
            # 1. Base Intelligence Extraction (from Layer-3 Scores)
            layer3_confidence = self._extract_metric(flat_data, ["confidence", "reliability"], 50.0)
            
            trend_score = self._extract_metric(flat_data, ["overall_trend_score", "trend_strength", "trend_score"], 50.0)
            mom_score = self._extract_metric(flat_data, ["overall_momentum_score", "momentum_score"], 50.0)
            vol_score = self._extract_metric(flat_data, ["overall_volume_score", "liquidity_score"], 50.0)
            vola_safety = self._extract_metric(flat_data, ["volatility_quality", "overall_volatility_score"], 50.0)
            risk_raw = self._extract_metric(flat_data, ["risk_score", "distribution_risk", "danger"], 30.0)
            smc_score = self._extract_metric(flat_data, ["overall_smart_money_score", "institutional_footprint"], 50.0)
            inst_score = self._extract_metric(flat_data, ["overall_institutional_score", "institutional_buying"], 50.0)
            fund_score = self._extract_metric(flat_data, ["overall_fundamental_score", "business_quality_score"], 50.0)
            dist_score = self._extract_metric(flat_data, ["distribution_score", "distribution"], 20.0)

            # 2. Market Health Models
            trend_health = trend_score
            mom_health = mom_score
            liq_health = vol_score
            risk_health = self._normalize(100.0 - risk_raw) # Invert risk: High Risk = Low Health
            inst_health = self._normalize((inst_score * 0.6) + (smc_score * 0.4))
            fund_stab = fund_score
            
            market_stability = self._normalize((trend_health * 0.4) + (liq_health * 0.3) + (risk_health * 0.3))
            
            # The aggregated raw regime score
            regime_score = self._normalize(
                (trend_health * 0.30) + 
                (mom_health * 0.20) + 
                (liq_health * 0.15) + 
                (inst_health * 0.20) + 
                (risk_health * 0.15)
            )

            # 3. State & Bias Determination
            market_state = self._determine_market_state(regime_score)
            market_bias = self._determine_market_bias(regime_score)
            
            # 4. Conflict Detection
            self._detect_conflicts(
                trend_health, mom_health, vol_score, dist_score, 
                fund_stab, regime_score, smc_score, risk_raw
            )

            # 5. Permission Engine Formulation
            permissions, modes, overall_permission = self._calculate_permissions(
                market_state, regime_score, risk_health, inst_health, liq_health
            )

            # 6. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 60.0) * 100.0) # Layer-4 has massive coverage expectation
            
            evidence_graph = DecisionEvidence(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 7. Confidence Model Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            agreement_ratio = self._normalize(100.0 - abs(trend_health - mom_health))
            
            final_confidence = self._normalize(
                (layer3_confidence * 0.30) + 
                (evidence_graph.evidence_coverage * 0.10) + 
                (market_stability * 0.20) +
                (liq_health * 0.10) +
                (inst_health * 0.15) +
                (agreement_ratio * 0.15) - 
                (self._conflicts * 15)
            )
            
            regime_reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # 8. Synthesize Explanations
            self._generate_explanations(
                market_state, market_bias, overall_permission, modes, 
                trend_health, risk_health, inst_health, liq_health, fund_stab
            )
            
            # Apply hard restrictions based on confidence
            if final_confidence < 30.0:
                self._restrictions.append("Severe confidence degradation forces extreme capital preservation.")
                permissions["trade_permission"] = "No"
                permissions["position_scaling_permission"] = "Blocked"
                overall_permission = "Restricted"
                modes["capital_preservation_mode"] = True
                modes["aggressive_mode"] = False

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(DecisionStatus(status="SUCCESS", quality="VALID")),
                "decision": {
                    "market_state": market_state,
                    "market_bias": market_bias,
                    "market_permission": overall_permission
                },
                "market_health": {
                    "trend_health": round(trend_health, 2),
                    "momentum_health": round(mom_health, 2),
                    "liquidity_health": round(liq_health, 2),
                    "risk_health": round(risk_health, 2),
                    "institutional_health": round(inst_health, 2),
                    "fundamental_stability": round(fund_stab, 2),
                    "market_stability": round(market_stability, 2),
                    "regime_reliability": round(regime_reliability, 2)
                },
                "permissions": permissions,
                "modes": modes,
                "restrictions": sorted(list(set(self._restrictions))),
                "evidence_graph": asdict(evidence_graph),
                "warnings": sorted(list(set(self._warning_log))),
                "explanations": sorted(list(set(self._score_reasons))),
                "trace": asdict(trace),
                "decision_signature": {
                    "profile": self.config.profile_name,
                    "engine_version": self.config.version,
                    "schema_version": self.config.schema_version,
                    "api_version": self.config.api_version,
                    "gatekeeper_status": "ACTIVE"
                }
            }

            return self._sanitize_json(result)

        except Exception as e:
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # MARKET REGIME & PERMISSION MODELS
    # ---------------------------------------------------------

    def _determine_market_state(self, regime_score: float) -> str:
        t = self.config.thresholds
        if regime_score >= t.get("strong_bull_threshold", 85.0): return "Strong Bull"
        if regime_score >= t.get("bull_threshold", 65.0): return "Bull Market"
        if regime_score >= t.get("weak_bull_threshold", 55.0): return "Weak Bull"
        if regime_score >= t.get("sideways_threshold", 45.0): return "Sideways"
        if regime_score >= t.get("weak_bear_threshold", 35.0): return "Weak Bear"
        if regime_score >= t.get("bear_threshold", 20.0): return "Bear Market"
        return "Strong Bear"

    def _determine_market_bias(self, regime_score: float) -> str:
        if regime_score >= 55.0: return "Bullish"
        if regime_score <= 45.0: return "Bearish"
        return "Neutral"

    def _calculate_permissions(self, state: str, regime_score: float, risk_health: float, inst_health: float, liq_health: float) -> tuple[dict, dict, str]:
        """
        Determines the strict permission flags for Layer-4 downstream engines.
        """
        permissions = {
            "trade_permission": "Conditional",
            "investment_permission": "Watchlist",
            "swing_permission": "Limited",
            "long_term_permission": "Watchlist",
            "position_scaling_permission": "Blocked"
        }
        
        modes = {
            "aggressive_mode": False,
            "normal_mode": True,
            "defensive_mode": False,
            "capital_preservation_mode": False
        }
        
        overall_permission = "Limited"

        if state in ["Strong Bull", "Bull Market"]:
            permissions["trade_permission"] = "Yes"
            permissions["investment_permission"] = "Allowed"
            permissions["swing_permission"] = "Allowed"
            permissions["long_term_permission"] = "Allowed"
            overall_permission = "Allowed"
            
            if state == "Strong Bull" and inst_health > 75.0 and risk_health > 70.0:
                permissions["position_scaling_permission"] = "Allowed"
                modes["aggressive_mode"] = True
                modes["normal_mode"] = False
                self._positive_log.append("Market exceptionally favorable; Aggressive scaling unlocked.")
        
        elif state in ["Weak Bull", "Sideways"]:
            permissions["trade_permission"] = "Conditional"
            overall_permission = "Limited"
            modes["defensive_mode"] = True
            modes["normal_mode"] = False
            self._restrictions.append("Market lacks directional conviction; Swing holding periods must be reduced.")
            
        elif state in ["Weak Bear", "Bear Market", "Strong Bear"]:
            permissions["trade_permission"] = "No"
            permissions["investment_permission"] = "Avoid"
            permissions["swing_permission"] = "Blocked"
            permissions["long_term_permission"] = "Watchlist"
            overall_permission = "Restricted"
            modes["capital_preservation_mode"] = True
            modes["normal_mode"] = False
            self._restrictions.append("Bearish market structure; Long positions are strictly blocked.")
            if state == "Strong Bear":
                overall_permission = "Blocked"
                self._restrictions.append("Severe wealth destruction phase; Absolute capital preservation enforced.")

        # Overrides based on critical health failures
        if risk_health < 30.0:
            permissions["trade_permission"] = "No"
            modes["capital_preservation_mode"] = True
            modes["aggressive_mode"] = False
            overall_permission = "Blocked"
            self._restrictions.append("Extreme volatility/risk environment overrides all bullish permissions.")
            
        if liq_health < 30.0:
            permissions["position_scaling_permission"] = "Blocked"
            self._restrictions.append("Severe illiquidity prevents position scaling and increases slippage risk.")

        return permissions, modes, overall_permission


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & CONFLICT DETECTION
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, trend: float, mom: float, vol: float, dist: float, 
                          fund: float, regime: float, smc: float, risk_raw: float) -> None:
        """Evaluates logical paradoxes in broad market states."""
        
        # 1. Bull Trend + Bear Momentum
        if trend > 75 and mom < 35:
            self._conflicts += 1
            warn = "Conflict: Strong directional trend contradicts weakening underlying momentum."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Strong Volume + Distribution
        if vol > 75 and dist > 75:
            self._conflicts += 1
            warn = "Conflict: High liquidity/volume is masking severe institutional distribution."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Bullish Fundamentals + Bearish Market
        if fund > 75 and regime < 35:
            self._conflicts += 1
            warn = "Conflict: Strong fundamentals fighting a severely bearish macro market regime (Value Trap risk)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Strong Smart Money + Weak Trend
        if smc > 75 and trend < 35:
            self._conflicts += 1
            warn = "Conflict: Smart money accumulation occurring within a broken macro trend."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Bullish Regime + Extreme Risk
        if regime > 70 and risk_raw > 80:
            self._conflicts += 1
            warn = "Conflict: Bullish structural regime heavily compromised by extreme systemic risk."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _generate_explanations(self, state: str, bias: str, perm: str, modes: dict, 
                               trend: float, risk_health: float, inst: float, liq: float, fund: float) -> None:
        """Synthesizes human-readable institutional logic for Gatekeeper decisions."""
        
        if bias == "Bullish" and perm in ["Allowed", "Limited"]:
            self._score_reasons.append("Overall market structure favors long positions.")
        elif bias == "Bearish":
            self._score_reasons.append("Bearish market structure limits new long positions.")
            
        if inst > 75:
            self._score_reasons.append("Institutional participation strongly supports continuation.")
            
        if risk_health > 70:
            self._score_reasons.append("Market volatility and risk parameters remain highly acceptable.")
        elif risk_health < 35:
            self._score_reasons.append("Hostile risk environment severely reduces trade permission.")
            
        if liq > 70:
            self._score_reasons.append("Liquidity conditions remain healthy and supportive.")
            
        if trend > 75:
            self._score_reasons.append("Underlying trend integrity remains fully intact.")
            
        if fund > 75 and bias == "Bearish":
            self._score_reasons.append("Strong fundamentals provide a floor, but macro regime dictates caution.")
            
        if modes.get("aggressive_mode"):
            self._score_reasons.append("Confluence of safety and momentum unlocks Aggressive sizing mode.")
            
        if modes.get("capital_preservation_mode"):
            self._score_reasons.append("Gatekeeper has activated absolute Capital Preservation mode.")

        if self._conflicts > 0:
            self._score_reasons.append("Conflicting inter-layer signals reduce systemic confidence.")


    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        """Provides a strict, defensive failover state."""
        return {
            "status": asdict(DecisionStatus(status="FAILED", quality="INVALID")),
            "decision": {
                "market_state": "Unknown",
                "market_bias": "Neutral",
                "market_permission": "Blocked"
            },
            "market_health": {
                "trend_health": 50.0,
                "momentum_health": 50.0,
                "liquidity_health": 50.0,
                "risk_health": 0.0,
                "institutional_health": 50.0,
                "fundamental_stability": 50.0,
                "market_stability": 0.0,
                "regime_reliability": 0.0
            },
            "permissions": {
                "trade_permission": "No",
                "investment_permission": "Avoid",
                "swing_permission": "Blocked",
                "long_term_permission": "Blocked",
                "position_scaling_permission": "Blocked"
            },
            "modes": {
                "aggressive_mode": False,
                "normal_mode": False,
                "defensive_mode": False,
                "capital_preservation_mode": True
            },
            "restrictions": ["Fatal execution error in Gatekeeper. All systems defaulted to Blocked."],
            "evidence_graph": asdict(DecisionEvidence()),
            "warnings": ["Gatekeeper execution failed."],
            "explanations": ["Defaulted to maximum safety (Blocked) due to processing failure."],
            "trace": asdict(trace),
            "decision_signature": {
                "profile": self.config.profile_name,
                "engine_version": self.config.version,
                "schema_version": self.config.schema_version,
                "api_version": self.config.api_version,
                "gatekeeper_status": "FAILSAFE_LOCKDOWN"
            }
        }


# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_market_regime(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to evaluate market regime and determine trade permissions.
    
    Args:
        engine_outputs: The dictionary payload compiled from all Layer-3 Analyzers/Engines.
        
    Returns:
        A JSON-compatible dictionary containing the definitive Gatekeeper decision.
    """
    engine = MarketRegimeEngine()
    return engine.evaluate_market_regime(engine_outputs)
