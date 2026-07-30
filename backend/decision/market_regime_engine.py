"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: market_regime_engine.py

Institutional Market Regime Gatekeeper Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. This is the master gatekeeper for the entire 
Layer-4 Decision pipeline.
"""

import time
from typing import Any
from dataclasses import dataclass, asdict

from backend.decision.base_decision_engine import (
    BaseDecisionEngine,
    DecisionConfig,
    DecisionContext,
    DecisionStatusEnum,
    WarningSeverityEnum,
    DecisionTrace
)


def get_market_regime_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Regime_Filter_V3",
        version="3.0.0",
        stage="Layer-4: Decision Gatekeeper",
        schema_version="3.0",
        api_version="v6",
        decision_method="Multi-Dimensional Health & Risk Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={},
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

class MarketRegimeEngine(BaseDecisionEngine):
    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_market_regime_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_market_regime(engine_outputs)

    def evaluate_market_regime(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_REGIME_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_DATA")
            
            layer3_confidence = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            trend_score = self._dynamic_lookup(flat_data, ["overall_trend_score", "trend_strength", "trend_score"], 50.0)
            mom_score = self._dynamic_lookup(flat_data, ["overall_momentum_score", "momentum_score"], 50.0)
            vol_score = self._dynamic_lookup(flat_data, ["overall_volume_score", "liquidity_score"], 50.0)
            vola_safety = self._dynamic_lookup(flat_data, ["volatility_quality", "overall_volatility_score"], 50.0)
            risk_raw = self._dynamic_lookup(flat_data, ["risk_score", "distribution_risk", "danger"], 30.0)
            smc_score = self._dynamic_lookup(flat_data, ["overall_smart_money_score", "institutional_footprint"], 50.0)
            inst_score = self._dynamic_lookup(flat_data, ["overall_institutional_score", "institutional_buying"], 50.0)
            fund_score = self._dynamic_lookup(flat_data, ["overall_fundamental_score", "business_quality_score"], 50.0)
            dist_score = self._dynamic_lookup(flat_data, ["distribution_score", "distribution"], 20.0)

            trend_health = trend_score
            mom_health = mom_score
            liq_health = vol_score
            risk_health = self._normalize(100.0 - risk_raw)
            inst_health = self._normalize((inst_score * 0.6) + (smc_score * 0.4))
            fund_stab = fund_score
            
            market_stability = self._normalize((trend_health * 0.4) + (liq_health * 0.3) + (risk_health * 0.3))
            
            regime_score = self._normalize(
                (trend_health * 0.30) + 
                (mom_health * 0.20) + 
                (liq_health * 0.15) + 
                (inst_health * 0.20) + 
                (risk_health * 0.15)
            )

            market_state = self._determine_market_state(regime_score)
            market_bias = self._determine_market_bias(regime_score)
            
            self._detect_regime_conflicts(ctx, trend_health, mom_health, vol_score, dist_score, fund_stab, regime_score, smc_score, risk_raw)

            permissions, modes, overall_permission = self._calculate_permissions(ctx, market_state, risk_health, inst_health, liq_health)

            conflict_ratio = (ctx.conflicts / max(len(ctx.positive_evidence) + len(ctx.negative_evidence) + ctx.conflicts, 1)) * 100.0
            agreement_ratio = self._normalize(100.0 - abs(trend_health - mom_health))
            
            final_confidence = self._normalize(
                (layer3_confidence * 0.30) + 
                (market_stability * 0.20) +
                (liq_health * 0.10) +
                (inst_health * 0.15) +
                (agreement_ratio * 0.15) - 
                (ctx.conflicts * 15)
            )
            
            regime_reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            self._generate_explanations(ctx, market_state, market_bias, overall_permission, modes, trend_health, risk_health, inst_health, liq_health, fund_stab)
            
            if final_confidence < 30.0:
                self._add_restriction(ctx, "Severe confidence degradation forces extreme capital preservation.")
                permissions["trade_permission"] = "No"
                permissions["position_scaling_permission"] = "Blocked"
                overall_permission = "Restricted"
                modes["capital_preservation_mode"] = True
                modes["aggressive_mode"] = False

            trace = self._build_trace(engine_outputs, start_time, ctx, parsed_inputs)
            
            result = {
                "market_state": market_state,
                "market_bias": market_bias,
                "market_permission": overall_permission,
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
                "modes": modes
            }

            return self._build_output(
                status=DecisionStatusEnum.SUCCESS,
                status_msg="Market Regime formulated.",
                decision_payload=result,
                confidence=final_confidence,
                rating_score=regime_score,
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"MarketRegime Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))

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

    def _calculate_permissions(self, ctx: DecisionContext, state: str, risk_health: float, inst_health: float, liq_health: float) -> tuple[dict, dict, str]:
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
                self._add_evidence(ctx, "Market exceptionally favorable; Aggressive scaling unlocked.", "positive")
        
        elif state in ["Weak Bull", "Sideways"]:
            permissions["trade_permission"] = "Conditional"
            overall_permission = "Limited"
            modes["defensive_mode"] = True
            modes["normal_mode"] = False
            self._add_restriction(ctx, "Market lacks directional conviction; Swing holding periods must be reduced.")
            
        elif state in ["Weak Bear", "Bear Market", "Strong Bear"]:
            permissions["trade_permission"] = "No"
            permissions["investment_permission"] = "Avoid"
            permissions["swing_permission"] = "Blocked"
            permissions["long_term_permission"] = "Watchlist"
            overall_permission = "Restricted"
            modes["capital_preservation_mode"] = True
            modes["normal_mode"] = False
            self._add_restriction(ctx, "Bearish market structure; Long positions are strictly blocked.")
            if state == "Strong Bear":
                overall_permission = "Blocked"
                self._add_restriction(ctx, "Severe wealth destruction phase; Absolute capital preservation enforced.")

        if risk_health < 30.0:
            permissions["trade_permission"] = "No"
            modes["capital_preservation_mode"] = True
            modes["aggressive_mode"] = False
            overall_permission = "Blocked"
            self._add_restriction(ctx, "Extreme volatility/risk environment overrides all bullish permissions.")
            
        if liq_health < 30.0:
            permissions["position_scaling_permission"] = "Blocked"
            self._add_restriction(ctx, "Severe illiquidity prevents position scaling and increases slippage risk.")

        return permissions, modes, overall_permission

    def _detect_regime_conflicts(self, ctx: DecisionContext, trend: float, mom: float, vol: float, dist: float, fund: float, regime: float, smc: float, risk_raw: float) -> None:
        if trend > 75 and mom < 35:
            self._add_conflict(ctx, "Strong directional trend contradicts weakening underlying momentum.", penalty=15.0)
        if vol > 75 and dist > 75:
            self._add_conflict(ctx, "High liquidity/volume is masking severe institutional distribution.", penalty=15.0)
        if fund > 75 and regime < 35:
            self._add_conflict(ctx, "Strong fundamentals fighting a severely bearish macro market regime (Value Trap risk).", penalty=15.0)
        if smc > 75 and trend < 35:
            self._add_conflict(ctx, "Smart money accumulation occurring within a broken macro trend.", penalty=10.0)
        if regime > 70 and risk_raw > 80:
            self._add_conflict(ctx, "Bullish structural regime heavily compromised by extreme systemic risk.", penalty=15.0)

    def _generate_explanations(self, ctx: DecisionContext, state: str, bias: str, perm: str, modes: dict, trend: float, risk_health: float, inst: float, liq: float, fund: float) -> None:
        if bias == "Bullish" and perm in ["Allowed", "Limited"]:
            self._add_explanation(ctx, "Overall market structure favors long positions.")
        elif bias == "Bearish":
            self._add_explanation(ctx, "Bearish market structure limits new long positions.")
            
        if inst > 75:
            self._add_explanation(ctx, "Institutional participation strongly supports continuation.")
        if risk_health > 70:
            self._add_explanation(ctx, "Market volatility and risk parameters remain highly acceptable.")
        elif risk_health < 35:
            self._add_explanation(ctx, "Hostile risk environment severely reduces trade permission.")
        if liq > 70:
            self._add_explanation(ctx, "Liquidity conditions remain healthy and supportive.")
        if trend > 75:
            self._add_explanation(ctx, "Underlying trend integrity remains fully intact.")
        if fund > 75 and bias == "Bearish":
            self._add_explanation(ctx, "Strong fundamentals provide a floor, but macro regime dictates caution.")
        if modes.get("aggressive_mode"):
            self._add_explanation(ctx, "Confluence of safety and momentum unlocks Aggressive sizing mode.")
        if modes.get("capital_preservation_mode"):
            self._add_explanation(ctx, "Gatekeeper has activated absolute Capital Preservation mode.")
        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting inter-layer signals reduce systemic confidence.")

    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Fatal execution error in Gatekeeper.", WarningSeverityEnum.CRITICAL)
        self._add_restriction(empty_ctx, "All systems defaulted to Blocked.")
        
        payload = {
            "market_state": "Unknown",
            "market_bias": "Neutral",
            "market_permission": "Blocked",
            "market_health": {
                "trend_health": 50.0, "momentum_health": 50.0, "liquidity_health": 50.0,
                "risk_health": 0.0, "institutional_health": 50.0, "fundamental_stability": 50.0,
                "market_stability": 0.0, "regime_reliability": 0.0
            },
            "permissions": {
                "trade_permission": "No", "investment_permission": "Avoid",
                "swing_permission": "Blocked", "long_term_permission": "Blocked",
                "position_scaling_permission": "Blocked"
            },
            "modes": {
                "aggressive_mode": False, "normal_mode": False,
                "defensive_mode": False, "capital_preservation_mode": True
            }
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe emergency exit logic.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )
