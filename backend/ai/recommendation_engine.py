"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: recommendation_engine.py

Institutional Recommendation Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine (reusing framework components).
Operates strictly as the Final Absolute Investment Decision Boundary.
It synthesizes Master Scores, Rankings, and Layer-4 Actionability (Conviction) 
to output a definitive, human-understandable recommendation.

Key Responsibilities:
- Outputs Absolute Investment Advice (STRONG_BUY, BUY, ACCUMULATE, HOLD, SELL, etc.).
- Assigns Institutional Grades (AAA, AA, A, BBB, etc.).
- Consolidates Risk, Reward, and Market Permissions into a single actionable verdict.

Boundary Constraint: This engine DOES NOT calculate Entry, Target, StopLoss, 
Position Size, or Relative Ranking. It determines "What should we ultimately do?".
"""

import time
from enum import Enum
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


# =====================================================================
# ENUMS (Strict Output Typing)
# =====================================================================
class Recommendation(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    WATCH = "WATCH"
    REDUCE = "REDUCE"
    SELL = "SELL"
    AVOID = "AVOID"
    NONE = "NONE"

class RecommendationGrade(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    C = "C"
    D = "D"
    UNKNOWN = "UNKNOWN"

class RecommendationPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class RecommendationWindow(str, Enum):
    TODAY = "TODAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    MULTI_YEAR = "MULTI_YEAR"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    NONE = "NONE"

class MarketPermission(str, Enum):
    BLOCKED = "BLOCKED"
    RESTRICTED = "RESTRICTED"
    LIMITED = "LIMITED"
    CONDITIONAL = "CONDITIONAL"
    ALLOWED = "ALLOWED"
    UNKNOWN = "UNKNOWN"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 Portfolio / Explanation Parsers)
# =====================================================================
@dataclass(frozen=True)
class RecommendationProfile:
    recommendation_score: float
    execution_quality: float
    investment_quality: float
    risk_adjustment: float
    reward_adjustment: float
    overall_quality: float

@dataclass(frozen=True)
class RecommendationComponent:
    name: str
    score: float
    weight: float
    confidence: float

@dataclass(frozen=True)
class RecommendationReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class RecommendationChecklist:
    ranking_ok: bool
    risk_ok: bool
    reward_ok: bool
    confidence_ok: bool
    conviction_ok: bool
    allocation_ok: bool
    position_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_recommendation_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Recommendation_V3",
        version="3.0.0",
        stage="Layer-5: Final Recommendation",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Adaptive Absolute Synthesis",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "master_score": 0.25,
            "conviction": 0.20,
            "reward": 0.15,
            "risk_safety": 0.20,
            "confidence": 0.10,
            "institutional_fundamental": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "strong_buy_threshold": 85.0,
            "buy_threshold": 70.0,
            "accumulate_fund_threshold": 75.0,
            "hold_threshold": 50.0,
            "watch_threshold": 40.0,
            "sell_risk_threshold": 80.0,
            "adaptive_master_mult": 0.20,
            "adaptive_conviction_mult": 0.20,
            "adaptive_risk_mult": 0.15
        }
    )


# =====================================================================
# RECOMMENDATION DECISION ENGINE
# =====================================================================
class RecommendationEngine(BaseDecisionEngine):
    """
    Institutional Recommendation Engine.
    Synthesizes upstream Master Scores, Relative Rankings, and absolute Layer-4 
    Decisions to formulate a definitive Investment Advice (Buy/Sell/Hold/etc.).
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_recommendation_profile())

    def evaluate_recommendation(self, stock_output: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main execution pipeline for Final Recommendation Synthesis.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_RECOMMENDATION_ENGINE")
        
        trace = self._build_trace(stock_output, start_time, ctx, 0)

        if not isinstance(stock_output, dict) or not stock_output:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for extraction
            flat_data = self._flatten_dict(stock_output)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_L3_L4_L5_DATA")

            # 1. Identity & Gatekeeper Data
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()
            
            # 2. Extract L5 Ranking & Master Scores
            master_score = self._dynamic_lookup(flat_data, ["master_rank_score", "master_score"], 50.0)
            ranking_tier = self._dynamic_lookup_string(flat_data, ["ranking_tier"]).upper()
            
            # 3. Extract L4 Decisions (Conviction, Confidence, Risk, Reward, Exit)
            conviction = self._dynamic_lookup(flat_data, ["conviction_score", "actionability_score"], 50.0)
            confidence = self._dynamic_lookup(flat_data, ["confidence_score", "reliability_score"], 50.0)
            overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], 50.0)
            reward = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score"], 50.0)
            alloc_qual = self._dynamic_lookup(flat_data, ["allocation_quality"], 50.0)
            pos_qual = self._dynamic_lookup(flat_data, ["position_quality"], 50.0)
            
            risk_safety = self._normalize(100.0 - overall_risk)
            
            # Look for active EXIT signals from Exit Engine
            exit_action = self._dynamic_lookup_string(flat_data, ["exit_action", "exit_type"]).upper()
            
            # 4. Extract Key L3 Fundamentals / Technicals (for Accumulate logic)
            fund = self._dynamic_lookup(flat_data, ["fundamental_score", "compounder_score", "business_quality"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            mom = self._dynamic_lookup(flat_data, ["momentum_score", "momentum_strength"], 50.0)
            trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)

            # 5. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(master_score, conviction, risk_safety)

            # 6. Core Recommendation Models
            self._add_step(ctx, "COMPUTE_RECOMMENDATION_MODELS")
            
            base_rec_score = self._normalize(
                (master_score * dynamic_weights["master_score"]) +
                (conviction * dynamic_weights["conviction"]) +
                (reward * dynamic_weights["reward"]) +
                (risk_safety * dynamic_weights["risk_safety"]) +
                (confidence * dynamic_weights["confidence"]) +
                (((inst + fund) / 2.0) * dynamic_weights["institutional_fundamental"])
            )

            # Profile construction
            exec_quality = self._normalize((conviction * 0.6) + (mom * 0.4))
            inv_quality = self._normalize((fund * 0.5) + (inst * 0.3) + (trend * 0.2))
            
            rec_profile = RecommendationProfile(
                recommendation_score=round(base_rec_score, 2),
                execution_quality=round(exec_quality, 2),
                investment_quality=round(inv_quality, 2),
                risk_adjustment=round(overall_risk, 2),
                reward_adjustment=round(reward, 2),
                overall_quality=round((exec_quality + inv_quality) / 2.0, 2)
            )

            # Build Components for Layer-5 Explainability
            rec_components = [
                RecommendationComponent("Master Synthesis", round(master_score, 2), dynamic_weights.get("master_score", 0.25), round(confidence, 2)),
                RecommendationComponent("Execution Conviction", round(conviction, 2), dynamic_weights.get("conviction", 0.20), round(confidence, 2)),
                RecommendationComponent("Risk/Reward Alignment", round((reward + risk_safety)/2.0, 2), dynamic_weights.get("reward", 0.15) + dynamic_weights.get("risk_safety", 0.20), round(confidence, 2))
            ]

            # 7. Determine Final Recommendation & Grade
            self._add_step(ctx, "DETERMINE_RECOMMENDATION_AND_GRADE")
            recommendation, rec_priority, rec_window = self._determine_recommendation(
                base_rec_score, conviction, fund, mom, overall_risk, exit_action, is_permitted, ranking_tier
            )
            grade = self._determine_grade(base_rec_score, overall_risk, fund)

            # 8. Checklists
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = RecommendationChecklist(
                ranking_ok=(ranking_tier in ["ELITE", "PLATINUM", "GOLD"]),
                risk_ok=(overall_risk <= 60.0),
                reward_ok=(reward >= 50.0),
                confidence_ok=(confidence >= 50.0),
                conviction_ok=(conviction >= 50.0),
                allocation_ok=(alloc_qual >= 50.0),
                position_ok=(pos_qual >= 50.0)
            )

            # 9. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, recommendation, ranking_tier, conviction, overall_risk, reward, fund, is_permitted
            )

            # Adjust Score Post-Conflict
            final_rec_score = self._normalize(base_rec_score - (ctx.conflict_penalty * 0.5))
            
            # Provide an Absolute Recommendation Strength
            rec_strength = self._normalize((final_rec_score * 0.7) + (confidence * 0.3))

            # 10. Build Reason
            self._add_step(ctx, "BUILD_RECOMMENDATION_REASON")
            r_reason = RecommendationReason(
                primary=self._get_primary_reason(recommendation, ranking_tier, fund, exit_action),
                secondary=f"Grade: {grade.value} | Conviction: {round(conviction,1)}%",
                confidence=round(confidence, 2)
            )

            # 11. Explanations & Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, recommendation, ranking_tier, inst, fund, overall_risk, conviction, regime, is_permitted
            )

            # 12. Build Payload Dict
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "recommendation": recommendation.value,
                "recommendation_grade": grade.value,
                "recommendation_score": round(final_rec_score, 2),
                "recommendation_strength": round(rec_strength, 2),
                "recommendation_confidence": round(confidence, 2),
                "recommendation_priority": rec_priority.value,
                "recommendation_window": rec_window.value,
                "recommendation_profile": asdict(rec_profile),
                "recommendation_components": [asdict(c) for c in rec_components],
                "recommendation_reason": asdict(r_reason),
                "recommendation_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS
            status_msg = "Final Investment Recommendation formulated."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=confidence,
                rating_score=final_rec_score, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Recommendation Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(stock_output, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, master: float, conviction: float, risk_safety: float) -> dict[str, float]:
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["master_score"] += (master / 100.0) * t.get("adaptive_master_mult", 0.20)
        w["conviction"] += (conviction / 100.0) * t.get("adaptive_conviction_mult", 0.20)
        w["risk_safety"] += (risk_safety / 100.0) * t.get("adaptive_risk_mult", 0.15)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_recommendation(self, score: float, conviction: float, fund: float, mom: float, 
                                  risk: float, exit_action: str, permitted: bool, rank_tier: str):
        t = self.config.thresholds
        
        # 1. Forced Exits / Reductions (Overrides Buy Logic)
        if exit_action in ["FULL_EXIT", "EMERGENCY_EXIT", "TREND_EXIT", "INSTITUTIONAL_EXIT"]:
            return Recommendation.SELL, RecommendationPriority.CRITICAL, RecommendationWindow.TODAY
        if exit_action in ["REDUCE", "SCALE_OUT", "PROFIT_BOOKING", "PARTIAL_EXIT"]:
            return Recommendation.REDUCE, RecommendationPriority.HIGH, RecommendationWindow.SHORT_TERM

        # 2. Gatekeeper Blocks
        if not permitted or risk >= t.get("sell_risk_threshold", 80.0):
            return Recommendation.AVOID, RecommendationPriority.HIGH, RecommendationWindow.NONE

        # 3. Buy Logic Hierarchy
        if score >= t.get("strong_buy_threshold", 85.0) and conviction >= 80.0 and rank_tier in ["ELITE", "PLATINUM"]:
            return Recommendation.STRONG_BUY, RecommendationPriority.CRITICAL, RecommendationWindow.SHORT_TERM
            
        if score >= t.get("buy_threshold", 70.0) and conviction >= 60.0:
            return Recommendation.BUY, RecommendationPriority.HIGH, RecommendationWindow.MEDIUM_TERM
            
        # ACCUMULATE: Great business, but timing/momentum isn't perfect
        if fund >= t.get("accumulate_fund_threshold", 75.0) and mom < 60.0 and conviction < 60.0:
            return Recommendation.ACCUMULATE, RecommendationPriority.MEDIUM, RecommendationWindow.LONG_TERM
            
        if score >= t.get("hold_threshold", 50.0):
            return Recommendation.HOLD, RecommendationPriority.LOW, RecommendationWindow.OPEN
            
        if score >= t.get("watch_threshold", 40.0):
            return Recommendation.WATCH, RecommendationPriority.WATCH, RecommendationWindow.NONE
            
        return Recommendation.AVOID, RecommendationPriority.LOW, RecommendationWindow.NONE

    def _determine_grade(self, score: float, risk: float, fund: float) -> RecommendationGrade:
        """Assigns an institutional Alpha grade."""
        grade_score = self._normalize((score * 0.6) + (fund * 0.4) - (risk * 0.2))
        
        if grade_score >= 90.0: return RecommendationGrade.AAA
        if grade_score >= 80.0: return RecommendationGrade.AA
        if grade_score >= 70.0: return RecommendationGrade.A
        if grade_score >= 60.0: return RecommendationGrade.BBB
        if grade_score >= 50.0: return RecommendationGrade.BB
        if grade_score >= 40.0: return RecommendationGrade.B
        if grade_score >= 30.0: return RecommendationGrade.C
        return RecommendationGrade.D

    def _get_primary_reason(self, rec: Recommendation, rank_tier: str, fund: float, exit_action: str) -> str:
        if rec == Recommendation.STRONG_BUY: return "Elite ranking and exceptional conviction justify a Strong Buy."
        if rec == Recommendation.BUY: return "High-quality metrics and actionable conviction support Buy."
        if rec == Recommendation.ACCUMULATE: return "Excellent long-term business quality; scale in on weakness."
        if rec == Recommendation.HOLD: return "Existing metrics favor holding current position."
        if rec == Recommendation.REDUCE: return "Partial distribution or target hit warrants exposure reduction."
        if rec == Recommendation.SELL: return "Structural breakdown or exit trigger mandates position liquidation."
        if rec == Recommendation.AVOID: return "Unfavorable metrics or market restrictions prohibit investment."
        return "Metrics dictate observation only."

    def _detect_conflicts(self, ctx: DecisionContext, rec: Recommendation, rank_tier: str, 
                          conviction: float, risk: float, reward: float, fund: float, permitted: bool) -> None:
        """Identifies paradoxes in the final recommendation mapping."""
        
        if rec == Recommendation.STRONG_BUY and not permitted:
            self._add_conflict(ctx, "Strong Buy generated internally despite Market Gatekeeper block (Systemic Contradiction).", penalty=25.0)
            
        if rec == Recommendation.BUY and risk > 75.0:
            self._add_conflict(ctx, "Buy recommendation issued amidst extremely high structural risk.", penalty=15.0)
            
        if rec == Recommendation.ACCUMULATE and fund < 50.0:
            self._add_conflict(ctx, "Accumulate recommendation logically invalid given weak fundamental business quality.", penalty=20.0)
            
        if rec == Recommendation.SELL and reward > 85.0 and risk < 50.0:
            self._add_conflict(ctx, "Sell recommendation issued despite exceptional risk-adjusted reward potential.", penalty=15.0)
            
        if rank_tier == "ELITE" and rec == Recommendation.AVOID:
            self._add_conflict(ctx, "Elite relative rank contradicts an absolute Avoid recommendation.", penalty=20.0)

    def _generate_flags_and_explanations(self, ctx: DecisionContext, rec: Recommendation, 
                                         rank_tier: str, inst: float, fund: float, risk: float, 
                                         conviction: float, regime: float, permitted: bool) -> list[str]:
        flags = []
        
        if not permitted and rec != Recommendation.SELL:
            self._add_explanation(ctx, "Market permission restrictions prevent investment despite potentially favorable metrics.")
            flags.append("GATEKEEPER_AVOID_OVERRIDE")

        # Explanations
        if rec == Recommendation.STRONG_BUY:
            self._add_explanation(ctx, "Rare structural opportunity. Technicals, fundamentals, and risk-reward are impeccably aligned.")
        elif rec == Recommendation.ACCUMULATE:
            self._add_explanation(ctx, "Excellent business quality warrants accumulation despite suboptimal immediate technical timing.")
        elif rec == Recommendation.REDUCE:
            self._add_explanation(ctx, "Risk management protocols suggest reducing overall capital exposure.")
            
        if inst > 75.0:
            self._add_explanation(ctx, "Deep institutional footprint bolsters the final investment thesis.")
            
        if risk > 75.0:
            self._add_explanation(ctx, "High structural risk heavily suppresses recommendation strength.")
            flags.append("RISK_SUPPRESSION")
            
        if conviction < 40.0 and rec in [Recommendation.BUY, Recommendation.STRONG_BUY]:
            self._add_explanation(ctx, "Warning: Recommendation strength decoupled from execution conviction.")
            flags.append("CONVICTION_DIVERGENCE")
            
        if regime < 40.0 and rec in [Recommendation.BUY, Recommendation.ACCUMULATE]:
            self._add_explanation(ctx, "Bearish macro regime requires defensive execution despite bullish absolute recommendation.")
            flags.append("BEARISH_REGIME_HEADWINDS")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Final synthesis adjusted downward due to structural logic conflicts.")
            flags.append("RECOMMENDATION_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK (Single Stock)
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Recommendation Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "recommendation": Recommendation.NONE.value,
            "recommendation_grade": RecommendationGrade.UNKNOWN.value,
            "recommendation_score": 0.0,
            "recommendation_strength": 0.0,
            "recommendation_confidence": 0.0,
            "recommendation_priority": RecommendationPriority.NONE.value,
            "recommendation_window": RecommendationWindow.NONE.value,
            "recommendation_profile": asdict(RecommendationProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            "recommendation_components": [],
            "recommendation_reason": asdict(RecommendationReason("SYSTEM_FAILURE", "Engine crash on this candidate", 0.0)),
            "recommendation_checklist": asdict(RecommendationChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_RECOMMENDATION"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Recommendation evaluation failed. Candidate ignored.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_recommendation(stock_output: dict[str, Any] | None) -> dict[str, Any]:
    return RecommendationEngine().evaluate_recommendation(stock_output)
