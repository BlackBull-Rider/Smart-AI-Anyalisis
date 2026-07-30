"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: conviction_engine.py

Institutional Conviction Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Actionability and 
Decision Strength Boundary. 

Key Responsibilities:
- Differentiates Reliability (Confidence) from Actionability (Conviction).
- Assesses Execution Readiness, Capital Commitment, and Risk-Reward Alignment.
- Determines how aggressively a validated opportunity should be executed.

Boundary Constraint: This engine DOES NOT calculate Entry, Exit, Target, Stop Loss, 
Position Size, Allocation, or Reliability. It evaluates "Actionability".
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
    DecisionTrace,
    DecisionEvidence,
    DecisionStatus
)


# =====================================================================
# ENUMS (Strict Output Typing)
# =====================================================================
class MarketPermission(str, Enum):
    BLOCKED = "BLOCKED"
    RESTRICTED = "RESTRICTED"
    LIMITED = "LIMITED"
    CONDITIONAL = "CONDITIONAL"
    ALLOWED = "ALLOWED"
    UNKNOWN = "UNKNOWN"

class ConvictionLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    ELITE = "ELITE"
    UNKNOWN = "UNKNOWN"

class ConvictionAction(str, Enum):
    AVOID = "AVOID"
    WATCH = "WATCH"
    CONSIDER = "CONSIDER"
    EXECUTE = "EXECUTE"
    HIGH_CONVICTION = "HIGH_CONVICTION"
    MAX_CONVICTION = "MAX_CONVICTION"
    NONE = "NONE"

class ConvictionPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class ConvictionPolicy(str, Enum):
    DEFENSIVE = "DEFENSIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"
    COMPOUNDER = "COMPOUNDER"
    INSTITUTIONAL = "INSTITUTIONAL"
    UNKNOWN = "UNKNOWN"

class ConvictionWindow(str, Enum):
    TODAY = "TODAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    OPEN = "OPEN"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class ConvictionProfile:
    conviction_score: float
    decision_strength: float
    execution_readiness: float
    capital_commitment: float
    risk_reward_alignment: float
    conviction_quality: float

@dataclass(frozen=True)
class ConvictionComponent:
    name: str
    score: float
    weight: float
    confidence: float

@dataclass(frozen=True)
class ConvictionReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class ConvictionChecklist:
    confidence_ok: bool
    risk_ok: bool
    reward_ok: bool
    allocation_ok: bool
    position_ok: bool
    holding_ok: bool
    market_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_conviction_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Conviction_V3",
        version="3.0.0",
        stage="Layer-4: Actionability Validation",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Execution Readiness & Alignment Synthesis",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "reliability_confidence": 0.20,
            "risk_reward_alignment": 0.25,
            "execution_readiness": 0.20,
            "capital_commitment": 0.15,
            "holding_quality": 0.10,
            "institutional_strength": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "elite_conviction_threshold": 85.0,
            "very_high_conviction_threshold": 75.0,
            "high_conviction_threshold": 60.0,
            "moderate_conviction_threshold": 40.0,
            "adaptive_rr_mult": 0.25,
            "adaptive_readiness_mult": 0.20,
            "adaptive_commitment_mult": 0.15
        }
    )


# =====================================================================
# CONVICTION DECISION ENGINE
# =====================================================================
class ConvictionEngine(BaseDecisionEngine):
    """
    Institutional Conviction Decision Engine.
    Quantifies the mathematical Actionability of the decision pipeline by measuring 
    Risk-Reward Alignment, Capital Commitment Grades, and Execution Readiness.
    Differentiates "Confidence" (Data Reliability) from "Conviction" (Action Strength).
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_conviction_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_conviction(engine_outputs)

    def evaluate_conviction(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main execution pipeline for Actionability Validation.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_CONVICTION_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON to extract upstream decisions
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER4_DATA")

            # 1. Base Intelligence & Gatekeeper Permissions
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()
            
            # 2. Extract Upstream Decision Quality Vectors
            # Confidence (Reliability)
            reliability_score = self._dynamic_lookup(flat_data, ["reliability_score", "confidence_score"], 50.0)
            
            # Risk & Reward
            risk_qual = self._dynamic_lookup(flat_data, ["risk_quality", "risk_safety"], 50.0) # Inverted Risk
            overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], 50.0)
            reward_qual = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score"], 50.0)
            
            # Capital & Execution Vectors
            alloc_qual = self._dynamic_lookup(flat_data, ["allocation_quality"], 50.0)
            pos_qual = self._dynamic_lookup(flat_data, ["position_quality"], 50.0)
            entry_qual = self._dynamic_lookup(flat_data, ["entry_quality"], 50.0)
            holding_qual = self._dynamic_lookup(flat_data, ["holding_quality"], 50.0)
            
            # Structural & Contextual Vectors
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_stability", "market_regime"], 50.0)
            inst_rel = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            fund_rel = self._dynamic_lookup(flat_data, ["fundamental_score", "business_quality"], 50.0)
            holding_type = self._dynamic_lookup_string(flat_data, ["holding_type", "holding_window", "reward_model"])

            # 3. Compute Composite Conviction Sub-Models
            self._add_step(ctx, "CALCULATE_SUB_MODELS")
            
            # Risk-Reward Alignment
            rr_alignment = self._normalize((reward_qual * 0.6) + (risk_qual * 0.4))
            
            # Execution Readiness
            execution_readiness = self._normalize((entry_qual * 0.5) + (pos_qual * 0.5))
            
            # Capital Commitment Grade
            capital_commitment = self._normalize((alloc_qual * 0.6) + (holding_qual * 0.4))

            # 4. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(
                rr_alignment, execution_readiness, capital_commitment
            )

            # 5. Base Conviction Calculation
            self._add_step(ctx, "CALCULATE_BASE_CONVICTION")
            base_conviction_score = self._normalize(
                (reliability_score * dynamic_weights["reliability_confidence"]) +
                (rr_alignment * dynamic_weights["risk_reward_alignment"]) +
                (execution_readiness * dynamic_weights["execution_readiness"]) +
                (capital_commitment * dynamic_weights["capital_commitment"]) +
                (holding_qual * dynamic_weights["holding_quality"]) +
                (inst_rel * dynamic_weights["institutional_strength"])
            )

            # 6. Checklists and Categorizations
            self._add_step(ctx, "GENERATE_CHECKLIST_AND_CATEGORIZATIONS")
            checklist = ConvictionChecklist(
                confidence_ok=(reliability_score >= 50.0),
                risk_ok=(overall_risk <= 60.0),
                reward_ok=(reward_qual >= 50.0),
                allocation_ok=(alloc_qual >= 50.0),
                position_ok=(pos_qual >= 50.0),
                holding_ok=(holding_qual >= 50.0),
                market_ok=(regime >= 50.0)
            )
            
            level, action, priority, policy, window = self._determine_categorizations(
                base_conviction_score, is_permitted, rr_alignment, holding_type, inst_rel, fund_rel
            )

            # 7. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, level, overall_risk, alloc_qual, pos_qual, reliability_score, rr_alignment, regime
            )

            # Adjust conviction score based on systemic actionability conflicts
            conviction_score = self._normalize(base_conviction_score - (ctx.conflict_penalty * 0.5)) if is_permitted else 0.0
            
            # 8. Advanced Quality and Strength Models
            self._add_step(ctx, "CALCULATE_ADVANCED_METRICS")
            decision_strength = self._calculate_decision_strength(conviction_score, execution_readiness, rr_alignment)
            conviction_quality = self._normalize((conviction_score + decision_strength) / 2.0)

            # 9. Build Conviction Reason
            self._add_step(ctx, "BUILD_CONVICTION_REASON")
            c_reason = ConvictionReason(
                primary="Actionable conviction is structurally validated." if conviction_score >= 60 else "Actionability severely constrained despite analytical baseline.",
                secondary=f"Readiness: {round(execution_readiness,1)}% | RR Align: {round(rr_alignment,1)}%",
                confidence=round(reliability_score, 2)
            )

            # Build Profile
            conv_profile = ConvictionProfile(
                conviction_score=round(conviction_score, 2),
                decision_strength=round(decision_strength, 2),
                execution_readiness=round(execution_readiness, 2),
                capital_commitment=round(capital_commitment, 2),
                risk_reward_alignment=round(rr_alignment, 2),
                conviction_quality=round(conviction_quality, 2)
            )

            # 10. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, level, rr_alignment, reliability_score, capital_commitment, 
                holding_qual, inst_rel, regime, is_permitted
            )

            # 11. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "conviction_level": level.value,
                "conviction_action": action.value,
                "conviction_priority": priority.value,
                "conviction_policy": policy.value,
                "conviction_window": window.value,
                "conviction_score": round(conviction_score, 2),
                "decision_strength": round(decision_strength, 2),
                "actionability_score": round(conviction_quality, 2),
                "execution_readiness": round(execution_readiness, 2),
                "risk_reward_alignment": round(rr_alignment, 2),
                "capital_commitment_grade": round(capital_commitment, 2),
                "conviction_reason": asdict(c_reason),
                "conviction_profile": asdict(conv_profile),
                "conviction_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL
            status_msg = "Actionability Validation Complete." if is_permitted else "Conviction restricted by Gatekeeper Block."

            trace = self._build_trace(engine_outputs, start_time, ctx, parsed_inputs)
            
            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=reliability_score, # Confidence measures the underlying reliability
                rating_score=conviction_quality, # Overall Actionability Rating
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Conviction Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, rr_alignment: float, readiness: float, commitment: float) -> dict[str, float]:
        """Mathematically biases the formula towards strong execution and reward attributes."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["risk_reward_alignment"] += (rr_alignment / 100.0) * t.get("adaptive_rr_mult", 0.25)
        w["execution_readiness"] += (readiness / 100.0) * t.get("adaptive_readiness_mult", 0.20)
        w["capital_commitment"] += (commitment / 100.0) * t.get("adaptive_commitment_mult", 0.15)
        
        # Normalize
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_categorizations(self, score: float, permitted: bool, rr_align: float, 
                                   holding_type: str, inst: float, fund: float):
        if not permitted:
            return ConvictionLevel.UNKNOWN, ConvictionAction.AVOID, ConvictionPriority.NONE, ConvictionPolicy.UNKNOWN, ConvictionWindow.NONE
            
        t = self.config.thresholds
        
        # Determine Level and Action
        if score >= t.get("elite_conviction_threshold", 85.0) and rr_align >= 75.0:
            level = ConvictionLevel.ELITE
            action = ConvictionAction.MAX_CONVICTION
            priority = ConvictionPriority.CRITICAL
        elif score >= t.get("very_high_conviction_threshold", 75.0):
            level = ConvictionLevel.VERY_HIGH
            action = ConvictionAction.HIGH_CONVICTION
            priority = ConvictionPriority.HIGH
        elif score >= t.get("high_conviction_threshold", 60.0):
            level = ConvictionLevel.HIGH
            action = ConvictionAction.EXECUTE
            priority = ConvictionPriority.MEDIUM
        elif score >= t.get("moderate_conviction_threshold", 40.0):
            level = ConvictionLevel.MODERATE
            action = ConvictionAction.CONSIDER
            priority = ConvictionPriority.LOW
        else:
            level = ConvictionLevel.LOW
            action = ConvictionAction.WATCH
            priority = ConvictionPriority.WATCH
            
        # Determine Policy
        if "COMPOUND" in holding_type.upper() and fund > 75.0:
            policy = ConvictionPolicy.COMPOUNDER
            window = ConvictionWindow.LONG_TERM
        elif inst > 80.0:
            policy = ConvictionPolicy.INSTITUTIONAL
            window = ConvictionWindow.MEDIUM_TERM
        elif score > 75.0 and rr_align > 70.0:
            policy = ConvictionPolicy.AGGRESSIVE
            window = ConvictionWindow.SHORT_TERM
        elif score < 50.0:
            policy = ConvictionPolicy.DEFENSIVE
            window = ConvictionWindow.TODAY
        else:
            policy = ConvictionPolicy.BALANCED
            window = ConvictionWindow.OPEN
            
        return level, action, priority, policy, window

    def _calculate_decision_strength(self, conviction: float, readiness: float, rr_align: float) -> float:
        """Determines the absolute force behind the execution directive."""
        return self._normalize((conviction * 0.5) + (readiness * 0.25) + (rr_align * 0.25))

    def _detect_conflicts(self, ctx: DecisionContext, level: ConvictionLevel, risk: float, 
                          alloc: float, pos: float, conf: float, rr_align: float, regime: float) -> None:
        """Identifies paradoxes between actionability and underlying execution realities."""
        
        # 1. Elite Conviction + Bear Market
        if level in [ConvictionLevel.ELITE, ConvictionLevel.VERY_HIGH] and regime < 40.0:
            self._add_conflict(ctx, "Elite conviction flagged despite severely hostile bearish macro regime.", penalty=15.0)
            
        # 2. High Conviction + Poor Allocation
        if level in [ConvictionLevel.ELITE, ConvictionLevel.VERY_HIGH] and alloc < 40.0:
            self._add_conflict(ctx, "High actionability contradicts poor capital allocation quality.", penalty=20.0)
            
        # 3. High Conviction + Weak Position Size
        if level in [ConvictionLevel.HIGH, ConvictionLevel.VERY_HIGH] and pos < 40.0:
            self._add_conflict(ctx, "Execution strength high, but position size execution metrics are fundamentally weak.", penalty=15.0)
            
        # 4. High Conviction + Low Confidence
        if level in [ConvictionLevel.HIGH, ConvictionLevel.VERY_HIGH] and conf < 50.0:
            self._add_conflict(ctx, "Actionability heavily contradicts low underlying analytical reliability (Confidence).", penalty=25.0)
            
        # 5. Maximum Conviction + Negative Risk/Reward
        if level in [ConvictionLevel.ELITE, ConvictionLevel.VERY_HIGH] and rr_align < 40.0:
            self._add_conflict(ctx, "Maximum conviction execution proposed on negative risk-reward skew.", penalty=20.0)

    def _generate_flags_and_explanations(self, ctx: DecisionContext, level: ConvictionLevel, 
                                         rr_align: float, reliability: float, capital_commit: float, 
                                         holding_qual: float, inst: float, regime: float, permitted: bool) -> list[str]:
        flags = []
        
        if not permitted:
            self._add_explanation(ctx, "Actionability suppressed: Gatekeeper explicitly denies execution permission.")
            flags.append("GATEKEEPER_CONSTRAINT")
            return flags

        # Explanations of Conviction (Actionability)
        if level in [ConvictionLevel.ELITE, ConvictionLevel.VERY_HIGH]:
            self._add_explanation(ctx, "Multiple execution engines consistently support immediate maximum action.")
            
        if rr_align > 75.0:
            self._add_explanation(ctx, "Favorable downside risk significantly strengthens actionable conviction.")
        elif rr_align < 40.0:
            self._add_explanation(ctx, "Poor risk-reward alignment severely limits execution actionability.")
            flags.append("POOR_RR_ALIGNMENT")
            
        if inst > 75.0:
            self._add_explanation(ctx, "High institutional participation vastly increases execution confidence.")
            
        if capital_commit < 40.0:
            self._add_explanation(ctx, "Weak capital allocation structurally reduces actionable conviction.")
            flags.append("WEAK_CAPITAL_COMMITMENT")
            
        if holding_qual < 40.0:
            self._add_explanation(ctx, "Current holding profile explicitly limits aggressive execution commitment.")
            flags.append("POOR_HOLDING_QUALITY")
            
        if regime < 40.0:
            self._add_explanation(ctx, "Bearish macro market regime systematically suppresses overall conviction.")
            flags.append("BEARISH_REGIME_SUPPRESSION")
            
        if reliability > 80.0 and level in [ConvictionLevel.LOW, ConvictionLevel.MODERATE]:
            self._add_explanation(ctx, "Analysis is highly reliable, but structural constraints prevent aggressive action.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Actionability compromised by structural mathematical paradoxes. Adjusting defensively.")
            flags.append("ACTIONABILITY_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Conviction Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "conviction_level": ConvictionLevel.UNKNOWN.value,
            "conviction_action": ConvictionAction.AVOID.value,
            "conviction_priority": ConvictionPriority.NONE.value,
            "conviction_policy": ConvictionPolicy.UNKNOWN.value,
            "conviction_window": ConvictionWindow.NONE.value,
            "conviction_score": 0.0,
            "decision_strength": 0.0,
            "actionability_score": 0.0,
            "execution_readiness": 0.0,
            "risk_reward_alignment": 0.0,
            "capital_commitment_grade": 0.0,
            "conviction_reason": asdict(ConvictionReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "conviction_profile": asdict(ConvictionProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            "conviction_checklist": asdict(ConvictionChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_CONVICTION"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe.",
            decision_payload=payload,
            confidence=0.0, # Reliability
            rating_score=0.0, # Actionability
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_conviction(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return ConvictionEngine().evaluate_conviction(engine_outputs)
