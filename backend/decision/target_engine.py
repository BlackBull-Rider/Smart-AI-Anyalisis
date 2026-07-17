"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: target_engine.py

Institutional Target Decision Engine. (V3.1.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates purely to formulate the intelligent 
profit objective roadmap.

Key Upgrades from V3.0:
- Fully continuous adaptive weights.
- Structured DistanceModel and TargetLevel objects.
- Config-driven partial exit scaling profiles.
- Explicit sequencing in TargetActionPlan.
- Advanced confidence incorporating Projection Consistency & Structure Quality.

Boundary Constraint: This engine DOES NOT calculate Expected Return (CAGR), 
Stop Loss, Entry, Exit, or Position Size.
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
class TargetType(str, Enum):
    FIXED = "FIXED"
    STRUCTURAL = "STRUCTURAL"
    ATR = "ATR"
    TREND = "TREND"
    BREAKOUT = "BREAKOUT"
    LIQUIDITY = "LIQUIDITY"
    FIBONACCI = "FIBONACCI"
    SMART_MONEY = "SMART_MONEY"
    VOLATILITY = "VOLATILITY"
    OPEN = "OPEN"
    NONE = "NONE"

class TargetAction(str, Enum):
    SET = "SET"
    SCALE_OUT = "SCALE_OUT"
    HOLD = "HOLD"
    TRAIL = "TRAIL"
    FINAL_EXIT = "FINAL_EXIT"
    NONE = "NONE"

class TargetPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class TargetWindow(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    OPEN_ENDED = "OPEN_ENDED"
    NONE = "NONE"

class TrailingRecommendation(str, Enum):
    NONE = "NONE"
    ATR = "ATR"
    SWING = "SWING"
    STRUCTURE = "STRUCTURE"
    PERCENTAGE = "PERCENTAGE"
    ADAPTIVE = "ADAPTIVE"

class ProfitScaling(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"     
    BALANCED = "BALANCED"         
    CONSERVATIVE = "CONSERVATIVE" 
    NONE = "NONE"

class DistanceType(str, Enum):
    R_MULTIPLE = "R_MULTIPLE"
    PERCENTAGE = "PERCENTAGE"
    VOLATILITY_MULTIPLIER = "VOLATILITY_MULTIPLIER"
    PRICE_STRUCTURE = "PRICE_STRUCTURE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class DistanceModel:
    distance_type: str
    value: float

@dataclass(frozen=True)
class TargetLevel:
    name: str
    reference_type: str
    reference_name: str
    projection_model: str
    distance_model: DistanceModel
    confirmation: str
    confidence: float

@dataclass(frozen=True)
class TargetActionPlan:
    sequence_order: int
    action: str
    fraction: float
    trail_after: bool
    next_target: str

@dataclass(frozen=True)
class TargetReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class TargetHierarchy:
    target_1: TargetLevel | None
    target_2: TargetLevel | None
    target_3: TargetLevel | None
    final_target: TargetLevel | None

@dataclass(frozen=True)
class PartialExitPlan:
    scaling_strategy: str
    t1_fraction: float
    t2_fraction: float
    t3_fraction: float
    final_fraction: float

@dataclass(frozen=True)
class TargetChecklist:
    trend_quality: float
    momentum: float
    institutional_support: float
    breakout_strength: float
    risk_reward: float
    volatility: float
    market_permission: str


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_target_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Target_V3_1",
        version="3.1.0",
        stage="Layer-4: Target Decision",
        schema_version="3.1",
        api_version="v6",
        decision_method="Adaptive Structural Profit Scaling & Sequencing",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "trend": 0.25,
            "momentum": 0.20,
            "smart_money": 0.20,
            "breakout": 0.15,
            "volatility": 0.10,
            "reward_score": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "aggressive_scaling_threshold": 75.0,
            "conservative_volatility_threshold": 75.0,
            "adaptive_trend_mult": 0.15,
            "adaptive_mom_mult": 0.10,
            "adaptive_vol_mult": 0.15,
            "adaptive_smc_mult": 0.15,
            "adaptive_brk_mult": 0.10,
            "scaling_profile_aggressive": [0.20, 0.20, 0.30, 0.30],
            "scaling_profile_balanced": [0.25, 0.25, 0.25, 0.25],
            "scaling_profile_conservative": [0.50, 0.30, 0.20, 0.00]
        }
    )


# =====================================================================
# TARGET DECISION ENGINE
# =====================================================================
class TargetEngine(BaseDecisionEngine):
    """
    Institutional Target Decision Engine.
    Constructs the Profit Objective Roadmap (Where to take profit, how many targets, 
    fractional scaling, and trailing recommendations) using fused Layer-3 Intelligence.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_target_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_targets(engine_outputs)

    def evaluate_targets(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_TARGET_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_DATA")

            # 1. Base Intelligence & Gatekeeper Data
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            
            entry_type = self._dynamic_lookup_string(flat_data, ["entry_type", "entry_strength"])
            entry_valid = entry_type.upper() not in ["NO ENTRY", "NONE", "AVOID", "UNKNOWN"]
            
            # Structural & Risk/Reward Metrics
            trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
            mom = self._dynamic_lookup(flat_data, ["momentum_score", "momentum_strength"], 50.0)
            volatility = self._dynamic_lookup(flat_data, ["volatility_score", "atr_expansion", "volatility_risk"], 50.0)
            smc = self._dynamic_lookup(flat_data, ["smart_money_score", "institutional_footprint"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "institutional_buying"], 50.0)
            breakout = self._dynamic_lookup(flat_data, ["breakout_score", "breakout"], 50.0)
            reward = self._dynamic_lookup(flat_data, ["reward_score", "upside", "reward_potential"], 50.0)
            risk = self._dynamic_lookup(flat_data, ["risk_score", "danger", "systemic_risk"], 20.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)

            # 2. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_CONTINUOUS_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(trend, mom, volatility, smc, breakout)

            # 3. Target Quality Base
            base_target_quality = self._normalize(
                (trend * dynamic_weights["trend"]) +
                (mom * dynamic_weights["momentum"]) +
                (smc * dynamic_weights["smart_money"]) +
                (breakout * dynamic_weights["breakout"]) +
                (volatility * dynamic_weights["volatility"]) +
                (reward * dynamic_weights["reward_score"])
            )

            # 4. Generate Target Checklist
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = TargetChecklist(
                trend_quality=round(trend, 2),
                momentum=round(mom, 2),
                institutional_support=round(max(smc, inst), 2),
                breakout_strength=round(breakout, 2),
                risk_reward=round(reward, 2),
                volatility=round(volatility, 2),
                market_permission=permission_str.upper()
            )

            # 5. Determine Target Type, Scaling & Trailing
            self._add_step(ctx, "DETERMINE_TARGET_STRUCTURES")
            t_type, scaling, trailing, window, priority, reason = self._determine_target_structures(
                entry_valid, permission_str, trend, smc, breakout, volatility, reward, regime
            )

            # 6. Build Target Hierarchy & Partial Exit Plans
            self._add_step(ctx, "BUILD_HIERARCHY_AND_PLAN")
            hierarchy, partial_plan, action_plans = self._build_target_hierarchy(
                entry_valid, t_type, scaling, base_target_quality, volatility
            )

            # 7. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, trend, volatility, reward, regime, inst, 
                t_type, scaling
            )

            # 8. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            target_conf = self._calculate_advanced_confidence(
                ctx, layer3_conf, base_target_quality, parsed_inputs, trend, mom, inst, risk, breakout
            )
            
            t_reason = TargetReason(reason.primary, reason.secondary, round(target_conf, 2))

            # 9. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, t_type, scaling, trailing, entry_valid, volatility, mom, inst
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "target_type": t_type.value,
                "target_action": TargetAction.SCALE_OUT.value if entry_valid else TargetAction.NONE.value,
                "target_priority": priority.value,
                "target_window": window.value,
                "target_quality_score": round(base_target_quality, 2), # Clearly named for Layer-5
                "target_confidence": round(target_conf, 2),
                "target_reason": asdict(t_reason),
                "profit_scaling": scaling.value,
                "trailing_recommendation": trailing.value,
                "target_hierarchy": asdict(hierarchy),
                "partial_exit_plan": asdict(partial_plan),
                "target_action_plans": action_plans,
                "target_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if entry_valid else DecisionStatusEnum.NO_DATA
            status_msg = "Target Roadmap generated." if entry_valid else "No active entry. Target Engine idling."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=target_conf,
                rating_score=base_target_quality, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Target Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, trend: float, mom: float, vol: float, smc: float, brk: float) -> dict[str, float]:
        """Continuous mathematical scaling of weights instead of hardcoded if/else rules."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["trend"] += (trend / 100.0) * t.get("adaptive_trend_mult", 0.15)
        w["momentum"] += (mom / 100.0) * t.get("adaptive_mom_mult", 0.10)
        w["smart_money"] += (smc / 100.0) * t.get("adaptive_smc_mult", 0.15)
        w["breakout"] += (brk / 100.0) * t.get("adaptive_brk_mult", 0.10)
        w["volatility"] += (vol / 100.0) * t.get("adaptive_vol_mult", 0.15)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_target_structures(self, entry_valid: bool, perm_str: str, trend: float, 
                                     smc: float, brk: float, vol: float, reward: float, regime: float):
        t = self.config.thresholds
        
        if not entry_valid or "BLOCK" in perm_str.upper():
            return (
                TargetType.NONE, ProfitScaling.NONE, TrailingRecommendation.NONE, 
                TargetWindow.NONE, TargetPriority.NONE, 
                TargetReason("No active trade.", "Idle state.", 0.0)
            )

        # 1. Profit Scaling Logic
        if trend > t.get("aggressive_scaling_threshold", 75.0) and smc > 70.0 and regime > 60.0:
            scaling = ProfitScaling.AGGRESSIVE
            window = TargetWindow.OPEN_ENDED
        elif vol > t.get("conservative_volatility_threshold", 75.0) or regime < 40.0:
            scaling = ProfitScaling.CONSERVATIVE
            window = TargetWindow.SHORT_TERM
        else:
            scaling = ProfitScaling.BALANCED
            window = TargetWindow.MEDIUM_TERM

        # 2. Target Type & Trailing Logic
        if smc > 80.0:
            t_type = TargetType.LIQUIDITY
            trail = TrailingRecommendation.STRUCTURE
            priority = TargetPriority.HIGH
            reason = TargetReason("Smart Money accumulation identified.", "Liquidity pool projection.", 0.0)
        elif brk > 75.0:
            t_type = TargetType.BREAKOUT
            trail = TrailingRecommendation.SWING
            priority = TargetPriority.HIGH
            reason = TargetReason("Strong breakout confirmed.", "Measured move projection.", 0.0)
        elif trend > 75.0:
            t_type = TargetType.TREND
            trail = TrailingRecommendation.ADAPTIVE
            priority = TargetPriority.HIGH
            reason = TargetReason("Robust trend momentum.", "Trend extension projection.", 0.0)
        elif vol > 75.0:
            t_type = TargetType.ATR
            trail = TrailingRecommendation.ATR
            priority = TargetPriority.MEDIUM
            reason = TargetReason("High volatility regime.", "ATR-based defensive targets.", 0.0)
        else:
            t_type = TargetType.STRUCTURAL
            trail = TrailingRecommendation.PERCENTAGE
            priority = TargetPriority.MEDIUM
            reason = TargetReason("Standard market conditions.", "Next resistance structure.", 0.0)

        return t_type, scaling, trail, window, priority, reason

    def _build_target_hierarchy(self, entry_valid: bool, t_type: TargetType, scaling: ProfitScaling, 
                                base_qual: float, vol: float):
        """Constructs the explicit multi-tier target roadmap using structured objects."""
        if not entry_valid:
            return (TargetHierarchy(None, None, None, None), PartialExitPlan("NONE", 0.0, 0.0, 0.0, 0.0), [])
            
        conf = round(base_qual, 2)
        t_ref = t_type.value
        
        # Distance Model Generation
        dm_type = DistanceType.VOLATILITY_MULTIPLIER.value if vol > 60.0 else DistanceType.R_MULTIPLE.value
        
        t1 = TargetLevel("Target-1", t_ref, f"{t_ref}_Level_1", "Base Extension", DistanceModel(dm_type, 1.0), "Hit T1", conf)
        t2 = TargetLevel("Target-2", t_ref, f"{t_ref}_Level_2", "Primary Projection", DistanceModel(dm_type, 2.0), "Hit T2", conf)
        t3 = TargetLevel("Target-3", t_ref, f"{t_ref}_Level_3", "Exhaustion Zone", DistanceModel(dm_type, 3.0), "Hit T3", conf)
        final = TargetLevel("Final_Target", t_ref, "Extended_Projection", "Trailing Runner", DistanceModel(dm_type, 5.0), "Trail Stop Hit", conf)
        
        hierarchy = TargetHierarchy(t1, t2, t3, final)

        # Fractional Scaling logic (Config Driven)
        t = self.config.thresholds
        if scaling == ProfitScaling.AGGRESSIVE:
            fractions = t.get("scaling_profile_aggressive", [0.20, 0.20, 0.30, 0.30])
        elif scaling == ProfitScaling.CONSERVATIVE:
            fractions = t.get("scaling_profile_conservative", [0.50, 0.30, 0.20, 0.00])
        else:
            fractions = t.get("scaling_profile_balanced", [0.25, 0.25, 0.25, 0.25])
            
        plan = PartialExitPlan(scaling.value, fractions[0], fractions[1], fractions[2], fractions[3])

        # Build Action Plans with Explicit Sequence Ordering
        action_plans = []
        seq = 1
        
        if plan.t1_fraction > 0:
            action_plans.append(asdict(TargetActionPlan(seq, TargetAction.SCALE_OUT.value, plan.t1_fraction, True, "Target-2")))
            seq += 1
        if plan.t2_fraction > 0:
            action_plans.append(asdict(TargetActionPlan(seq, TargetAction.SCALE_OUT.value, plan.t2_fraction, True, "Target-3")))
            seq += 1
        if plan.t3_fraction > 0:
            action_plans.append(asdict(TargetActionPlan(seq, TargetAction.SCALE_OUT.value, plan.t3_fraction, True, "Final_Target")))
            seq += 1
        if plan.final_fraction > 0:
            action_plans.append(asdict(TargetActionPlan(seq, TargetAction.FINAL_EXIT.value, plan.final_fraction, False, "NONE")))

        return hierarchy, plan, action_plans

    def _detect_conflicts(self, ctx: DecisionContext, trend: float, vol: float, reward: float, 
                          regime: float, inst: float, t_type: TargetType, scaling: ProfitScaling) -> None:
        """Identifies logical paradoxes that contradict the target mapping."""
        
        # 1. Weak Trend + Far Target
        if trend < 40.0 and scaling == ProfitScaling.AGGRESSIVE:
            self._add_conflict(ctx, "Weak directional trend contradicts an aggressive scaling (Extended Target) plan.", penalty=15.0)
            
        # 2. Strong Trend + Conservative Targets
        if trend > 80.0 and scaling == ProfitScaling.CONSERVATIVE:
            self._add_conflict(ctx, "Exceptionally strong trend contradicts conservative/early profit taking.", penalty=10.0)
            
        # 3. Institutional Buying + Very Early Exit
        if inst > 80.0 and scaling == ProfitScaling.CONSERVATIVE:
            self._add_conflict(ctx, "Strong institutional accumulation contradicts early structural exit plans.", penalty=10.0)

        # 4. Low Volatility + Very Wide Targets
        if vol < 30.0 and t_type.value == "EXTENDED":
            self._add_conflict(ctx, "Low volatility regime cannot mathematically support extremely wide targets.", penalty=15.0)
            
        # 5. Bear Market + Aggressive Final Target
        if regime < 35.0 and scaling == ProfitScaling.AGGRESSIVE:
            self._add_conflict(ctx, "Bearish macro regime contradicts aggressive long-term hold for extended targets.", penalty=15.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, base_qual: float, 
                                       parsed_count: int, trend: float, mom: float, inst: float, risk: float, breakout: float) -> float:
        """Determines how structurally reliable the target roadmap is."""
        
        projection_consistency = self._normalize(100.0 - abs(trend - breakout))
        structure_quality = self._normalize((trend + mom + inst + breakout) / 4.0)
        historical_projection_accuracy = layer3_conf
        
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        risk_alignment = self._normalize(100.0 - risk)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        
        return self._normalize(
            (historical_projection_accuracy * 0.20) + 
            (projection_consistency * 0.20) + 
            (structure_quality * 0.20) + 
            (evidence_density * 0.15) + 
            (risk_alignment * 0.15) + 
            (signal_stability * 0.10)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, t_type: TargetType, 
                                         scaling: ProfitScaling, trailing: TrailingRecommendation, 
                                         entry_valid: bool, vol: float, mom: float, inst: float) -> list[str]:
        flags = []
        
        if not entry_valid:
            self._add_explanation(ctx, "No active entry evaluated. Target engine idling.")
            return flags

        # Explanations
        if t_type == TargetType.TREND or scaling == ProfitScaling.AGGRESSIVE:
            self._add_explanation(ctx, "Targets extended due to strong underlying trend and market structure.")
        elif t_type == TargetType.LIQUIDITY:
            self._add_explanation(ctx, "Smart Money liquidity pools provide final objective framework.")
        elif t_type == TargetType.BREAKOUT:
            self._add_explanation(ctx, "Measured move breakout projection dictates target geometry.")
            
        if inst > 75.0:
            self._add_explanation(ctx, "Institutional accumulation supports holding positions for extended objectives.")
            
        if mom < 40.0:
            self._add_explanation(ctx, "Momentum deterioration limits realistic upside projections.")
            flags.append("MOMENTUM_WEAKNESS_CAP")
            
        if vol > 75.0:
            self._add_explanation(ctx, "High volatility regime requires staggered, defensive target exits.")
            flags.append("HIGH_VOLATILITY_STAGGER")
            
        if trailing != TrailingRecommendation.NONE:
            self._add_explanation(ctx, f"{trailing.value} trailing stop recommended to protect unrealized gains post-T1.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting sub-system signals reduce target projection probability.")
            flags.append("CONFLICTING_TARGET_PROJECTIONS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Target Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "target_type": TargetType.NONE.value,
            "target_action": TargetAction.NONE.value,
            "target_priority": TargetPriority.NONE.value,
            "target_window": TargetWindow.NONE.value,
            "target_quality_score": 0.0,
            "target_confidence": 0.0,
            "target_reason": asdict(TargetReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "profit_scaling": ProfitScaling.NONE.value,
            "trailing_recommendation": TrailingRecommendation.NONE.value,
            "target_hierarchy": asdict(TargetHierarchy(None, None, None, None)),
            "partial_exit_plan": asdict(PartialExitPlan("NONE", 0.0, 0.0, 0.0, 0.0)),
            "target_action_plans": [],
            "target_checklist": asdict(TargetChecklist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "UNKNOWN")),
            "risk_flags": ["SYSTEM_FAILURE_TARGETS"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_targets(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return TargetEngine().evaluate_targets(engine_outputs)
