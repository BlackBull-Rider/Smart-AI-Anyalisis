"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: allocation_engine.py

Institutional Capital Allocation Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Capital Policy 
boundary. It evaluates Risk, Reward, Conviction, and Regime to define 
capital exposure percentages and scale-in roadmaps.

Boundary Constraint: This engine DOES NOT calculate exact share/contract quantities, 
entry/exit points, stop losses, or targets. It outputs exposure PERCENTAGES.
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

class AllocationLevel(str, Enum):
    ZERO = "ZERO"
    SMALL = "SMALL"
    MODERATE = "MODERATE"
    LARGE = "LARGE"
    AGGRESSIVE = "AGGRESSIVE"
    MAXIMUM = "MAXIMUM"
    UNKNOWN = "UNKNOWN"

class AllocationAction(str, Enum):
    SKIP = "SKIP"
    WATCH = "WATCH"
    SMALL_POSITION = "SMALL_POSITION"
    NORMAL_POSITION = "NORMAL_POSITION"
    INCREASE = "INCREASE"
    FULL_ALLOCATION = "FULL_ALLOCATION"
    NONE = "NONE"

class AllocationPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class AllocationPolicy(str, Enum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"
    COMPOUNDER = "COMPOUNDER"
    DEFENSIVE = "DEFENSIVE"
    UNKNOWN = "UNKNOWN"

class AllocationWindow(str, Enum):
    TODAY = "TODAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    OPEN = "OPEN"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 Portfolio / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class AllocationProfile:
    initial_allocation_pct: float     # Initial % of the assigned portfolio slice
    maximum_allocation_pct: float     # Max % of the assigned portfolio slice
    reserve_cash_pct: float           # % of slice kept as reserve
    portfolio_exposure_pct: float     # Absolute maximum % of entire portfolio
    sector_exposure_pct: float        # Assumed absolute max % of sector
    allocation_quality: float

@dataclass(frozen=True)
class ScaleInPlan:
    first_entry_pct: float            # e.g. 50%
    second_entry_pct: float           # e.g. 30%
    third_entry_pct: float            # e.g. 20%
    conditions: str

@dataclass(frozen=True)
class AllocationReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class AllocationChecklist:
    risk_ok: bool
    reward_ok: bool
    confidence_ok: bool
    conviction_ok: bool
    market_ok: bool
    holding_ok: bool
    portfolio_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_allocation_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Allocation_V3",
        version="3.0.0",
        stage="Layer-4: Allocation Decision",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Adaptive Capital Sizing Policy",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "reward_quality": 0.25,
            "risk_safety": 0.25,
            "conviction": 0.20,
            "institutional_support": 0.15,
            "market_regime": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "max_portfolio_exposure_limit": 10.0, # Max 10% of portfolio per asset
            "max_sector_exposure_limit": 25.0,    # Max 25% per sector
            "aggressive_alloc_threshold": 80.0,
            "moderate_alloc_threshold": 60.0,
            "small_alloc_threshold": 40.0,
            "adaptive_risk_mult": 0.20,
            "adaptive_reward_mult": 0.20,
            "adaptive_conviction_mult": 0.15
        }
    )


# =====================================================================
# ALLOCATION DECISION ENGINE
# =====================================================================
class AllocationEngine(BaseDecisionEngine):
    """
    Institutional Capital Allocation Engine.
    Computes absolute and relative portfolio exposure limits, capital deployment 
    schedules (scale-in plans), and cash reserves based on upstream Risk-Reward profiles.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_allocation_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_allocation(engine_outputs)

    def evaluate_allocation(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main decision execution pipeline for Capital Allocation Policy.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_ALLOCATION_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER4_DATA")

            # 1. Base Intelligence & Permissions
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()
            
            # 2. Extract Key Upstream Scores
            overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], 50.0)
            risk_safety = self._normalize(100.0 - overall_risk) # Invert for weight calculations
            
            reward_quality = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score", "upside_probability"], 50.0)
            conviction = self._dynamic_lookup(flat_data, ["confidence", "conviction", "holding_confidence"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            fund = self._dynamic_lookup(flat_data, ["fundamental_score", "business_quality"], 50.0)
            
            holding_type = self._dynamic_lookup_string(flat_data, ["holding_type", "holding_window"]).upper()

            # 3. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(risk_safety, reward_quality, conviction)

            # 4. Base Allocation Quality
            self._add_step(ctx, "CALCULATE_ALLOCATION_QUALITY")
            base_allocation_quality = self._normalize(
                (reward_quality * dynamic_weights["reward_quality"]) +
                (risk_safety * dynamic_weights["risk_safety"]) +
                (conviction * dynamic_weights["conviction"]) +
                (inst * dynamic_weights["institutional_support"]) +
                (regime * dynamic_weights["market_regime"])
            )

            if not is_permitted:
                base_allocation_quality = 0.0

            # 5. Determine Policy & Profile
            self._add_step(ctx, "DETERMINE_POLICY_AND_PROFILE")
            policy = self._determine_allocation_policy(base_allocation_quality, overall_risk, regime, holding_type)
            alloc_profile = self._calculate_allocation_percentages(base_allocation_quality, policy, is_permitted)

            # 6. Build Scale-In Roadmap
            self._add_step(ctx, "BUILD_SCALE_IN_PLAN")
            scale_in = self._build_scale_in_plan(policy, is_permitted)

            # 7. Checklists and Categorizations
            self._add_step(ctx, "GENERATE_CHECKLIST_AND_CATEGORIZATIONS")
            checklist = AllocationChecklist(
                risk_ok=(overall_risk < 60.0),
                reward_ok=(reward_quality > 60.0),
                confidence_ok=(conviction > 60.0),
                conviction_ok=(base_allocation_quality > 50.0),
                market_ok=(regime > 50.0),
                holding_ok=("UNKNOWN" not in holding_type and "NONE" not in holding_type),
                portfolio_ok=True # Assuming macro portfolio constraints are cleared
            )
            
            level, action, priority, window = self._determine_categorizations(
                base_allocation_quality, is_permitted, policy
            )

            # 8. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, level, overall_risk, conviction, regime, reward_quality, inst
            )

            # Force adjust quality down if conflicts exist
            allocation_quality = self._normalize(base_allocation_quality - (ctx.conflict_penalty * 0.5)) if is_permitted else 0.0

            # 9. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            alloc_conf = self._calculate_advanced_confidence(
                ctx, conviction, allocation_quality, parsed_inputs, risk_safety, reward_quality, regime
            )
            
            a_reason = AllocationReason(
                primary="Capital policy formulated based on positive risk/reward skew." if is_permitted and allocation_quality > 50 else "Allocation restricted due to hostile metrics or Gatekeeper limits.",
                secondary=f"Policy: {policy.value} | Max Exp: {alloc_profile.portfolio_exposure_pct}%",
                confidence=round(alloc_conf, 2)
            )

            # 10. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, level, policy, overall_risk, reward_quality, conviction, regime, is_permitted
            )

            # 11. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "allocation_level": level.value,
                "allocation_action": action.value,
                "allocation_priority": priority.value,
                "allocation_policy": policy.value,
                "allocation_window": window.value,
                "allocation_quality": round(allocation_quality, 2),
                "allocation_confidence": round(alloc_conf, 2),
                "allocation_reason": asdict(a_reason),
                "allocation_profile": asdict(alloc_profile),
                "scale_in_plan": asdict(scale_in),
                "allocation_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL
            status_msg = "Allocation Policy formulated." if is_permitted else "Allocation Blocked by Upstream Gatekeepers."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=alloc_conf,
                rating_score=allocation_quality, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Allocation Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, risk_safety: float, reward_qual: float, conviction: float) -> dict[str, float]:
        """Mathematically biases the allocation formula towards the dominant safety/reward metric."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["risk_safety"] += (risk_safety / 100.0) * t.get("adaptive_risk_mult", 0.20)
        w["reward_quality"] += (reward_qual / 100.0) * t.get("adaptive_reward_mult", 0.20)
        w["conviction"] += (conviction / 100.0) * t.get("adaptive_conviction_mult", 0.15)
        
        # Normalize weights
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_allocation_policy(self, qual: float, risk: float, regime: float, holding: str) -> AllocationPolicy:
        if "COMPOUND" in holding and qual > 75.0:
            return AllocationPolicy.COMPOUNDER
        if risk > 65.0 or regime < 45.0:
            return AllocationPolicy.DEFENSIVE
        if qual > 80.0 and risk < 40.0:
            return AllocationPolicy.AGGRESSIVE
        if qual < 50.0:
            return AllocationPolicy.CONSERVATIVE
        return AllocationPolicy.BALANCED

    def _calculate_allocation_percentages(self, qual: float, policy: AllocationPolicy, permitted: bool) -> AllocationProfile:
        t = self.config.thresholds
        max_abs_exposure = t.get("max_portfolio_exposure_limit", 10.0)
        
        if not permitted:
            return AllocationProfile(0.0, 0.0, 100.0, 0.0, 0.0, 0.0)

        # Base portfolio exposure based on quality
        portfolio_exp = self._clamp((qual / 100.0) * max_abs_exposure, 1.0, max_abs_exposure)
        sector_exp = self._clamp(portfolio_exp * 2.5, 1.0, t.get("max_sector_exposure_limit", 25.0))
        
        # Relative sizing based on policy
        if policy == AllocationPolicy.AGGRESSIVE:
            init_pct, max_pct, reserve = 100.0, 100.0, 0.0 # Full deployment
        elif policy == AllocationPolicy.COMPOUNDER:
            init_pct, max_pct, reserve = 25.0, 100.0, 75.0 # SIP/Scale-in heavily
        elif policy == AllocationPolicy.DEFENSIVE:
            init_pct, max_pct, reserve = 50.0, 50.0, 50.0  # Deploy half, keep half cash
        elif policy == AllocationPolicy.CONSERVATIVE:
            init_pct, max_pct, reserve = 30.0, 60.0, 40.0  # Cautious
        else: # BALANCED
            init_pct, max_pct, reserve = 50.0, 100.0, 50.0 # Deploy half now, half later
            
        return AllocationProfile(
            initial_allocation_pct=round(init_pct, 2),
            maximum_allocation_pct=round(max_pct, 2),
            reserve_cash_pct=round(reserve, 2),
            portfolio_exposure_pct=round(portfolio_exp, 2),
            sector_exposure_pct=round(sector_exp, 2),
            allocation_quality=round(qual, 2)
        )

    def _build_scale_in_plan(self, policy: AllocationPolicy, permitted: bool) -> ScaleInPlan:
        if not permitted:
            return ScaleInPlan(0.0, 0.0, 0.0, "Blocked")
            
        if policy == AllocationPolicy.AGGRESSIVE:
            return ScaleInPlan(100.0, 0.0, 0.0, "All-in on initial entry signal.")
        elif policy == AllocationPolicy.COMPOUNDER:
            return ScaleInPlan(25.0, 25.0, 50.0, "Time-based SIP or structural pullback accumulation.")
        elif policy == AllocationPolicy.DEFENSIVE:
            return ScaleInPlan(100.0, 0.0, 0.0, "No scaling. Strict one-time defensive entry.")
        elif policy == AllocationPolicy.CONSERVATIVE:
            return ScaleInPlan(50.0, 50.0, 0.0, "Second tranche on strong continuation confirmation.")
        else: # BALANCED
            return ScaleInPlan(50.0, 30.0, 20.0, "Scale in on minor pullbacks/retests.")

    def _determine_categorizations(self, qual: float, permitted: bool, policy: AllocationPolicy):
        if not permitted:
            return AllocationLevel.ZERO, AllocationAction.SKIP, AllocationPriority.NONE, AllocationWindow.NONE
            
        t = self.config.thresholds
        
        # Level & Action
        if qual >= t.get("aggressive_alloc_threshold", 80.0):
            level = AllocationLevel.AGGRESSIVE if policy == AllocationPolicy.AGGRESSIVE else AllocationLevel.LARGE
            action = AllocationAction.FULL_ALLOCATION
            priority = AllocationPriority.CRITICAL
            window = AllocationWindow.TODAY
        elif qual >= t.get("moderate_alloc_threshold", 60.0):
            level = AllocationLevel.MODERATE
            action = AllocationAction.NORMAL_POSITION
            priority = AllocationPriority.HIGH
            window = AllocationWindow.SHORT_TERM
        elif qual >= t.get("small_alloc_threshold", 40.0):
            level = AllocationLevel.SMALL
            action = AllocationAction.SMALL_POSITION
            priority = AllocationPriority.MEDIUM
            window = AllocationWindow.OPEN
        else:
            level = AllocationLevel.ZERO
            action = AllocationAction.WATCH
            priority = AllocationPriority.LOW
            window = AllocationWindow.NONE
            
        return level, action, priority, window

    def _detect_conflicts(self, ctx: DecisionContext, level: AllocationLevel, risk: float, 
                          conviction: float, regime: float, reward: float, inst: float) -> None:
        """Identifies contradictions between allocation aggressiveness and systemic metrics."""
        
        # 1. High Allocation + Extreme Risk
        if level in [AllocationLevel.LARGE, AllocationLevel.AGGRESSIVE] and risk > 75.0:
            self._add_conflict(ctx, "Aggressive capital allocation proposed amidst extreme systemic/asset risk.", penalty=20.0)
            
        # 2. High Allocation + Low Conviction
        if level in [AllocationLevel.LARGE, AllocationLevel.AGGRESSIVE] and conviction < 40.0:
            self._add_conflict(ctx, "High allocation mathematically invalid given low upstream conviction scores.", penalty=15.0)
            
        # 3. Aggressive Allocation + Bear Market
        if level in [AllocationLevel.AGGRESSIVE] and regime < 40.0:
            self._add_conflict(ctx, "Aggressive portfolio exposure contradicts hostile bearish macro regime.", penalty=15.0)
            
        # 4. Large Allocation + Poor Reward
        if level in [AllocationLevel.MODERATE, AllocationLevel.LARGE] and reward < 40.0:
            self._add_conflict(ctx, "Capital deployment scaled up despite structurally poor risk/reward profiles.", penalty=15.0)
            
        # 5. Large Allocation + Weak Institutional Support
        if level in [AllocationLevel.LARGE, AllocationLevel.AGGRESSIVE] and inst < 35.0:
            self._add_conflict(ctx, "Large capital deployment lacks the required underlying institutional flow support.", penalty=10.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, upstream_conviction: float, 
                                       alloc_qual: float, parsed_count: int, risk_safety: float, 
                                       reward_qual: float, regime: float) -> float:
        """Determines how structurally reliable the capital allocation policy is."""
        
        engine_agreement = self._normalize(100.0 - abs(risk_safety - reward_qual))
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        market_stability = self._normalize(regime)
        
        return self._normalize(
            (upstream_conviction * 0.20) + (alloc_qual * 0.20) + 
            (engine_agreement * 0.20) + (evidence_density * 0.15) + 
            (market_stability * 0.15) + (signal_stability * 0.10)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, level: AllocationLevel, 
                                         policy: AllocationPolicy, risk: float, reward: float, 
                                         conviction: float, regime: float, permitted: bool) -> list[str]:
        flags = []
        
        if not permitted:
            self._add_explanation(ctx, "Allocation restricted to Zero: Upstream Gatekeeper denies market permission.")
            flags.append("GATEKEEPER_CONSTRAINT")
            return flags

        # Explanations
        if risk < 40.0:
            self._add_explanation(ctx, "Low structural portfolio risk mathematically supports larger capital allocation.")
        else:
            self._add_explanation(ctx, "Elevated risk environment necessitates defensive cash buffering.")
            flags.append("RISK_BUFFERING_ACTIVE")
            
        if reward > 75.0 and policy != AllocationPolicy.DEFENSIVE:
            self._add_explanation(ctx, "Current risk-adjusted reward profile strongly justifies progressive scaling.")
            
        if conviction > 75.0:
            self._add_explanation(ctx, "High upstream conviction permits normal/elevated allocation deployment.")
        elif conviction < 40.0:
            self._add_explanation(ctx, "Low conviction caps initial allocation. Mandatory fractional scaling enforced.")
            flags.append("CONVICTION_CAPPED")
            
        if regime < 40.0:
            self._add_explanation(ctx, "Bearish macro regime strictly enforces higher reserve cash requirements.")
            flags.append("REGIME_DEFENSE_MODE")

        if policy == AllocationPolicy.COMPOUNDER:
            self._add_explanation(ctx, "Compounder logic dictates long-term structured systematic accumulation (SIP).")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting evaluation metrics degrade maximum safe allocation exposure.")
            flags.append("ALLOCATION_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Allocation Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "allocation_level": AllocationLevel.ZERO.value,
            "allocation_action": AllocationAction.SKIP.value,
            "allocation_priority": AllocationPriority.NONE.value,
            "allocation_policy": AllocationPolicy.UNKNOWN.value,
            "allocation_window": AllocationWindow.NONE.value,
            "allocation_quality": 0.0,
            "allocation_confidence": 0.0,
            "allocation_reason": asdict(AllocationReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "allocation_profile": asdict(AllocationProfile(0.0, 0.0, 100.0, 0.0, 0.0, 0.0)),
            "scale_in_plan": asdict(ScaleInPlan(0.0, 0.0, 0.0, "Engine Failure")),
            "allocation_checklist": asdict(AllocationChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_ALLOCATION"]
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
def evaluate_allocation(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return AllocationEngine().evaluate_allocation(engine_outputs)
