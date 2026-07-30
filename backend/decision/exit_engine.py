"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: exit_engine.py

Institutional Exit Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates as the premier capital protection gate.
Converts fused Layer-3 Intelligence and Market Regime permissions into highly 
structured, deterministic institutional exit frameworks.

Key Upgrades from V2:
- ExitAction Object (Intent-based execution: REDUCE, FULL_EXIT, HEDGE with fractions).
- Market Regime is Advisory (Blocks new entries, but evaluates existing holds gracefully).
- Continuous Adaptive Weighting (No hardcoded if/else rules).
- ExitReason & ExitTrigger structured objects.
- Granular Risk & Emergency Metrics (Liquidity Collapse, Gap, Circuit Risk).
- Profit Booking strictly consumes upstream Target/Reward engine signals.
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

class ExitType(str, Enum):
    TREND_EXIT = "TREND_EXIT"
    MOMENTUM_EXIT = "MOMENTUM_EXIT"
    PROFIT_BOOKING = "PROFIT_BOOKING"
    SMART_MONEY_EXIT = "SMART_MONEY_EXIT"
    INSTITUTIONAL_EXIT = "INSTITUTIONAL_EXIT"
    RISK_EXIT = "RISK_EXIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    FULL_EXIT = "FULL_EXIT"
    SCALE_OUT = "SCALE_OUT"
    ROTATE = "ROTATE"
    HEDGE = "HEDGE"
    NO_EXIT = "NO_EXIT"

class ExitWindow(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    TODAY_CLOSE = "TODAY_CLOSE"
    NEXT_OPEN = "NEXT_OPEN"
    THIS_WEEK = "THIS_WEEK"
    WAIT_CONFIRMATION = "WAIT_CONFIRMATION"
    MONITOR = "MONITOR"
    NONE = "NONE"

class ExitPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class ExitActionType(str, Enum):
    FULL_EXIT = "FULL_EXIT"
    REDUCE = "REDUCE"
    HEDGE = "HEDGE"
    HOLD = "HOLD"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class ExitTrigger:
    trigger_condition: str
    confirmation_metric: str
    urgency_level: str

@dataclass(frozen=True)
class ExitReason:
    primary_reason: str
    secondary_reason: str
    confidence: float

@dataclass(frozen=True)
class ExitAction:
    action: ExitActionType
    fraction: float  # 1.0 = 100%, 0.50 = 50%
    intent_reason: str

@dataclass(frozen=True)
class ExitChecklist:
    trend_weakness_score: float
    momentum_weakness_score: float
    distribution_score: float
    institutional_selling_score: float
    systemic_risk_score: float
    target_achievement_score: float
    regime_hostility_score: float


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_exit_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Exit_V3",
        version="3.0.0",
        stage="Layer-4: Exit Decision",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Adaptive Risk-Weighted Exit Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "trend_weakness": 0.20,
            "momentum_weakness": 0.15,
            "distribution": 0.20,
            "institutional_selling": 0.20,
            "risk_environment": 0.15,
            "market_regime_weakness": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "emergency_risk_threshold": 85.0,
            "profit_booking_threshold": 80.0,
            "scale_out_threshold": 60.0,
            "full_exit_threshold": 85.0,
            "adaptive_risk_multiplier": 0.20,
            "adaptive_dist_multiplier": 0.15
        }
    )


# =====================================================================
# EXIT DECISION ENGINE
# =====================================================================
class ExitEngine(BaseDecisionEngine):
    """
    Institutional Exit Decision Engine.
    Evaluates deterioration in trend, momentum, institutional backing, and broad 
    market regimes to determine institutional exit intent (Action & Fraction).
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_exit_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_exit(engine_outputs)

    def evaluate_exit(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_EXIT_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_AND_GATEKEEPER_DATA")

            # 1. Gatekeeper / Market Permission Check (Advisory Only for Exit)
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            market_permission = self._map_market_permission(permission_str)

            # 2. Extract Base Intelligence from Layer-3 & Upstream Target Engines
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            
            # Trend & Momentum Weakness (Inverted from Health)
            trend_health = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength", "trend_alignment"], 50.0)
            trend_weakness = self._normalize(100.0 - trend_health)
            mom_health = self._dynamic_lookup(flat_data, ["momentum_score", "momentum_strength"], 50.0)
            mom_weakness = self._normalize(100.0 - mom_health)
            
            # Regime Hostility (Inverted from Health)
            regime_health = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
            regime_hostility = self._normalize(100.0 - regime_health)
            
            # Market Permission Modifier (Blocked regime means hostile, but doesn't force instant liquidation if trend holds)
            if market_permission in [MarketPermission.BLOCKED, MarketPermission.RESTRICTED]:
                regime_hostility = self._clamp(regime_hostility + 30.0)

            # Explicit Risk / Negative Metrics
            dist_risk = self._dynamic_lookup(flat_data, ["distribution_score", "selling_pressure", "distribution"], 20.0)
            inst_sell = self._dynamic_lookup(flat_data, ["institutional_selling", "fii_selling", "outflow"], 20.0)
            
            # Emergency Granular Risk Metrics (Consumed from Risk Engine)
            volatility_risk = self._dynamic_lookup(flat_data, ["volatility_risk", "atr_expansion", "danger"], 20.0)
            liquidity_risk = self._dynamic_lookup(flat_data, ["liquidity_risk", "liquidity_collapse", "slippage"], 20.0)
            gap_risk = self._dynamic_lookup(flat_data, ["gap_risk", "overnight_risk"], 20.0)
            circuit_risk = self._dynamic_lookup(flat_data, ["circuit_risk", "lock_limit"], 20.0)
            systemic_risk = self._normalize((volatility_risk * 0.4) + (liquidity_risk * 0.3) + (gap_risk * 0.2) + (circuit_risk * 0.1))

            # Profit Booking Signals (Consumed from Target/Reward Engine)
            target_achieved = self._dynamic_lookup(flat_data, ["target_hit", "reward_realized", "target_achievement"], 0.0)

            # 3. Continuous Adaptive Weights Calculation
            self._add_step(ctx, "CALCULATE_CONTINUOUS_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(
                trend_weakness, dist_risk, inst_sell, systemic_risk, regime_hostility
            )

            # 4. Exit Necessity Model Calculation (0 = Safe Hold, 100 = Immediate Liquidation)
            self._add_step(ctx, "CALCULATE_EXIT_NECESSITY")
            exit_necessity = self._calculate_exit_necessity(
                trend_weakness, mom_weakness, dist_risk, inst_sell, systemic_risk, regime_hostility, dynamic_weights
            )

            # 5. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, trend_health, dist_risk, mom_health, regime_hostility, 
                inst_sell, systemic_risk, exit_necessity, target_achieved
            )

            # Adjust necessity slightly by conflicts to prevent whiplash
            adjusted_exit_necessity = self._clamp(exit_necessity - (ctx.conflict_penalty * 0.4))

            # 6. Generate Exit Checklist (Percentage Based)
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = ExitChecklist(
                trend_weakness_score=round(trend_weakness, 2),
                momentum_weakness_score=round(mom_weakness, 2),
                distribution_score=round(dist_risk, 2),
                institutional_selling_score=round(inst_sell, 2),
                systemic_risk_score=round(systemic_risk, 2),
                target_achievement_score=round(target_achieved, 2),
                regime_hostility_score=round(regime_hostility, 2)
            )

            # 7. Determine Exit Structures (Type, Action, Priority, Window, Trigger, Reason)
            self._add_step(ctx, "DETERMINE_EXIT_STRUCTURES")
            (exit_type, exit_action, exit_priority, exit_window, exit_trigger, exit_reason) = self._determine_exit_structures(
                adjusted_exit_necessity, checklist, trend_health, liquidity_risk, gap_risk, circuit_risk
            )

            # 8. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            exit_confidence = self._calculate_advanced_confidence(
                ctx, layer3_conf, adjusted_exit_necessity, trend_health, mom_health, 
                dist_risk, inst_sell, parsed_inputs
            )

            # Update ExitReason confidence
            exit_reason = ExitReason(
                primary_reason=exit_reason.primary_reason, 
                secondary_reason=exit_reason.secondary_reason, 
                confidence=round(exit_confidence, 2)
            )

            # 9. Generate Flags and Institutional Explanations
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, exit_type, exit_action, checklist, liquidity_risk, gap_risk, circuit_risk, market_permission
            )

            # 10. Build Final Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "exit_type": exit_type.value,
                "exit_priority": exit_priority.value,
                "exit_window": exit_window.value,
                "market_permission_advisory": market_permission.value,
                "exit_action": asdict(exit_action),
                "exit_reason": asdict(exit_reason),
                "exit_trigger": asdict(exit_trigger),
                "exit_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            trace = self._build_trace(engine_outputs, start_time, ctx, parsed_inputs)
            
            return self._build_output(
                status=DecisionStatusEnum.SUCCESS,
                status_msg="Exit intent constructed successfully.",
                decision_payload=decision_payload,
                confidence=exit_confidence,
                rating_score=adjusted_exit_necessity, # Note: High Score = High Exit Necessity
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Exit Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & ADVANCED MODELS
    # ---------------------------------------------------------

    def _map_market_permission(self, perm_str: str) -> MarketPermission:
        p = perm_str.upper()
        if "BLOCK" in p: return MarketPermission.BLOCKED
        if "RESTRICT" in p: return MarketPermission.RESTRICTED
        if "LIMIT" in p: return MarketPermission.LIMITED
        if "CONDITION" in p: return MarketPermission.CONDITIONAL
        if "ALLOW" in p or "YES" in p: return MarketPermission.ALLOWED
        return MarketPermission.UNKNOWN

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, tw: float, dist: float, inst: float, sys_risk: float, regime: float) -> dict[str, float]:
        """Continuous mathematical scaling of weights instead of hardcoded if/else rules."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        # Scale risk weight linearly based on how severe the systemic risk is
        w["risk_environment"] += (sys_risk / 100.0) * t.get("adaptive_risk_multiplier", 0.20)
        
        # Scale distribution impact
        w["distribution"] += (dist / 100.0) * t.get("adaptive_dist_multiplier", 0.15)
        
        # Normalize back to 1.0
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _calculate_exit_necessity(self, tw: float, mw: float, dist: float, inst: float, 
                                  sys_risk: float, regime_hostility: float, w: dict) -> float:
        """Calculates the aggregate pressure to exit. 0 = Safe Hold. 100 = Immediate Liquidation."""
        base = (
            (tw * w.get("trend_weakness", 0.20)) +
            (mw * w.get("momentum_weakness", 0.15)) +
            (dist * w.get("distribution", 0.20)) +
            (inst * w.get("institutional_selling", 0.20)) +
            (sys_risk * w.get("risk_environment", 0.15)) +
            (regime_hostility * w.get("market_regime_weakness", 0.10))
        )
        return self._normalize(base)

    def _determine_exit_structures(self, necessity: float, chk: ExitChecklist, trend_health: float, 
                                   liq_risk: float, gap_risk: float, circuit_risk: float):
        """Maps specific failure vectors into definitive institutional exit intents and actions."""
        t = self.config.thresholds

        # 1. Profit Booking (Triggered by Upstream Target/Reward Engine)
        if chk.target_achievement_score >= t.get("profit_booking_threshold", 80.0):
            action_frac = 1.0 if trend_health < 40.0 else 0.50 # Full exit if trend is dead, else Scale Out 50%
            action_type = ExitActionType.FULL_EXIT if action_frac == 1.0 else ExitActionType.REDUCE
            return (
                ExitType.PROFIT_BOOKING,
                ExitAction(action_type, action_frac, "Upstream Target Realized"),
                ExitPriority.HIGH, ExitWindow.TODAY_CLOSE,
                ExitTrigger("Profit Target Achieved", "Reward Engine Signal", "HIGH"),
                ExitReason("Profit objective reached.", "Capital reallocation triggered.", 0.0)
            )

        # 2. Emergency Triggers (Liquidity Collapse, Circuit, Gap)
        if liq_risk > 85.0 or gap_risk > 85.0 or circuit_risk > 85.0 or chk.systemic_risk_score > t.get("emergency_risk_threshold", 85.0):
            return (
                ExitType.EMERGENCY_EXIT,
                ExitAction(ExitActionType.FULL_EXIT, 1.0, "Absolute Capital Preservation"),
                ExitPriority.CRITICAL, ExitWindow.IMMEDIATE,
                ExitTrigger("Systemic Danger Metric Breached", "Risk Engine Override", "CRITICAL"),
                ExitReason("Extreme systemic volatility or liquidity collapse.", "Immediate capital protection mandated.", 0.0)
            )

        # 3. Institutional / Smart Money Dump
        if chk.institutional_selling_score > 80.0 and chk.distribution_score > 80.0:
            return (
                ExitType.INSTITUTIONAL_EXIT,
                ExitAction(ExitActionType.FULL_EXIT, 1.0, "Dumping position to match smart money"),
                ExitPriority.HIGH, ExitWindow.NEXT_OPEN,
                ExitTrigger("Heavy Institutional Outflow", "Volume Distribution Confirmation", "HIGH"),
                ExitReason("Severe institutional selling detected.", "Overhead supply distribution.", 0.0)
            )

        # 4. Market Regime Hostility (Hedge or Rotate)
        if chk.regime_hostility_score > 85.0 and trend_health > 60.0:
            return (
                ExitType.HEDGE,
                ExitAction(ExitActionType.HEDGE, 0.50, "Macro is hostile, but micro trend survives"),
                ExitPriority.MEDIUM, ExitWindow.THIS_WEEK,
                ExitTrigger("Macro Regime Collapse", "Local Trend Still Valid", "MEDIUM"),
                ExitReason("Bearish market regime forces structural hedging.", "Local trend prevents full exit.", 0.0)
            )

        # 5. Core Necessity Scaling (Full Exit vs Scale Out)
        if necessity >= t.get("full_exit_threshold", 85.0):
            return (
                ExitType.TREND_EXIT,
                ExitAction(ExitActionType.FULL_EXIT, 1.0, "Structural collapse confirmed"),
                ExitPriority.HIGH, ExitWindow.TODAY_CLOSE,
                ExitTrigger("Trend & Momentum Breakdown", "Technical Floor Breached", "HIGH"),
                ExitReason("Macro trend deterioration dictates full liquidation.", "Momentum exhaust confirmed.", 0.0)
            )
            
        if necessity >= t.get("scale_out_threshold", 60.0):
            fraction = self._clamp((necessity - 50.0) / 50.0, 0.25, 0.75) # Dynamically scales fraction
            return (
                ExitType.SCALE_OUT,
                ExitAction(ExitActionType.REDUCE, round(fraction, 2), "Scaling out to reduce exposure"),
                ExitPriority.WATCH, ExitWindow.MONITOR,
                ExitTrigger("Accumulating Structural Weakness", "Partial Distribution", "WATCH"),
                ExitReason("Progressive weakness suggests scaling down.", "Risk mitigation.", 0.0)
            )

        # Default Hold
        return (
            ExitType.NO_EXIT,
            ExitAction(ExitActionType.HOLD, 0.0, "Structure remains viable"),
            ExitPriority.NONE, ExitWindow.NONE,
            ExitTrigger("Trend intact", "No Exit Required", "LOW"),
            ExitReason("Metrics remain healthy.", "Holding structure fully intact.", 0.0)
        )

    def _detect_conflicts(self, ctx: DecisionContext, trend_health: float, dist: float, mom_health: float, 
                          regime_hostility: float, inst: float, sys_risk: float, necessity: float, target_achieved: float) -> None:
        """Identifies logical paradoxes that make exit decisions murky."""
        if trend_health > 75.0 and dist > 75.0:
            self._add_conflict(ctx, "Strong superficial trend masking heavy institutional distribution (Trap).", penalty=15.0)
        if mom_health > 75.0 and regime_hostility > 75.0:
            self._add_conflict(ctx, "High asset momentum fighting a severely bearish macro market.", penalty=15.0)
        if necessity < 40.0 and sys_risk > 80.0:
            self._add_conflict(ctx, "Metrics suggest holding, but systemic risk is dangerously high.", penalty=20.0)
        if target_achieved > 80.0 and trend_health > 85.0:
            self._add_conflict(ctx, "Target hit but trend remains exceptionally strong (Early exit risk).", penalty=10.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, necessity: float, 
                                       trend_health: float, mom_health: float, dist: float, 
                                       inst: float, parsed_count: int) -> float:
        """Determines how reliable this exit signal is using Evidence Density and Engine Agreement."""
        
        # Engine Agreement: Align weakness if exiting, align strength if holding
        if necessity > 50.0:
            engine_agreement = self._normalize(100.0 - abs(dist - inst))
        else:
            engine_agreement = self._normalize(100.0 - abs(trend_health - mom_health))
            
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        historical_stability = layer3_conf # Assuming Layer-3 feeds long-term reliability
        
        return self._normalize(
            (historical_stability * 0.20) + (evidence_density * 0.20) + 
            (engine_agreement * 0.30) + (signal_stability * 0.30)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, type_enum: ExitType, 
                                         action: ExitAction, chk: ExitChecklist, 
                                         liq_risk: float, gap_risk: float, circuit_risk: float, 
                                         perm: MarketPermission) -> list[str]:
        flags = []
        
        if perm in [MarketPermission.BLOCKED, MarketPermission.RESTRICTED]:
            self._add_explanation(ctx, "Gatekeeper signals macro hostility. Advisory constraint applied to existing positions.")
            flags.append("MACRO_REGIME_HOSTILE")

        if type_enum == ExitType.NO_EXIT:
            self._add_explanation(ctx, "Core structural health remains intact. No institutional exit required.")
            return flags

        if type_enum == ExitType.PROFIT_BOOKING:
            self._add_explanation(ctx, f"Target realization triggered. {action.fraction*100}% reduction advised.")
            
        if action.action == ExitActionType.HEDGE:
            self._add_explanation(ctx, "Trend survives but macro warrants hedging strategy.")

        if chk.distribution_score > 75.0 or chk.institutional_selling_score > 75.0:
            self._add_explanation(ctx, "Significant institutional distribution detected.")
            flags.append("INSTITUTIONAL_OUTFLOW")

        if liq_risk > 85.0:
            self._add_warning(ctx, "Liquidity collapse imminent. Slippage risk extreme.", WarningSeverityEnum.CRITICAL)
            flags.append("LIQUIDITY_COLLAPSE_RISK")
            
        if gap_risk > 85.0 or circuit_risk > 85.0:
            self._add_warning(ctx, "Overnight/Circuit danger detected. Emergency exit triggered.", WarningSeverityEnum.CRITICAL)
            flags.append("GAP_CIRCUIT_DANGER")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting technical signals reduce absolute exit timing confidence.")
            flags.append("CONFLICTING_SIGNALS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Exit Engine.", WarningSeverityEnum.CRITICAL)
        self._add_restriction(empty_ctx, "EXIT ALGORITHMS OFFLINE. MANUAL INTERVENTION REQUIRED.")
        
        payload = {
            "exit_type": ExitType.EMERGENCY_EXIT.value,
            "exit_reason": asdict(ExitReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "exit_priority": ExitPriority.CRITICAL.value,
            "exit_urgency": "Maximum",
            "exit_window": ExitWindow.IMMEDIATE.value,
            "market_permission_advisory": MarketPermission.UNKNOWN.value,
            "exit_action": asdict(ExitAction(ExitActionType.FULL_EXIT, 1.0, "System Failsafe Triggered")),
            "exit_trigger": asdict(ExitTrigger("Engine Failure", "Failsafe Triggered", "CRITICAL")),
            "exit_checklist": asdict(ExitChecklist(100.0, 100.0, 100.0, 100.0, 100.0, 0.0, 100.0)),
            "risk_flags": ["SYSTEM_FAILURE_EXIT"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe emergency exit logic.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=100.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_exit(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return ExitEngine().evaluate_exit(engine_outputs)
