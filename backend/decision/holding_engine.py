"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: holding_engine.py

Institutional Holding Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the temporal and 
conditional position management framework.

Key Responsibilities:
- Determines Expected Holding Duration and Time Horizon.
- Establishes Monitoring Frequency and Review Schedules.
- Defines Continuation and Termination Conditions based on structural integrity.

Boundary Constraint: This engine DOES NOT calculate Entry, Exit, Target, Stop Loss, 
Position Size, or Expected Return (CAGR). It evaluates TIME and CONDITION, not PRICE.
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

class HoldingType(str, Enum):
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITIONAL = "POSITIONAL"
    LONG_TERM = "LONG_TERM"
    COMPOUNDER = "COMPOUNDER"
    OPEN = "OPEN"
    NONE = "NONE"

class HoldingAction(str, Enum):
    HOLD = "HOLD"
    REVIEW = "REVIEW"
    REDUCE = "REDUCE"
    ROTATE = "ROTATE"
    EXIT_PREPARE = "EXIT_PREPARE"
    NONE = "NONE"

class HoldingPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class HoldingWindow(str, Enum):
    TODAY = "TODAY"
    THIS_WEEK = "THIS_WEEK"
    THIS_MONTH = "THIS_MONTH"
    OPEN_ENDED = "OPEN_ENDED"
    UNTIL_STRUCTURE_BREAK = "UNTIL_STRUCTURE_BREAK"
    NONE = "NONE"

class ReviewFrequency(str, Enum):
    EACH_CANDLE = "EACH_CANDLE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Portfolio Managers)
# =====================================================================
@dataclass(frozen=True)
class HoldingPeriod:
    minimum_duration: str
    expected_duration: str
    maximum_duration: str
    duration_type: str

@dataclass(frozen=True)
class HoldingCondition:
    condition: str
    importance: str
    confidence: float

@dataclass(frozen=True)
class MonitoringPlan:
    review_frequency: str
    required_confirmations: int
    termination_trigger: str
    adaptive: bool

@dataclass(frozen=True)
class HoldingReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class HoldingChecklist:
    trend_quality: float
    momentum_quality: float
    institutional_support: float
    business_quality: float
    risk: float
    reward: float
    market_regime: str


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_holding_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Holding_V3",
        version="3.0.0",
        stage="Layer-4: Holding Decision",
        schema_version="3.0",
        api_version="v6",
        decision_method="Adaptive Temporal and Conditional Assessment",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "trend": 0.25,
            "institutional": 0.20,
            "fundamental": 0.15,
            "momentum": 0.15,
            "regime": 0.15,
            "risk": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "compounder_fundamental_threshold": 80.0,
            "long_term_trend_threshold": 75.0,
            "swing_momentum_threshold": 70.0,
            "adaptive_trend_mult": 0.15,
            "adaptive_inst_mult": 0.10,
            "adaptive_fund_mult": 0.15,
            "adaptive_regime_mult": 0.10,
            "adaptive_risk_mult": 0.10
        }
    )


# =====================================================================
# HOLDING DECISION ENGINE
# =====================================================================
class HoldingEngine(BaseDecisionEngine):
    """
    Institutional Holding Decision Engine.
    Evaluates the temporal viability of an active position, dictating how long 
    capital should remain deployed and establishing rigorous review schedules.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_holding_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_holding(engine_outputs)

    def evaluate_holding(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main decision execution pipeline for Holding formulation.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_HOLDING_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_DATA")

            # 1. Base Intelligence & Gatekeeper Data
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            market_permission = self._map_market_permission(permission_str)
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            
            entry_type = self._dynamic_lookup_string(flat_data, ["entry_type", "entry_strength"])
            entry_valid = entry_type.upper() not in ["NO ENTRY", "NONE", "AVOID", "UNKNOWN"]
            
            # Structural, Momentum & Fundamental Metrics
            trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
            mom = self._dynamic_lookup(flat_data, ["momentum_score", "momentum_strength"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            fund = self._dynamic_lookup(flat_data, ["fundamental_score", "compounder_score", "business_quality"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
            risk = self._dynamic_lookup(flat_data, ["risk_score", "volatility_risk", "danger"], 20.0)
            reward = self._dynamic_lookup(flat_data, ["reward_score", "upside"], 50.0)

            # 2. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_CONTINUOUS_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(trend, inst, fund, regime, risk)

            # 3. Holding Quality Base Calculation
            # Risk is inverted for quality calculation (High Risk = Low Quality contribution)
            risk_alignment = self._normalize(100.0 - risk)
            
            base_holding_quality = self._normalize(
                (trend * dynamic_weights["trend"]) +
                (mom * dynamic_weights["momentum"]) +
                (inst * dynamic_weights["institutional"]) +
                (fund * dynamic_weights["fundamental"]) +
                (regime * dynamic_weights["regime"]) +
                (risk_alignment * dynamic_weights["risk"])
            )

            # 4. Generate Holding Checklist
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = HoldingChecklist(
                trend_quality=round(trend, 2),
                momentum_quality=round(mom, 2),
                institutional_support=round(inst, 2),
                business_quality=round(fund, 2),
                risk=round(risk, 2),
                reward=round(reward, 2),
                market_regime=market_permission.value
            )

            # 5. Determine Temporal Structures (Type, Action, Period, Monitoring)
            self._add_step(ctx, "DETERMINE_HOLDING_STRUCTURES")
            h_type, h_action, h_window, h_period, h_freq, h_priority, reason = self._determine_holding_structures(
                entry_valid, market_permission, trend, mom, inst, fund, regime, risk
            )

            # 6. Formulate Continuation & Termination Conditions
            self._add_step(ctx, "FORMULATE_CONDITIONS_AND_MONITORING")
            cont_cond, term_cond, monitoring_plan = self._build_monitoring_framework(
                h_type, h_freq, trend, fund, inst, risk
            )

            # 7. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, trend, fund, risk, regime, inst, h_type, h_period
            )

            # 8. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            holding_conf = self._calculate_advanced_confidence(
                ctx, layer3_conf, base_holding_quality, parsed_inputs, trend, mom, fund, risk
            )
            
            h_reason = HoldingReason(reason.primary, reason.secondary, round(holding_conf, 2))

            # 9. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, h_type, h_action, h_freq, entry_valid, trend, mom, inst, fund, risk, regime
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "holding_status": "ACTIVE" if entry_valid and h_action == HoldingAction.HOLD else "IDLE/REVIEW",
                "holding_action": h_action.value,
                "holding_type": h_type.value,
                "holding_priority": h_priority.value,
                "holding_window": h_window.value,
                "holding_quality": round(base_holding_quality, 2),
                "holding_confidence": round(holding_conf, 2),
                "holding_reason": asdict(h_reason),
                "holding_period": asdict(h_period),
                "review_frequency": h_freq.value,
                "monitoring_plan": asdict(monitoring_plan),
                "continuation_conditions": [asdict(c) for c in cont_cond],
                "termination_conditions": [asdict(c) for c in term_cond],
                "holding_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if entry_valid else DecisionStatusEnum.NO_DATA
            status_msg = "Holding Strategy generated." if entry_valid else "No active entry. Holding Engine idling."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=holding_conf,
                rating_score=base_holding_quality, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Holding Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
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

    def _calculate_continuous_weights(self, trend: float, inst: float, fund: float, regime: float, risk: float) -> dict[str, float]:
        """Continuous mathematical scaling of weights for temporal durability."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        # Scaling based on long-term conviction drivers
        w["trend"] += (trend / 100.0) * t.get("adaptive_trend_mult", 0.15)
        w["institutional"] += (inst / 100.0) * t.get("adaptive_inst_mult", 0.10)
        w["fundamental"] += (fund / 100.0) * t.get("adaptive_fund_mult", 0.15)
        w["regime"] += (regime / 100.0) * t.get("adaptive_regime_mult", 0.10)
        
        # If risk is extremely low, it increases in weight to anchor stability
        risk_safety = self._normalize(100.0 - risk)
        w["risk"] += (risk_safety / 100.0) * t.get("adaptive_risk_mult", 0.10)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_holding_structures(self, entry_valid: bool, perm: MarketPermission, trend: float, 
                                      mom: float, inst: float, fund: float, regime: float, risk: float):
        t = self.config.thresholds
        
        if not entry_valid or perm in [MarketPermission.BLOCKED, MarketPermission.RESTRICTED]:
            return (
                HoldingType.NONE, HoldingAction.NONE, HoldingWindow.NONE, 
                HoldingPeriod("N/A", "N/A", "N/A", "NONE"),
                ReviewFrequency.NONE, HoldingPriority.NONE,
                HoldingReason("No active trade permitted.", "Idle state.", 0.0)
            )

        # 1. Compounder / Decadal Holding
        if fund > t.get("compounder_fundamental_threshold", 80.0) and inst > 70.0 and regime > 60.0:
            return (
                HoldingType.COMPOUNDER, HoldingAction.HOLD, HoldingWindow.OPEN_ENDED,
                HoldingPeriod("1 Year", "5-10 Years", "Decades", "YEARS_TO_DECADES"),
                ReviewFrequency.MONTHLY, HoldingPriority.HIGH,
                HoldingReason("Exceptional fundamental quality.", "Multi-year compounding capability.", 0.0)
            )

        # 2. Long Term Trend Holding
        if trend > t.get("long_term_trend_threshold", 75.0) and regime > 65.0:
            return (
                HoldingType.LONG_TERM, HoldingAction.HOLD, HoldingWindow.UNTIL_STRUCTURE_BREAK,
                HoldingPeriod("3 Months", "1-3 Years", "5 Years", "MONTHS_TO_YEARS"),
                ReviewFrequency.WEEKLY, HoldingPriority.HIGH,
                HoldingReason("Macro trend alignment.", "Riding secular market cycle.", 0.0)
            )

        # 3. Positional Holding
        if trend > 60.0 and inst > 60.0:
            return (
                HoldingType.POSITIONAL, HoldingAction.HOLD, HoldingWindow.THIS_MONTH,
                HoldingPeriod("1 Week", "1-3 Months", "6 Months", "WEEKS_TO_MONTHS"),
                ReviewFrequency.WEEKLY, HoldingPriority.MEDIUM,
                HoldingReason("Mid-term structural alignment.", "Capturing primary market leg.", 0.0)
            )

        # 4. Swing Holding
        if mom > t.get("swing_momentum_threshold", 70.0):
            action = HoldingAction.REVIEW if risk > 60.0 else HoldingAction.HOLD
            freq = ReviewFrequency.DAILY if mom < 80.0 else ReviewFrequency.EACH_CANDLE
            return (
                HoldingType.SWING, action, HoldingWindow.THIS_WEEK,
                HoldingPeriod("1 Day", "3-10 Days", "3 Weeks", "DAYS_TO_WEEKS"),
                freq, HoldingPriority.MEDIUM,
                HoldingReason("Strong tactical momentum.", "Capturing rapid price discovery.", 0.0)
            )

        # 5. Intraday / Defensive Holding
        return (
            HoldingType.INTRADAY, HoldingAction.REVIEW, HoldingWindow.TODAY,
            HoldingPeriod("1 Hour", "End of Day", "Next Open", "HOURS_TO_DAYS"),
            ReviewFrequency.EACH_CANDLE, HoldingPriority.WATCH,
            HoldingReason("Weak temporal conviction.", "Defensive/Tactical exposure only.", 0.0)
        )

    def _build_monitoring_framework(self, h_type: HoldingType, h_freq: ReviewFrequency, 
                                    trend: float, fund: float, inst: float, risk: float):
        """Constructs strictly condition-based continuation and termination parameters."""
        
        cont_cond = []
        term_cond = []

        if h_type in [HoldingType.COMPOUNDER, HoldingType.LONG_TERM]:
            cont_cond.append(HoldingCondition("Quarterly earnings support fundamental thesis", "HIGH", round(fund, 2)))
            cont_cond.append(HoldingCondition("Institutional ownership remains stable or grows", "HIGH", round(inst, 2)))
            term_cond.append(HoldingCondition("Macro regime definitively flips bearish", "CRITICAL", 95.0))
            term_cond.append(HoldingCondition("Structural breakdown of secular trend", "CRITICAL", round(trend, 2)))
            term_cond.append(HoldingCondition("Severe corporate governance violation", "CRITICAL", 99.0))
        elif h_type in [HoldingType.SWING, HoldingType.POSITIONAL]:
            cont_cond.append(HoldingCondition("Trend alignment remains intact", "HIGH", round(trend, 2)))
            cont_cond.append(HoldingCondition("Price maintains position above dynamic moving averages", "MEDIUM", 80.0))
            term_cond.append(HoldingCondition("Momentum divergence confirmed on primary timeframe", "HIGH", 85.0))
            term_cond.append(HoldingCondition("Volume distribution prints consecutive negative nodes", "HIGH", 80.0))
        else:
            cont_cond.append(HoldingCondition("Intraday VWAP/Momentum sustained", "HIGH", 85.0))
            term_cond.append(HoldingCondition("Risk/ATR spike invalidates setup", "CRITICAL", round(risk, 2)))
            term_cond.append(HoldingCondition("Session close approaches", "HIGH", 90.0))

        plan = MonitoringPlan(
            review_frequency=h_freq.value,
            required_confirmations=2 if h_type in [HoldingType.LONG_TERM, HoldingType.COMPOUNDER] else 1,
            termination_trigger="Any Critical Termination Condition Met",
            adaptive=(risk > 60.0) # Adaptive monitoring if risk is high
        )

        return cont_cond, term_cond, plan

    def _detect_conflicts(self, ctx: DecisionContext, trend: float, fund: float, risk: float, 
                          regime: float, inst: float, h_type: HoldingType, h_period: HoldingPeriod) -> None:
        """Identifies logical paradoxes that contradict the expected time horizon."""
        
        # 1. Weak Trend + Long Hold
        if trend < 40.0 and h_type in [HoldingType.POSITIONAL, HoldingType.LONG_TERM]:
            self._add_conflict(ctx, "Weak directional trend contradicts a positional/long-term holding thesis.", penalty=15.0)
            
        # 2. Strong Trend + Very Short Hold
        if trend > 85.0 and h_type == HoldingType.INTRADAY:
            self._add_conflict(ctx, "Exceptionally strong macro trend contradicts micro/intraday holding limitation.", penalty=10.0)
            
        # 3. High Risk + Compound Holding
        if risk > 75.0 and h_type == HoldingType.COMPOUNDER:
            self._add_conflict(ctx, "High structural volatility/risk contradicts the stability required for multi-year compounding.", penalty=20.0)
            
        # 4. Bear Market + Open Ended Hold
        if regime < 35.0 and "OPEN" in h_period.duration_type:
            self._add_conflict(ctx, "Bearish macro regime contradicts open-ended, untethered holding duration.", penalty=15.0)
            
        # 5. Institutional Selling + Long Hold
        if inst < 30.0 and h_type in [HoldingType.LONG_TERM, HoldingType.COMPOUNDER]:
            self._add_conflict(ctx, "Lack of institutional ownership contradicts extreme long-term holding viability.", penalty=15.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, base_qual: float, 
                                       parsed_count: int, trend: float, mom: float, fund: float, risk: float) -> float:
        """Determines how structurally reliable the time-horizon roadmap is."""
        
        # Temporal stability metrics
        trend_stability = self._normalize(trend)
        momentum_stability = self._normalize(mom)
        risk_alignment = self._normalize(100.0 - risk)
        
        engine_agreement = self._normalize(100.0 - abs(trend - fund))
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        
        return self._normalize(
            (layer3_conf * 0.15) + (base_qual * 0.20) + 
            (trend_stability * 0.15) + (momentum_stability * 0.10) +
            (engine_agreement * 0.15) + (evidence_density * 0.10) + 
            (risk_alignment * 0.10) + (signal_stability * 0.05)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, h_type: HoldingType, 
                                         h_action: HoldingAction, h_freq: ReviewFrequency, 
                                         entry_valid: bool, trend: float, mom: float, 
                                         inst: float, fund: float, risk: float, regime: float) -> list[str]:
        flags = []
        
        if not entry_valid:
            self._add_explanation(ctx, "No active entry evaluated. Holding engine idling.")
            return flags

        # Explanations
        if trend > 75.0:
            self._add_explanation(ctx, "Trend remains structurally intact, supporting extended temporal holding.")
        elif trend < 40.0:
            self._add_explanation(ctx, "Trend deterioration severely limits holding duration.")
            flags.append("RESTRICTED_DURATION_TREND")
            
        if fund > 80.0 and h_type == HoldingType.COMPOUNDER:
            self._add_explanation(ctx, "Compound fundamentals justify extreme long-term holding strategies.")
            
        if inst > 75.0:
            self._add_explanation(ctx, "Institutional accumulation supports extended position maintenance.")
            
        if mom < 40.0 and h_type in [HoldingType.SWING, HoldingType.INTRADAY]:
            self._add_explanation(ctx, "Weak momentum requires high-frequency review and active monitoring.")
            flags.append("ACTIVE_MONITORING_REQUIRED")
            
        if risk > 75.0:
            self._add_explanation(ctx, "Elevated risk profile mathematically limits safe holding duration.")
            flags.append("DURATION_CAPPED_BY_RISK")
            
        if regime < 40.0:
            self._add_explanation(ctx, "Bearish/Hostile regime demands active monitoring and defensive holding constraints.")
            flags.append("DEFENSIVE_HOLD_REGIME")

        if h_freq in [ReviewFrequency.EACH_CANDLE, ReviewFrequency.DAILY]:
            self._add_explanation(ctx, f"High urgency {h_freq.value} monitoring schedule activated.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting structural signals reduce temporal holding confidence.")
            flags.append("CONFLICTING_TEMPORAL_SIGNALS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Holding Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "holding_status": "IDLE/ERROR",
            "holding_action": HoldingAction.NONE.value,
            "holding_type": HoldingType.NONE.value,
            "holding_priority": HoldingPriority.NONE.value,
            "holding_window": HoldingWindow.NONE.value,
            "holding_quality": 0.0,
            "holding_confidence": 0.0,
            "holding_reason": asdict(HoldingReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "holding_period": asdict(HoldingPeriod("N/A", "N/A", "N/A", "ERROR")),
            "review_frequency": ReviewFrequency.NONE.value,
            "monitoring_plan": asdict(MonitoringPlan("NONE", 0, "ERROR", False)),
            "continuation_conditions": [],
            "termination_conditions": [],
            "holding_checklist": asdict(HoldingChecklist(0.0, 0.0, 0.0, 0.0, 100.0, 0.0, "UNKNOWN")),
            "risk_flags": ["SYSTEM_FAILURE_HOLDING"]
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
def evaluate_holding(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return HoldingEngine().evaluate_holding(engine_outputs)
