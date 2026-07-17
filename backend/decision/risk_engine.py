"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: risk_engine.py

Institutional Risk Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the risk assessment 
and policy definition boundary.

Key Responsibilities:
- Determines Overall Risk, Risk Profile, and Risk Level.
- Establishes Monitoring Policies based on market and asset danger.
- Defines Tail-risk, Systemic Risk, and Asset-specific Risk.

Boundary Constraint: This engine DOES NOT calculate Entry, Exit, Target, Stop Loss, 
Position Size, Allocation, Expected Return, or CAGR. It evaluates RISK EXPOSURE only.
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
class RiskLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"

class RiskAction(str, Enum):
    ACCEPT = "ACCEPT"
    MONITOR = "MONITOR"
    REDUCE = "REDUCE"
    HEDGE = "HEDGE"
    AVOID = "AVOID"
    NONE = "NONE"

class RiskPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class RiskWindow(str, Enum):
    INTRADAY = "INTRADAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    OPEN = "OPEN"
    NONE = "NONE"

class MonitoringLevel(str, Enum):
    NORMAL = "NORMAL"
    ENHANCED = "ENHANCED"
    CONTINUOUS = "CONTINUOUS"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class RiskProfile:
    overall_risk: float
    drawdown_risk: float
    gap_risk: float
    volatility_risk: float
    liquidity_risk: float
    institutional_risk: float
    business_risk: float
    trend_risk: float
    systemic_risk: float

@dataclass(frozen=True)
class RiskComponent:
    name: str
    score: float
    weight: float
    confidence: float

@dataclass(frozen=True)
class RiskReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class RiskChecklist:
    trend_risk_acceptable: bool
    volatility_risk_acceptable: bool
    liquidity_risk_acceptable: bool
    institutional_risk_acceptable: bool
    business_risk_acceptable: bool
    market_risk_acceptable: bool
    holding_risk_acceptable: bool

@dataclass(frozen=True)
class MonitoringPlan:
    frequency: str
    critical_events: list[str]
    adaptive: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_risk_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Risk_V3",
        version="3.0.0",
        stage="Layer-4: Risk Assessment",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Adaptive Multi-Dimensional Risk Evaluation",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "trend_risk": 0.15,
            "volatility_risk": 0.20,
            "liquidity_risk": 0.15,
            "institutional_risk": 0.15,
            "business_risk": 0.10,
            "systemic_risk": 0.15,
            "gap_risk": 0.05,
            "drawdown_risk": 0.05
        },
        thresholds={
            "conflict_penalty": 15.0,
            "extreme_risk_threshold": 85.0,
            "very_high_risk_threshold": 75.0,
            "high_risk_threshold": 60.0,
            "moderate_risk_threshold": 40.0,
            "low_risk_threshold": 20.0,
            "adaptive_vol_mult": 0.20,
            "adaptive_liq_mult": 0.15,
            "adaptive_sys_mult": 0.20,
            "adaptive_tail_mult": 0.25
        }
    )


# =====================================================================
# RISK DECISION ENGINE
# =====================================================================
class RiskEngine(BaseDecisionEngine):
    """
    Institutional Risk Decision Engine.
    Quantifies and categorizes the holistic risk profile of a trade or position.
    It establishes monitoring levels and outputs advisory actions (Accept, Hedge, Avoid).
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_risk_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_risk(engine_outputs)

    def evaluate_risk(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main decision execution pipeline for Risk Assessment.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_RISK_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_AND_LAYER4_DATA")

            # 1. Base Intelligence Extraction
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            
            # 2. Convert Upstream Health Scores to Risk Scores (Inversion)
            # High Health = Low Risk -> Risk = 100 - Health
            trend_health = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
            trend_risk = self._normalize(100.0 - trend_health)
            
            inst_health = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            inst_risk = self._normalize(100.0 - inst_health)
            
            fund_health = self._dynamic_lookup(flat_data, ["fundamental_score", "business_quality", "compounder_score"], 50.0)
            business_risk = self._normalize(100.0 - fund_health)
            
            regime_health = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
            systemic_risk = self._normalize(100.0 - regime_health)
            
            liq_health = self._dynamic_lookup(flat_data, ["liquidity_score", "volume_score"], 50.0)
            liquidity_risk = self._normalize(100.0 - liq_health)

            # Direct Risk Metrics (High = High Risk)
            volatility_risk = self._dynamic_lookup(flat_data, ["volatility_risk", "volatility_score", "atr_expansion", "danger"], 30.0)
            gap_risk = self._dynamic_lookup(flat_data, ["gap_risk", "overnight_risk"], 20.0)
            drawdown_risk = self._dynamic_lookup(flat_data, ["drawdown_risk", "max_drawdown"], 20.0)

            # Holding duration context
            holding_type = self._dynamic_lookup_string(flat_data, ["holding_type", "holding_window"]).upper()

            # 3. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(
                volatility_risk, liquidity_risk, systemic_risk, gap_risk, drawdown_risk
            )

            # 4. Composite Risk Calculation
            self._add_step(ctx, "COMPUTE_OVERALL_RISK")
            overall_risk = self._normalize(
                (trend_risk * dynamic_weights["trend_risk"]) +
                (volatility_risk * dynamic_weights["volatility_risk"]) +
                (liquidity_risk * dynamic_weights["liquidity_risk"]) +
                (inst_risk * dynamic_weights["institutional_risk"]) +
                (business_risk * dynamic_weights["business_risk"]) +
                (systemic_risk * dynamic_weights["systemic_risk"]) +
                (gap_risk * dynamic_weights["gap_risk"]) +
                (drawdown_risk * dynamic_weights["drawdown_risk"])
            )
            
            # Risk Quality (Inverted Overall Risk, used for engine's rating_score)
            risk_quality = self._normalize(100.0 - overall_risk)

            # 5. Build Structured Profiles
            self._add_step(ctx, "BUILD_RISK_PROFILES")
            risk_profile = RiskProfile(
                overall_risk=round(overall_risk, 2),
                drawdown_risk=round(drawdown_risk, 2),
                gap_risk=round(gap_risk, 2),
                volatility_risk=round(volatility_risk, 2),
                liquidity_risk=round(liquidity_risk, 2),
                institutional_risk=round(inst_risk, 2),
                business_risk=round(business_risk, 2),
                trend_risk=round(trend_risk, 2),
                systemic_risk=round(systemic_risk, 2)
            )

            # 6. Determine Categorizations and Plans
            self._add_step(ctx, "DETERMINE_RISK_LEVEL_AND_PLAN")
            risk_level, risk_action, risk_priority = self._determine_risk_level(overall_risk, systemic_risk, liquidity_risk)
            monitoring_level, monitoring_plan = self._build_monitoring_plan(risk_level, volatility_risk, gap_risk, holding_type)
            risk_window = self._determine_risk_window(holding_type)

            # 7. Generate Checklist
            checklist = RiskChecklist(
                trend_risk_acceptable=(trend_risk < 60.0),
                volatility_risk_acceptable=(volatility_risk < 75.0),
                liquidity_risk_acceptable=(liquidity_risk < 60.0),
                institutional_risk_acceptable=(inst_risk < 70.0),
                business_risk_acceptable=(business_risk < 75.0),
                market_risk_acceptable=(systemic_risk < 65.0),
                holding_risk_acceptable=(overall_risk < 80.0)
            )

            # 8. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            reward_health = self._dynamic_lookup(flat_data, ["reward_score", "upside", "target_potential"], 50.0)
            self._detect_conflicts(
                ctx, trend_health, overall_risk, reward_health, regime_health, 
                systemic_risk, fund_health, business_risk, inst_health, inst_risk
            )

            # Apply slight smoothing if conflicts are high
            adjusted_risk = self._clamp(overall_risk + (ctx.conflict_penalty * 0.3))
            risk_quality = self._normalize(100.0 - adjusted_risk)

            # 9. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            risk_conf = self._calculate_advanced_confidence(
                ctx, layer3_conf, parsed_inputs, trend_health, volatility_risk, inst_health, regime_health
            )
            
            risk_reason = RiskReason(
                "Aggregated multi-dimensional risk analysis complete.",
                f"Driven by volatility ({round(volatility_risk,1)}%) and systemic factors ({round(systemic_risk,1)}%).",
                round(risk_conf, 2)
            )

            # 10. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, risk_level, risk_action, volatility_risk, inst_risk, systemic_risk, liquidity_risk, gap_risk, business_risk
            )

            # 11. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "overall_risk": round(adjusted_risk, 2),
                "risk_quality": round(risk_quality, 2),
                "risk_category": risk_level.value,
                "risk_profile": asdict(risk_profile),
                "risk_priority": risk_priority.value,
                "risk_window": risk_window.value,
                "risk_action": risk_action.value,
                "monitoring_level": monitoring_level.value,
                "risk_confidence": round(risk_conf, 2),
                "risk_reason": asdict(risk_reason),
                "risk_checklist": asdict(checklist),
                "monitoring_plan": asdict(monitoring_plan),
                "risk_flags": risk_flags
            }

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=DecisionStatusEnum.SUCCESS,
                status_msg="Risk Assessment generated successfully.",
                decision_payload=decision_payload,
                confidence=risk_conf,
                rating_score=risk_quality, # Engine rating based on safety, not risk magnitude
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Risk Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, vol: float, liq: float, sys: float, gap: float, dd: float) -> dict[str, float]:
        """Continuous scaling shifting weight toward the most critical danger."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["volatility_risk"] += (vol / 100.0) * t.get("adaptive_vol_mult", 0.20)
        w["liquidity_risk"] += (liq / 100.0) * t.get("adaptive_liq_mult", 0.15)
        w["systemic_risk"] += (sys / 100.0) * t.get("adaptive_sys_mult", 0.20)
        
        # Tail risks (Gap & Drawdown)
        tail_risk_avg = (gap + dd) / 2.0
        w["gap_risk"] += (tail_risk_avg / 100.0) * (t.get("adaptive_tail_mult", 0.25) / 2)
        w["drawdown_risk"] += (tail_risk_avg / 100.0) * (t.get("adaptive_tail_mult", 0.25) / 2)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_risk_level(self, risk: float, sys: float, liq: float):
        t = self.config.thresholds
        
        if risk >= t.get("extreme_risk_threshold", 85.0) or sys > 90.0 or liq > 90.0:
            return RiskLevel.EXTREME, RiskAction.AVOID, RiskPriority.CRITICAL
        if risk >= t.get("very_high_risk_threshold", 75.0):
            return RiskLevel.VERY_HIGH, RiskAction.HEDGE, RiskPriority.HIGH
        if risk >= t.get("high_risk_threshold", 60.0):
            return RiskLevel.HIGH, RiskAction.REDUCE, RiskPriority.MEDIUM
        if risk >= t.get("moderate_risk_threshold", 40.0):
            return RiskLevel.MODERATE, RiskAction.MONITOR, RiskPriority.LOW
        if risk >= t.get("low_risk_threshold", 20.0):
            return RiskLevel.LOW, RiskAction.ACCEPT, RiskPriority.WATCH
            
        return RiskLevel.VERY_LOW, RiskAction.ACCEPT, RiskPriority.NONE

    def _determine_risk_window(self, holding_type: str) -> RiskWindow:
        h = holding_type.upper()
        if "INTRADAY" in h: return RiskWindow.INTRADAY
        if "SWING" in h: return RiskWindow.SHORT_TERM
        if "POSITIONAL" in h: return RiskWindow.MEDIUM_TERM
        if "LONG" in h or "COMPOUND" in h: return RiskWindow.LONG_TERM
        if "OPEN" in h: return RiskWindow.OPEN
        return RiskWindow.NONE

    def _build_monitoring_plan(self, level: RiskLevel, vol: float, gap: float, holding: str):
        if level in [RiskLevel.EXTREME, RiskLevel.VERY_HIGH]:
            m_level = MonitoringLevel.CONTINUOUS
            freq = "Tick/Minute"
            adaptive = True
            events = ["Liquidity Drop", "Volatility Expansion", "Regime Shift"]
        elif level == RiskLevel.HIGH or gap > 60.0:
            m_level = MonitoringLevel.ENHANCED
            freq = "Hourly"
            adaptive = True
            events = ["Overnight Gap", "Macro News Release", "Support Breach"]
        elif "EVENT" in holding:
            m_level = MonitoringLevel.EVENT_DRIVEN
            freq = "As Scheduled"
            adaptive = False
            events = ["Earnings", "Fed Rate", "Data Print"]
        else:
            m_level = MonitoringLevel.NORMAL
            freq = "Daily Close"
            adaptive = False
            events = ["Weekly Close", "Structural Breakdown"]
            
        return m_level, MonitoringPlan(freq, events, adaptive)

    def _detect_conflicts(self, ctx: DecisionContext, trend: float, risk: float, reward: float, 
                          regime: float, sys_risk: float, fund: float, biz_risk: float, 
                          inst: float, inst_risk: float) -> None:
        """Identifies paradoxes between asset performance and underlying risk structure."""
        
        # 1. Strong Trend + Extreme Risk
        if trend > 80.0 and risk > 80.0:
            self._add_conflict(ctx, "Strong directional trend contradicts extreme aggregate risk profile (Blow-off risk).", penalty=15.0)
            
        # 2. High Reward + Extreme Risk
        if reward > 80.0 and risk > 85.0:
            self._add_conflict(ctx, "High reward potential paired with extreme risk denotes an asymmetric volatility trap.", penalty=10.0)
            
        # 3. Bull Regime + High Systemic Risk
        if regime > 75.0 and sys_risk > 75.0:
            self._add_conflict(ctx, "Bullish market regime paradoxically accompanied by elevated systemic risk metrics.", penalty=15.0)
            
        # 4. Strong Fundamentals + Extreme Business Risk
        if fund > 80.0 and biz_risk > 75.0:
            self._add_conflict(ctx, "Reported strong fundamentals conflict with high underlying business risk evaluations.", penalty=20.0)

        # 5. Institutional Buying + Very High Institutional Risk
        if inst > 80.0 and inst_risk > 75.0:
            self._add_conflict(ctx, "Institutional buying metrics conflict with high institutional outflow/risk metrics.", penalty=15.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, 
                                       parsed_count: int, trend: float, vol: float, inst: float, regime: float) -> float:
        """Determines the mathematical reliability of the assessed risk."""
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        
        # Stability of core pillars
        trend_stability = self._normalize(trend)
        vol_stability = self._normalize(100.0 - vol)
        market_stability = self._normalize(regime)
        inst_reliability = self._normalize(inst)
        
        engine_agreement = self._normalize((trend_stability + vol_stability + market_stability + inst_reliability) / 4.0)
        
        return self._normalize(
            (layer3_conf * 0.15) + 
            (evidence_density * 0.15) + 
            (engine_agreement * 0.30) + 
            (signal_stability * 0.40)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, level: RiskLevel, action: RiskAction, 
                                         vol: float, inst_risk: float, sys_risk: float, liq: float, 
                                         gap: float, biz: float) -> list[str]:
        flags = []
        
        if action == RiskAction.AVOID:
            self._add_explanation(ctx, "Asset risk profile is mathematically hostile. Absolute avoidance recommended.")
            flags.append("AVOID_ASSET")
        elif action == RiskAction.HEDGE:
            self._add_explanation(ctx, "Elevated risk profile warrants direct delta or volatility hedging.")
            flags.append("HEDGING_REQUIRED")
        elif action == RiskAction.REDUCE:
            self._add_explanation(ctx, "Risk density advises reduction of standard position sizing vectors.")

        if vol > 70.0:
            self._add_explanation(ctx, "Elevated volatility increases absolute portfolio tail-risk.")
            flags.append("ELEVATED_VOLATILITY")
            
        if inst_risk < 40.0:
            self._add_explanation(ctx, "Strong institutional participation structurally lowers ownership risk.")
        elif inst_risk > 75.0:
            self._add_explanation(ctx, "High institutional distribution risk detected.")
            flags.append("INSTITUTIONAL_DISTRIBUTION_RISK")
            
        if sys_risk < 40.0:
            self._add_explanation(ctx, "Market regime remains stable and structurally supportive.")
        elif sys_risk > 75.0:
            self._add_explanation(ctx, "Systemic macro regime risk is heavily elevated.")
            flags.append("SYSTEMIC_MACRO_RISK")
            
        if liq > 70.0:
            self._add_explanation(ctx, "Liquidity degradation requires enhanced execution monitoring.")
            self._add_warning(ctx, "Slippage and liquidity sweep risk elevated.", WarningSeverityEnum.RISK)
            flags.append("LIQUIDITY_DANGER")
            
        if gap > 70.0:
            self._add_explanation(ctx, "Overnight gap risk remains elevated; adjust closing positions accordingly.")
            flags.append("GAP_RISK_ELEVATED")
            
        if biz < 40.0:
            self._add_explanation(ctx, "Robust business fundamentals mathematically reduce long-term downside risk.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Internal paradoxes between performance and risk lower evaluation confidence.")
            flags.append("RISK_ASSESSMENT_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Risk Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "overall_risk": 100.0, # Assume maximum risk on failure
            "risk_quality": 0.0,
            "risk_category": RiskLevel.EXTREME.value,
            "risk_profile": asdict(RiskProfile(100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0)),
            "risk_priority": RiskPriority.CRITICAL.value,
            "risk_window": RiskWindow.NONE.value,
            "risk_action": RiskAction.AVOID.value,
            "monitoring_level": MonitoringLevel.CONTINUOUS.value,
            "risk_confidence": 0.0,
            "risk_reason": asdict(RiskReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "risk_checklist": asdict(RiskChecklist(False, False, False, False, False, False, False)),
            "monitoring_plan": asdict(MonitoringPlan("TICK", ["ENGINE_FAILURE"], True)),
            "risk_flags": ["SYSTEM_FAILURE_RISK_MAXIMUM"]
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
def evaluate_risk(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return RiskEngine().evaluate_risk(engine_outputs)
