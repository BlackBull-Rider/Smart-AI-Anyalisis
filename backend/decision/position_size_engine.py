"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: position_size_engine.py

Institutional Position Size Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly to compute absolute execution 
quantities (units/shares) and fractional scale-in plans based on upstream Capital 
Allocation, Entry Price, Stop Distance, and Risk limits.

Boundary Constraint: This engine DOES NOT calculate Capital Allocation %, Entry levels, 
Stop Loss levels, or Targets. It outputs Executable Quantities and Trade Risk Profiles.
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

class PositionLevel(str, Enum):
    ZERO = "ZERO"
    MICRO = "MICRO"
    SMALL = "SMALL"
    STANDARD = "STANDARD"
    LARGE = "LARGE"
    MAXIMUM = "MAXIMUM"
    UNKNOWN = "UNKNOWN"

class PositionAction(str, Enum):
    SKIP = "SKIP"
    WATCH = "WATCH"
    OPEN_SMALL = "OPEN_SMALL"
    OPEN_STANDARD = "OPEN_STANDARD"
    OPEN_LARGE = "OPEN_LARGE"
    FULL_POSITION = "FULL_POSITION"
    NONE = "NONE"

class PositionPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class PositionPolicy(str, Enum):
    FIXED = "FIXED"
    RISK_BASED = "RISK_BASED"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    ATR_BASED = "ATR_BASED"
    COMPOUNDER = "COMPOUNDER"
    DEFENSIVE = "DEFENSIVE"
    UNKNOWN = "UNKNOWN"

class PositionWindow(str, Enum):
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
class PositionProfile:
    recommended_quantity: float
    maximum_quantity: float
    initial_quantity: float
    capital_utilized_pct: float
    portfolio_risk_pct: float
    position_quality: float

@dataclass(frozen=True)
class ScalePlan:
    first_quantity: float
    second_quantity: float
    third_quantity: float
    conditions: str

@dataclass(frozen=True)
class RiskContribution:
    risk_per_trade_pct: float
    capital_at_risk: float
    stop_distance_pct: float
    expected_loss: float

@dataclass(frozen=True)
class PositionReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class PositionChecklist:
    allocation_ok: bool
    risk_ok: bool
    entry_ok: bool
    stoploss_ok: bool
    confidence_ok: bool
    portfolio_ok: bool
    liquidity_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_position_size_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_PositionSize_V3",
        version="3.0.0",
        stage="Layer-4: Position Size Decision",
        schema_version="3.0",
        api_version="v6",
        decision_method="Risk-Parity Execution Sizing Framework",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "allocation_quality": 0.25,
            "risk_safety": 0.20,
            "entry_quality": 0.20,
            "stop_quality": 0.15,
            "liquidity": 0.20
        },
        thresholds={
            "conflict_penalty": 15.0,
            "max_portfolio_risk_per_trade_pct": 2.0,   # Institutional max risk constraint
            "default_portfolio_size": 100000.0,        # Fallback if unprovided
            "standard_stop_distance_pct": 5.0,
            "liquidity_cap_threshold": 60.0,
            "adaptive_alloc_mult": 0.20,
            "adaptive_risk_mult": 0.20,
            "adaptive_liq_mult": 0.15
        }
    )


# =====================================================================
# POSITION SIZE DECISION ENGINE
# =====================================================================
class PositionSizeEngine(BaseDecisionEngine):
    """
    Institutional Position Size Engine.
    Converts capital allocation percentages and stop-loss geometries into 
    executable quantities (shares/units) using Risk-Parity dynamics.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_position_size_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_position_size(engine_outputs)

    def evaluate_position_size(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main decision execution pipeline for absolute Position Sizing.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_POSITION_SIZE_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER4_DATA")

            # 1. Base Intelligence & Permissions
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()
            
            # 2. Extract Upstream Sizing Variables
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            alloc_pct = self._dynamic_lookup(flat_data, ["initial_allocation_pct", "allocation_pct", "portfolio_exposure_pct"], 0.0)
            overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], 50.0)
            alloc_qual = self._dynamic_lookup(flat_data, ["allocation_quality", "allocation_score"], 50.0)
            entry_qual = self._dynamic_lookup(flat_data, ["entry_quality", "entry_score"], 50.0)
            stop_qual = self._dynamic_lookup(flat_data, ["stoploss_confidence", "protection_quality"], 50.0)
            liquidity = self._dynamic_lookup(flat_data, ["liquidity_score", "float_quality"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
            
            risk_safety = self._normalize(100.0 - overall_risk)

            # Extract Execution Pricing (Required for exact unit calculation)
            entry_price = self._dynamic_lookup(flat_data, ["entry_price", "cmp"], 0.0)
            stop_price = self._dynamic_lookup(flat_data, ["stoploss_distance", "stop_price"], 0.0)
            portfolio_size = self._dynamic_lookup(flat_data, ["portfolio_size", "capital_base"], self.config.thresholds.get("default_portfolio_size", 100000.0))

            # 3. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(alloc_qual, risk_safety, liquidity)

            # 4. Base Position Quality Calculation
            self._add_step(ctx, "CALCULATE_POSITION_QUALITY")
            base_pos_quality = self._normalize(
                (alloc_qual * dynamic_weights["allocation_quality"]) +
                (risk_safety * dynamic_weights["risk_safety"]) +
                (entry_qual * dynamic_weights["entry_quality"]) +
                (stop_qual * dynamic_weights["stop_quality"]) +
                (liquidity * dynamic_weights["liquidity"])
            )

            if not is_permitted or alloc_pct == 0.0:
                base_pos_quality = 0.0

            # 5. Core Institutional Mathematics (Risk-Parity Sizing)
            self._add_step(ctx, "CALCULATE_CORE_MATH")
            pos_profile, risk_contrib, policy = self._calculate_sizing_math(
                ctx, alloc_pct, overall_risk, entry_price, stop_price, portfolio_size, liquidity, base_pos_quality, is_permitted
            )

            # 6. Build Scale-in Architecture
            self._add_step(ctx, "BUILD_SCALE_IN_PLAN")
            scale_plan = self._build_scale_plan(flat_data, pos_profile.recommended_quantity, is_permitted)

            # 7. Checklists and Categorizations
            self._add_step(ctx, "GENERATE_CHECKLIST_AND_CATEGORIZATIONS")
            checklist = PositionChecklist(
                allocation_ok=(alloc_pct > 0.0),
                risk_ok=(overall_risk < 70.0),
                entry_ok=(entry_qual > 50.0),
                stoploss_ok=(stop_qual > 50.0),
                confidence_ok=(layer3_conf > 50.0),
                portfolio_ok=(risk_contrib.risk_per_trade_pct <= self.config.thresholds.get("max_portfolio_risk_per_trade_pct", 2.0)),
                liquidity_ok=(liquidity > 40.0)
            )
            
            level, action, priority, window = self._determine_categorizations(
                base_pos_quality, pos_profile.capital_utilized_pct, is_permitted, policy
            )

            # 8. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, level, overall_risk, risk_contrib.stop_distance_pct, liquidity, layer3_conf, regime
            )

            # Reduce quality post-conflict
            position_quality = self._normalize(base_pos_quality - (ctx.conflict_penalty * 0.4)) if is_permitted else 0.0

            # 9. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            pos_conf = self._calculate_advanced_confidence(
                ctx, layer3_conf, position_quality, parsed_inputs, alloc_qual, risk_safety, entry_qual, liquidity
            )
            
            p_reason = PositionReason(
                primary="Risk-parity bounds strictly limit maximum executable quantity." if risk_contrib.stop_distance_pct > 5.0 else "Capital allocation supports a standard mathematically sized position.",
                secondary=f"Policy: {policy.value} | Ptf Risk: {risk_contrib.risk_per_trade_pct}%",
                confidence=round(pos_conf, 2)
            )

            # 10. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, level, policy, overall_risk, risk_contrib, liquidity, layer3_conf, regime, is_permitted, entry_price
            )

            # 11. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "position_level": level.value,
                "position_action": action.value,
                "position_priority": priority.value,
                "position_policy": policy.value,
                "position_window": window.value,
                "position_quality": round(position_quality, 2),
                "position_confidence": round(pos_conf, 2),
                "position_reason": asdict(p_reason),
                "position_profile": asdict(pos_profile),
                "scale_plan": asdict(scale_plan),
                "risk_contribution": asdict(risk_contrib),
                "position_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL
            status_msg = "Sizing execution calculated." if is_permitted else "Sizing Blocked by Upstream Gatekeepers."

            trace = self._build_trace(engine_outputs, start_time, ctx, parsed_inputs)
            
            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=pos_conf,
                rating_score=position_quality, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"PositionSize Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MATHEMATICS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, alloc_qual: float, risk_safety: float, liq: float) -> dict[str, float]:
        """Mathematically biases weighting toward available liquidity and risk."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["allocation_quality"] += (alloc_qual / 100.0) * t.get("adaptive_alloc_mult", 0.20)
        w["risk_safety"] += (risk_safety / 100.0) * t.get("adaptive_risk_mult", 0.20)
        
        # If liquidity drops, it dominates sizing limits
        if liq < t.get("liquidity_cap_threshold", 60.0):
            liq_penalty = self._normalize(100.0 - liq)
            w["liquidity"] += (liq_penalty / 100.0) * t.get("adaptive_liq_mult", 0.15)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _calculate_sizing_math(self, ctx: DecisionContext, alloc_pct: float, risk: float, entry: float, 
                               stop: float, ptf_size: float, liq: float, qual: float, permitted: bool):
        """
        The Core Institutional Formula:
        Evaluates Fixed Allocation vs Risk-Based Sizing (Risk-Parity).
        """
        t = self.config.thresholds
        max_ptf_risk = t.get("max_portfolio_risk_per_trade_pct", 2.0)
        
        if not permitted or alloc_pct == 0.0:
            return PositionProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), RiskContribution(0.0, 0.0, 0.0, 0.0), PositionPolicy.UNKNOWN

        # Fallback values if absolute execution prices are missing
        stop_dist_pct = t.get("standard_stop_distance_pct", 5.0)
        if entry > 0 and stop > 0 and entry != stop:
            stop_dist_pct = self._normalize(abs(entry - stop) / entry * 100.0)

        allocated_capital = ptf_size * (alloc_pct / 100.0)
        
        # Determine target risk per trade based on overall risk score
        target_risk_pct = self._clamp(max_ptf_risk * ((100.0 - risk) / 100.0), 0.25, max_ptf_risk)
        capital_at_risk = ptf_size * (target_risk_pct / 100.0)
        
        # Quantity calculation based on risk parity: Q = Risk$ / StopDistance$
        # If entry is missing, we compute a normalized 'notional unit' relative to portfolio
        if entry > 0:
            stop_dollar_dist = entry * (stop_dist_pct / 100.0)
            recommended_qty = capital_at_risk / stop_dollar_dist if stop_dollar_dist > 0 else 0.0
            
            # Bound quantity by actual allocated capital limitation
            max_qty_by_capital = allocated_capital / entry
            recommended_qty = min(recommended_qty, max_qty_by_capital)
        else:
            # Normalized theoretical sizing if execution prices are abstracted
            recommended_qty = allocated_capital / 100.0 # Base 100 nominal price

        # Liquidity capping
        if liq < 50.0:
            recommended_qty *= (liq / 100.0) # Reduce qty linearly with poor liquidity
            
        policy = PositionPolicy.RISK_BASED if stop_dist_pct > 2.0 else PositionPolicy.FIXED
        if "COMPOUND" in self._dynamic_lookup_string({}, []): # Stub checking holding
            policy = PositionPolicy.COMPOUNDER

        # Populate output structures
        actual_utilization = (recommended_qty * (entry if entry > 0 else 100.0)) / ptf_size * 100.0
        expected_loss = capital_at_risk

        pos_profile = PositionProfile(
            recommended_quantity=round(recommended_qty, 2),
            maximum_quantity=round(recommended_qty * 1.2, 2), # Minor buffer for discretion
            initial_quantity=round(recommended_qty * 0.5, 2), # Assuming default 2-tranche
            capital_utilized_pct=round(actual_utilization, 2),
            portfolio_risk_pct=round(target_risk_pct, 2),
            position_quality=round(qual, 2)
        )
        
        risk_contrib = RiskContribution(
            risk_per_trade_pct=round(target_risk_pct, 2),
            capital_at_risk=round(expected_loss, 2),
            stop_distance_pct=round(stop_dist_pct, 2),
            expected_loss=round(expected_loss, 2)
        )
        
        return pos_profile, risk_contrib, policy

    def _build_scale_plan(self, flat_data: dict[str, Any], recommended_qty: float, permitted: bool) -> ScalePlan:
        """Determines fractional lot sizing based on upstream scaling strings."""
        if not permitted or recommended_qty == 0.0:
            return ScalePlan(0.0, 0.0, 0.0, "Blocked")
            
        scale_in = self._dynamic_lookup_string(flat_data, ["scale_in_plan", "conditions", "allocation_action"])
        
        if "ALL-IN" in scale_in.upper() or "FULL_ALLOCATION" in scale_in.upper():
            return ScalePlan(round(recommended_qty, 2), 0.0, 0.0, "Full quantity execution on trigger.")
        elif "PULLBACK" in scale_in.upper() or "COMPOUNDER" in scale_in.upper():
            q1 = round(recommended_qty * 0.25, 2)
            q2 = round(recommended_qty * 0.25, 2)
            q3 = round(recommended_qty * 0.50, 2)
            return ScalePlan(q1, q2, q3, "Structured multi-tranche accumulation.")
        else: # Standard Balanced (50/30/20)
            q1 = round(recommended_qty * 0.50, 2)
            q2 = round(recommended_qty * 0.30, 2)
            q3 = round(recommended_qty * 0.20, 2)
            return ScalePlan(q1, q2, q3, "Standard progressive scaling.")

    def _determine_categorizations(self, qual: float, util_pct: float, permitted: bool, policy: PositionPolicy):
        if not permitted or util_pct == 0.0:
            return PositionLevel.ZERO, PositionAction.SKIP, PositionPriority.NONE, PositionWindow.NONE
            
        # Level based on actual capital utilized
        if util_pct > 8.0:
            level = PositionLevel.MAXIMUM
            action = PositionAction.FULL_POSITION
            priority = PositionPriority.CRITICAL
        elif util_pct > 5.0:
            level = PositionLevel.LARGE
            action = PositionAction.OPEN_LARGE
            priority = PositionPriority.HIGH
        elif util_pct > 2.0:
            level = PositionLevel.STANDARD
            action = PositionAction.OPEN_STANDARD
            priority = PositionPriority.MEDIUM
        else:
            level = PositionLevel.SMALL
            action = PositionAction.OPEN_SMALL
            priority = PositionPriority.LOW

        window = PositionWindow.OPEN if policy == PositionPolicy.COMPOUNDER else PositionWindow.SHORT_TERM
            
        return level, action, priority, window

    def _detect_conflicts(self, ctx: DecisionContext, level: PositionLevel, risk: float, 
                          stop_dist: float, liq: float, conf: float, regime: float) -> None:
        """Identifies paradoxes between computed sizes and systemic boundaries."""
        
        if level in [PositionLevel.LARGE, PositionLevel.MAXIMUM] and risk > 75.0:
            self._add_conflict(ctx, "Large absolute quantity contradicts extreme structural risk profile.", penalty=15.0)
            
        if level in [PositionLevel.LARGE, PositionLevel.MAXIMUM] and stop_dist > 15.0:
            self._add_conflict(ctx, "Large quantity computed despite extremely wide stop loss distance (Risk-Parity failure).", penalty=20.0)
            
        if level in [PositionLevel.MAXIMUM] and liq < 40.0:
            self._add_conflict(ctx, "Maximum position size contradicts extremely poor liquidity metrics (Slippage danger).", penalty=20.0)
            
        if level in [PositionLevel.LARGE, PositionLevel.STANDARD] and conf < 40.0:
            self._add_conflict(ctx, "High executable quantity contradicts low upstream confidence.", penalty=15.0)
            
        if level in [PositionLevel.LARGE, PositionLevel.MAXIMUM] and regime < 35.0:
            self._add_conflict(ctx, "Full position sizing contradicts highly hostile bearish market regime.", penalty=15.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, 
                                       pos_qual: float, parsed_count: int, alloc_qual: float, 
                                       risk_safety: float, entry_qual: float, liq: float) -> float:
        """Determines structural mathematical reliability of the generated quantities."""
        engine_agreement = self._normalize(100.0 - abs(alloc_qual - risk_safety))
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        execution_reliability = self._normalize((entry_qual + liq) / 2.0)
        
        return self._normalize(
            (layer3_conf * 0.15) + (pos_qual * 0.20) + 
            (engine_agreement * 0.15) + (evidence_density * 0.10) + 
            (execution_reliability * 0.20) + (signal_stability * 0.20)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, level: PositionLevel, 
                                         policy: PositionPolicy, risk: float, rc: RiskContribution, 
                                         liq: float, conf: float, regime: float, permitted: bool, 
                                         entry_price: float) -> list[str]:
        flags = []
        
        if not permitted:
            self._add_explanation(ctx, "Position sizing bypassed: Upstream Gatekeeper denies market execution.")
            flags.append("GATEKEEPER_CONSTRAINT")
            return flags
            
        if entry_price == 0.0:
            self._add_warning(ctx, "Exact entry/stop prices missing. Sizing calculated nominally against base capital.", WarningSeverityEnum.ADVISORY)
            flags.append("NOMINAL_PRICING_USED")

        # Explanations
        if level in [PositionLevel.STANDARD, PositionLevel.LARGE]:
            self._add_explanation(ctx, "Capital allocation mathematically supports a standard institutional position size.")
            
        if rc.stop_distance_pct > 10.0:
            self._add_explanation(ctx, "Wide stop-loss distance automatically reduces executable quantity via risk-parity mechanics.")
            flags.append("WIDE_STOP_REDUCTION")
            
        if conf > 75.0:
            self._add_explanation(ctx, "Strong systemic conviction permits progressive tranches and scaling.")
        elif conf < 40.0:
            self._add_explanation(ctx, "Low structural confidence dictates a reduction in initial executable quantity.")
            flags.append("CONFIDENCE_REDUCTION")
            
        if liq < 50.0:
            self._add_explanation(ctx, "Weak liquidity metrics actively limit total tradable quantity to prevent slippage.")
            flags.append("LIQUIDITY_QUANTITY_CAP")
            
        if regime < 40.0:
            self._add_explanation(ctx, "Bearish macro regime strictly caps total nominal position exposure.")

        if rc.risk_per_trade_pct <= self.config.thresholds.get("max_portfolio_risk_per_trade_pct", 2.0):
            self._add_explanation(ctx, "Computed risk-per-trade remains safely within strict institutional limits.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting sizing vectors detected. Execution quantities adjusted defensively.")
            flags.append("SIZING_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Position Size Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "position_level": PositionLevel.ZERO.value,
            "position_action": PositionAction.SKIP.value,
            "position_priority": PositionPriority.NONE.value,
            "position_policy": PositionPolicy.UNKNOWN.value,
            "position_window": PositionWindow.NONE.value,
            "position_quality": 0.0,
            "position_confidence": 0.0,
            "position_reason": asdict(PositionReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "position_profile": asdict(PositionProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            "scale_plan": asdict(ScalePlan(0.0, 0.0, 0.0, "Engine Failure")),
            "risk_contribution": asdict(RiskContribution(0.0, 0.0, 0.0, 0.0)),
            "position_checklist": asdict(PositionChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_SIZING"]
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
def evaluate_position_size(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return PositionSizeEngine().evaluate_position_size(engine_outputs)
