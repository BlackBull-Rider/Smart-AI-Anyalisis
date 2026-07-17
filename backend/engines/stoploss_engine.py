"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: stoploss_engine.py

Institutional Stop Loss Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the capital protection 
and risk mitigation boundary. 

Key Responsibilities:
- Stop Loss Strategy (Type, Action, Zone, Trailing Policy).
- Protection placement logic without calculating raw price points or OHLCV indicators.
- Dynamic capital protection scaling based on volatility and institutional footprint.

Boundary Constraint: This engine DOES NOT calculate Entry, Exit, Target, Position Size, 
or Technical Indicators.
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
    EvidenceGraph,
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

class StopLossType(str, Enum):
    INITIAL = "INITIAL"
    ATR = "ATR"
    STRUCTURAL = "STRUCTURAL"
    SWING_LOW = "SWING_LOW"
    ORDER_BLOCK = "ORDER_BLOCK"
    LIQUIDITY = "LIQUIDITY"
    BREAKEVEN = "BREAKEVEN"
    TRAILING = "TRAILING"
    TIME_BASED = "TIME_BASED"
    VOLATILITY = "VOLATILITY"
    NO_STOPLOSS = "NO_STOPLOSS"

class StopLossAction(str, Enum):
    SET = "SET"
    MOVE = "MOVE"
    TRAIL = "TRAIL"
    TIGHTEN = "TIGHTEN"
    LOOSEN = "LOOSEN"
    BREAKEVEN = "BREAKEVEN"
    HOLD = "HOLD"
    NONE = "NONE"

class StopLossPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class StopLossWindow(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    AFTER_ENTRY = "AFTER_ENTRY"
    AFTER_CONFIRMATION = "AFTER_CONFIRMATION"
    AFTER_BREAKOUT = "AFTER_BREAKOUT"
    MONITOR = "MONITOR"
    NONE = "NONE"

class CapitalProtectionLevel(str, Enum):
    MAXIMUM = "MAXIMUM"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class StopLossZone:
    reference_type: str
    reference_name: str
    buffer_percent: float
    execution_style: str

@dataclass(frozen=True)
class StopLossTrigger:
    trigger_condition: str
    confirmation_metric: str
    urgency: str

@dataclass(frozen=True)
class TrailingPolicy:
    policy: str
    step_percent: float
    activation_level: str
    dynamic: bool

@dataclass(frozen=True)
class StopLossReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class ProtectionProfile:
    volatility_protection: float
    gap_protection: float
    liquidity_protection: float
    drawdown_protection: float
    systemic_protection: float

@dataclass(frozen=True)
class EntryProtectionChecklist:
    trend_alignment: float
    market_permission: str
    risk_level: float
    entry_valid: bool
    volatility: float
    institutional_support: float


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_stoploss_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_StopLoss_V3",
        version="3.0.0",
        stage="Layer-4: StopLoss Decision",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Adaptive Volatility & Structure Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "volatility": 0.25,
            "support_structure": 0.25,
            "smart_money": 0.20,
            "trend": 0.15,
            "liquidity": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "extreme_volatility": 80.0,
            "strong_institutional": 75.0,
            "tight_stop_buffer": 0.5,
            "wide_stop_buffer": 2.5,
            "atr_multiplier_high": 3.0,
            "atr_multiplier_low": 1.5
        }
    )


# =====================================================================
# STOP LOSS DECISION ENGINE
# =====================================================================
class StopLossEngine(BaseDecisionEngine):
    """
    Institutional Stop Loss Decision Engine.
    Determines protection strategies, dynamic trailing rules, and capital 
    preservation zones utilizing fused Layer-3 technical and risk metrics.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_stoploss_profile())

    def evaluate_stoploss(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_STOPLOSS_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for schema-agnostic extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_DATA")

            # 1. Base Intelligence & Permissions Extraction
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            market_permission = self._map_market_permission(permission_str)
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)

            # Entry Status (to determine if we are protecting a new or existing position)
            entry_type = self._dynamic_lookup_string(flat_data, ["entry_type", "entry_strength"])
            entry_valid = entry_type.upper() not in ["NO ENTRY", "NO_ENTRY", "NONE", "AVOID", "UNKNOWN"]
            
            # Structural & Risk Metrics
            trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
            volatility = self._dynamic_lookup(flat_data, ["volatility_score", "atr_expansion", "volatility_risk"], 50.0)
            support = self._dynamic_lookup(flat_data, ["support_score", "support_resistance"], 50.0)
            smc = self._dynamic_lookup(flat_data, ["smart_money_score", "order_block", "institutional_footprint"], 50.0)
            liquidity = self._dynamic_lookup(flat_data, ["liquidity_score", "float_quality"], 50.0)
            gap_risk = self._dynamic_lookup(flat_data, ["gap_risk", "overnight_risk"], 20.0)
            sys_risk = self._dynamic_lookup(flat_data, ["systemic_risk", "risk_score"], 20.0)
            breakout = self._dynamic_lookup(flat_data, ["breakout_score", "breakout"], 50.0)

            # Active position status (optional inputs from Holding/Exit engines)
            in_profit = self._dynamic_lookup(flat_data, ["in_profit", "unrealized_pnl_positive"], 0.0) > 50.0
            large_profit = self._dynamic_lookup(flat_data, ["large_profit", "target_near"], 0.0) > 75.0

            # 2. Risk Profiles & Checklists
            self._add_step(ctx, "BUILD_PROTECTION_PROFILE")
            protection_profile = self._build_protection_profile(volatility, gap_risk, liquidity, sys_risk)
            
            checklist = EntryProtectionChecklist(
                trend_alignment=round(trend, 2),
                market_permission=market_permission.value,
                risk_level=round(sys_risk, 2),
                entry_valid=entry_valid,
                volatility=round(volatility, 2),
                institutional_support=round(smc, 2)
            )

            # 3. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(volatility, support, smc, trend, liquidity)

            # 4. Protection Quality Base
            base_protection_quality = self._normalize(
                (volatility * dynamic_weights["volatility"]) +
                (support * dynamic_weights["support_structure"]) +
                (smc * dynamic_weights["smart_money"]) +
                (trend * dynamic_weights["trend"]) +
                (liquidity * dynamic_weights["liquidity"])
            )

            # 5. Determine Capital Protection Level
            self._add_step(ctx, "DETERMINE_PROTECTION_LEVEL")
            protection_level = self._determine_protection_level(sys_risk, volatility, gap_risk, in_profit)

            # 6. Determine StopLoss Structures (Type, Action, Zone, Trailing)
            self._add_step(ctx, "DETERMINE_STOPLOSS_STRUCTURES")
            sl_type, sl_action, sl_zone, sl_dist, sl_buffer, sl_window, sl_priority, trailing_policy, sl_reason = self._determine_stoploss_structures(
                entry_valid, market_permission, volatility, smc, support, breakout, in_profit, large_profit, protection_level
            )

            # 7. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, volatility, sl_buffer, trend, sl_dist, smc, 
                protection_level, market_permission, sys_risk
            )

            # 8. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            sl_confidence = self._calculate_advanced_confidence(
                ctx, layer3_conf, base_protection_quality, parsed_inputs, smc, support
            )
            
            sl_reason = StopLossReason(sl_reason.primary, sl_reason.secondary, round(sl_confidence, 2))

            # 9. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, sl_type, sl_action, trailing_policy, protection_profile, market_permission, entry_valid
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "market_permission": market_permission.value,
                "stoploss_type": sl_type.value,
                "stoploss_action": sl_action.value,
                "stoploss_reason": asdict(sl_reason),
                "stoploss_zone": asdict(sl_zone),
                "stoploss_distance": sl_dist,
                "stoploss_buffer": round(sl_buffer, 2),
                "stoploss_trigger": asdict(StopLossTrigger("Price Breaches Buffer Zone", "Close Below Zone", "HIGH")),
                "stoploss_priority": sl_priority.value,
                "stoploss_window": sl_window.value,
                "trailing_policy": asdict(trailing_policy),
                "capital_protection_level": protection_level.value,
                "stoploss_confidence": round(sl_confidence, 2),
                "protection_profile": asdict(protection_profile),
                "entry_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if entry_valid or sl_action != StopLossAction.NONE else DecisionStatusEnum.NO_DATA
            status_msg = "Stop Loss framework generated." if entry_valid else "No active entry. Stop Loss idling."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=sl_confidence,
                rating_score=base_protection_quality, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"StopLoss Engine execution failed: {str(e)}", exc_info=True)
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

    def _build_protection_profile(self, vol: float, gap: float, liq: float, sys_risk: float) -> ProtectionProfile:
        return ProtectionProfile(
            volatility_protection=self._normalize(vol * 1.2),
            gap_protection=self._normalize(gap * 1.5),
            liquidity_protection=self._normalize(100.0 - liq),
            drawdown_protection=self._normalize(sys_risk * 1.1),
            systemic_protection=self._normalize((vol + gap + sys_risk) / 3.0)
        )

    def _calculate_continuous_weights(self, vol: float, sup: float, smc: float, trend: float, liq: float) -> dict[str, float]:
        w = dict(self.config.base_weights)
        
        # Volatility commands the stop distance
        if vol > 70.0: w["volatility"] += (vol / 100.0) * 0.15; w["trend"] -= 0.05
        # Smart money sets the floor
        if smc > 75.0: w["smart_money"] += 0.10; w["liquidity"] -= 0.05
        # Structural support dictates zone
        if sup > 80.0: w["support_structure"] += 0.10; w["trend"] -= 0.05
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_protection_level(self, sys_risk: float, vol: float, gap: float, in_profit: bool) -> CapitalProtectionLevel:
        if sys_risk > 80.0 or gap > 80.0:
            return CapitalProtectionLevel.MAXIMUM
        if vol > 75.0 or (sys_risk > 60.0 and not in_profit):
            return CapitalProtectionLevel.HIGH
        if in_profit:
            return CapitalProtectionLevel.NORMAL
        return CapitalProtectionLevel.LOW

    def _determine_stoploss_structures(self, entry_valid: bool, perm: MarketPermission, 
                                       vol: float, smc: float, sup: float, breakout: float, 
                                       in_profit: bool, large_profit: bool, level: CapitalProtectionLevel):
        """Maps risk and structural intelligence into exact stop loss mechanics."""
        t = self.config.thresholds
        
        # 1. No Active Trade / Blocked
        if not entry_valid or perm in [MarketPermission.BLOCKED, MarketPermission.RESTRICTED]:
            return (
                StopLossType.NO_STOPLOSS, StopLossAction.NONE, 
                StopLossZone("NONE", "N/A", 0.0, "N/A"), "0.0", 0.0,
                StopLossWindow.NONE, StopLossPriority.NONE,
                TrailingPolicy("NO_TRAILING", 0.0, "N/A", False),
                StopLossReason("No active valid entry.", "Engine idling.", 0.0)
            )

        # 2. Profit Locking (Breakeven or Trailing)
        if large_profit:
            return (
                StopLossType.TRAILING, StopLossAction.TRAIL,
                StopLossZone("SWING_LOW", "Dynamic Pivot Low", 0.5, "HARD_STOP"),
                "Trailing Distance", 0.5, StopLossWindow.MONITOR, StopLossPriority.HIGH,
                TrailingPolicy("ATR_TRAIL", 1.5, "Profit Target 1 Reached", True),
                StopLossReason("Large profit secured.", "Trailing stop activated to maximize gain.", 0.0)
            )
        if in_profit and level in [CapitalProtectionLevel.MAXIMUM, CapitalProtectionLevel.HIGH]:
            return (
                StopLossType.BREAKEVEN, StopLossAction.BREAKEVEN,
                StopLossZone("ENTRY_PRICE", "Breakeven Plus Spread", 0.1, "HARD_STOP"),
                "0.0", 0.1, StopLossWindow.IMMEDIATE, StopLossPriority.CRITICAL,
                TrailingPolicy("NO_TRAILING", 0.0, "N/A", False),
                StopLossReason("Position in profit during high risk.", "Capital protection enforced at breakeven.", 0.0)
            )

        # 3. High Volatility -> Adaptive ATR
        if vol > t.get("extreme_volatility", 80.0):
            buffer = t.get("wide_stop_buffer", 2.5)
            dist_str = f"{t.get('atr_multiplier_high', 3.0)}x ATR"
            return (
                StopLossType.ATR, StopLossAction.SET,
                StopLossZone("ATR", "Volatility Adjusted Band", buffer, "HARD_STOP"),
                dist_str, buffer, StopLossWindow.AFTER_ENTRY, StopLossPriority.HIGH,
                TrailingPolicy("VOLATILITY_ADAPTIVE", buffer, "Post-Entry", True),
                StopLossReason("Extreme volatility detected.", "ATR expansion requires wider protective band.", 0.0)
            )

        # 4. Smart Money / Institutional Core
        if smc > t.get("strong_institutional", 75.0):
            buffer = 0.75
            return (
                StopLossType.ORDER_BLOCK, StopLossAction.SET,
                StopLossZone("ORDER_BLOCK", "Institutional OB Floor", buffer, "CLOSE_BELOW_ZONE"),
                "Below OB", buffer, StopLossWindow.AFTER_ENTRY, StopLossPriority.HIGH,
                TrailingPolicy("STRUCTURE_TRAIL", 0.0, "OB Shift", False),
                StopLossReason("Smart money accumulation present.", "Order Block offers strongest structural protection.", 0.0)
            )

        # 5. Breakout structural support
        if breakout > 70.0:
            buffer = 1.0
            return (
                StopLossType.STRUCTURAL, StopLossAction.SET,
                StopLossZone("BREAKOUT_ORIGIN", "Previous Resistance now Support", buffer, "HARD_STOP"),
                "Below Breakout Base", buffer, StopLossWindow.AFTER_BREAKOUT, StopLossPriority.MEDIUM,
                TrailingPolicy("SWING_TRAIL", 0.0, "New Swing High", True),
                StopLossReason("Breakout structure intact.", "Stop placed below breakout origin.", 0.0)
            )

        # Default standard swing low
        return (
            StopLossType.SWING_LOW, StopLossAction.SET,
            StopLossZone("SWING_LOW", "Recent Swing Low", t.get("tight_stop_buffer", 0.5), "HARD_STOP"),
            "1.5x ATR", t.get("tight_stop_buffer", 0.5), StopLossWindow.AFTER_ENTRY, StopLossPriority.MEDIUM,
            TrailingPolicy("PERCENTAGE_TRAIL", 2.0, "1:1 RR Hit", False),
            StopLossReason("Standard trend mechanics.", "Swing low structural defense.", 0.0)
        )

    def _detect_conflicts(self, ctx: DecisionContext, vol: float, buffer: float, trend: float, 
                          dist: str, smc: float, level: CapitalProtectionLevel, perm: MarketPermission, sys_risk: float) -> None:
        
        # 1. High Volatility + Tight Stop
        if vol > 75.0 and buffer < 1.0:
            self._add_conflict(ctx, "High volatility regime contradicts extremely tight stop loss buffer (Whipsaw risk).", penalty=20.0)
            
        # 2. Weak Trend + Wide Stop
        if trend < 40.0 and "3.0x" in str(dist):
            self._add_conflict(ctx, "Weak directional trend contradicts wide structural stop distance.", penalty=15.0)
            
        # 3. Institutional Entry + Weak Protection
        if smc > 75.0 and level == CapitalProtectionLevel.LOW:
            self._add_conflict(ctx, "Institutional Order Block entry contradicts weak capital protection settings.", penalty=15.0)
            
        # 4. Bull Market + Extreme Tight Stop
        if perm == MarketPermission.ALLOWED and buffer < 0.2:
            self._add_conflict(ctx, "Strong bull regime contradicted by microscopically tight stop buffer.", penalty=10.0)
            
        # 5. High Systemic Risk + Loose Trail
        if sys_risk > 80.0 and level in [CapitalProtectionLevel.NORMAL, CapitalProtectionLevel.LOW]:
            self._add_conflict(ctx, "Extreme systemic risk contradicts relaxed trailing/protection levels.", penalty=20.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, base_qual: float, 
                                       parsed_count: int, smc: float, sup: float) -> float:
        """Determines how structurally reliable the stop loss zone is."""
        engine_agreement = self._normalize(100.0 - abs(smc - sup))
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        protection_quality = base_qual
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        
        return self._normalize(
            (layer3_conf * 0.15) + (protection_quality * 0.30) + 
            (engine_agreement * 0.25) + (evidence_density * 0.15) + (signal_stability * 0.15)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, sl_type: StopLossType, 
                                         sl_action: StopLossAction, trail: TrailingPolicy, 
                                         prof: ProtectionProfile, perm: MarketPermission, entry_valid: bool) -> list[str]:
        flags = []
        
        if not entry_valid:
            self._add_explanation(ctx, "No active entry evaluated. Stop loss engine idling.")
            return flags

        # Explanations
        if sl_type == StopLossType.ATR:
            self._add_explanation(ctx, "ATR expansion requires wider, volatility-adjusted protection.")
        elif sl_type == StopLossType.ORDER_BLOCK:
            self._add_explanation(ctx, "Institutional Order Block offers strongest structural defense against liquidity sweeps.")
        elif sl_action == StopLossAction.BREAKEVEN:
            self._add_explanation(ctx, "Break-even action executed to definitively protect capital in hostile regime.")
        
        if trail.policy != "NO_TRAILING":
            self._add_explanation(ctx, f"Dynamic {trail.policy} activated for progressive profit protection.")

        if prof.liquidity_protection > 80.0:
            self._add_explanation(ctx, "Liquidity sweep risk detected; buffer widened.")
            flags.append("LIQUIDITY_SWEEP_RISK")
            
        if prof.gap_protection > 80.0:
            self._add_explanation(ctx, "High overnight gap risk requires structural cushioning.")
            flags.append("ELEVATED_GAP_RISK")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting volatility/structural signals lower stop loss precision.")
            flags.append("CONFLICTING_PROTECTION_SIGNALS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in StopLoss Engine.", WarningSeverityEnum.CRITICAL)
        self._add_restriction(empty_ctx, "PROTECTION SYSTEMS OFFLINE. FLAT POSITION RECOMMENDED.")
        
        payload = {
            "market_permission": MarketPermission.UNKNOWN.value,
            "stoploss_type": StopLossType.EMERGENCY_EXIT.value if hasattr(StopLossType, 'EMERGENCY_EXIT') else "UNKNOWN",
            "stoploss_action": StopLossAction.NONE.value,
            "stoploss_reason": asdict(StopLossReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "stoploss_zone": asdict(StopLossZone("NONE", "N/A", 0.0, "HARD_STOP")),
            "stoploss_distance": "0.0",
            "stoploss_buffer": 0.0,
            "stoploss_trigger": asdict(StopLossTrigger("Engine Failure", "Failsafe", "CRITICAL")),
            "stoploss_priority": StopLossPriority.CRITICAL.value,
            "stoploss_window": StopLossWindow.IMMEDIATE.value,
            "trailing_policy": asdict(TrailingPolicy("NO_TRAILING", 0.0, "N/A", False)),
            "capital_protection_level": CapitalProtectionLevel.MAXIMUM.value,
            "stoploss_confidence": 0.0,
            "protection_profile": asdict(ProtectionProfile(100.0, 100.0, 100.0, 100.0, 100.0)),
            "entry_checklist": asdict(EntryProtectionChecklist(0.0, "UNKNOWN", 100.0, False, 100.0, 0.0)),
            "risk_flags": ["SYSTEM_FAILURE_PROTECTION"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe risk logic.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_stoploss(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return StopLossEngine().evaluate_stoploss(engine_outputs)
