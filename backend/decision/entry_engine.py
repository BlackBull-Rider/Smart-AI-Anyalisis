"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: entry_engine.py

Institutional Entry Decision Engine. (V2 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates as the premier execution gate.
Converts fused Layer-3 Intelligence and Gatekeeper permissions into a 
highly structured, deterministic institutional entry framework.

Key Upgrades from V1:
- Structured Objects (EntryZone, EntryTrigger, RiskProfile, EntryChecklist)
- Strict Enums (EntryType, MarketPermission, CapitalAggression, etc.)
- Adaptive Weight Engine
- Advanced Granular Confidence & Risk Models
"""

import time
from enum import Enum
from typing import Any
from dataclasses import dataclass, asdict

from backend.decision.base_decision_engine import (
    BaseDecisionEngine,
    DecisionConfig,
    DecisionContext,
    DecisionTrace,
    DecisionStatusEnum,
    WarningSeverityEnum
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

class EntryType(str, Enum):
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    RETEST = "RETEST"
    INSTITUTIONAL = "INSTITUTIONAL"
    MOMENTUM = "MOMENTUM"
    CONSERVATIVE = "CONSERVATIVE"
    NO_ENTRY = "NO_ENTRY"

class EntryWindow(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    TODAY = "TODAY"
    ONE_TO_THREE_DAYS = "1_TO_3_DAYS"
    WAIT_PULLBACK = "WAIT_PULLBACK"
    WAIT_CONFIRMATION = "WAIT_CONFIRMATION"
    N_A = "N/A"

class CapitalAggression(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"
    AVOID = "AVOID"

class ExpectedHold(str, Enum):
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITIONAL = "POSITIONAL"
    LONG = "LONG"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class EntryZone:
    type: str
    reference: str
    offset_pct: float
    execution_style: str

@dataclass(frozen=True)
class EntryTrigger:
    trigger_condition: str
    confirmation_metric: str
    confirmation_count_required: int
    confirmation_strength: str

@dataclass(frozen=True)
class RiskProfile:
    structural_risk: float
    volatility_risk: float
    liquidity_risk: float
    execution_risk: float
    gap_risk: float
    composite_risk: float

@dataclass(frozen=True)
class EntryChecklist:
    trend_aligned: bool
    momentum_confirmed: bool
    volume_supported: bool
    institutional_backing: bool
    smart_money_aligned: bool
    risk_acceptable: bool
    reward_favorable: bool
    market_permitted: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_entry_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Entry_V2",
        version="2.2.0",
        stage="Layer-4: Entry Decision",
        schema_version="2.0",
        api_version="v6",
        decision_method="Structured Adaptive Entry Framework",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "trend": 0.20,
            "momentum": 0.15,
            "volume": 0.15,
            "institutional": 0.15,
            "smart_money": 0.15,
            "breakout": 0.10,
            "support": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "checklist_pass_threshold": 65.0,
            "high_quality_threshold": 75.0,
            "immediate_priority": 85.0
        }
    )


# =====================================================================
# ENTRY DECISION ENGINE
# =====================================================================
class EntryEngine(BaseDecisionEngine):
    """
    Institutional Entry Decision Engine.
    Converts Layer-3 Intelligence and Gatekeeper permissions into highly 
    structured, multi-dimensional Entry Objects.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_entry_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_entry(engine_outputs)

    def evaluate_entry(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_ENTRY_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_DATA")

            # 1. Gatekeeper / Market Permission Check (Strict Enum Mapping)
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            market_permission = self._map_market_permission(permission_str)
            is_permitted = market_permission in [MarketPermission.ALLOWED, MarketPermission.CONDITIONAL, MarketPermission.LIMITED]

            # 2. Extract Base Intelligence from Layer-3
            trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength", "trend_alignment"], 50.0)
            mom = self._dynamic_lookup(flat_data, ["momentum_score", "momentum_strength"], 50.0)
            vol = self._dynamic_lookup(flat_data, ["volume_score", "volume_confirmation", "delivery"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "institutional_buying"], 50.0)
            smc = self._dynamic_lookup(flat_data, ["smart_money_score", "accumulation"], 50.0)
            brk = self._dynamic_lookup(flat_data, ["breakout_score", "breakout"], 50.0)
            sup = self._dynamic_lookup(flat_data, ["support", "support_score", "clearance"], 50.0)
            rew = self._dynamic_lookup(flat_data, ["reward_score", "reward_potential"], 50.0)
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)
            
            # Granular Risk Extraction
            risk_profile = self._build_risk_profile(flat_data)

            # 3. Adaptive Weights Calculation
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_adaptive_weights(trend, mom, vol, inst, smc, brk, sup)

            # 4. Entry Quality Model Calculation
            self._add_step(ctx, "CALCULATE_ENTRY_QUALITY")
            entry_quality = self._calculate_entry_quality(
                trend, mom, vol, inst, smc, brk, sup, dynamic_weights, is_permitted, risk_profile.composite_risk
            )

            # 5. Determine Entry Structures (Type, Zone, Trigger, Window, Aggression, Hold)
            self._add_step(ctx, "DETERMINE_ENTRY_STRUCTURES")
            entry_type, entry_zone, entry_trigger, entry_window, aggression, hold = self._determine_entry_structures(
                is_permitted, entry_quality, brk, sup, trend, mom, inst, smc, risk_profile
            )

            # 6. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(ctx, trend, mom, brk, vol, inst, smc, risk_profile.composite_risk, rew, market_permission)

            # 7. Entry Checklist Generation
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = self._generate_checklist(trend, mom, vol, inst, smc, risk_profile.composite_risk, rew, is_permitted)

            # 8. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            entry_confidence = self._calculate_advanced_confidence(
                ctx, layer3_conf, entry_quality, trend, mom, inst, smc, parsed_inputs, is_permitted
            )

            # 9. Generate Flags and Institutional Explanations
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, market_permission, entry_quality, checklist, risk_profile, entry_type
            )
            
            # Force penalize quality by dynamic conflicts
            final_quality = self._normalize(entry_quality - ctx.conflict_penalty) if is_permitted else 0.0
            entry_strength = "Strong" if final_quality >= 75.0 else ("Moderate" if final_quality >= 50.0 else "Weak")
            if not is_permitted: entry_strength = "None"
            
            entry_priority = self._determine_entry_priority(final_quality, is_permitted)

            # 10. Build Final Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "entry_type": entry_type.value,
                "entry_window": entry_window.value,
                "capital_aggression": aggression.value,
                "expected_hold": hold.value,
                "entry_quality": round(final_quality, 2),
                "entry_strength": entry_strength,
                "entry_priority": entry_priority,
                "market_permission": market_permission.value,
                "entry_zone": asdict(entry_zone),
                "entry_trigger": asdict(entry_trigger),
                "entry_checklist": asdict(checklist),
                "risk_profile": asdict(risk_profile),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL
            status_msg = "Entry framework constructed." if is_permitted else "Entry blocked by Gatekeeper."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=entry_confidence,
                rating_score=final_quality,
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Entry Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & ADVANCED MODELS
    # ---------------------------------------------------------

    def _map_market_permission(self, perm_str: str) -> MarketPermission:
        p = perm_str.upper()
        if "ALLOW" in p or "YES" in p: return MarketPermission.ALLOWED
        if "CONDITION" in p: return MarketPermission.CONDITIONAL
        if "LIMIT" in p: return MarketPermission.LIMITED
        if "RESTRICT" in p: return MarketPermission.RESTRICTED
        return MarketPermission.BLOCKED

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _build_risk_profile(self, flat_data: dict[str, Any]) -> RiskProfile:
        struct_risk = self._dynamic_lookup(flat_data, ["structural_risk", "regime_risk", "fundamental_risk"], 30.0)
        vol_risk = self._dynamic_lookup(flat_data, ["volatility_risk", "atr_expansion", "danger"], 30.0)
        liq_risk = self._dynamic_lookup(flat_data, ["liquidity_risk", "slippage"], 20.0)
        exec_risk = self._dynamic_lookup(flat_data, ["execution_risk", "spread"], 20.0)
        gap_risk = self._dynamic_lookup(flat_data, ["gap_risk", "overnight_risk"], 20.0)
        
        comp_risk = self._normalize((struct_risk * 0.3) + (vol_risk * 0.3) + (liq_risk * 0.2) + (gap_risk * 0.1) + (exec_risk * 0.1))
        
        return RiskProfile(
            round(struct_risk, 2), round(vol_risk, 2), round(liq_risk, 2), 
            round(exec_risk, 2), round(gap_risk, 2), round(comp_risk, 2)
        )

    def _calculate_adaptive_weights(self, trend: float, mom: float, vol: float, inst: float, smc: float, brk: float, sup: float) -> dict[str, float]:
        w = dict(self.config.base_weights)
        if trend > 80: w["trend"] += 0.10; w["support"] -= 0.10
        if mom > 85: w["momentum"] += 0.10; w["breakout"] -= 0.10
        if vol > 80: w["volume"] += 0.05; w["trend"] -= 0.05
        if inst > 85 or smc > 85: w["institutional"] += 0.10; w["momentum"] -= 0.10
        if brk > 80: w["breakout"] += 0.10; w["support"] -= 0.10
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _calculate_entry_quality(self, trend: float, mom: float, vol: float, inst: float, smc: float, 
                                 brk: float, sup: float, w: dict, is_permitted: bool, comp_risk: float) -> float:
        if not is_permitted: return 0.0
        base = (trend*w["trend"] + mom*w["momentum"] + vol*w["volume"] + 
                inst*w["institutional"] + smc*w["smart_money"] + 
                brk*w["breakout"] + sup*w["support"])
        return self._normalize(base - (comp_risk * 0.15))

    def _determine_entry_structures(self, is_permitted: bool, quality: float, brk: float, sup: float, 
                                    trend: float, mom: float, inst: float, smc: float, risk: RiskProfile):
        if not is_permitted or quality < 30.0:
            return (
                EntryType.NO_ENTRY,
                EntryZone("NONE", "N/A", 0.0, "BLOCKED"),
                EntryTrigger("NONE", "N/A", 0, "NONE"),
                EntryWindow.N_A,
                CapitalAggression.AVOID,
                ExpectedHold.NONE
            )

        aggression = CapitalAggression.NORMAL
        if quality > 80.0 and risk.composite_risk < 30.0: aggression = CapitalAggression.AGGRESSIVE
        elif risk.composite_risk > 60.0: aggression = CapitalAggression.DEFENSIVE

        # Breakout Architecture
        if brk > 75.0 and mom > 70.0:
            return (
                EntryType.BREAKOUT,
                EntryZone("breakout_level", "Resistance Breakout", 0.5, "CMP_ON_CLOSE"),
                EntryTrigger("Price Close > Breakout Level", "Volume Surge", 1, "HIGH"),
                EntryWindow.IMMEDIATE, aggression, ExpectedHold.SWING
            )
        # Pullback Architecture
        if sup > 75.0 and trend > 70.0:
            return (
                EntryType.PULLBACK,
                EntryZone("support_level", "Dynamic / Static Support", 0.2, "LIMIT_OR_CMP_ON_REJECTION"),
                EntryTrigger("Bullish Reversal Candle at Support", "Momentum Divergence Reset", 1, "MEDIUM"),
                EntryWindow.ONE_TO_THREE_DAYS, aggression, ExpectedHold.POSITIONAL
            )
        # Institutional / Smart Money
        if inst > 75.0 or smc > 75.0:
            return (
                EntryType.INSTITUTIONAL,
                EntryZone("order_block", "Smart Money Accumulation Zone", 0.0, "LIMIT_IN_ZONE"),
                EntryTrigger("Price Enters Discount OB", "Institutional Footprint Print", 2, "HIGH"),
                EntryWindow.WAIT_PULLBACK, aggression, ExpectedHold.LONG
            )
        # Momentum Continuation
        if trend > 70.0 and mom > 70.0:
            return (
                EntryType.MOMENTUM,
                EntryZone("momentum_vector", "Current Trend Vector", 0.0, "CMP_IMMEDIATE"),
                EntryTrigger("Momentum Oscillators Unbroken", "Trend Alignment", 1, "HIGH"),
                EntryWindow.IMMEDIATE, aggression, ExpectedHold.INTRADAY if risk.gap_risk > 60 else ExpectedHold.SWING
            )
            
        return (
            EntryType.CONSERVATIVE,
            EntryZone("fair_value", "Discount Zone", 0.0, "WAIT_FOR_SETUP"),
            EntryTrigger("Multiple Timeframe Alignment", "Broad Market Consensus", 3, "HIGH"),
            EntryWindow.WAIT_CONFIRMATION, CapitalAggression.DEFENSIVE, ExpectedHold.SWING
        )

    def _generate_checklist(self, trend: float, mom: float, vol: float, inst: float, smc: float, 
                            risk: float, reward: float, is_permitted: bool) -> EntryChecklist:
        t = self.config.thresholds.get("checklist_pass_threshold", 65.0)
        return EntryChecklist(
            trend_aligned=(trend >= t),
            momentum_confirmed=(mom >= t),
            volume_supported=(vol >= t),
            institutional_backing=(inst >= t),
            smart_money_aligned=(smc >= t),
            risk_acceptable=(risk < 40.0),
            reward_favorable=(reward >= t),
            market_permitted=is_permitted
        )

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, quality: float, 
    DecisionTrace,
                                       trend: float, mom: float, inst: float, smc: float, 
                                       parsed_count: int, is_permitted: bool) -> float:
        if not is_permitted: return 0.0
        
        engine_agreement = self._normalize(100.0 - (abs(trend - mom) * 0.5) - (abs(inst - smc) * 0.5))
        evidence_density = self._clamp((len(ctx.positive_evidence) / max(1, parsed_count)) * 200.0) # Scaled metric
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        inst_reliability = self._normalize((inst * 0.6) + (smc * 0.4))
        
        return self._normalize(
            (layer3_conf * 0.20) + (engine_agreement * 0.20) + (evidence_density * 0.15) + 
            (signal_stability * 0.20) + (inst_reliability * 0.25)
        )

    def _detect_conflicts(self, ctx: DecisionContext, trend: float, mom: float, brk: float, vol: float, 
    DecisionTrace,
                          inst: float, smc: float, risk: float, rew: float, perm: MarketPermission) -> None:
        if trend > 80.0 and perm in [MarketPermission.BLOCKED, MarketPermission.RESTRICTED]:
            self._add_conflict(ctx, "Bullish Trend exists, but Market Regime Gatekeeper has BLOCKED entry.", penalty=50.0)
        if trend > 75.0 and mom < 40.0:
            self._add_conflict(ctx, "Strong directional trend contradicts weakening momentum.", penalty=15.0)
        if brk > 70.0 and vol < 40.0:
            self._add_conflict(ctx, "Breakout detected without confirming volume (Fake-out Risk).", penalty=20.0)
        if smc > 75.0 and inst < 40.0:
            self._add_conflict(ctx, "Smart Money bullish, but broad Institutional Participation missing.", penalty=10.0)
        if rew > 75.0 and risk > 75.0:
            self._add_conflict(ctx, "High reward potential overshadowed by extreme structural/execution risk.", penalty=15.0)

    def _determine_entry_priority(self, quality: float, is_permitted: bool) -> str:
        if not is_permitted: return "Avoid"
        t = self.config.thresholds
        if quality >= t.get("immediate_priority", 85.0): return "Immediate"
        if quality >= t.get("high_quality_threshold", 75.0): return "High"
        if quality >= 60.0: return "Medium"
        if quality >= 45.0: return "Watchlist"
        return "Avoid"

    def _generate_flags_and_explanations(self, ctx: DecisionContext, perm: MarketPermission, 
    DecisionTrace,
                                         quality: float, chk: EntryChecklist, risk: RiskProfile, type_enum: EntryType) -> list[str]:
        flags = []
        
        if perm in [MarketPermission.BLOCKED, MarketPermission.RESTRICTED]:
            self._add_explanation(ctx, "Entry evaluating blocked by Layer-4 Market Regime Gatekeeper.")
            self._add_restriction(ctx, "Trade entry strictly prohibited.")
            flags.append("GATEKEEPER_BLOCKED")
            return flags

        if chk.trend_aligned and chk.momentum_confirmed:
            self._add_explanation(ctx, "Trend and momentum align for a highly robust entry vector.")
        if chk.institutional_backing and chk.smart_money_aligned:
            self._add_explanation(ctx, "Combined Smart Money and Institutional footprints strongly endorse accumulation.")

        if risk.composite_risk > 60.0:
            self._add_warning(ctx, "Elevated composite risk dictates defensive positioning.", WarningSeverityEnum.RISK)
            flags.append("HIGH_COMPOSITE_RISK")
        if risk.gap_risk > 60.0:
            flags.append("HIGH_OVERNIGHT_GAP_RISK")
        if risk.liquidity_risk > 60.0:
            flags.append("ELEVATED_SLIPPAGE_RISK")

        if type_enum == EntryType.BREAKOUT and not chk.volume_supported:
            flags.append("UNCONFIRMED_BREAKOUT")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Inter-layer signal conflicts mathematically degrade entry reliability.")
            flags.append("SIGNAL_CONFLICTS_PRESENT")

        if quality >= 80.0 and len(flags) == 0:
            self._add_explanation(ctx, "Pristine alignment across all technical and institutional matrices. Optimal entry conditions met.")

        return flags


    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure. Entry defaulted to BLOCKED.", WarningSeverityEnum.CRITICAL)
        self._add_restriction(empty_ctx, "ENTRY SYSTEM LOCKDOWN.")
        
        payload = {
            "entry_type": EntryType.NO_ENTRY.value,
            "entry_window": EntryWindow.N_A.value,
            "capital_aggression": CapitalAggression.AVOID.value,
            "expected_hold": ExpectedHold.NONE.value,
            "entry_quality": 0.0,
            "entry_strength": "None",
            "entry_priority": "Avoid",
            "market_permission": MarketPermission.BLOCKED.value,
            "entry_zone": asdict(EntryZone("NONE", "N/A", 0.0, "BLOCKED")),
            "entry_trigger": asdict(EntryTrigger("NONE", "N/A", 0, "NONE")),
            "entry_checklist": asdict(EntryChecklist(False,False,False,False,False,False,False,False)),
            "risk_profile": asdict(RiskProfile(100.0, 100.0, 100.0, 100.0, 100.0, 100.0)),
            "risk_flags": ["SYSTEM_FAILURE_BLOCK"]
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
def evaluate_entry(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return EntryEngine().evaluate_entry(engine_outputs)
