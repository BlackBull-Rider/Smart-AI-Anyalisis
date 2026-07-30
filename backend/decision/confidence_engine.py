"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: confidence_engine.py

Institutional Confidence Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Reliability Validation Boundary.

Key Responsibilities:
- Measures Engine Agreement across Entry, Target, Risk, Reward, and Allocation.
- Assesses Evidence Density and Signal Stability.
- Outputs the absolute Statistical Reliability of the systemic analysis.

Boundary Constraint: This engine DOES NOT calculate Execution Conviction (Actionability), 
Entry/Exit triggers, or Sizing. It evaluates "How reliable is the data and analysis?"
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

class ConfidenceLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXCEPTIONAL = "EXCEPTIONAL"
    UNKNOWN = "UNKNOWN"

class ConfidenceAction(str, Enum):
    IGNORE = "IGNORE"
    WATCH = "WATCH"
    ACCEPT = "ACCEPT"
    TRUST = "TRUST"
    HIGH_TRUST = "HIGH_TRUST"
    NONE = "NONE"

class ConfidencePriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class ConfidenceWindow(str, Enum):
    INTRADAY = "INTRADAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    OPEN = "OPEN"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class ConfidenceProfile:
    confidence_score: float
    reliability_score: float
    engine_agreement: float
    signal_stability: float
    evidence_strength: float
    conflict_score: float

@dataclass(frozen=True)
class ConfidenceReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class ConfidenceChecklist:
    trend_agreement: bool
    risk_agreement: bool
    reward_agreement: bool
    allocation_agreement: bool
    position_agreement: bool
    market_agreement: bool
    evidence_ok: bool

@dataclass(frozen=True)
class ConfidenceComponent:
    name: str
    score: float
    weight: float
    confidence: float


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_confidence_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Confidence_V3",
        version="3.0.0",
        stage="Layer-4: Reliability Validation",
        schema_version="3.0",
        api_version="v6",
        decision_method="Continuous Engine Agreement & Evidence Synthesis",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "engine_agreement": 0.30,
            "evidence_strength": 0.20,
            "signal_stability": 0.20,
            "institutional_reliability": 0.15,
            "fundamental_reliability": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "exceptional_confidence_threshold": 85.0,
            "very_high_confidence_threshold": 75.0,
            "high_confidence_threshold": 60.0,
            "acceptable_evidence_threshold": 50.0,
            "adaptive_agreement_mult": 0.20,
            "adaptive_evidence_mult": 0.15,
            "adaptive_stability_mult": 0.20
        }
    )


# =====================================================================
# CONFIDENCE DECISION ENGINE
# =====================================================================
class ConfidenceEngine(BaseDecisionEngine):
    """
    Institutional Confidence Decision Engine.
    Quantifies the mathematical reliability of the decision pipeline by measuring 
    Engine Agreement, Evidence Density, and Systemic Conflict Ratios.
    Does NOT calculate actionable Conviction.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_confidence_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_confidence(engine_outputs)

    def evaluate_confidence(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main execution pipeline for Reliability Validation.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_CONFIDENCE_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON to extract upstream decisions
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER4_DATA")

            # 1. Base Intelligence & Permissions
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()
            
            # 2. Extract Upstream Quality/Confidence Vectors
            entry_qual = self._dynamic_lookup(flat_data, ["entry_quality"], 50.0)
            risk_qual = self._dynamic_lookup(flat_data, ["risk_quality", "risk_safety"], 50.0) # Inverted Risk
            reward_qual = self._dynamic_lookup(flat_data, ["reward_quality"], 50.0)
            alloc_qual = self._dynamic_lookup(flat_data, ["allocation_quality"], 50.0)
            pos_qual = self._dynamic_lookup(flat_data, ["position_quality"], 50.0)
            stop_qual = self._dynamic_lookup(flat_data, ["stop_quality", "protection_quality"], 50.0)
            target_qual = self._dynamic_lookup(flat_data, ["target_quality_score"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_stability"], 50.0)

            # Upstream Layer-3 Reliability Vectors
            trend_rel = self._dynamic_lookup(flat_data, ["trend_score", "trend_reliability"], 50.0)
            mom_rel = self._dynamic_lookup(flat_data, ["momentum_score"], 50.0)
            inst_rel = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            fund_rel = self._dynamic_lookup(flat_data, ["fundamental_score", "business_quality"], 50.0)
            
            # Global Upstream Conflict Extraction (If passed from other engines)
            upstream_conflicts = self._dynamic_lookup(flat_data, ["conflict_count", "conflict_ratio", "conflicts"], 0.0)
            evidence_cov = self._dynamic_lookup(flat_data, ["evidence_coverage"], 50.0)

            # 3. Calculate Core Systemic Reliability Metrics
            self._add_step(ctx, "CALCULATE_AGREEMENT_AND_EVIDENCE")
            engine_agreement = self._calculate_engine_agreement(
                entry_qual, risk_qual, reward_qual, alloc_qual, pos_qual, stop_qual, target_qual
            )
            evidence_strength = self._calculate_evidence_strength(parsed_inputs, evidence_cov)
            
            # Conflict Score (0 = No Conflicts, 100 = Maximum Systemic Chaos)
            conflict_score = self._normalize(upstream_conflicts * 10.0) # Scaling upstream raw counts to %
            signal_stability = self._normalize(100.0 - conflict_score)

            # 4. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(engine_agreement, evidence_strength, signal_stability)

            # 5. Base Reliability Calculation
            self._add_step(ctx, "CALCULATE_BASE_RELIABILITY")
            base_confidence_score = self._normalize(
                (engine_agreement * dynamic_weights["engine_agreement"]) +
                (evidence_strength * dynamic_weights["evidence_strength"]) +
                (signal_stability * dynamic_weights["signal_stability"]) +
                (inst_rel * dynamic_weights["institutional_reliability"]) +
                (fund_rel * dynamic_weights["fundamental_reliability"])
            )

            # 6. Checklists and Categorizations
            self._add_step(ctx, "GENERATE_CHECKLIST_AND_CATEGORIZATIONS")
            checklist = ConfidenceChecklist(
                trend_agreement=(abs(trend_rel - mom_rel) < 30.0),
                risk_agreement=(abs(risk_qual - stop_qual) < 30.0),
                reward_agreement=(abs(reward_qual - target_qual) < 30.0),
                allocation_agreement=(abs(alloc_qual - risk_qual) < 30.0),
                position_agreement=(abs(pos_qual - entry_qual) < 30.0),
                market_agreement=(regime > 50.0),
                evidence_ok=(evidence_strength >= self.config.thresholds.get("acceptable_evidence_threshold", 50.0))
            )
            
            level, action, priority, window = self._determine_categorizations(
                base_confidence_score, is_permitted, evidence_strength
            )

            # 7. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            overall_risk_magnitude = self._normalize(100.0 - risk_qual) # Recover raw risk
            self._detect_conflicts(
                ctx, level, overall_risk_magnitude, evidence_strength, is_permitted, engine_agreement, conflict_score
            )

            # Adjust confidence score based on intra-engine conflicts
            confidence_score = self._normalize(base_confidence_score - (ctx.conflict_penalty * 0.4)) if is_permitted else 0.0
            
            # The Reliability Score is a smoothed version explicitly for downstream Conviction Engine
            reliability_score = self._normalize((confidence_score * 0.7) + (engine_agreement * 0.3))

            # 8. Build Confidence Reason
            self._add_step(ctx, "BUILD_CONFIDENCE_REASON")
            c_reason = ConfidenceReason(
                primary="Systemic reliability validated." if confidence_score > 60 else "Systemic analysis lacks robust mathematical reliability.",
                secondary=f"Agreement: {round(engine_agreement,1)}% | Evidence: {round(evidence_strength,1)}%",
                confidence=round(confidence_score, 2)
            )

            # Build Profile
            conf_profile = ConfidenceProfile(
                confidence_score=round(confidence_score, 2),
                reliability_score=round(reliability_score, 2),
                engine_agreement=round(engine_agreement, 2),
                signal_stability=round(signal_stability, 2),
                evidence_strength=round(evidence_strength, 2),
                conflict_score=round(conflict_score, 2)
            )

            # 9. Generate Explanations and Risk Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, level, engine_agreement, evidence_strength, signal_stability, inst_rel, fund_rel, is_permitted
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "confidence_level": level.value,
                "confidence_action": action.value,
                "confidence_priority": priority.value,
                "confidence_window": window.value,
                "confidence_score": round(confidence_score, 2),
                "reliability_score": round(reliability_score, 2), # Explicitly provided for Conviction Engine
                "confidence_reason": asdict(c_reason),
                "confidence_profile": asdict(conf_profile),
                "confidence_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL
            status_msg = "Reliability Validation Complete." if is_permitted else "Confidence restricted by Gatekeeper Block."

            trace = self._build_trace(engine_outputs, start_time, ctx, parsed_inputs)
            
            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=reliability_score,
                rating_score=confidence_score, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Confidence Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_engine_agreement(self, entry: float, risk_safe: float, reward: float, 
                                    alloc: float, pos: float, stop: float, target: float) -> float:
        """
        Calculates mathematical agreement across Layer-4 Engines.
        Lower deviation = Higher Agreement.
        """
        scores = [entry, risk_safe, reward, alloc, pos, stop, target]
        valid_scores = [s for s in scores if s > 0.0] # Ignore inactive engines
        
        if not valid_scores:
            return 0.0
            
        mean_val = sum(valid_scores) / len(valid_scores)
        variance = sum((s - mean_val) ** 2 for s in valid_scores) / len(valid_scores)
        std_dev = variance ** 0.5
        
        # Max standard deviation roughly 50 for 0-100 scale. So (std_dev * 2) caps at ~100.
        agreement = self._normalize(100.0 - (std_dev * 2.0))
        return agreement

    def _calculate_evidence_strength(self, parsed_inputs: int, upstream_cov: float) -> float:
        """Combines raw parsing volume with semantic upstream coverage."""
        volume_score = self._clamp((parsed_inputs / 200.0) * 100.0) # Assumes ~200 inputs = 100% density
        return self._normalize((volume_score * 0.4) + (upstream_cov * 0.6))

    def _calculate_continuous_weights(self, agreement: float, evidence: float, stability: float) -> dict[str, float]:
        """Mathematically adjusts weights towards the most dominant reliability factor."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["engine_agreement"] += (agreement / 100.0) * t.get("adaptive_agreement_mult", 0.20)
        w["evidence_strength"] += (evidence / 100.0) * t.get("adaptive_evidence_mult", 0.15)
        w["signal_stability"] += (stability / 100.0) * t.get("adaptive_stability_mult", 0.20)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_categorizations(self, score: float, permitted: bool, evidence: float):
        if not permitted:
            return ConfidenceLevel.UNKNOWN, ConfidenceAction.IGNORE, ConfidencePriority.NONE, ConfidenceWindow.NONE
            
        t = self.config.thresholds
        
        # Confidence determines Trust (not execution action)
        if score >= t.get("exceptional_confidence_threshold", 85.0) and evidence >= 70.0:
            level = ConfidenceLevel.EXCEPTIONAL
            action = ConfidenceAction.HIGH_TRUST
            priority = ConfidencePriority.CRITICAL
        elif score >= t.get("very_high_confidence_threshold", 75.0):
            level = ConfidenceLevel.VERY_HIGH
            action = ConfidenceAction.TRUST
            priority = ConfidencePriority.HIGH
        elif score >= t.get("high_confidence_threshold", 60.0):
            level = ConfidenceLevel.HIGH
            action = ConfidenceAction.ACCEPT
            priority = ConfidencePriority.MEDIUM
        elif score >= 40.0:
            level = ConfidenceLevel.MODERATE
            action = ConfidenceAction.WATCH
            priority = ConfidencePriority.LOW
        else:
            level = ConfidenceLevel.LOW
            action = ConfidenceAction.IGNORE
            priority = ConfidencePriority.WATCH
            
        window = ConfidenceWindow.OPEN if score >= 60.0 else ConfidenceWindow.SHORT_TERM
            
        return level, action, priority, window

    def _detect_conflicts(self, ctx: DecisionContext, level: ConfidenceLevel, risk_magnitude: float, 
                          evidence: float, permitted: bool, agreement: float, conflict_score: float) -> None:
        """Identifies paradoxes between systemic reliability and risk environments."""
        
        # 1. High Confidence + Extreme Risk
        if level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.EXCEPTIONAL] and risk_magnitude > 80.0:
            self._add_conflict(ctx, "Exceptional confidence reported amidst extreme structural market risk.", penalty=15.0)
            
        # 2. High Confidence + Weak Evidence
        if level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.EXCEPTIONAL] and evidence < 40.0:
            self._add_conflict(ctx, "High systemic confidence mathematically invalid given extremely weak evidence density.", penalty=20.0)
            
        # 3. Exceptional Confidence + Bear Market Restriction
        if level in [ConfidenceLevel.EXCEPTIONAL] and not permitted:
            self._add_conflict(ctx, "Mathematical confidence is high, but Gatekeeper has restricted market engagement.", penalty=10.0)
            
        # 4. High Confidence + Low Engine Agreement
        if level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH] and agreement < 40.0:
            self._add_conflict(ctx, "Reported high confidence contradicts severe lack of cross-engine agreement.", penalty=15.0)
            
        # 5. High Confidence + High Conflict Ratio
        if level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH] and conflict_score > 60.0:
            self._add_conflict(ctx, "Confidence inflated despite a very high systemic conflict ratio.", penalty=20.0)

    def _generate_flags_and_explanations(self, ctx: DecisionContext, level: ConfidenceLevel, 
                                         agreement: float, evidence: float, stability: float, 
                                         inst_rel: float, fund_rel: float, permitted: bool) -> list[str]:
        flags = []
        
        if not permitted:
            self._add_explanation(ctx, "Systemic confidence assessment constrained by Market Regime Gatekeeper.")
            flags.append("GATEKEEPER_CONSTRAINT")
            return flags

        # Explanations of Reliability (Not Actionability)
        if agreement > 75.0:
            self._add_explanation(ctx, "Multiple independent engines strongly converge on the same systemic evaluation.")
        elif agreement < 40.0:
            self._add_explanation(ctx, "Divergence between risk, reward, and entry engines mathematically lowers reliability.")
            flags.append("LOW_ENGINE_AGREEMENT")
            
        if evidence > 70.0:
            self._add_explanation(ctx, "High evidence density structurally validates the underlying analysis.")
        elif evidence < 40.0:
            self._add_explanation(ctx, "Sparse or incomplete data sets severely limit overall analytical reliability.")
            flags.append("WEAK_EVIDENCE_DENSITY")
            
        if stability > 75.0:
            self._add_explanation(ctx, "Risk and reward assessments remain internally consistent without logical paradoxes.")
        else:
            self._add_explanation(ctx, "High systemic conflict ratios significantly degrade output stability.")
            flags.append("POOR_SIGNAL_STABILITY")
            
        if inst_rel > 75.0:
            self._add_explanation(ctx, "Confirmed institutional accumulation anchors analytical reliability.")
            
        if fund_rel > 75.0:
            self._add_explanation(ctx, "Robust fundamental backing provides a high-reliability structural floor.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Mathematical paradoxes detected. Reliability metrics adjusted defensively.")
            flags.append("RELIABILITY_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Confidence Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "confidence_level": ConfidenceLevel.UNKNOWN.value,
            "confidence_action": ConfidenceAction.IGNORE.value,
            "confidence_priority": ConfidencePriority.NONE.value,
            "confidence_window": ConfidenceWindow.NONE.value,
            "confidence_score": 0.0,
            "reliability_score": 0.0,
            "confidence_reason": asdict(ConfidenceReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "confidence_profile": asdict(ConfidenceProfile(0.0, 0.0, 0.0, 0.0, 0.0, 100.0)),
            "confidence_checklist": asdict(ConfidenceChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_CONFIDENCE"]
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
def evaluate_confidence(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return ConfidenceEngine().evaluate_confidence(engine_outputs)
