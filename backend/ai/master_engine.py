"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: master_engine.py

Institutional Master AI Engine. (V3.1.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Layer-5 Gatekeeper and 
Decision Fusion Boundary.

Key Upgrades from V3.0:
- Acts purely as a Fusion Engine (Does NOT calculate a new Master Score).
- Measures Evidence Completeness based on 12 explicit Layer-4 Engines.
- Replaces math-based conflict penalties with deterministic Status overrides (CONFLICTED).
- Uses 'None' instead of 'UNKNOWN' for missing data safety.
- Trade Profile uses structured ActionStatus (Action + Confidence).

Boundary Constraint: This engine NEVER creates new Entry, Target, Risk, or Size metrics, 
nor does it recalculate new weighted master scores. It ONLY fuses existing Layer-4 outputs, 
validates systemic consistency, and determines the Canonical Status and Grade.
"""

import time
from enum import Enum
from typing import Any, Optional
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
class MasterGrade(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    C = "C"
    D = "D"

class MasterStatus(str, Enum):
    APPROVED = "APPROVED"         # Fully aligned execution pipeline
    RESTRICTED = "RESTRICTED"     # Gatekeeper Blocked
    CONFLICTED = "CONFLICTED"     # Inter-engine paradoxes detected
    WATCH = "WATCH"               # Setup forming, execution not ready
    INCOMPLETE = "INCOMPLETE"     # Missing critical L4 Data

class MarketPermission(str, Enum):
    BLOCKED = "BLOCKED"
    RESTRICTED = "RESTRICTED"
    LIMITED = "LIMITED"
    CONDITIONAL = "CONDITIONAL"
    ALLOWED = "ALLOWED"


# =====================================================================
# STRUCTURED OBJECTS (For Downstream Layer-5 Engines)
# =====================================================================
@dataclass(frozen=True)
class ActionStatus:
    status: Optional[str]
    confidence: Optional[float]

@dataclass(frozen=True)
class MasterProfile:
    master_grade: str
    master_confidence: Optional[float]
    master_conviction: Optional[float]
    execution_readiness: Optional[float]
    opportunity_rating: Optional[float]
    risk_rating: Optional[float]
    systemic_alignment_pct: float

@dataclass(frozen=True)
class TradeProfile:
    market_permission: Optional[str]
    entry: ActionStatus
    exit: ActionStatus
    stoploss: ActionStatus
    target: ActionStatus

@dataclass(frozen=True)
class CapitalProfile:
    holding_type: Optional[str]
    allocation_level: Optional[str]
    position_level: Optional[str]
    risk_per_trade_pct: Optional[float]

@dataclass(frozen=True)
class ConflictSummary:
    conflict_count: int
    critical_conflicts: int
    resolution_status: str

@dataclass(frozen=True)
class EvidenceSummary:
    expected_engines: int
    received_engines: int
    data_completeness_pct: float


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_master_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Master_AI_V3_1",
        version="3.1.0",
        stage="Layer-5: AI Fusion & Synthesis",
        schema_version="3.1",
        api_version="v6",
        decision_method="Canonical Multi-Engine Fusion and Conflict Matrix",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={}, # Weights removed: Master Engine does not calculate scores
        thresholds={
            "expected_l4_engines": 12,
            "aaa_threshold": 85.0, # Uses L4 composites for grading
            "aa_threshold": 75.0,
            "a_threshold": 65.0,
            "bbb_threshold": 55.0,
            "bb_threshold": 45.0
        }
    )


# =====================================================================
# MASTER AI DECISION ENGINE
# =====================================================================
class MasterEngine(BaseDecisionEngine):
    """
    Institutional Master AI Engine.
    The Single Source of Truth for Layer-5. Fuses all Layer-4 outputs into a canonical 
    matrix, evaluates inter-engine alignment, and generates the Canonical Payload 
    WITHOUT regenerating new analytical scores.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_master_profile())

    def evaluate_master(self, layer4_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main execution pipeline for Canonical Decision Fusion.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_MASTER_FUSION")
        
        trace = self._build_trace(layer4_outputs, start_time, ctx, 0)

        if not isinstance(layer4_outputs, dict) or not layer4_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Check Data Completeness explicitly based on expected engine outputs
            expected_engines = self.config.thresholds.get("expected_l4_engines", 12)
            received_engines = len([k for k in layer4_outputs.keys() if "engine" in k.lower() or isinstance(layer4_outputs[k], dict)])
            completeness_pct = round(self._clamp((received_engines / expected_engines) * 100.0), 2)

            flat_data = self._flatten_dict(layer4_outputs)
            self._add_step(ctx, "EXTRACT_LAYER4_MATRICES")

            # 1. Market & Gatekeeper Extraction
            market_perm_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            market_perm = self._map_market_permission(market_perm_str)
            is_permitted = market_perm in [MarketPermission.ALLOWED, MarketPermission.CONDITIONAL]

            # 2. Extract Core L4 Numeric Quantifications (Preserved exactly as received)
            l4_conviction = self._dynamic_lookup(flat_data, ["conviction_score", "actionability_score"], None)
            l4_confidence = self._dynamic_lookup(flat_data, ["confidence_score", "reliability_score"], None)
            l4_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], None)
            l4_reward = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score"], None)
            l4_entry_qual = self._dynamic_lookup(flat_data, ["entry_quality", "execution_readiness"], None)

            # 3. Extract Execution Matrix (Action + Confidence)
            entry_status = ActionStatus(
                self._dynamic_lookup_string(flat_data, ["entry_action", "entry_status"]),
                l4_entry_qual
            )
            exit_status = ActionStatus(
                self._dynamic_lookup_string(flat_data, ["exit_action", "exit_status"]),
                self._dynamic_lookup(flat_data, ["exit_confidence"], None)
            )
            sl_status = ActionStatus(
                self._dynamic_lookup_string(flat_data, ["stoploss_action", "stoploss_type"]),
                self._dynamic_lookup(flat_data, ["stoploss_confidence"], None)
            )
            target_status = ActionStatus(
                self._dynamic_lookup_string(flat_data, ["target_action", "target_type"]),
                self._dynamic_lookup(flat_data, ["target_confidence"], None)
            )

            # 4. Extract Capital Matrix
            holding_type = self._dynamic_lookup_string(flat_data, ["holding_type", "holding_window"])
            alloc_level = self._dynamic_lookup_string(flat_data, ["allocation_level"])
            pos_level = self._dynamic_lookup_string(flat_data, ["position_level"])
            risk_per_trade = self._dynamic_lookup(flat_data, ["risk_per_trade_pct"], None)

            # 5. Inter-Engine Systemic Conflict Resolution
            self._add_step(ctx, "VALIDATE_SYSTEMIC_CONSISTENCY")
            critical_conflicts = self._detect_conflicts(
                ctx, l4_conviction, l4_confidence, l4_risk, entry_status.status, is_permitted
            )

            # Systemic Alignment Calculation (Agreement between engines, no math penalty)
            alignment_pct = self._calculate_systemic_alignment(l4_conviction, l4_confidence, l4_risk, l4_reward)

            # 6. Determine Master Status and Grade
            self._add_step(ctx, "DETERMINE_CANONICAL_STATUS_AND_GRADE")
            master_status = self._determine_master_status(is_permitted, critical_conflicts, completeness_pct, l4_entry_qual)
            master_grade = self._determine_master_grade(l4_conviction, l4_reward, l4_risk)

            # 7. Construct Sub-Profiles
            self._add_step(ctx, "BUILD_PROFILES")
            m_profile = MasterProfile(
                master_grade=master_grade.value,
                master_confidence=round(l4_confidence, 2) if l4_confidence else None,
                master_conviction=round(l4_conviction, 2) if l4_conviction else None,
                execution_readiness=round(l4_entry_qual, 2) if l4_entry_qual else None,
                opportunity_rating=round(l4_reward, 2) if l4_reward else None,
                risk_rating=round(l4_risk, 2) if l4_risk else None,
                systemic_alignment_pct=round(alignment_pct, 2)
            )
            
            t_profile = TradeProfile(
                market_permission=market_perm.value if market_perm else None, 
                entry=entry_status, 
                exit=exit_status, 
                stoploss=sl_status, 
                target=target_status
            )
            
            c_profile = CapitalProfile(
                holding_type=holding_type, 
                allocation_level=alloc_level, 
                position_level=pos_level, 
                risk_per_trade_pct=round(risk_per_trade, 2) if risk_per_trade else None
            )
            
            c_summary = ConflictSummary(
                conflict_count=ctx.conflicts,
                critical_conflicts=critical_conflicts,
                resolution_status="Systemic Conflicts Logged" if critical_conflicts > 0 else "Fully Aligned"
            )

            e_summary = EvidenceSummary(
                expected_engines=expected_engines,
                received_engines=received_engines,
                data_completeness_pct=completeness_pct
            )

            # 8. Build Payload
            self._add_step(ctx, "BUILD_CANONICAL_PAYLOAD")
            decision_payload = {
                "master_status": master_status.value,
                "master_grade": master_grade.value,
                "systemic_alignment_pct": round(alignment_pct, 2),
                "master_profile": asdict(m_profile),
                "trade_profile": asdict(t_profile),
                "capital_profile": asdict(c_profile),
                "conflict_summary": asdict(c_summary),
                "evidence_summary": asdict(e_summary)
            }

            status_enum = DecisionStatusEnum.SUCCESS if critical_conflicts == 0 else DecisionStatusEnum.PARTIAL
            status_msg = "Canonical Master Matrix successfully generated."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = received_engines

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=l4_confidence or 0.0,
                rating_score=alignment_pct, # Replacing arbitrary master_score with systemic alignment
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Master Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # DECISION FUSION & VALIDATION
    # ---------------------------------------------------------

    def _map_market_permission(self, perm_str: str | None) -> Optional[MarketPermission]:
        if not perm_str: return None
        p = perm_str.upper()
        if "BLOCK" in p: return MarketPermission.BLOCKED
        if "RESTRICT" in p: return MarketPermission.RESTRICTED
        if "LIMIT" in p: return MarketPermission.LIMITED
        if "CONDITION" in p: return MarketPermission.CONDITIONAL
        if "ALLOW" in p or "YES" in p: return MarketPermission.ALLOWED
        return None

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> Optional[str]:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                if value.upper() in ["UNKNOWN", "NONE", "N/A"]:
                    return None
                return value
        return None

    def _detect_conflicts(self, ctx: DecisionContext, conv: float | None, conf: float | None, 
                          risk: float | None, entry_status: str | None, is_permitted: bool) -> int:
        """
        Cross-checks Layer-4 engines for logical paradoxes without modifying native scores.
        """
        critical_conflicts = 0

        # 1. Gatekeeper Override (Critical)
        if not is_permitted and entry_status and "EXECUTE" in entry_status.upper():
            self._add_conflict(ctx, "CRITICAL: Entry Engine triggered execution despite Market Gatekeeper Blockade.", penalty=0.0)
            critical_conflicts += 1
            
        # 2. Conviction vs Confidence Mismatch
        if conv and conf and conv > 80.0 and conf < 40.0:
            self._add_conflict(ctx, "Actionability (Conviction) is extreme while Data Reliability (Confidence) is weak.", penalty=0.0)
            critical_conflicts += 1
            
        # 3. Actionability vs Extreme Risk
        if conv and risk and conv > 75.0 and risk > 80.0:
            self._add_conflict(ctx, "High Execution Conviction is heavily contradicted by Extreme L4 Risk.", penalty=0.0)
            critical_conflicts += 1

        return critical_conflicts

    def _calculate_systemic_alignment(self, conv: float | None, conf: float | None, risk: float | None, reward: float | None) -> float:
        """Calculates mathematical consistency between core L4 engines without creating a new master score."""
        valid_scores = []
        if conv is not None: valid_scores.append(conv)
        if conf is not None: valid_scores.append(conf)
        if reward is not None: valid_scores.append(reward)
        if risk is not None: valid_scores.append(100.0 - risk) # Invert risk to align with positive metrics
        
        if len(valid_scores) < 2: return 0.0
            
        mean_val = sum(valid_scores) / len(valid_scores)
        variance = sum((s - mean_val) ** 2 for s in valid_scores) / len(valid_scores)
        std_dev = variance ** 0.5
        
        # Max standard deviation roughly 50. So (std_dev * 2) caps at ~100.
        return self._normalize(100.0 - (std_dev * 2.0))

    def _determine_master_status(self, is_permitted: bool, critical_conflicts: int, completeness_pct: float, entry_readiness: float | None) -> MasterStatus:
        if not is_permitted:
            return MasterStatus.RESTRICTED
        if completeness_pct < 50.0:
            return MasterStatus.INCOMPLETE
        if critical_conflicts > 0:
            return MasterStatus.CONFLICTED
        if entry_readiness and entry_readiness >= 60.0:
            return MasterStatus.APPROVED
            
        return MasterStatus.WATCH

    def _determine_master_grade(self, conv: float | None, reward: float | None, risk: float | None) -> MasterGrade:
        """Assigns an Institutional Alpha Grade based strictly on L4 matrix."""
        t = self.config.thresholds
        
        # If critical data is missing, fail safely
        if conv is None or risk is None:
            return MasterGrade.D
            
        composite_eval = (conv * 0.5) + ((reward or 50.0) * 0.3) + ((100.0 - risk) * 0.2)
        
        if composite_eval >= t.get("aaa_threshold", 85.0): return MasterGrade.AAA
        if composite_eval >= t.get("aa_threshold", 75.0): return MasterGrade.AA
        if composite_eval >= t.get("a_threshold", 65.0): return MasterGrade.A
        if composite_eval >= t.get("bbb_threshold", 55.0): return MasterGrade.BBB
        if composite_eval >= t.get("bb_threshold", 45.0): return MasterGrade.BB
        if composite_eval >= 35.0: return MasterGrade.B
        if composite_eval >= 20.0: return MasterGrade.C
        
        return MasterGrade.D


    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Master Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "master_status": MasterStatus.INCOMPLETE.value,
            "master_grade": MasterGrade.D.value,
            "systemic_alignment_pct": 0.0,
            "master_profile": asdict(MasterProfile(MasterGrade.D.value, None, None, None, None, None, 0.0)),
            "trade_profile": asdict(TradeProfile(None, ActionStatus(None, None), ActionStatus(None, None), ActionStatus(None, None), ActionStatus(None, None))),
            "capital_profile": asdict(CapitalProfile(None, None, None, None)),
            "conflict_summary": asdict(ConflictSummary(0, 0, "SYSTEM FAILURE")),
            "evidence_summary": asdict(EvidenceSummary(12, 0, 0.0))
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Master fusion failed. Canonical payload aborted.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_master(layer4_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return MasterEngine().evaluate_master(layer4_outputs)
