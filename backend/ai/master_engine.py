"""
GREEN BULL RIDER V6
Layer-5: Master AI Engine
Module: backend/ai/master_engine.py
"""

import time
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, asdict

from backend.decision.base_decision_engine import (
    BaseDecisionEngine,
    DecisionContext,
    DecisionStatusEnum,
    WarningSeverityEnum,
    DecisionTrace
)

# [তোমার অরিজিনাল Enum এবং Dataclass গুলো এখানে হুবহু আছে]
class MasterGrade(str, Enum):
    AAA = "AAA"; AA = "AA"; A = "A"; BBB = "BBB"; BB = "BB"; B = "B"; C = "C"; D = "D"

class MasterStatus(str, Enum):
    APPROVED = "APPROVED"; RESTRICTED = "RESTRICTED"; CONFLICTED = "CONFLICTED"; WATCH = "WATCH"; INCOMPLETE = "INCOMPLETE"

class MarketPermission(str, Enum):
    BLOCKED = "BLOCKED"; RESTRICTED = "RESTRICTED"; LIMITED = "LIMITED"; CONDITIONAL = "CONDITIONAL"; ALLOWED = "ALLOWED"

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

def get_master_profile() -> Any:
    from backend.decision.base_decision_engine import DecisionConfig
    return DecisionConfig(
        profile_name="Institutional_Master_AI_V3_1",
        version="3.1.0",
        stage="Layer-5: AI Fusion & Synthesis",
        schema_version="3.1",
        api_version="v6",
        decision_method="Canonical Multi-Engine Fusion and Conflict Matrix",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={},
        thresholds={"expected_l4_engines": 12, "aaa_threshold": 85.0, "aa_threshold": 75.0, "a_threshold": 65.0, "bbb_threshold": 55.0, "bb_threshold": 45.0}
    )

class MasterEngine(BaseDecisionEngine):
    def __init__(self, config: Any = None):
        super().__init__(config or get_master_profile())

    # --- ফিক্সড মেথড: ইমপ্লিমেন্টিং অ্যাবস্ট্রাক্ট ইভ্যালুয়েট ---
    def evaluate(self, layer4_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_master(layer4_outputs)

    def evaluate_master(self, layer4_outputs: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_MASTER_FUSION")
        
        trace = self._build_trace(layer4_outputs, start_time, ctx, 0)
        if not isinstance(layer4_outputs, dict) or not layer4_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            expected_engines = self.config.thresholds.get("expected_l4_engines", 12)
            received_engines = len([k for k in layer4_outputs.keys() if "engine" in k.lower() or isinstance(layer4_outputs[k], dict)])
            completeness_pct = round(self._clamp((received_engines / expected_engines) * 100.0), 2)

            flat_data = self._flatten_dict(layer4_outputs)
            self._add_step(ctx, "EXTRACT_LAYER4_MATRICES")

            market_perm_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            market_perm = self._map_market_permission(market_perm_str)
            is_permitted = market_perm in [MarketPermission.ALLOWED, MarketPermission.CONDITIONAL]

            l4_conviction = self._dynamic_lookup(flat_data, ["conviction_score", "actionability_score"], None)
            l4_confidence = self._dynamic_lookup(flat_data, ["confidence_score", "reliability_score"], None)
            l4_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], None)
            l4_reward = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score"], None)
            l4_entry_qual = self._dynamic_lookup(flat_data, ["entry_quality", "execution_readiness"], None)

            entry_status = ActionStatus(self._dynamic_lookup_string(flat_data, ["entry_action", "entry_status"]), l4_entry_qual)
            exit_status = ActionStatus(self._dynamic_lookup_string(flat_data, ["exit_action", "exit_status"]), self._dynamic_lookup(flat_data, ["exit_confidence"], None))
            sl_status = ActionStatus(self._dynamic_lookup_string(flat_data, ["stoploss_action", "stoploss_type"]), self._dynamic_lookup(flat_data, ["stoploss_confidence"], None))
            target_status = ActionStatus(self._dynamic_lookup_string(flat_data, ["target_action", "target_type"]), self._dynamic_lookup(flat_data, ["target_confidence"], None))

            holding_type = self._dynamic_lookup_string(flat_data, ["holding_type", "holding_window"])
            alloc_level = self._dynamic_lookup_string(flat_data, ["allocation_level"])
            pos_level = self._dynamic_lookup_string(flat_data, ["position_level"])
            risk_per_trade = self._dynamic_lookup(flat_data, ["risk_per_trade_pct"], None)

            self._add_step(ctx, "VALIDATE_SYSTEMIC_CONSISTENCY")
            critical_conflicts = self._detect_conflicts(ctx, l4_conviction, l4_confidence, l4_risk, entry_status.status, is_permitted)
            alignment_pct = self._calculate_systemic_alignment(l4_conviction, l4_confidence, l4_risk, l4_reward)

            self._add_step(ctx, "DETERMINE_CANONICAL_STATUS_AND_GRADE")
            master_status = self._determine_master_status(is_permitted, critical_conflicts, completeness_pct, l4_entry_qual)
            master_grade = self._determine_master_grade(l4_conviction, l4_reward, l4_risk)

            m_profile = MasterProfile(master_grade.value, round(l4_confidence, 2) if l4_confidence else None, round(l4_conviction, 2) if l4_conviction else None, round(l4_entry_qual, 2) if l4_entry_qual else None, round(l4_reward, 2) if l4_reward else None, round(l4_risk, 2) if l4_risk else None, round(alignment_pct, 2))
            t_profile = TradeProfile(market_perm.value if market_perm else None, entry_status, exit_status, sl_status, target_status)
            c_profile = CapitalProfile(holding_type, alloc_level, pos_level, round(risk_per_trade, 2) if risk_per_trade else None)
            c_summary = ConflictSummary(ctx.conflicts, critical_conflicts, "Systemic Conflicts Logged" if critical_conflicts > 0 else "Fully Aligned")
            e_summary = EvidenceSummary(expected_engines, received_engines, completeness_pct)

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

            return self._build_output(DecisionStatusEnum.SUCCESS if critical_conflicts == 0 else DecisionStatusEnum.PARTIAL, "Success", decision_payload, l4_confidence or 0.0, alignment_pct, ctx, trace)
        except Exception as e:
            self.logger.error(f"Master Engine error: {e}", exc_info=True)
            return self._sanitize_json(self._build_fallback(trace))

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
                if value.upper() in ["UNKNOWN", "NONE", "N/A"]: return None
                return value
        return None

    def _detect_conflicts(self, ctx: DecisionContext, conv: float | None, conf: float | None, risk: float | None, entry_status: str | None, is_permitted: bool) -> int:
        critical_conflicts = 0
        if not is_permitted and entry_status and "EXECUTE" in entry_status.upper():
            self._add_conflict(ctx, "CRITICAL: Entry Engine triggered execution despite Market Gatekeeper Blockade.", penalty=0.0)
            critical_conflicts += 1
        return critical_conflicts

    def _calculate_systemic_alignment(self, conv: float | None, conf: float | None, risk: float | None, reward: float | None) -> float:
        valid_scores = [s for s in [conv, conf, reward, 100.0 - risk] if s is not None]
        if len(valid_scores) < 2: return 0.0
        mean_val = sum(valid_scores) / len(valid_scores)
        variance = sum((s - mean_val) ** 2 for s in valid_scores) / len(valid_scores)
        return self._normalize(100.0 - ((variance ** 0.5) * 2.0))

    def _determine_master_status(self, is_permitted: bool, critical_conflicts: int, completeness_pct: float, entry_readiness: float | None) -> MasterStatus:
        if not is_permitted: return MasterStatus.RESTRICTED
        if completeness_pct < 50.0: return MasterStatus.INCOMPLETE
        if critical_conflicts > 0: return MasterStatus.CONFLICTED
        if entry_readiness and entry_readiness >= 60.0: return MasterStatus.APPROVED
        return MasterStatus.WATCH

    def _determine_master_grade(self, conv: float | None, reward: float | None, risk: float | None) -> MasterGrade:
        if conv is None or risk is None: return MasterGrade.D
        composite_eval = (conv * 0.5) + ((reward or 50.0) * 0.3) + ((100.0 - risk) * 0.2)
        if composite_eval >= 85.0: return MasterGrade.AAA
        if composite_eval >= 75.0: return MasterGrade.AA
        if composite_eval >= 65.0: return MasterGrade.A
        return MasterGrade.D

    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        return {"status": "FAILED"}
