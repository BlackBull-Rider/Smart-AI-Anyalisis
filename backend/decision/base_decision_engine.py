"""
GREEN BULL RIDER V6
Layer-4: Decision Engine Framework
Module: base_decision_engine.py

Institutional-Grade Base Framework for all Layer-4 Decision Engines.
Provides shared infrastructure for validation, decision routing, evidence 
graphing, conflict management, deterministic tracing, and JSON sanitization.

Strictly adheres to thread-safety, pure functions, and deterministic 
execution. Contains NO indicator logic, NO OHLCV processing, and NO AI.
"""

import math
import time
import uuid
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Callable
from dataclasses import dataclass, field, asdict


# =====================================================================
# ENUMS
# =====================================================================

class DecisionStatusEnum(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    INVALID_INPUT = "INVALID_INPUT"
    NO_DATA = "NO_DATA"
    CONFLICT = "CONFLICT"

class DecisionRatingEnum(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    WEAK = "WEAK"
    AVOID = "AVOID"
    UNKNOWN = "UNKNOWN"

class WarningSeverityEnum(str, Enum):
    CRITICAL = "CRITICAL"
    ADVISORY = "ADVISORY"
    RISK = "RISK"
    RESTRICTION = "RESTRICTION"


# =====================================================================
# DATACLASSES (Immutable / Data Structures)
# =====================================================================

@dataclass(frozen=True)
class DecisionConfig:
    """Core configuration profile for a Decision Engine."""
    profile_name: str
    version: str
    stage: str = "Layer-4: Decision"
    schema_version: str = "1.0"
    api_version: str = "v6"
    decision_method: str = "Deterministic Multi-Layer Fusion"
    normalization_method: str = "Min-Max Clamp (0-100)"
    base_weights: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class DecisionStatus:
    status: DecisionStatusEnum
    quality: str
    message: str = "Execution completed"

@dataclass(frozen=True)
class DecisionRating:
    rating: DecisionRatingEnum
    score: float

@dataclass(frozen=True)
class DecisionTrace:
    input_hash: str
    decision_id: str
    pipeline_stage: str
    execution_time_ms: float
    steps_executed: list[str]
    inputs_parsed: int

@dataclass(frozen=True)
class DecisionEvidence:
    evidence_score: float
    positive_count: int
    negative_count: int
    neutral_count: int
    conflict_count: int
    evidence_coverage: float
    confidence_contribution: float

@dataclass(frozen=True)
class DecisionWarning:
    severity: WarningSeverityEnum
    message: str

@dataclass(frozen=True)
class DecisionSignature:
    engine_name: str
    profile: str
    engine_version: str
    schema_version: str
    api_version: str
    timestamp: str
    decision_uuid: str

@dataclass(frozen=True)
class DecisionResult:
    """The standard definitive output schema for ALL Layer-4 Decision Engines."""
    status: dict[str, Any]
    decision: dict[str, Any]
    confidence: float
    rating: dict[str, Any]
    evidence_graph: dict[str, Any]
    warnings: list[dict[str, Any]]
    restrictions: list[str]
    explanations: list[str]
    trace: dict[str, Any]
    decision_signature: dict[str, Any]


# =====================================================================
# THREAD-SAFE EXECUTION CONTEXT
# =====================================================================

@dataclass
class DecisionContext:
    """
    Thread-safe state container injected into a single execution run.
    Ensures the engine remains stateless and purely functional.
    """
    positive_evidence: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    neutral_evidence: list[str] = field(default_factory=list)
    conflicts: int = 0
    conflict_penalty: float = 0.0
    warnings: list[DecisionWarning] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


# =====================================================================
# BASE DECISION ENGINE (Abstract Framework)
# =====================================================================

class BaseDecisionEngine(ABC):
    """
    Abstract Base Class for all Layer-4 Institutional Decision Engines.
    Provides rigorous, deterministic framework utilities.
    """

    def __init__(self, config: DecisionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # Ensure loggers don't duplicate if instantiated repeatedly
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @abstractmethod
    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main execution pipeline. Must be implemented by specific Decision Engines.
        """
        pass

    # ---------------------------------------------------------
    # VALIDATION HELPERS
    # ---------------------------------------------------------
    
    def _validate_input(self, payload: Any, required_keys: list[str] | None = None) -> bool:
        """Strict validation of incoming Layer-3 payloads."""
        if not isinstance(payload, dict) or not payload:
            return False
        if required_keys:
            flat_keys = self._flatten_dict(payload).keys()
            for key in required_keys:
                if not any(key in fk for fk in flat_keys):
                    return False
        return True

    def _is_valid_numeric(self, value: Any) -> bool:
        """Validates if a value is safely numeric (not NaN/Inf)."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return not (math.isnan(value) or math.isinf(value))
        return False

    # ---------------------------------------------------------
    # JSON & DICTIONARY EXTRACTION HELPERS
    # ---------------------------------------------------------

    def _safe_get(self, payload: dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely retrieves a top-level key."""
        return payload.get(key, default)

    def _nested_get(self, payload: dict[str, Any], keys_list: list[str], default: Any = None) -> Any:
        """Safely navigates a nested dictionary structure."""
        current = payload
        for key in keys_list:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def _flatten_dict(self, d: dict[str, Any], parent_key: str = '', sep: str = '_') -> dict[str, Any]:
        """Recursively flattens JSON for schema-agnostic extraction."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key.lower(), v))
        return dict(items)

    def _dynamic_lookup(self, flat_payload: dict[str, Any], keywords: list[str], default: float = 50.0) -> float:
        """
        Dynamic field lookup resilient to schema changes. 
        Searches flattened keys for keyword matches and averages the results.
        """
        extracted = []
        for key, value in flat_payload.items():
            if any(kw in key for kw in keywords):
                if self._is_valid_numeric(value):
                    val = float(value)
                    # Auto-scale 0-1 metrics to 0-100
                    if 0.0 <= val <= 1.0 and val != 0.0:
                        val *= 100.0
                    extracted.append(val)
        
        if not extracted:
            return default
        return self._clamp(sum(extracted) / len(extracted), 0.0, 100.0)

    # ---------------------------------------------------------
    # NORMALIZATION & MATH HELPERS
    # ---------------------------------------------------------

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """Strictly clamps a value between min and max bounds."""
        if not self._is_valid_numeric(value):
            return min_val
        return max(min_val, min(max_val, float(value)))

    def _normalize(self, value: float) -> float:
        """Standard 0-100 Normalization wrapper."""
        return self._clamp(value, 0.0, 100.0)

    def _scale(self, value: float, scale_factor: float) -> float:
        """Safely scales a numeric value."""
        if not self._is_valid_numeric(value):
            return 0.0
        return value * scale_factor

    def _weighted_average(self, items: list[tuple[float, float]]) -> float:
        """
        Calculates a weighted average safely.
        Args: items is a list of tuples (value, weight)
        """
        total_weight = sum(w for _, w in items)
        if total_weight == 0:
            return 50.0
        weighted_sum = sum(v * w for v, w in items if self._is_valid_numeric(v))
        return self._clamp(weighted_sum / total_weight)

    def _determine_rating(self, score: float) -> DecisionRatingEnum:
        """Standard institutional rating curve mapping."""
        if score >= 85.0: return DecisionRatingEnum.EXCELLENT
        if score >= 65.0: return DecisionRatingEnum.GOOD
        if score >= 50.0: return DecisionRatingEnum.FAIR
        if score >= 35.0: return DecisionRatingEnum.WEAK
        if score >= 20.0: return DecisionRatingEnum.AVOID
        return DecisionRatingEnum.UNKNOWN

    # ---------------------------------------------------------
    # CONTEXT MANAGERS (Evidence, Conflicts, Warnings)
    # ---------------------------------------------------------
    
    def _add_evidence(self, ctx: DecisionContext, message: str, evidence_type: str = "positive") -> None:
        """Registers evidence to the execution context."""
        if evidence_type == "positive":
            ctx.positive_evidence.append(message)
        elif evidence_type == "negative":
            ctx.negative_evidence.append(message)
        else:
            ctx.neutral_evidence.append(message)

    def _add_conflict(self, ctx: DecisionContext, message: str, penalty: float = 0.0) -> None:
        """Registers a logical conflict, increments counter, and applies penalty."""
        ctx.conflicts += 1
        ctx.conflict_penalty += penalty
        warning_msg = f"Conflict Detected: {message}"
        self._add_warning(ctx, warning_msg, WarningSeverityEnum.CRITICAL)
        self._add_explanation(ctx, warning_msg)

    def _add_warning(self, ctx: DecisionContext, message: str, severity: WarningSeverityEnum = WarningSeverityEnum.ADVISORY) -> None:
        """Registers a system or market warning."""
        ctx.warnings.append(DecisionWarning(severity=severity, message=message))

    def _add_restriction(self, ctx: DecisionContext, restriction: str) -> None:
        """Applies a strict trade/investment restriction."""
        ctx.restrictions.append(restriction)

    def _add_explanation(self, ctx: DecisionContext, explanation: str) -> None:
        """Adds to the institutional explainability graph."""
        ctx.explanations.append(explanation)

    def _add_step(self, ctx: DecisionContext, step_name: str) -> None:
        """Traces the execution sequence for debugging and determinism."""
        ctx.steps.append(step_name)

    # ---------------------------------------------------------
    # BUILDERS & FORMATTERS
    # ---------------------------------------------------------

    def _build_evidence_graph(self, ctx: DecisionContext, total_expected_fields: int, actual_parsed: int) -> DecisionEvidence:
        """Synthesizes the accumulated contextual evidence into an Evidence Graph."""
        pos = len(ctx.positive_evidence)
        neg = len(ctx.negative_evidence)
        neu = len(ctx.neutral_evidence)
        conflicts = ctx.conflicts
        
        coverage = self._clamp((actual_parsed / max(1, total_expected_fields)) * 100.0)
        
        # Base 50, +5 for pos, -5 for neg, -15 for conflicts
        raw_score = 50.0 + (pos * 5.0) - (neg * 5.0) - (conflicts * 15.0)
        score = self._normalize(raw_score)
        
        # Contribution to final confidence
        total_signals = pos + neg + neu + conflicts
        contribution = self._clamp((total_signals / 20.0) * 100.0)
        
        return DecisionEvidence(
            evidence_score=round(score, 2),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            conflict_count=conflicts,
            evidence_coverage=round(coverage, 2),
            confidence_contribution=round(contribution, 2)
        )

    def _build_trace(self, payload: Any, start_time: float, ctx: DecisionContext, parsed_count: int) -> DecisionTrace:
        """Constructs the deterministic execution trace."""
        input_str = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else "{}"
        input_hash = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
        
        decision_id = f"DEC-{input_hash[:12]}"
        exec_time = round((time.perf_counter() - start_time) * 1000, 4)
        
        return DecisionTrace(
            input_hash=input_hash,
            decision_id=decision_id,
            pipeline_stage=self.config.stage,
            execution_time_ms=exec_time,
            steps_executed=ctx.steps,
            inputs_parsed=parsed_count
        )

    def _build_signature(self) -> DecisionSignature:
        """Constructs the immutable digital signature of the Decision Engine."""
        return DecisionSignature(
            engine_name=self.__class__.__name__,
            profile=self.config.profile_name,
            engine_version=self.config.version,
            schema_version=self.config.schema_version,
            api_version=self.config.api_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision_uuid=str(uuid.uuid4())
        )

    def _sanitize_json(self, obj: Any) -> Any:
        """
        Recursive JSON sanitizer.
        Converts NaN, Infinity, -Infinity to None to ensure strict JSON serialization compliance.
        Safely casts unrecognized objects to strings.
        """
        if isinstance(obj, dict):
            return {str(k): self._sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_json(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return str(obj)

    def _build_output(self, 
                      status: DecisionStatusEnum,
                      status_msg: str,
                      decision_payload: dict[str, Any],
                      confidence: float,
                      rating_score: float,
                      ctx: DecisionContext,
                      trace: DecisionTrace) -> dict[str, Any]:
        """Constructs the final, standardized DecisionResult object."""
        
        rating_enum = self._determine_rating(rating_score)
        
        result = DecisionResult(
            status=asdict(DecisionStatus(status=status, quality="VALID" if status == DecisionStatusEnum.SUCCESS else "INVALID", message=status_msg)),
            decision=decision_payload,
            confidence=round(self._normalize(confidence), 2),
            rating=asdict(DecisionRating(rating=rating_enum, score=round(rating_score, 2))),
            evidence_graph=asdict(self._build_evidence_graph(ctx, 50, trace.inputs_parsed)),
            warnings=[asdict(w) for w in set(ctx.warnings)],  # Deduplicate
            restrictions=sorted(list(set(ctx.restrictions))),
            explanations=sorted(list(set(ctx.explanations))),
            trace=asdict(trace),
            decision_signature=asdict(self._build_signature())
        )
        
        return self._sanitize_json(asdict(result))

    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        """Provides a safe, deterministic failover state for catastrophic errors."""
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure. Engine defaulted to defensive fallback.", WarningSeverityEnum.CRITICAL)
        self._add_restriction(empty_ctx, "ALL TRADING/INVESTMENT BLOCKED DUE TO SYSTEM FAILURE.")
        self._add_explanation(empty_ctx, "Decision engine encountered an unrecoverable error during processing.")
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe.",
            decision_payload={"market_permission": "Blocked", "trade_permission": "No"},
            confidence=0.0,
            rating_score=0.0,
            ctx=empty_ctx,
            trace=trace
        )
