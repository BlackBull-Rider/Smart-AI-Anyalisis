"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine Base Framework
Module: base_engine.py
"""

import math
import time
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class EngineConfig:
    profile_name: str
    version: str
    stage: str = "Layer-3: Scoring"
    schema_version: str = "1.0"
    api_version: str = "v6"
    scoring_method: str = "Adaptive Evidence-Weighted"
    normalization_method: str = "Min-Max Clamp (0-100)"
    base_weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class EvidenceGraph:
    evidence_score: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    conflict_count: int = 0
    evidence_coverage: float = 0.0

@dataclass(frozen=True)
class PipelineTrace:
    input_hash: str
    processing_id: str
    pipeline_stage: str
    execution_time_ms: float
    analyzer_version: str
    analyzer_timestamp: str
    analyzer_hash: str

@dataclass(frozen=True)
class OutputStatus:
    status: str
    quality: str

# FIX 1: Removed frozen=True so child engines can update weighted_score later
@dataclass
class ScoreBreakdown:
    raw_score: float
    normalized_score: float
    weighted_score: float
    penalty: float
    bonus: float
    final_score: float

class BaseEngine(ABC):
    RATINGS = (
        (95.0, "Elite"), (90.0, "Exceptional"), (80.0, "Very Strong"),
        (70.0, "Strong"), (60.0, "Good"), (50.0, "Neutral"),
        (35.0, "Weak"), (20.0, "Poor"), (0.0, "Very Weak")
    )

    def __init__(self, config: EngineConfig):
        self.config = config
        self._evidence_log: List[str] = []
        self._warning_log: List[str] = []
        self._positive_log: List[str] = []
        self._negative_log: List[str] = []
        self._score_reasons: List[str] = []
        self._conflicts = 0

    @abstractmethod
    def calculate(self, payload: Union[Dict[str, Any], None]) -> Dict[str, Any]:
        pass

    def _extract_meta(self, payload: Any) -> Dict[str, str]:
        """Aggressively hunts for L2 Metadata across nested dicts."""
        if not isinstance(payload, dict): return {}
        if "metadata" in payload: return payload["metadata"]
        for v in payload.values():
            if isinstance(v, dict) and "metadata" in v:
                return v["metadata"]
        return {}

    def _generate_trace(self, payload: Any, start_time: float, prefix: str) -> PipelineTrace:
        try:
            input_str = json.dumps(payload, sort_keys=True, default=str) if isinstance(payload, dict) else "{}"
            input_hash = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
        except Exception:
            input_hash = hashlib.sha256(str(payload).encode('utf-8')).hexdigest()

        meta = self._extract_meta(payload)

        return PipelineTrace(
            input_hash=input_hash,
            processing_id=f"{prefix}-{input_hash[:12]}",
            pipeline_stage=self.config.stage,
            execution_time_ms=round((time.perf_counter() - start_time) * 1000, 4),
            analyzer_version=meta.get("analyzer_version", "UNKNOWN"),
            analyzer_timestamp=meta.get("timestamp", "UNKNOWN"),
            analyzer_hash=meta.get("analyzer_hash", "UNKNOWN")
        )

    def _determine_rating(self, score: float) -> str:
        for threshold, rating_str in self.RATINGS:
            if score >= threshold: return rating_str
        return "Very Weak"

    def _normalize(self, value: float) -> float:
        try:
            if math.isnan(value) or math.isinf(value): return 50.0
            return max(0.0, min(100.0, float(value)))
        except: return 50.0

    def _sanitize_json(self, obj: Any) -> Any:
        if isinstance(obj, dict): return {str(k): self._sanitize_json(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [self._sanitize_json(v) for v in obj]
        elif isinstance(obj, float): return None if math.isnan(obj) or math.isinf(obj) else obj
        elif isinstance(obj, (int, str, bool, type(None))): return obj
        return str(obj)

    def _safe_div(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        try:
            return float(numerator) / float(denominator) if denominator != 0 else default
        except:
            return default

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        items: List[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key.lower(), v))
        return dict(items)

    def _extract_metric(self, flat_data: Dict[str, Any], target_keywords: List[str], default: float) -> float:
        extracted = []
        for key, value in flat_data.items():
            if any(keyword in key for keyword in target_keywords):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    try:
                        if not math.isnan(value) and not math.isinf(value):
                            val = float(value) * 100.0 if 0.0 < float(value) <= 1.0 else float(value)
                            extracted.append(val)
                    except: continue
        if not extracted: return default
        return self._normalize(sum(extracted) / len(extracted))

    def _contains_keyword(self, flat_data: Dict[str, Any], key_targets: List[str], val_targets: List[str]) -> bool:
        for key, value in flat_data.items():
            if any(k in key for k in key_targets):
                if isinstance(value, str) and any(v in value.lower() for v in val_targets): return True
                if isinstance(value, bool):
                    if value is True and any(v in ["true", "yes", "high"] for v in val_targets): return True
                    if value is False and any(v in ["false", "no", "low"] for v in val_targets): return True
        return False

    def _extract_upstream_text(self, flat_data: Dict[str, Any]) -> None:
        for key, value in flat_data.items():
            if isinstance(value, str) and len(value) > 5:
                if any(kw in key for kw in ['reason', 'explanation', 'evidence', 'context', 'summary', 'warning']):
                    msg = f"L2 Context: {value}"
                    if msg not in self._evidence_log:
                        self._evidence_log.append(msg)

    def _compute_component(self, flat_data: Dict[str, Any], keys: List[str], pos_kw: str, neg_kw: str, name: str, weight: float = 0.0) -> ScoreBreakdown:
        """Computes score breakdown with mutually exclusive logic and optional weighting."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0

        pos_targets = [pos_kw, "true", "yes", "high", "strong", "bullish", "aligned", "excellent", "wide", "healthy", "efficient", "stable", "cleared", "increasing", "attractive", "huge", "accumulation", "consistent"]
        neg_targets = [neg_kw, "false", "no", "low", "weak", "bearish", "divergent", "choppy", "poor", "none", "declining", "inefficient", "erratic", "rejected", "decreasing", "expensive", "negative", "volatile", "distribution", "unstable"]

        is_pos = self._contains_keyword(flat_data, keys, pos_targets)
        is_neg = self._contains_keyword(flat_data, keys, neg_targets)

        # FIX 2 & 5: STRICT MUTUAL EXCLUSION AND DEDUPLICATION
        if is_pos and is_neg:
            self._conflicts += 1
            msg = f"Conflict: Contradictory signals detected for {name}."
            if msg not in self._warning_log:
                self._warning_log.append(msg)
        elif is_pos:
            bonus = 10.0
            msg = f"Strong/Positive {name} validated."
            if msg not in self._positive_log:
                self._positive_log.append(msg)
        elif is_neg:
            penalty = 12.0
            msg = f"Weak/Negative {name} detected."
            if msg not in self._negative_log:
                self._negative_log.append(msg)

        final = self._normalize(raw + bonus - penalty)
        weighted_score = round(final * weight, 2) if weight > 0 else 0.0

        return ScoreBreakdown(round(raw, 2), round(raw, 2), weighted_score, round(penalty, 2), round(bonus, 2), round(final, 2))

    def _compute_inverse_component(self, flat_data: Dict[str, Any], keys: List[str], pos_kw: str, neg_kw: str, name: str, weight: float = 0.0) -> ScoreBreakdown:
        raw_risk = self._extract_metric(flat_data, keys, 20.0)
        inverted_raw = self._normalize(100.0 - raw_risk)
        bonus, penalty = 0.0, 0.0

        pos_targets = [pos_kw, "false", "no", "low", "none", "attractive", "cheap", "safe"]
        neg_targets = [neg_kw, "true", "yes", "high", "extreme", "severe", "expensive", "danger", "fade", "heavy"]

        is_pos = self._contains_keyword(flat_data, keys, pos_targets)
        is_neg = self._contains_keyword(flat_data, keys, neg_targets)

        # FIX 2 & 5: STRICT MUTUAL EXCLUSION AND DEDUPLICATION
        if is_pos and is_neg:
            self._conflicts += 1
            msg = f"Conflict: Contradictory risk signals detected for {name}."
            if msg not in self._warning_log:
                self._warning_log.append(msg)
        elif is_pos:
            bonus = 10.0
            msg = f"Favorable/Low risk state for {name} validated."
            if msg not in self._positive_log:
                self._positive_log.append(msg)
        elif is_neg:
            penalty = 15.0
            msg = f"High risk/Negative state for {name} detected."
            if msg not in self._negative_log:
                self._negative_log.append(msg)

        final = self._normalize(inverted_raw + bonus - penalty)
        weighted_score = round(final * weight, 2) if weight > 0 else 0.0

        return ScoreBreakdown(round(inverted_raw, 2), round(inverted_raw, 2), weighted_score, round(penalty, 2), round(bonus, 2), round(final, 2))

