"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine Base Framework
Module: base_engine.py

Provides the core abstract architecture for all Layer-3 Scoring Engines.
Handles determinism, metadata, trace generation, sanitation, and core utilities.
"""

import math
import time
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# =====================================================================
# MASTER DATA CLASSES
# =====================================================================
@dataclass
class EngineConfig:
    profile_name: str
    version: str
    stage: str = "Layer-3: Scoring"
    schema_version: str = "1.0"
    api_version: str = "v6"
    scoring_method: str = "Adaptive Evidence-Weighted"
    normalization_method: str = "Min-Max Clamp (0-100)"
    base_weights: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

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


# =====================================================================
# BASE ENGINE CLASS
# =====================================================================
class BaseEngine(ABC):
    """Abstract Base Class for all Layer-3 Scoring Engines."""

    RATINGS = (
        (95.0, "Elite"), (90.0, "Exceptional"), (80.0, "Very Strong"),
        (70.0, "Strong"), (60.0, "Good"), (50.0, "Neutral"),
        (35.0, "Weak"), (20.0, "Poor"), (0.0, "Very Weak")
    )

    def __init__(self, config: EngineConfig):
        self.config = config
        self._evidence_log: list[str] = []
        self._warning_log: list[str] = []
        self._positive_log: list[str] = []
        self._negative_log: list[str] = []
        self._score_reasons: list[str] = []
        self._conflicts = 0

    @abstractmethod
    def calculate(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        """Must be implemented by child classes (Volume, Trend, etc.)."""
        pass

    # ---------------------------------------------------------
    # UTILITIES & PIPELINE TRACING
    # ---------------------------------------------------------
    def _generate_trace(self, payload: Any, start_time: float, prefix: str) -> PipelineTrace:
        input_str = json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else "{}"
        input_hash = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
        
        # Extract Analyzer Upstream Metadata
        meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        
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

    # ---------------------------------------------------------
    # INTELLIGENCE EXTRACTORS
    # ---------------------------------------------------------
    def _flatten_dict(self, d: dict[str, Any], parent_key: str = '', sep: str = '_') -> dict[str, Any]:
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key.lower(), v))
        return dict(items)

    def _extract_metric(self, flat_data: dict[str, Any], target_keywords: list[str], default: float) -> float:
        extracted = []
        for key, value in flat_data.items():
            if any(keyword in key for keyword in target_keywords):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isnan(value) and not math.isinf(value):
                        val = float(value) * 100.0 if 0.0 < float(value) <= 1.0 else float(value)
                        extracted.append(val)
        if not extracted: return default
        return self._normalize(sum(extracted) / len(extracted))

    def _contains_keyword(self, flat_data: dict[str, Any], key_targets: list[str], val_targets: list[str]) -> bool:
        for key, value in flat_data.items():
            if any(k in key for k in key_targets):
                if isinstance(value, str) and any(v in value.lower() for v in val_targets): return True
                if isinstance(value, bool):
                    if value is True and any(v in ["true", "yes", "high"] for v in val_targets): return True
                    if value is False and any(v in ["false", "no", "low"] for v in val_targets): return True
        return False
