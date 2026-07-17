"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: volume_engine.py

Institutional Volume Scoring Implementation inheriting from BaseEngine.
Features Adaptive Weights and Dynamic Evidence Graphs.
"""

import time
from typing import Any
from dataclasses import dataclass, asdict
from backend.engines.base_engine import BaseEngine, EngineConfig, EvidenceGraph, OutputStatus

# =====================================================================
# ENGINE PROFILE (Configurable & Swappable)
# =====================================================================
def get_institutional_volume_profile() -> EngineConfig:
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        base_weights={
            "quality": 0.20,
            "confirmation": 0.20,
            "institutional": 0.35, # Highly skewed for institutional profile
            "liquidity": 0.25
        },
        thresholds={"conflict_penalty": 15.0, "divergence_penalty": 10.0}
    )

@dataclass(frozen=True)
class ScoreBreakdown:
    raw_score: float
    normalized_score: float
    weighted_score: float
    penalty: float
    bonus: float
    final_score: float

# =====================================================================
# VOLUME ENGINE (Inherits BaseEngine)
# =====================================================================
class VolumeEngine(BaseEngine):
    
    def __init__(self, config: EngineConfig = None):
        super().__init__(config or get_institutional_volume_profile())

    def calculate(self, volume_json: dict[str, Any] | None) -> dict[str, Any]:
        start_time = time.perf_counter()
        trace = self._generate_trace(volume_json, start_time, "VOL")

        if not isinstance(volume_json, dict) or not volume_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            flat_data = self._flatten_dict(volume_json)
            
            # 1. Component Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty"], 50.0)
            q_comp = self._compute_component(flat_data, ["quality", "delivery"], "high", "low", "delivery")
            c_comp = self._compute_component(flat_data, ["confirmation", "trend"], "detected", "divergence", "confirmation")
            i_comp = self._compute_component(flat_data, ["institutional", "smart_money"], "inflow", "outflow", "institutional footprint")
            l_comp = self._compute_component(flat_data, ["liquidity", "turnover"], "high", "dry", "liquidity")
            
            acc = self._extract_metric(flat_data, ["accumulation"], 50.0)
            dist = self._extract_metric(flat_data, ["distribution"], 20.0)

            # 2. Conflict Detection & Cross Validation
            if i_comp.final_score > 80 and l_comp.final_score < 40:
                self._conflicts += 1
                self._score_reasons.append("Conflict: High institutional activity but low liquidity (Trap Risk).")

            # 3. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 20.0) * 100) # Assuming 20 expected fields
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=coverage
            )

            # 4. Adaptive Weight Engine
            # Redistribute weight dynamically if Liquidity is dangerously low
            dynamic_weights = dict(self.config.base_weights)
            if l_comp.final_score < 30:
                self._score_reasons.append("Adaptive Shift: Liquidity weight increased due to dry environment.")
                dynamic_weights["liquidity"] += 0.15
                dynamic_weights["institutional"] -= 0.15

            # 5. Calculate Weighted Final Score
            acc_dist_bias = self._normalize(50.0 + (acc - dist) / 2.0)
            base_score = (
                (q_comp.final_score * dynamic_weights["quality"]) +
                (c_comp.final_score * dynamic_weights["confirmation"]) +
                (i_comp.final_score * dynamic_weights["institutional"]) +
                (l_comp.final_score * dynamic_weights["liquidity"])
            )

            # 6. Final Confidence Calculation (The Hybrid Model)
            final_confidence = self._normalize(
                (analyzer_confidence * 0.4) + 
                (evidence_graph.evidence_coverage * 0.3) + 
                (min(total_evidence * 10, 100) * 0.3) - 
                (self._conflicts * 20)
            )

            final_volume_score = self._normalize(base_score * (0.5 + (final_confidence / 200.0)))

            # Build Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall": round(final_volume_score, 2),
                    "rating": self._determine_rating(final_volume_score),
                    "confidence": round(final_confidence, 2),
                    "accumulation_bias": round(acc_dist_bias, 2)
                },
                "components": {
                    "quality": asdict(q_comp),
                    "confirmation": asdict(c_comp),
                    "institutional": asdict(i_comp),
                    "liquidity": asdict(l_comp)
                },
                "evidence_graph": asdict(evidence_graph),
                "explanations": {
                    "reasons": sorted(list(set(self._score_reasons))),
                    "positive_signals": sorted(list(set(self._positive_log))),
                    "negative_signals": sorted(list(set(self._negative_log)))
                },
                "trace": asdict(trace),
                "engine_signature": {
                    "profile": self.config.profile_name,
                    "engine_version": self.config.version,
                    "schema_version": self.config.schema_version,
                    "api_version": self.config.api_version,
                    "adaptive_weights_applied": dynamic_weights
                }
            }

            return self._sanitize_json(result)

        except Exception as e:
            return self._sanitize_json(self._build_fallback(trace))

    # ---------------------------------------------------------
    # COMPONENT BUILDER (Internal)
    # ---------------------------------------------------------
    def _compute_component(self, flat_data: dict, keys: list, pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw]):
            bonus = 10.0
            self._positive_log.append(f"Strong {name} validated.")
        if self._contains_keyword(flat_data, keys, [neg_kw]):
            penalty = 15.0
            self._negative_log.append(f"Weak {name} detected.")
            
        final = self._normalize(raw + bonus - penalty)
        return ScoreBreakdown(raw, raw, 0.0, penalty, bonus, final)

    def _build_fallback(self, trace: PipelineTrace) -> dict:
        return {
            "status": asdict(OutputStatus(status="FAILED", quality="INVALID")),
            "scores": {"overall": 50.0, "rating": "Neutral", "confidence": 0.0},
            "explanations": {"reasons": ["Fatal execution error. Defaulted to neutral."]},
            "trace": asdict(trace)
        }

# =====================================================================
# PUBLIC API
# =====================================================================
def calculate_volume_score(volume_json: dict[str, Any] | None) -> dict[str, Any]:
    return VolumeEngine().calculate(volume_json)
