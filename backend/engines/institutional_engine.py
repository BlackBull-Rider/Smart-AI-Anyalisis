"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: institutional_engine.py

Institutional Activity Scoring Implementation.
Inherits from BaseEngine. Converts Institutional Activity analyzer intelligence
(FII/DII, Promoters, Delivery, Block Deals) into deterministic, institutional-grade scores
using adaptive weights and evidence graphs.
"""

import time
from typing import Any
from dataclasses import dataclass, asdict

from backend.engines.base_engine import (
    BaseEngine,
    EngineConfig,
    EvidenceGraph,
    OutputStatus,
    PipelineTrace
)


# =====================================================================
# ENGINE PROFILE (Configurable & Swappable)
# =====================================================================
def get_institutional_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Institutional Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Adaptive Evidence-Weighted",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "institutional_activity": 0.30,
            "promoter_activity": 0.20,
            "delivery": 0.15,
            "accumulation": 0.15,
            "distribution": 0.10,
            "shareholding_quality": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "delivery_penalty": 10.0,
            "ownership_penalty": 12.0
        }
    )


@dataclass(frozen=True)
class ScoreBreakdown:
    """Immutable sub-component score details."""
    raw_score: float
    normalized_score: float
    weighted_score: float
    penalty: float
    bonus: float
    final_score: float


# =====================================================================
# INSTITUTIONAL ENGINE (Inherits BaseEngine)
# =====================================================================
class InstitutionalEngine(BaseEngine):
    """
    Institutional Activity Scoring Engine.
    Transforms Layer-2 Institutional Analyzer JSON into deterministic Layer-3 scores.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_institutional_profile())

    def calculate(self, institutional_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            institutional_json: The dictionary payload from Layer-2 Institutional Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(institutional_json, start_time, "INST")

        if not isinstance(institutional_json, dict) or not institutional_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(institutional_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Derived primary metrics for granularity
            inst_buying_score = self._extract_metric(flat_data, ["institutional_buying", "fii_buying", "dii_buying"], 50.0)
            inst_selling_score = self._extract_metric(flat_data, ["institutional_selling", "fii_selling", "dii_selling"], 50.0)
            promoter_conf_score = self._extract_metric(flat_data, ["promoter_confidence", "insider_confidence"], 50.0)
            ownership_stability = self._extract_metric(flat_data, ["ownership_stability", "concentration", "retention"], 50.0)
            dist_raw_score = self._extract_metric(flat_data, ["distribution", "selling_pressure", "dumping"], 20.0)
            acc_raw_score = self._extract_metric(flat_data, ["accumulation", "buying_pressure", "inflow"], 50.0)
            
            # 2. Component Extractions
            inst_comp = self._compute_component(
                flat_data, ["institutional", "fii", "dii", "mutual_fund", "insurance"], "buying", "selling", "Institutional Activity"
            )
            promoter_comp = self._compute_component(
                flat_data, ["promoter", "insider"], "buying", "selling", "Promoter Activity"
            )
            delivery_comp = self._compute_component(
                flat_data, ["delivery", "block", "bulk"], "high", "low", "Delivery Quality"
            )
            acc_comp = self._compute_component(
                flat_data, ["accumulation"], "strong", "weak", "Accumulation"
            )
            # Distribution is inverted for component scoring (high distribution = low safety score)
            dist_comp = self._compute_inverse_component(
                flat_data, ["distribution", "dumping"], "weak", "strong", "Distribution Safety"
            )
            shareholding_comp = self._compute_component(
                flat_data, ["shareholding", "ownership", "float"], "stable", "unstable", "Shareholding Quality"
            )

            # 3. Conflict Detection & Cross Validation
            self._detect_conflicts(
                flat_data, acc_raw_score, dist_raw_score, inst_buying_score, inst_selling_score, 
                promoter_comp.final_score, delivery_comp.final_score, shareholding_comp.final_score
            )

            # 4. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 30.0) * 100.0) # Assuming ~30 expected data points
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 5. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(
                inst_comp.final_score, promoter_comp.final_score, delivery_comp.final_score, 
                acc_comp.final_score, shareholding_comp.final_score
            )

            # 6. Calculate Weighted Final Score
            base_score = (
                (inst_comp.final_score * dynamic_weights["institutional_activity"]) +
                (promoter_comp.final_score * dynamic_weights["promoter_activity"]) +
                (delivery_comp.final_score * dynamic_weights["delivery"]) +
                (acc_comp.final_score * dynamic_weights["accumulation"]) +
                (dist_comp.final_score * dynamic_weights["distribution"]) +
                (shareholding_comp.final_score * dynamic_weights["shareholding_quality"])
            )

            # Apply general structural penalties for ownership collapse
            ownership_penalty = 0.0
            if shareholding_comp.final_score < 30 and promoter_comp.final_score < 30:
                ownership_penalty = self.config.thresholds.get("ownership_penalty", 12.0)
                self._score_reasons.append(f"Applied penalty of {ownership_penalty} due to collapsing ownership structure.")

            # 7. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            
            # Confidence utilizes ownership stability as a core anchor
            component_agreement = 100.0 - abs(inst_comp.final_score - promoter_comp.final_score)
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.25) + 
                (evidence_graph.evidence_coverage * 0.20) + 
                (min(total_evidence * 10, 100) * 0.15) +
                (ownership_stability * 0.20) +
                (component_agreement * 0.20) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_institutional_score = self._normalize((base_score * (0.5 + (final_confidence / 200.0))) - ownership_penalty)
            
            # Synthesize contextual explanations based on processed metrics
            self._generate_explanations(
                acc_comp.final_score, promoter_comp.final_score, delivery_comp.final_score, 
                dist_raw_score, shareholding_comp.final_score
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 8. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_institutional_score": round(final_institutional_score, 2),
                    "institutional_rating": self._determine_rating(final_institutional_score),
                    "confidence": round(final_confidence, 2),
                    "accumulation_score": round(acc_raw_score, 2),
                    "distribution_score": round(dist_raw_score, 2),
                    "institutional_buying_score": round(inst_buying_score, 2),
                    "institutional_selling_score": round(inst_selling_score, 2),
                    "delivery_quality_score": round(delivery_comp.final_score, 2),
                    "promoter_confidence_score": round(promoter_conf_score, 2),
                    "shareholding_quality_score": round(shareholding_comp.final_score, 2),
                    "ownership_stability_score": round(ownership_stability, 2),
                    "institutional_reliability": round(reliability, 2)
                },
                "components": {
                    "institutional_activity": asdict(inst_comp),
                    "promoter_activity": asdict(promoter_comp),
                    "delivery": asdict(delivery_comp),
                    "accumulation": asdict(acc_comp),
                    "distribution": asdict(dist_comp),
                    "shareholding_quality": asdict(shareholding_comp)
                },
                "evidence_graph": asdict(evidence_graph),
                "explanations": {
                    "reasons": sorted(list(set(self._score_reasons))),
                    "positive_signals": sorted(list(set(self._positive_log))),
                    "negative_signals": sorted(list(set(self._negative_log))),
                    "warnings": sorted(list(set(self._warning_log))),
                    "evidence": sorted(list(set(self._evidence_log)))
                },
                "trace": asdict(trace),
                "engine_signature": {
                    "profile": self.config.profile_name,
                    "engine_version": self.config.version,
                    "schema_version": self.config.schema_version,
                    "api_version": self.config.api_version,
                    "adaptive_weights_applied": {k: round(v, 4) for k, v in dynamic_weights.items()}
                }
            }

            return self._sanitize_json(result)

        except Exception as e:
            # Silent fallback generation on fatal logic crash to protect pipeline
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, flat_data: dict[str, Any], acc: float, dist: float, inst_buy: float, 
                          inst_sell: float, prom: float, delivery: float, shareholding: float) -> None:
        """Evaluates logical paradoxes in Institutional states."""
        
        # 1. Strong Accumulation + Strong Distribution
        if acc > 75 and dist > 75:
            self._conflicts += 1
            warn = "Conflict: Simultaneous strong accumulation and distribution metrics (Chaotic turnover)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Promoter Buying + Heavy Institutional Selling
        if prom > 75 and inst_sell > 75:
            self._conflicts += 1
            warn = "Conflict: Promoters are buying into heavy institutional distribution."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. High Delivery + Heavy Distribution
        if delivery > 75 and dist > 75:
            self._conflicts += 1
            warn = "Conflict: High delivery percentages correspond with severe distribution (Strong dumping)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Strong Institutional Buying + Promoter Selling
        if inst_buy > 75 and prom < 25:
            self._conflicts += 1
            warn = "Conflict: Institutional accumulation occurring while Promoters are selling off."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Rising/Stable Shareholding + Large Insider Selling
        if shareholding > 75 and self._contains_keyword(flat_data, ["insider", "promoter"], ["large_sell", "dumping", "heavy_sell"]):
            self._conflicts += 1
            warn = "Conflict: Supposed stable ownership structure contradicts large insider selling events."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, inst_score: float, prom_score: float, delivery_score: float, 
                                    acc_score: float, shareholding_score: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on current state severity."""
        weights = dict(self.config.base_weights)

        # Shift 1: Strong Institutional Activity dominates sentiment
        if inst_score > 80:
            self._score_reasons.append("Adaptive Shift: Strong Institutional Buying detected, increasing Institutional weight.")
            weights["institutional_activity"] += 0.10
            weights["distribution"] -= 0.10

        # Shift 2: Strong Accumulation signals robust buying
        if acc_score > 80:
            self._score_reasons.append("Adaptive Shift: Strong accumulation detected, scaling Accumulation weight.")
            weights["accumulation"] += 0.10
            weights["delivery"] -= 0.05
            weights["promoter_activity"] -= 0.05

        # Shift 3: High Delivery confirms genuine holding
        if delivery_score > 80:
            self._score_reasons.append("Adaptive Shift: Exceptional delivery percentages detected, increasing Delivery weight.")
            weights["delivery"] += 0.10
            weights["institutional_activity"] -= 0.10

        # Shift 4: Active Promoter Engagement
        if prom_score > 85:
            self._score_reasons.append("Adaptive Shift: Heavy Promoter buying identified, scaling Promoter weight.")
            weights["promoter_activity"] += 0.15
            weights["shareholding_quality"] -= 0.05
            weights["distribution"] -= 0.10
            
        # Shift 5: Rock-solid Shareholding Stability acts as an anchor
        if shareholding_score > 85:
            self._score_reasons.append("Adaptive Shift: Extremely stable ownership structure, increasing Shareholding weight.")
            weights["shareholding_quality"] += 0.10
            weights["institutional_activity"] -= 0.10

        # Safety clamp before normalization
        for key in weights:
            weights[key] = max(0.0, weights[key])

        # Normalize weights to exactly 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = self.config.base_weights

        return weights

    def _generate_explanations(self, acc: float, prom: float, delivery: float, dist: float, ownership: float) -> None:
        """Synthesizes human-readable logic for final execution state."""
        if acc > 75:
            self._score_reasons.append("Strong institutional accumulation detected.")
        if prom > 75:
            self._score_reasons.append("Promoters continue increasing ownership.")
        elif prom < 30:
            self._score_reasons.append("Promoter selling reduces conviction.")
            
        if delivery > 75 and acc > 70:
            self._score_reasons.append("Delivery percentage confirms genuine accumulation.")
            
        if dist > 75:
            self._score_reasons.append("Heavy institutional distribution detected.")
            
        if ownership > 80:
            self._score_reasons.append("Ownership structure remains highly stable.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Conflicting institutional signals reduce overall confidence.")


    # ---------------------------------------------------------
    # COMPONENT BUILDER (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        # Bullish / Positive Context
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "increase"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        # Bearish / Negative Context
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "decrease"]):
            penalty = 12.0
            self._negative_log.append(f"Weak/Negative {name} detected.")
            
        final = self._normalize(raw + bonus - penalty)
        
        return ScoreBreakdown(
            raw_score=round(raw, 2),
            normalized_score=round(raw, 2),
            weighted_score=0.0, 
            penalty=round(penalty, 2),
            bonus=round(bonus, 2),
            final_score=round(final, 2)
        )
        
    def _compute_inverse_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """
        Extracts and evaluates an inversely correlated component (e.g., Distribution).
        High raw input means High Risk. Component final score maps High Risk to Low Score (Safe = 100).
        """
        raw_risk = self._extract_metric(flat_data, keys, 20.0)
        inverted_raw = self._normalize(100.0 - raw_risk)
        
        bonus, penalty = 0.0, 0.0
        
        # If it contains "weak" (pos_kw), risk is low, safety gets a bonus
        if self._contains_keyword(flat_data, keys, [pos_kw, "false", "no", "low"]):
            bonus = 10.0
            self._positive_log.append(f"Low risk regarding {name} validated.")
            
        # If it contains "strong" (neg_kw), risk is high, safety takes a penalty
        if self._contains_keyword(flat_data, keys, [neg_kw, "true", "yes", "high", "heavy"]):
            penalty = 15.0
            self._negative_log.append(f"High risk regarding {name} detected.")
            
        final = self._normalize(inverted_raw + bonus - penalty)
        
        return ScoreBreakdown(
            raw_score=round(inverted_raw, 2),
            normalized_score=round(inverted_raw, 2),
            weighted_score=0.0, 
            penalty=round(penalty, 2),
            bonus=round(bonus, 2),
            final_score=round(final, 2)
        )

    def _build_fallback(self, trace: PipelineTrace) -> dict[str, Any]:
        """Provides a safe, deterministic failover state."""
        return {
            "status": asdict(OutputStatus(status="FAILED", quality="INVALID")),
            "scores": {
                "overall_institutional_score": 50.0, 
                "institutional_rating": "Neutral", 
                "confidence": 0.0,
                "accumulation_score": 50.0,
                "distribution_score": 50.0,
                "institutional_buying_score": 50.0,
                "institutional_selling_score": 50.0,
                "delivery_quality_score": 50.0,
                "promoter_confidence_score": 50.0,
                "shareholding_quality_score": 50.0,
                "ownership_stability_score": 50.0,
                "institutional_reliability": 0.0
            },
            "components": {},
            "evidence_graph": asdict(EvidenceGraph()),
            "explanations": {"reasons": ["Fatal execution error. Defaulted to neutral state."]},
            "trace": asdict(trace),
            "engine_signature": {
                "profile": self.config.profile_name,
                "engine_version": self.config.version,
                "schema_version": self.config.schema_version,
                "api_version": self.config.api_version
            }
        }


# =====================================================================
# PUBLIC API
# =====================================================================
def calculate_institutional_score(institutional_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional investment score.
    
    Args:
        institutional_json: The dictionary payload from Layer-2 Institutional Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = InstitutionalEngine()
    return engine.calculate(institutional_json)
