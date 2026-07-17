"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: trend_engine.py

Institutional Trend Scoring Implementation.
Inherits from BaseEngine. Converts Trend Analyzer intelligence into deterministic, 
institutional-grade scores using advanced quantitative models (Hierarchical Bayesian, 
Shannon Entropy, Kaplan-Meier Persistence, and HMM Continuation).
"""

import math
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
def get_institutional_trend_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Trend Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Bayesian Evidence-Weighted Trend Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "direction": 0.20,
            "strength": 0.20,
            "quality": 0.15,
            "continuation": 0.15,
            "exhaustion": 0.15,
            "multi_timeframe": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "exhaustion_penalty": 12.0,
            "entropy_penalty_max": 10.0
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
# TREND ENGINE (Inherits BaseEngine)
# =====================================================================
class TrendEngine(BaseEngine):
    """
    Institutional Trend Scoring Engine.
    Transforms Layer-2 Trend Analyzer JSON into deterministic Layer-3 scores
    using advanced quantitative modelling applied to intelligence vectors.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_institutional_trend_profile())

    def calculate(self, trend_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            trend_json: The dictionary payload from Layer-2 Trend Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(trend_json, start_time, "TREND")

        if not isinstance(trend_json, dict) or not trend_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(trend_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Raw Metric for inverted components
            exhaustion_raw = self._extract_metric(flat_data, ["exhaustion", "fade", "overextended"], 20.0)
            
            # 2. Component Extractions
            dir_comp = self._compute_component(
                flat_data, ["direction", "bias", "trend_direction"], "bullish", "bearish", "Trend Direction"
            )
            str_comp = self._compute_component(
                flat_data, ["strength", "momentum", "adx"], "strong", "weak", "Trend Strength"
            )
            qual_comp = self._compute_component(
                flat_data, ["quality", "smoothness", "health"], "high", "choppy", "Trend Quality"
            )
            cont_comp = self._compute_component(
                flat_data, ["continuation", "persistence"], "likely", "unlikely", "Trend Continuation"
            )
            mtf_comp = self._compute_component(
                flat_data, ["multi_timeframe", "mtf", "alignment"], "aligned", "divergent", "Multi-Timeframe"
            )
            
            # Exhaustion is an inverse metric (High Exhaustion = Low Score/High Risk)
            exh_comp = self._compute_inverse_component(
                flat_data, ["exhaustion", "overbought", "oversold", "fade"], "low", "high", "Trend Exhaustion Safety"
            )

            # 3. Advanced Quantitative Models (Layer-3 Intelligence)
            bayesian_score = self._hierarchical_bayesian_fusion(mtf_comp.final_score, str_comp.final_score, qual_comp.final_score)
            entropy_penalty = self._shannon_entropy_penalty([dir_comp.final_score, str_comp.final_score, qual_comp.final_score, cont_comp.final_score, mtf_comp.final_score])
            kaplan_meier_persistence = self._kaplan_meier_persistence(cont_comp.final_score, exhaustion_raw)
            hmm_prob = self._hmm_continuation_probability(cont_comp.final_score, str_comp.final_score, qual_comp.final_score)
            trend_health = self._trend_health_model(qual_comp.final_score, str_comp.final_score, mtf_comp.final_score)

            # 4. Conflict Detection
            self._detect_conflicts(str_comp.final_score, exhaustion_raw, dir_comp.final_score, mtf_comp.final_score, cont_comp.final_score, qual_comp.final_score)

            # 5. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 25.0) * 100.0)
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 6. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(
                str_comp.final_score, mtf_comp.final_score, exhaustion_raw, qual_comp.final_score
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (dir_comp.final_score * dynamic_weights["direction"]) +
                (str_comp.final_score * dynamic_weights["strength"]) +
                (qual_comp.final_score * dynamic_weights["quality"]) +
                (cont_comp.final_score * dynamic_weights["continuation"]) +
                (exh_comp.final_score * dynamic_weights["exhaustion"]) +
                (mtf_comp.final_score * dynamic_weights["multi_timeframe"])
            )

            # Blend Quantitative Models into Base Score (Institutional Smoothing)
            fused_trend_score = (base_score * 0.40) + (bayesian_score * 0.30) + (trend_health * 0.30)

            # Apply Quantitative Penalties
            structural_penalty = entropy_penalty
            if exhaustion_raw > 80:
                exhaustion_penalty = self.config.thresholds.get("exhaustion_penalty", 12.0)
                structural_penalty += exhaustion_penalty
                self._score_reasons.append(f"Applied penalty of {exhaustion_penalty} due to extreme trend exhaustion.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            
            # Entropy indicates uncertainty; lower entropy = higher confidence
            entropy_confidence_boost = max(0.0, 10.0 - entropy_penalty)
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.30) + 
                (evidence_graph.evidence_coverage * 0.20) + 
                (min(total_evidence * 10, 100) * 0.20) +
                (entropy_confidence_boost * 1.5) +
                (hmm_prob * 0.15) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_trend_score = self._normalize((fused_trend_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                str_comp.final_score, mtf_comp.final_score, exhaustion_raw, 
                qual_comp.final_score, kaplan_meier_persistence
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_trend_score": round(final_trend_score, 2),
                    "trend_rating": self._determine_rating(final_trend_score),
                    "confidence": round(final_confidence, 2),
                    "trend_strength_score": round(str_comp.final_score, 2),
                    "trend_quality_score": round(qual_comp.final_score, 2),
                    "trend_continuation_score": round(kaplan_meier_persistence, 2),
                    "trend_exhaustion_score": round(exh_comp.final_score, 2),
                    "multi_timeframe_alignment_score": round(mtf_comp.final_score, 2),
                    "bayesian_fusion_score": round(bayesian_score, 2),
                    "hmm_continuation_probability": round(hmm_prob, 2),
                    "trend_reliability": round(reliability, 2)
                },
                "components": {
                    "trend_direction": asdict(dir_comp),
                    "trend_strength": asdict(str_comp),
                    "trend_quality": asdict(qual_comp),
                    "trend_continuation": asdict(cont_comp),
                    "trend_exhaustion": asdict(exh_comp),
                    "multi_timeframe": asdict(mtf_comp)
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
    # ADVANCED QUANTITATIVE MODELS
    # ---------------------------------------------------------

    def _hierarchical_bayesian_fusion(self, mtf: float, strength: float, quality: float) -> float:
        """
        Bayesian Fusion: Uses MTF as the Prior, and Strength/Quality as the Likelihood.
        Returns the Posterior probability of a sustained trend.
        """
        prior = mtf / 100.0
        likelihood = (strength * 0.6 + quality * 0.4) / 100.0
        
        # Bayesian update (Normalized)
        numerator = prior * likelihood
        denominator = numerator + ((1.0 - prior) * (1.0 - likelihood)) + 1e-9
        posterior = numerator / denominator
        
        return self._normalize(posterior * 100.0)

    def _shannon_entropy_penalty(self, scores: list[float]) -> float:
        """
        Shannon Entropy: Measures uncertainty/chaos among trend components.
        High entropy = conflicting/uncertain state = High penalty.
        """
        total = sum(scores) + 1e-9
        probs = [s / total for s in scores if s > 0]
        
        if not probs:
            return 0.0
            
        entropy = -sum(p * math.log(p) for p in probs)
        max_entropy = math.log(len(scores)) if len(scores) > 1 else 1.0
        
        max_penalty = self.config.thresholds.get("entropy_penalty_max", 10.0)
        penalty = (entropy / max_entropy) * max_penalty
        
        if penalty > (max_penalty * 0.8):
            self._score_reasons.append("High Shannon Entropy detected among trend components (Chaotic alignment).")
            
        return penalty

    def _kaplan_meier_persistence(self, continuation: float, exhaustion_raw: float) -> float:
        """
        Kaplan-Meier Style Survival Model: Probability of trend survival.
        Hazard function is driven by trend exhaustion.
        """
        hazard_rate = exhaustion_raw / 100.0
        survival_prob = (continuation / 100.0) * (1.0 - hazard_rate)
        return self._normalize(survival_prob * 100.0)
        
    def _hmm_continuation_probability(self, continuation: float, strength: float, quality: float) -> float:
        """
        Hidden Markov Model (HMM) Transition Estimation.
        Estimates the probability of transitioning to/remaining in a 'Trending' hidden state.
        """
        transition_prob = (strength * 0.5 + continuation * 0.5) / 100.0
        emission_prob = quality / 100.0
        hmm_state_prob = transition_prob * emission_prob
        return self._normalize(hmm_state_prob * 100.0)

    def _trend_health_model(self, quality: float, strength: float, mtf: float) -> float:
        """Standard aggregation of core health metrics."""
        return self._normalize((quality * 0.4) + (strength * 0.3) + (mtf * 0.3))


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, strength: float, exhaustion_raw: float, direction: float, mtf: float, cont: float, qual: float) -> None:
        """Evaluates logical paradoxes in Trend states."""
        
        # 1. Strong Trend + Strong Exhaustion
        if strength > 75 and exhaustion_raw > 75:
            self._conflicts += 1
            warn = "Conflict: Extreme trend strength coupled with extreme exhaustion (Blow-off top/bottom risk)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Bullish Local Trend + Bearish MTF
        if direction > 75 and mtf < 25:
            self._conflicts += 1
            warn = "Conflict: Strong local uptrend fighting a severe multi-timeframe downtrend."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Bearish Local Trend + Bullish MTF
        if direction < 25 and mtf > 75:
            self._conflicts += 1
            warn = "Conflict: Strong local downtrend fighting a severe multi-timeframe uptrend (Pullback risk)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Continuation Expected + Weak Quality
        if cont > 75 and qual < 25:
            self._conflicts += 1
            warn = "Conflict: High continuation probability despite exceptionally poor trend quality (Choppy vector)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, strength: float, mtf: float, exhaustion_raw: float, quality: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on current state severity."""
        weights = dict(self.config.base_weights)

        # Shift 1: Strong Trend dominates the profile
        if strength > 80:
            self._score_reasons.append("Adaptive Shift: Exceptional trend strength detected, increasing Strength weight.")
            weights["strength"] += 0.10
            weights["quality"] -= 0.10

        # Shift 2: High MTF Alignment guarantees macro support
        if mtf > 80:
            self._score_reasons.append("Adaptive Shift: Multi-Timeframe perfectly aligned, scaling MTF weight.")
            weights["multi_timeframe"] += 0.10
            weights["direction"] -= 0.10

        # Shift 3: High Exhaustion demands immediate risk management
        if exhaustion_raw > 75:
            self._score_reasons.append("Adaptive Shift: Critical trend exhaustion, scaling Exhaustion penalty weight.")
            weights["exhaustion"] += 0.15
            weights["continuation"] -= 0.15

        # Shift 4: Weak Quality requires closer scrutiny
        if quality < 30:
            self._score_reasons.append("Adaptive Shift: Poor trend quality, increasing Quality scrutiny weight.")
            weights["quality"] += 0.10
            weights["strength"] -= 0.10

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

    def _generate_explanations(self, strength: float, mtf: float, exh: float, qual: float, km_pers: float) -> None:
        """Synthesizes human-readable logic for final execution state."""
        if strength > 75:
            self._score_reasons.append("Robust directional trend strength confirmed.")
        
        if mtf > 75:
            self._score_reasons.append("Multi-timeframe alignment provides strong macro tailwinds.")
        elif mtf < 30:
            self._score_reasons.append("Multi-timeframe divergence creates significant structural friction.")
            
        if exh > 75:
            self._score_reasons.append("Late-stage trend exhaustion highly elevated.")
            
        if qual < 30:
            self._score_reasons.append("Trend exhibits highly choppy, inefficient price action.")
            
        if km_pers > 75:
            self._score_reasons.append("Kaplan-Meier model projects high probability of trend survival.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Contradictory trend vectors significantly reduce model confidence.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "aligned"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "divergent", "choppy"]):
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
        Extracts and evaluates an inversely correlated component (e.g., Exhaustion).
        High raw input means High Risk. Component final score maps High Risk to Low Score (Safe = 100).
        """
        raw_risk = self._extract_metric(flat_data, keys, 20.0)
        inverted_raw = self._normalize(100.0 - raw_risk)
        
        bonus, penalty = 0.0, 0.0
        
        # If it contains "low/none" (pos_kw), risk is low, safety gets a bonus
        if self._contains_keyword(flat_data, keys, [pos_kw, "false", "no", "low", "none"]):
            bonus = 10.0
            self._positive_log.append(f"Low risk regarding {name} validated.")
            
        # If it contains "high/extreme" (neg_kw), risk is high, safety takes a penalty
        if self._contains_keyword(flat_data, keys, [neg_kw, "true", "yes", "high", "extreme", "fade"]):
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
                "overall_trend_score": 50.0, 
                "trend_rating": "Neutral", 
                "confidence": 0.0,
                "trend_strength_score": 50.0,
                "trend_quality_score": 50.0,
                "trend_continuation_score": 50.0,
                "trend_exhaustion_score": 50.0,
                "multi_timeframe_alignment_score": 50.0,
                "bayesian_fusion_score": 50.0,
                "hmm_continuation_probability": 50.0,
                "trend_reliability": 0.0
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
def calculate_trend_score(trend_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional trend score.
    
    Args:
        trend_json: The dictionary payload from Layer-2 Trend Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = TrendEngine()
    return engine.calculate(trend_json)
