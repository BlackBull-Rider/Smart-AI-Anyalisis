"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: momentum_engine.py

Institutional Momentum Scoring Implementation.
Inherits from BaseEngine. Converts Momentum Analyzer intelligence into deterministic, 
institutional-grade scores using advanced quantitative models (Hierarchical Bayesian, 
Momentum Physics, Markov Transitions, and Shannon Entropy).
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
def get_institutional_momentum_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Momentum Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Bayesian Physics-Fused Momentum Evaluation",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "strength": 0.25,
            "acceleration": 0.15,
            "slowdown": 0.15,
            "rsi": 0.15,
            "macd": 0.15,
            "divergence": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "divergence_penalty": 12.0,
            "entropy_penalty_max": 10.0,
            "collapse_penalty": 15.0
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
# MOMENTUM ENGINE (Inherits BaseEngine)
# =====================================================================
class MomentumEngine(BaseEngine):
    """
    Institutional Momentum Scoring Engine.
    Transforms Layer-2 Momentum Analyzer JSON into deterministic Layer-3 scores
    using advanced quantitative models applied strictly to intelligence vectors.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_institutional_momentum_profile())

    def calculate(self, momentum_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            momentum_json: The dictionary payload from Layer-2 Momentum Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(momentum_json, start_time, "MOM")

        if not isinstance(momentum_json, dict) or not momentum_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(momentum_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Raw Metrics for Inverse Components
            slowdown_raw = self._extract_metric(flat_data, ["slowdown", "deceleration", "fading", "decay"], 20.0)
            divergence_raw = self._extract_metric(flat_data, ["divergence", "anomaly", "divergent"], 20.0)
            
            # 2. Component Extractions
            str_comp = self._compute_component(
                flat_data, ["strength", "momentum_strength", "power"], "strong", "weak", "Momentum Strength"
            )
            accel_comp = self._compute_component(
                flat_data, ["acceleration", "increasing", "surge"], "high", "low", "Momentum Acceleration"
            )
            rsi_comp = self._compute_component(
                flat_data, ["rsi", "relative_strength"], "bullish", "bearish", "RSI Consensus"
            )
            macd_comp = self._compute_component(
                flat_data, ["macd", "convergence"], "bullish", "bearish", "MACD Consensus"
            )
            
            # Inverse Components (High Raw = High Risk = Low Safety Score)
            slow_comp = self._compute_inverse_component(
                flat_data, ["slowdown", "decay"], "low", "high", "Momentum Slowdown Safety"
            )
            div_comp = self._compute_inverse_component(
                flat_data, ["divergence", "anomaly"], "none", "strong", "Divergence Safety"
            )

            # 3. Advanced Quantitative Models (Layer-3 Intelligence)
            bayesian_score = self._hierarchical_bayesian_fusion(str_comp.final_score, rsi_comp.final_score, macd_comp.final_score)
            physics_score = self._momentum_physics_model(str_comp.final_score, accel_comp.final_score, slowdown_raw)
            markov_prob = self._markov_transition_model(str_comp.final_score, rsi_comp.final_score, macd_comp.final_score)
            entropy_penalty = self._shannon_entropy_penalty([
                str_comp.final_score, accel_comp.final_score, rsi_comp.final_score, 
                macd_comp.final_score, slow_comp.final_score, div_comp.final_score
            ])
            health_score = self._momentum_health_model(str_comp.final_score, rsi_comp.final_score, macd_comp.final_score, div_comp.final_score)

            # 4. Conflict Detection
            self._detect_conflicts(
                str_comp.final_score, rsi_comp.final_score, macd_comp.final_score, 
                divergence_raw, accel_comp.final_score, slowdown_raw
            )

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
                rsi_comp.final_score, macd_comp.final_score, divergence_raw, accel_comp.final_score
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (str_comp.final_score * dynamic_weights["strength"]) +
                (accel_comp.final_score * dynamic_weights["acceleration"]) +
                (slow_comp.final_score * dynamic_weights["slowdown"]) +
                (rsi_comp.final_score * dynamic_weights["rsi"]) +
                (macd_comp.final_score * dynamic_weights["macd"]) +
                (div_comp.final_score * dynamic_weights["divergence"])
            )

            # Blend Quantitative Models into Base Score (Institutional Smoothing)
            fused_momentum_score = (base_score * 0.35) + (physics_score * 0.30) + (bayesian_score * 0.20) + (health_score * 0.15)

            # Apply Quantitative Penalties
            structural_penalty = entropy_penalty
            if divergence_raw > 80:
                div_penalty = self.config.thresholds.get("divergence_penalty", 12.0)
                structural_penalty += div_penalty
                self._score_reasons.append(f"Applied penalty of {div_penalty} due to severe momentum divergence.")
            if slowdown_raw > 85 and accel_comp.final_score < 30:
                col_penalty = self.config.thresholds.get("collapse_penalty", 15.0)
                structural_penalty += col_penalty
                self._score_reasons.append(f"Applied penalty of {col_penalty} due to imminent momentum collapse.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            entropy_confidence_boost = max(0.0, 10.0 - entropy_penalty)
            component_agreement = 100.0 - abs(rsi_comp.final_score - macd_comp.final_score)
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.25) + 
                (evidence_graph.evidence_coverage * 0.15) + 
                (min(total_evidence * 10, 100) * 0.15) +
                (entropy_confidence_boost * 1.5) +
                (markov_prob * 0.15) +
                (component_agreement * 0.15) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_overall_score = self._normalize((fused_momentum_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                str_comp.final_score, accel_comp.final_score, rsi_comp.final_score,
                macd_comp.final_score, divergence_raw, slowdown_raw, physics_score
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_momentum_score": round(final_overall_score, 2),
                    "momentum_rating": self._determine_rating(final_overall_score),
                    "momentum_confidence": round(final_confidence, 2),
                    "momentum_strength_score": round(str_comp.final_score, 2),
                    "momentum_acceleration_score": round(accel_comp.final_score, 2),
                    "momentum_slowdown_score": round(slow_comp.final_score, 2),
                    "rsi_consensus_score": round(rsi_comp.final_score, 2),
                    "macd_consensus_score": round(macd_comp.final_score, 2),
                    "divergence_score": round(div_comp.final_score, 2),
                    "momentum_reliability": round(reliability, 2)
                },
                "components": {
                    "momentum_strength": asdict(str_comp),
                    "acceleration": asdict(accel_comp),
                    "slowdown": asdict(slow_comp),
                    "rsi": asdict(rsi_comp),
                    "macd": asdict(macd_comp),
                    "divergence": asdict(div_comp)
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
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # ADVANCED QUANTITATIVE MODELS
    # ---------------------------------------------------------

    def _hierarchical_bayesian_fusion(self, strength: float, rsi: float, macd: float) -> float:
        """
        Bayesian Fusion: Uses Momentum Strength as Prior, and RSI/MACD consensus as Likelihood.
        """
        prior = strength / 100.0
        likelihood = (rsi * 0.5 + macd * 0.5) / 100.0
        
        numerator = prior * likelihood
        denominator = numerator + ((1.0 - prior) * (1.0 - likelihood)) + 1e-9
        posterior = numerator / denominator
        
        return self._normalize(posterior * 100.0)

    def _momentum_physics_model(self, velocity: float, accel: float, friction_raw: float) -> float:
        """
        Momentum Physics: Kinetic Energy = (Velocity + (Acceleration/2)) * (1 - Friction).
        Friction is driven by raw momentum slowdown metrics.
        """
        v = velocity / 100.0
        a = accel / 100.0
        f = friction_raw / 100.0
        
        kinetic_potential = (v + (a * 0.5)) * max(0.0, 1.0 - f)
        # Scale back to 0-100 (Max theoretical is 1.5, so we normalize)
        return self._normalize((kinetic_potential / 1.5) * 100.0)

    def _markov_transition_model(self, strength: float, rsi: float, macd: float) -> float:
        """
        Markov Transition Estimation: Probability of transitioning into/maintaining a 'High Momentum' state.
        """
        transition_prob = (strength * 0.4 + rsi * 0.3 + macd * 0.3) / 100.0
        return self._normalize(transition_prob * 100.0)

    def _shannon_entropy_penalty(self, scores: list[float]) -> float:
        """
        Shannon Entropy: Measures uncertainty/chaos among momentum components.
        High entropy = conflicting oscillator readings = High penalty.
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
            self._score_reasons.append("High Shannon Entropy detected among momentum oscillators (Chaotic consensus).")
            
        return penalty

    def _momentum_health_model(self, strength: float, rsi: float, macd: float, divergence_safety: float) -> float:
        """Standard aggregation of core structural health metrics."""
        return self._normalize((strength * 0.4) + (rsi * 0.2) + (macd * 0.2) + (divergence_safety * 0.2))


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, strength: float, rsi: float, macd: float, div_raw: float, accel: float, slow_raw: float) -> None:
        """Evaluates logical paradoxes in Momentum states."""
        
        # 1. RSI and MACD extreme contradiction
        if (rsi > 75 and macd < 25) or (rsi < 25 and macd > 75):
            self._conflicts += 1
            warn = "Conflict: Extreme contradiction between RSI and MACD consensus."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Strong Momentum + Strong Bearish Divergence
        if strength > 75 and div_raw > 75:
            self._conflicts += 1
            warn = "Conflict: High momentum strength printing against severe bearish divergence."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Acceleration + Momentum Collapse
        if accel > 75 and slow_raw > 75:
            self._conflicts += 1
            warn = "Conflict: Paradoxical state of high acceleration and severe momentum decay."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, rsi: float, macd: float, div_raw: float, accel: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on current state severity."""
        weights = dict(self.config.base_weights)

        # Shift 1: Strong RSI alignment
        if rsi > 80 or rsi < 20:
            self._score_reasons.append("Adaptive Shift: Extreme RSI consensus dictates increased RSI weight.")
            weights["rsi"] += 0.10
            weights["macd"] -= 0.10

        # Shift 2: Strong MACD alignment
        if macd > 80 or macd < 20:
            self._score_reasons.append("Adaptive Shift: Strong MACD structural consensus, scaling MACD weight.")
            weights["macd"] += 0.10
            weights["rsi"] -= 0.10

        # Shift 3: Severe Divergence dictates risk protocol
        if div_raw > 75:
            self._score_reasons.append("Adaptive Shift: Severe divergence detected, increasing Divergence Risk weight.")
            weights["divergence"] += 0.15
            weights["strength"] -= 0.15

        # Shift 4: Explosive Acceleration
        if accel > 80:
            self._score_reasons.append("Adaptive Shift: Explosive momentum acceleration detected, increasing Acceleration weight.")
            weights["acceleration"] += 0.15
            weights["slowdown"] -= 0.15

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

    def _generate_explanations(self, strength: float, accel: float, rsi: float, macd: float, 
                               div_raw: float, slow_raw: float, phys_score: float) -> None:
        """Synthesizes human-readable logic for final execution state."""
        if strength > 75:
            self._score_reasons.append("Underlying momentum strength remains highly robust.")
        
        if accel > 75:
            self._score_reasons.append("Momentum is rapidly accelerating (Surge state).")
            
        if rsi > 75 and macd > 75:
            self._score_reasons.append("RSI and MACD confirm synchronized bullish momentum.")
            
        if div_raw > 75:
            self._score_reasons.append("Severe momentum divergence poses high reversal risk.")
            
        if slow_raw > 75:
            self._score_reasons.append("Momentum decay/slowdown indicates possible exhaustion.")
            
        if phys_score > 75:
            self._score_reasons.append("Physics model projects strong kinetic continuation.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Contradictory momentum oscillators reduce overall confidence.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "bullish", "increasing"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "bearish", "decreasing"]):
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
        Extracts and evaluates an inversely correlated component (e.g., Divergence, Slowdown).
        High raw input means High Risk. Component final score maps High Risk to Low Score (Safe = 100).
        """
        raw_risk = self._extract_metric(flat_data, keys, 20.0)
        inverted_raw = self._normalize(100.0 - raw_risk)
        
        bonus, penalty = 0.0, 0.0
        
        # Low risk -> Bonus
        if self._contains_keyword(flat_data, keys, [pos_kw, "false", "no", "low", "none"]):
            bonus = 10.0
            self._positive_log.append(f"Low risk regarding {name} validated.")
            
        # High risk -> Penalty
        if self._contains_keyword(flat_data, keys, [neg_kw, "true", "yes", "high", "extreme", "severe"]):
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
                "overall_momentum_score": 50.0, 
                "momentum_rating": "Neutral", 
                "momentum_confidence": 0.0,
                "momentum_strength_score": 50.0,
                "momentum_acceleration_score": 50.0,
                "momentum_slowdown_score": 50.0,
                "rsi_consensus_score": 50.0,
                "macd_consensus_score": 50.0,
                "divergence_score": 50.0,
                "momentum_reliability": 0.0
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
def calculate_momentum_score(momentum_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional momentum score.
    
    Args:
        momentum_json: The dictionary payload from Layer-2 Momentum Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = MomentumEngine()
    return engine.calculate(momentum_json)
