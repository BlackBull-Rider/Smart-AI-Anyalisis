"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: volatility_engine.py

Institutional Volatility Scoring Implementation.
Inherits from BaseEngine. Features Adaptive Weights, Dynamic Evidence Graphs,
and Volatility Regime Conflict Detection.
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
def get_institutional_volatility_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Volatility Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Adaptive Evidence-Weighted",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "atr": 0.20,
            "expansion": 0.20,
            "compression": 0.20,
            "breakout": 0.25,
            "historical": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "divergence_penalty": 10.0,
            "risk_penalty": 12.0
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
# VOLATILITY ENGINE (Inherits BaseEngine)
# =====================================================================
class VolatilityEngine(BaseEngine):
    """
    Institutional Volatility Scoring Engine.
    Transforms Layer-2 Volatility Analyzer JSON into deterministic Layer-3 scores.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_institutional_volatility_profile())

    def calculate(self, volatility_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            volatility_json: The dictionary payload from Layer-2 Volatility Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(volatility_json, start_time, "VOLATILITY")

        if not isinstance(volatility_json, dict) or not volatility_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(volatility_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty"], 50.0)
            risk_score = self._extract_metric(flat_data, ["risk", "danger", "ulcer", "extreme"], 30.0)
            
            # 2. Component Extractions (ATR, Expansion, Compression, Breakout, Historical)
            atr_comp = self._compute_component(
                flat_data, ["atr", "average_true_range", "range"], "expansion", "contraction", "ATR Quality"
            )
            exp_comp = self._compute_component(
                flat_data, ["expansion", "widening", "increase"], "high", "low", "Expansion"
            )
            comp_comp = self._compute_component(
                flat_data, ["compression", "squeeze", "contraction", "narrow"], "strong", "weak", "Compression"
            )
            brk_comp = self._compute_component(
                flat_data, ["breakout", "probability", "break"], "high", "false", "Breakout"
            )
            hist_comp = self._compute_component(
                flat_data, ["historical", "hv", "chaikin"], "stable", "erratic", "Historical Volatility"
            )

            # 3. Conflict Detection & Cross Validation
            self._detect_conflicts(comp_comp.final_score, exp_comp.final_score, brk_comp.final_score, atr_comp.final_score, risk_score)

            # 4. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 25.0) * 100.0) # Assuming ~25 expected data points
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 5. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(brk_comp.final_score, comp_comp.final_score, risk_score)

            # 6. Calculate Weighted Final Score
            base_score = (
                (atr_comp.final_score * dynamic_weights["atr"]) +
                (exp_comp.final_score * dynamic_weights["expansion"]) +
                (comp_comp.final_score * dynamic_weights["compression"]) +
                (brk_comp.final_score * dynamic_weights["breakout"]) +
                (hist_comp.final_score * dynamic_weights["historical"])
            )

            # Apply Risk Penalty if Volatility Risk is dangerously high
            risk_penalty = 0.0
            if risk_score > 80:
                risk_penalty = self.config.thresholds.get("risk_penalty", 10.0)
                self._score_reasons.append(f"Applied penalty of {risk_penalty} due to extreme volatility risk.")

            # 7. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            final_confidence = self._normalize(
                (analyzer_confidence * 0.40) + 
                (evidence_graph.evidence_coverage * 0.30) + 
                (min(total_evidence * 10, 100) * 0.30) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (risk_score * 0.2))

            # Execute final risk-adjusted normalization
            final_volatility_score = self._normalize((base_score * (0.5 + (final_confidence / 200.0))) - risk_penalty)

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 8. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_volatility_score": round(final_volatility_score, 2),
                    "volatility_rating": self._determine_rating(final_volatility_score),
                    "confidence": round(final_confidence, 2),
                    "risk_volatility": round(risk_score, 2),
                    "breakout_score": round(brk_comp.final_score, 2),
                    "expansion_score": round(exp_comp.final_score, 2),
                    "compression_score": round(comp_comp.final_score, 2),
                    "atr_quality": round(atr_comp.final_score, 2),
                    "historical_volatility_score": round(hist_comp.final_score, 2),
                    "volatility_reliability": round(reliability, 2)
                },
                "components": {
                    "atr": asdict(atr_comp),
                    "expansion": asdict(exp_comp),
                    "compression": asdict(comp_comp),
                    "breakout": asdict(brk_comp),
                    "historical_volatility": asdict(hist_comp)
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
    
    def _detect_conflicts(self, comp_score: float, exp_score: float, brk_score: float, atr_score: float, risk_score: float) -> None:
        """Evaluates logical paradoxes in volatility states."""
        # 1. Squeeze + Expansion paradox
        if comp_score > 75 and exp_score > 75:
            self._conflicts += 1
            warn = "Conflict: Simultaneous strong compression and expansion detected."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Breakout without structural ATR support
        if brk_score > 75 and atr_score < 25:
            self._conflicts += 1
            warn = "Conflict: High breakout probability paired with abnormally low ATR structure."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Expansion + Extreme Risk (Unstable Bullishness)
        if exp_score > 70 and risk_score > 85:
            self._conflicts += 1
            warn = "Conflict: Bullish expansion contradicted by extreme volatility risk."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, brk_score: float, comp_score: float, risk_score: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on current state severity."""
        weights = dict(self.config.base_weights)

        # Shift 1: High Breakout shifts focus to Actionable Events
        if brk_score > 75:
            self._score_reasons.append("Adaptive Shift: Breakout probability elevated, scaling Breakout weight.")
            weights["breakout"] += 0.15
            weights["historical"] -= 0.15

        # Shift 2: Squeeze/Compression shifts focus to Future Expansion
        if comp_score > 80:
            self._score_reasons.append("Adaptive Shift: Strong compression detected, increasing Expansion probability weight.")
            weights["expansion"] += 0.15
            weights["atr"] -= 0.15

        # Shift 3: Extreme Risk shifts focus to Historical Stability
        if risk_score > 80:
            self._score_reasons.append("Adaptive Shift: Extreme volatility risk detected, increasing Historical baseline weight.")
            weights["historical"] += 0.20
            weights["breakout"] -= 0.10
            weights["expansion"] -= 0.10

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

    # ---------------------------------------------------------
    # COMPONENT BUILDER (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a specific sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high"]):
            bonus = 10.0
            self._positive_log.append(f"Strong {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low"]):
            penalty = 12.0
            self._negative_log.append(f"Weak/Negative {name} detected.")
            
        final = self._normalize(raw + bonus - penalty)
        
        # Calculate nominal weighted score (Pre-Adaptive Distribution)
        # We pass 0.0 for weighted_score here as the true adaptive weight is applied later.
        return ScoreBreakdown(
            raw_score=round(raw, 2),
            normalized_score=round(raw, 2),
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
                "overall_volatility_score": 50.0, 
                "volatility_rating": "Neutral", 
                "confidence": 0.0,
                "risk_volatility": 50.0,
                "breakout_score": 50.0,
                "expansion_score": 50.0,
                "compression_score": 50.0,
                "atr_quality": 50.0,
                "historical_volatility_score": 50.0,
                "volatility_reliability": 0.0
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
def calculate_volatility_score(volatility_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional volatility score.
    
    Args:
        volatility_json: The dictionary payload from Layer-2 Volatility Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = VolatilityEngine()
    return engine.calculate(volatility_json)
