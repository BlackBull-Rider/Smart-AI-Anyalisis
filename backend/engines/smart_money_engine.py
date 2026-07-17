"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: smart_money_engine.py

Institutional Smart Money Scoring Implementation.
Inherits from BaseEngine. Converts Smart Money Concepts (SMC) analyzer intelligence
into deterministic, institutional-grade scores using adaptive weights and evidence graphs.
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
def get_institutional_smart_money_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Smart Money Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Adaptive Evidence-Weighted",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "market_structure": 0.25,
            "liquidity": 0.20,
            "order_block": 0.20,
            "fair_value_gap": 0.15,
            "institutional_footprint": 0.15,
            "premium_discount": 0.05
        },
        thresholds={
            "conflict_penalty": 15.0,
            "liquidity_penalty": 10.0,
            "structure_penalty": 12.0
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
# SMART MONEY ENGINE (Inherits BaseEngine)
# =====================================================================
class SmartMoneyEngine(BaseEngine):
    """
    Institutional Smart Money Scoring Engine.
    Transforms Layer-2 SMC Analyzer JSON into deterministic Layer-3 scores.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_institutional_smart_money_profile())

    def calculate(self, smart_money_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            smart_money_json: The dictionary payload from Layer-2 Smart Money Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(smart_money_json, start_time, "SMC")

        if not isinstance(smart_money_json, dict) or not smart_money_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(smart_money_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Additional derived primary metrics
            sweep_score = self._extract_metric(flat_data, ["sweep", "grab"], 50.0)
            efficiency_score = self._extract_metric(flat_data, ["efficiency", "inefficiency"], 50.0)
            
            # 2. Component Extractions
            ms_comp = self._compute_component(
                flat_data, ["structure", "bos", "choch", "trend"], "bullish", "bearish", "Market Structure"
            )
            liq_comp = self._compute_component(
                flat_data, ["liquidity", "equal_high", "equal_low"], "bullish", "bearish", "Liquidity"
            )
            ob_comp = self._compute_component(
                flat_data, ["order_block", "breaker", "mitigation"], "bullish", "bearish", "Order Block"
            )
            fvg_comp = self._compute_component(
                flat_data, ["fair_value_gap", "fvg", "imbalance"], "bullish", "bearish", "Fair Value Gap"
            )
            inst_comp = self._compute_component(
                flat_data, ["footprint", "smart_money", "displacement"], "strong", "outflow", "Institutional Footprint"
            )
            pd_comp = self._compute_component(
                flat_data, ["zone", "premium", "discount"], "discount", "premium", "Premium/Discount Zone"
            )

            # 3. Pre-weight calculation of base direction to pass to conflict detector
            pre_bias = (ms_comp.final_score + ob_comp.final_score + inst_comp.final_score) / 3.0

            # 4. Conflict Detection & Cross Validation
            self._detect_conflicts(flat_data, pre_bias)

            # 5. Build Evidence Graph
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

            # 6. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(
                ms_comp.final_score, ob_comp.final_score, liq_comp.final_score, inst_comp.final_score, efficiency_score
            )

            # 7. Calculate Weighted Final Score
            base_score = (
                (ms_comp.final_score * dynamic_weights["market_structure"]) +
                (liq_comp.final_score * dynamic_weights["liquidity"]) +
                (ob_comp.final_score * dynamic_weights["order_block"]) +
                (fvg_comp.final_score * dynamic_weights["fair_value_gap"]) +
                (inst_comp.final_score * dynamic_weights["institutional_footprint"]) +
                (pd_comp.final_score * dynamic_weights["premium_discount"])
            )

            # Apply general structure penalty if structural integrity is heavily compromised
            structure_penalty = 0.0
            if ms_comp.final_score < 30 and ob_comp.final_score > 70:
                structure_penalty = self.config.thresholds.get("structure_penalty", 12.0)
                self._score_reasons.append(f"Applied penalty of {structure_penalty} due to conflicting market structure.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            
            # Confidence heavily relies on alignment between structure and institutional footprint
            component_agreement = 100.0 - abs(ms_comp.final_score - inst_comp.final_score)
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.30) + 
                (evidence_graph.evidence_coverage * 0.20) + 
                (min(total_evidence * 10, 100) * 0.20) +
                (component_agreement * 0.30) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_smc_score = self._normalize((base_score * (0.5 + (final_confidence / 200.0))) - structure_penalty)

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_smart_money_score": round(final_smc_score, 2),
                    "smart_money_rating": self._determine_rating(final_smc_score),
                    "confidence": round(final_confidence, 2),
                    "market_structure_score": round(ms_comp.final_score, 2),
                    "liquidity_score": round(liq_comp.final_score, 2),
                    "order_block_score": round(ob_comp.final_score, 2),
                    "fair_value_gap_score": round(fvg_comp.final_score, 2),
                    "liquidity_sweep_score": round(sweep_score, 2),
                    "institutional_footprint_score": round(inst_comp.final_score, 2),
                    "premium_discount_score": round(pd_comp.final_score, 2),
                    "market_efficiency_score": round(efficiency_score, 2),
                    "smart_money_reliability": round(reliability, 2)
                },
                "components": {
                    "market_structure": asdict(ms_comp),
                    "liquidity": asdict(liq_comp),
                    "order_block": asdict(ob_comp),
                    "fair_value_gap": asdict(fvg_comp),
                    "institutional_footprint": asdict(inst_comp),
                    "premium_discount": asdict(pd_comp)
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
    
    def _detect_conflicts(self, flat_data: dict[str, Any], pre_bias: float) -> None:
        """Evaluates logical paradoxes in Smart Money states."""
        
        # 1. Bullish BOS + Bearish CHOCH paradox
        if self._contains_keyword(flat_data, ["bos"], ["bullish", "strong"]) and \
           self._contains_keyword(flat_data, ["choch"], ["bearish", "down"]):
            self._conflicts += 1
            warn = "Conflict: Market structure paradox (Bullish BOS alongside Bearish CHOCH)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Bullish OB + Bearish Liquidity Sweep
        if self._contains_keyword(flat_data, ["order_block"], ["bullish"]) and \
           self._contains_keyword(flat_data, ["sweep", "liquidity"], ["bearish", "down"]):
            self._conflicts += 1
            warn = "Conflict: Bullish Order Block compromised by strong bearish liquidity sweep."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Premium Zone + Strong Buying Bias
        if self._contains_keyword(flat_data, ["zone", "premium"], ["premium", "expensive"]) and pre_bias > 75:
            self._conflicts += 1
            warn = "Conflict: High buying bias occurring within a deep Premium (expensive) zone."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Discount Zone + Strong Selling Bias
        if self._contains_keyword(flat_data, ["zone", "discount"], ["discount", "cheap"]) and pre_bias < 25:
            self._conflicts += 1
            warn = "Conflict: High selling bias occurring within a deep Discount (cheap) zone."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Equal Highs + Liquidity Grab Down
        if self._contains_keyword(flat_data, ["equal_high", "eqh"], ["true", "detected"]) and \
           self._contains_keyword(flat_data, ["sweep", "grab"], ["down", "bearish"]):
            self._conflicts += 1
            warn = "Conflict: Un-swept equal highs persist despite a bearish liquidity grab."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, ms_score: float, ob_score: float, liq_score: float, inst_score: float, eff_score: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on current state severity."""
        weights = dict(self.config.base_weights)

        # Shift 1: Strong Order Block overrides standard structure weight
        if ob_score > 80:
            self._score_reasons.append("Adaptive Shift: Strong Order Block detected, scaling Order Block weight.")
            weights["order_block"] += 0.10
            weights["fair_value_gap"] -= 0.10

        # Shift 2: Strong Market Structure validates trend continuity
        if ms_score > 80:
            self._score_reasons.append("Adaptive Shift: Strong Market Structure identified, increasing Market Structure weight.")
            weights["market_structure"] += 0.10
            weights["premium_discount"] -= 0.05
            weights["fair_value_gap"] -= 0.05

        # Shift 3: Strong Liquidity action indicates manipulation/mitigation
        if liq_score > 80:
            self._score_reasons.append("Adaptive Shift: Major liquidity event detected, increasing Liquidity weight.")
            weights["liquidity"] += 0.10
            weights["market_structure"] -= 0.10

        # Shift 4: Institutional footprint dominates all other signals
        if inst_score > 85:
            self._score_reasons.append("Adaptive Shift: Massive institutional footprint detected, scaling Institutional weight.")
            weights["institutional_footprint"] += 0.15
            weights["liquidity"] -= 0.05
            weights["order_block"] -= 0.10
            
        # Shift 5: Extreme market inefficiency prioritizes FVG fills
        if eff_score < 25:
            self._score_reasons.append("Adaptive Shift: Extreme market inefficiency detected, scaling Fair Value Gap weight.")
            weights["fair_value_gap"] += 0.10
            weights["institutional_footprint"] -= 0.10

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
        """Extracts and evaluates a specific SMC sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        # Bullish / Positive Context
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong"]):
            bonus = 10.0
            self._positive_log.append(f"Bullish/Strong {name} validated.")
            
        # Bearish / Negative Context
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak"]):
            penalty = 12.0
            self._negative_log.append(f"Bearish/Weak {name} detected.")
            
        final = self._normalize(raw + bonus - penalty)
        
        # Calculate nominal weighted score (Pre-Adaptive Distribution)
        # Passed as 0.0 initially, adaptive weighting takes over in final aggregation.
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
                "overall_smart_money_score": 50.0, 
                "smart_money_rating": "Neutral", 
                "confidence": 0.0,
                "market_structure_score": 50.0,
                "liquidity_score": 50.0,
                "order_block_score": 50.0,
                "fair_value_gap_score": 50.0,
                "liquidity_sweep_score": 50.0,
                "institutional_footprint_score": 50.0,
                "premium_discount_score": 50.0,
                "market_efficiency_score": 50.0,
                "smart_money_reliability": 0.0
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
def calculate_smart_money_score(smart_money_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional smart money score.
    
    Args:
        smart_money_json: The dictionary payload from Layer-2 Smart Money Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = SmartMoneyEngine()
    return engine.calculate(smart_money_json)
