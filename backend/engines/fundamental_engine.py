"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: fundamental_engine.py

Institutional Fundamental Scoring Implementation.
Inherits from BaseEngine. Converts Fundamental Analyzer intelligence into deterministic, 
institutional-grade scores using advanced quantitative models (Business Quality, 
Financial Strength, Growth Sustainability, Cash Flow Quality) applied strictly to 
intelligence vectors.
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
def get_fundamental_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Fundamental Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Adaptive Evidence-Weighted Fundamental Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "business_quality": 0.20,
            "financial_quality": 0.20,
            "growth": 0.15,
            "profitability": 0.10,
            "valuation": 0.10,
            "cash_flow": 0.10,
            "management": 0.05,
            "capital_allocation": 0.05,
            "economic_moat": 0.03,
            "governance": 0.02
        },
        thresholds={
            "conflict_penalty": 15.0,
            "debt_penalty": 12.0,
            "governance_penalty": 20.0,
            "cash_flow_penalty": 15.0
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
# FUNDAMENTAL ENGINE (Inherits BaseEngine)
# =====================================================================
class FundamentalEngine(BaseEngine):
    """
    Institutional Fundamental Scoring Engine.
    Transforms Layer-2 Fundamental Analyzer JSON into deterministic Layer-3 scores
    using advanced quantitative models applied strictly to intelligence vectors.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_fundamental_profile())

    def calculate(self, fundamental_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            fundamental_json: The dictionary payload from Layer-2 Fundamental Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(fundamental_json, start_time, "FUND")

        if not isinstance(fundamental_json, dict) or not fundamental_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(fundamental_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Sub-metrics for Conflict Detection & Advanced Models
            debt_risk = self._extract_metric(flat_data, ["debt", "leverage", "gearing"], 20.0)
            roe_score = self._extract_metric(flat_data, ["roe", "return_on_equity"], 50.0)
            roce_score = self._extract_metric(flat_data, ["roce", "return_on_capital"], 50.0)
            margin_score = self._extract_metric(flat_data, ["margin", "operating_margin", "net_margin"], 50.0)
            eps_growth = self._extract_metric(flat_data, ["eps", "earnings_growth"], 50.0)
            rev_growth = self._extract_metric(flat_data, ["revenue", "sales_growth"], 50.0)
            
            # 2. Component Extractions
            bq_comp = self._compute_component(
                flat_data, ["business", "quality", "model"], "excellent", "weak", "Business Quality"
            )
            fq_comp = self._compute_component(
                flat_data, ["financial", "balance_sheet", "strength"], "healthy", "weak", "Financial Quality"
            )
            gw_comp = self._compute_component(
                flat_data, ["growth", "expansion"], "strong", "declining", "Growth"
            )
            pr_comp = self._compute_component(
                flat_data, ["profitability", "profit"], "high", "low", "Profitability"
            )
            # Valuation: High score = Attractive/Cheap, Low score = Expensive/Overvalued
            val_comp = self._compute_component(
                flat_data, ["valuation", "pe", "pb", "cheap"], "attractive", "expensive", "Valuation"
            )
            cf_comp = self._compute_component(
                flat_data, ["cash_flow", "fcf", "operating_cash"], "strong", "negative", "Cash Flow"
            )
            mg_comp = self._compute_component(
                flat_data, ["management", "leadership", "promoter_quality"], "excellent", "poor", "Management Quality"
            )
            ca_comp = self._compute_component(
                flat_data, ["capital_allocation", "allocation"], "efficient", "inefficient", "Capital Allocation"
            )
            moat_comp = self._compute_component(
                flat_data, ["moat", "competitive", "advantage"], "wide", "none", "Economic Moat"
            )
            gov_comp = self._compute_component(
                flat_data, ["governance", "ethics", "board"], "strong", "poor", "Governance"
            )

            # 3. Advanced Quantitative Models (Layer-3 Intelligence)
            business_stability = self._business_stability_model(bq_comp.final_score, moat_comp.final_score, gov_comp.final_score)
            financial_strength = self._financial_strength_model(fq_comp.final_score, pr_comp.final_score, cf_comp.final_score, debt_risk)
            growth_sustainability = self._growth_sustainability_model(gw_comp.final_score, margin_score, roce_score)
            quality_consistency = self._quality_consistency_model(mg_comp.final_score, ca_comp.final_score, gov_comp.final_score)

            # 4. Conflict Detection
            self._detect_conflicts(
                pr_comp.final_score, cf_comp.final_score, roe_score, roce_score, 
                gw_comp.final_score, margin_score, val_comp.final_score, bq_comp.final_score, 
                debt_risk, eps_growth, rev_growth
            )

            # 5. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 35.0) * 100.0)
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 6. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(
                bq_comp.final_score, roe_score, roce_score, gw_comp.final_score, 
                cf_comp.final_score, moat_comp.final_score, gov_comp.final_score
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (bq_comp.final_score * dynamic_weights["business_quality"]) +
                (fq_comp.final_score * dynamic_weights["financial_quality"]) +
                (gw_comp.final_score * dynamic_weights["growth"]) +
                (pr_comp.final_score * dynamic_weights["profitability"]) +
                (val_comp.final_score * dynamic_weights["valuation"]) +
                (cf_comp.final_score * dynamic_weights["cash_flow"]) +
                (mg_comp.final_score * dynamic_weights["management"]) +
                (ca_comp.final_score * dynamic_weights["capital_allocation"]) +
                (moat_comp.final_score * dynamic_weights["economic_moat"]) +
                (gov_comp.final_score * dynamic_weights["governance"])
            )

            # Blend Advanced Models into Base Score (Institutional Smoothing)
            fused_fundamental_score = (base_score * 0.40) + (business_stability * 0.20) + (financial_strength * 0.20) + (growth_sustainability * 0.20)

            # Apply Structural Penalties
            structural_penalty = 0.0
            if debt_risk > 75:
                debt_pen = self.config.thresholds.get("debt_penalty", 12.0)
                structural_penalty += debt_pen
                self._score_reasons.append(f"Applied penalty of {debt_pen} due to heavy debt load weakening stability.")
            
            if gov_comp.final_score < 30:
                gov_pen = self.config.thresholds.get("governance_penalty", 20.0)
                structural_penalty += gov_pen
                self._score_reasons.append(f"Applied massive penalty of {gov_pen} due to severe corporate governance concerns.")
                
            if cf_comp.final_score < 30:
                cf_pen = self.config.thresholds.get("cash_flow_penalty", 15.0)
                structural_penalty += cf_pen
                self._score_reasons.append(f"Applied penalty of {cf_pen} due to highly negative/weak cash flows.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            component_agreement = quality_consistency
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.25) + 
                (evidence_graph.evidence_coverage * 0.15) + 
                (min(total_evidence * 10, 100) * 0.15) +
                (business_stability * 0.15) +
                (gov_comp.final_score * 0.15) +
                (component_agreement * 0.15) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_overall_score = self._normalize((fused_fundamental_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                bq_comp.final_score, fq_comp.final_score, gw_comp.final_score,
                margin_score, cf_comp.final_score, mg_comp.final_score,
                ca_comp.final_score, moat_comp.final_score, gov_comp.final_score, debt_risk
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_fundamental_score": round(final_overall_score, 2),
                    "fundamental_rating": self._determine_rating(final_overall_score),
                    "confidence": round(final_confidence, 2),
                    "business_quality_score": round(bq_comp.final_score, 2),
                    "financial_quality_score": round(fq_comp.final_score, 2),
                    "growth_score": round(gw_comp.final_score, 2),
                    "profitability_score": round(pr_comp.final_score, 2),
                    "valuation_score": round(val_comp.final_score, 2),
                    "cash_flow_score": round(cf_comp.final_score, 2),
                    "management_quality_score": round(mg_comp.final_score, 2),
                    "capital_allocation_score": round(ca_comp.final_score, 2),
                    "economic_moat_score": round(moat_comp.final_score, 2),
                    "governance_score": round(gov_comp.final_score, 2),
                    "fundamental_reliability": round(reliability, 2)
                },
                "components": {
                    "business_quality": asdict(bq_comp),
                    "financial_quality": asdict(fq_comp),
                    "growth": asdict(gw_comp),
                    "profitability": asdict(pr_comp),
                    "valuation": asdict(val_comp),
                    "cash_flow": asdict(cf_comp),
                    "management": asdict(mg_comp),
                    "capital_allocation": asdict(ca_comp),
                    "economic_moat": asdict(moat_comp),
                    "governance": asdict(gov_comp)
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

    def _business_stability_model(self, bq: float, moat: float, gov: float) -> float:
        """Models the structural stability of the underlying business."""
        return self._normalize((bq * 0.5) + (moat * 0.3) + (gov * 0.2))
        
    def _financial_strength_model(self, fq: float, prof: float, cf: float, debt_risk: float) -> float:
        """Models absolute financial robustness penalizing extreme debt."""
        base_strength = (fq * 0.4) + (prof * 0.3) + (cf * 0.3)
        return self._normalize(base_strength - (debt_risk * 0.2))

    def _growth_sustainability_model(self, growth: float, margins: float, roce: float) -> float:
        """Models whether growth is value-accretive (supported by margins/ROCE)."""
        return self._normalize((growth * 0.5) + (margins * 0.25) + (roce * 0.25))

    def _quality_consistency_model(self, mgmt: float, cap_alloc: float, gov: float) -> float:
        """Measures the agreement and quality of human-driven capital parameters."""
        return self._normalize(100.0 - (abs(mgmt - gov) * 0.5) - (abs(cap_alloc - gov) * 0.5))


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, prof: float, cf: float, roe: float, roce: float, gw: float, 
                          margins: float, val: float, bq: float, debt_risk: float, 
                          eps_growth: float, rev_growth: float) -> None:
        """Evaluates logical paradoxes in Fundamental states."""
        
        # 1. High Profit + Negative Cash Flow (Earnings Quality Issue)
        if prof > 75 and cf < 30:
            self._conflicts += 1
            warn = "Conflict: High profitability reported alongside highly negative/weak cash flows (Earnings manipulation risk)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. High ROE + Weak ROCE (Over-leveraged)
        if roe > 75 and roce < 40:
            self._conflicts += 1
            warn = "Conflict: High ROE paired with weak ROCE suggests returns are driven purely by dangerous leverage."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Strong Growth + Weak Margins (Profitless Growth)
        if gw > 75 and margins < 30:
            self._conflicts += 1
            warn = "Conflict: Strong top-line growth is destroying value due to severely weak margins."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Cheap Valuation + Weak Business Quality (Value Trap)
        if val > 75 and bq < 30:
            self._conflicts += 1
            warn = "Conflict: Attractive valuation paired with weak business fundamentals (Classic Value Trap)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Excellent Profitability + Heavy Debt (Systemic Risk)
        if prof > 80 and debt_risk > 80:
            self._conflicts += 1
            warn = "Conflict: Excellent apparent profitability threatened by critical debt load."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 6. Strong EPS Growth + Declining Revenue (Financial Engineering)
        if eps_growth > 75 and rev_growth < 30:
            self._conflicts += 1
            warn = "Conflict: Strong EPS growth alongside declining revenue (Risk of buybacks masking operational decline)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, bq: float, roe: float, roce: float, gw: float, 
                                    cf: float, moat: float, gov: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on extreme fundamental states."""
        weights = dict(self.config.base_weights)

        # Shift 1: Excellent Business Quality acts as an anchor
        if bq > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Business Quality detected, increasing Business weight.")
            weights["business_quality"] += 0.10
            weights["valuation"] -= 0.10

        # Shift 2: Exceptional ROE & ROCE indicates premium financial health
        if roe > 80 and roce > 80:
            self._score_reasons.append("Adaptive Shift: Exceptional ROE & ROCE detected, scaling Financial Quality weight.")
            weights["financial_quality"] += 0.10
            weights["profitability"] -= 0.10

        # Shift 3: Very Strong Growth momentum
        if gw > 80:
            self._score_reasons.append("Adaptive Shift: Very strong growth identified, increasing Growth weight.")
            weights["growth"] += 0.10
            weights["valuation"] -= 0.05
            weights["management"] -= 0.05

        # Shift 4: Excellent Free Cash Flow secures survival
        if cf > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Cash Flow generation, scaling Cash Flow weight.")
            weights["cash_flow"] += 0.10
            weights["profitability"] -= 0.10
            
        # Shift 5: Wide Economic Moat ensures longevity
        if moat > 85:
            self._score_reasons.append("Adaptive Shift: Wide economic moat detected, increasing Moat weight.")
            weights["economic_moat"] += 0.07
            weights["business_quality"] -= 0.07
            
        # Shift 6: Poor Governance takes absolute priority
        if gov < 30:
            self._score_reasons.append("Adaptive Shift: Poor governance dictates severe scrutiny, increasing Governance weight.")
            weights["governance"] += 0.15
            weights["growth"] -= 0.15

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

    def _generate_explanations(self, bq: float, fq: float, gw: float, margins: float, 
                               cf: float, mgmt: float, cap_alloc: float, moat: float, 
                               gov: float, debt_risk: float) -> None:
        """Synthesizes human-readable institutional logic for final execution state."""
        if bq > 75:
            self._score_reasons.append("Business quality remains exceptional.")
        if fq > 75:
            self._score_reasons.append("Financial position is extremely healthy.")
            
        if gw > 75:
            self._score_reasons.append("Revenue and earnings growth remain consistent.")
            
        if margins > 75:
            self._score_reasons.append("Operating and net margins continue expanding.")
            
        if cf > 75:
            self._score_reasons.append("Cash flow strongly supports reported earnings.")
            
        if mgmt > 75:
            self._score_reasons.append("Management quality and leadership are excellent.")
            
        if cap_alloc > 75:
            self._score_reasons.append("Capital allocation remains highly efficient.")
            
        if moat > 75:
            self._score_reasons.append("Wide economic moat supports long-term growth prospects.")
            
        if gov < 30:
            self._score_reasons.append("Governance concerns severely reduce institutional confidence.")
            
        if debt_risk > 75:
            self._score_reasons.append("Heavy debt profile weakens long-term structural stability.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Contradictory financial signals indicate structural friction.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "excellent", "attractive", "wide", "healthy", "efficient"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "poor", "expensive", "none", "declining", "inefficient", "negative"]):
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

    def _build_fallback(self, trace: PipelineTrace) -> dict[str, Any]:
        """Provides a safe, deterministic failover state."""
        return {
            "status": asdict(OutputStatus(status="FAILED", quality="INVALID")),
            "scores": {
                "overall_fundamental_score": 50.0, 
                "fundamental_rating": "Neutral", 
                "confidence": 0.0,
                "business_quality_score": 50.0,
                "financial_quality_score": 50.0,
                "growth_score": 50.0,
                "profitability_score": 50.0,
                "valuation_score": 50.0,
                "cash_flow_score": 50.0,
                "management_quality_score": 50.0,
                "capital_allocation_score": 50.0,
                "economic_moat_score": 50.0,
                "governance_score": 50.0,
                "fundamental_reliability": 0.0
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
def calculate_fundamental_score(fundamental_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional fundamental score.
    
    Args:
        fundamental_json: The dictionary payload from Layer-2 Fundamental Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = FundamentalEngine()
    return engine.calculate(fundamental_json)
