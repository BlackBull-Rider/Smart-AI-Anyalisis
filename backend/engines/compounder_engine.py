"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: compounder_engine.py

Institutional Long-Term Compounder Scoring Implementation.
Inherits from BaseEngine. Converts Fundamental and Institutional Analyzer intelligence 
into deterministic, institutional-grade scores to identify companies capable of 
long-term wealth creation. Applies advanced quantitative models strictly to 
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
def get_compounder_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Long-Term Compounder Scoring."""
    return EngineConfig(
        profile_name="Institutional_Compounder",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Adaptive Evidence-Weighted Compounding Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "business_quality": 0.18,
            "financial_strength": 0.16,
            "growth_sustainability": 0.15,
            "profitability": 0.10,
            "cash_flow": 0.10,
            "capital_allocation": 0.08,
            "economic_moat": 0.08,
            "management": 0.05,
            "competitive_advantage": 0.04,
            "business_stability": 0.03,
            "institutional_confidence": 0.02,
            "wealth_creation": 0.01
        },
        thresholds={
            "conflict_penalty": 15.0,
            "governance_penalty": 20.0,
            "cash_flow_penalty": 15.0,
            "debt_penalty": 12.0,
            "growth_penalty": 10.0
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
# COMPOUNDER ENGINE (Inherits BaseEngine)
# =====================================================================
class CompounderEngine(BaseEngine):
    """
    Institutional Compounder Scoring Engine.
    Transforms Layer-2 Analyzer JSON into deterministic Layer-3 scores
    using advanced quantitative models focusing on long-term wealth creation.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_compounder_profile())

    def calculate(self, compounder_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            compounder_json: The dictionary payload from Layer-2 Analyzers.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(compounder_json, start_time, "COMP")

        if not isinstance(compounder_json, dict) or not compounder_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(compounder_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Sub-metrics for Conflict Detection & Wealth Models
            debt_risk = self._extract_metric(flat_data, ["debt", "leverage", "gearing", "borrowing"], 20.0)
            roe_score = self._extract_metric(flat_data, ["roe", "return_on_equity"], 50.0)
            roce_score = self._extract_metric(flat_data, ["roce", "return_on_capital"], 50.0)
            margin_score = self._extract_metric(flat_data, ["margin", "operating_margin", "net_margin"], 50.0)
            gov_score = self._extract_metric(flat_data, ["governance", "ethics", "board"], 50.0)
            innovation_score = self._extract_metric(flat_data, ["innovation", "r_and_d", "scalability"], 50.0)
            leadership_score = self._extract_metric(flat_data, ["leadership", "market_share", "dominance"], 50.0)
            
            # 2. Component Extractions
            bq_comp = self._compute_component(
                flat_data, ["business_quality", "quality", "model"], "excellent", "weak", "Business Quality"
            )
            fs_comp = self._compute_component(
                flat_data, ["financial_strength", "financial_quality", "balance_sheet"], "healthy", "weak", "Financial Strength"
            )
            gw_comp = self._compute_component(
                flat_data, ["growth", "expansion", "revenue_growth", "eps_growth"], "consistent", "declining", "Growth Sustainability"
            )
            pr_comp = self._compute_component(
                flat_data, ["profitability", "profit"], "high", "low", "Profitability"
            )
            cf_comp = self._compute_component(
                flat_data, ["cash_flow", "fcf", "operating_cash"], "strong", "negative", "Cash Flow"
            )
            ca_comp = self._compute_component(
                flat_data, ["capital_allocation", "reinvestment"], "efficient", "inefficient", "Capital Allocation"
            )
            moat_comp = self._compute_component(
                flat_data, ["moat", "economic_moat"], "wide", "none", "Economic Moat"
            )
            mg_comp = self._compute_component(
                flat_data, ["management", "promoter_quality"], "excellent", "poor", "Management Quality"
            )
            cadv_comp = self._compute_component(
                flat_data, ["competitive_advantage", "pricing_power", "brand"], "strong", "weak", "Competitive Advantage"
            )
            bstab_comp = self._compute_component(
                flat_data, ["business_stability", "stability", "longevity"], "stable", "volatile", "Business Stability"
            )
            inst_comp = self._compute_component(
                flat_data, ["institutional", "ownership", "fii", "dii"], "strong", "weak", "Institutional Confidence"
            )
            wc_comp = self._compute_component(
                flat_data, ["wealth_creation", "compounding", "value_creation"], "high", "low", "Wealth Creation"
            )

            # 3. Advanced Compounder Models (Wealth Creation Synthesis)
            biz_longevity = self._business_longevity_model(bq_comp.final_score, moat_comp.final_score, bstab_comp.final_score)
            growth_consistency = self._growth_consistency_model(gw_comp.final_score, margin_score, cf_comp.final_score)
            cap_compounding = self._capital_compounding_score(roce_score, pr_comp.final_score, cf_comp.final_score)
            reinvest_qual = self._reinvestment_quality_score(ca_comp.final_score, gw_comp.final_score, cf_comp.final_score)
            mgmt_exec = self._management_execution_score(mg_comp.final_score, ca_comp.final_score, gov_score)
            moat_str = self._economic_moat_strength(moat_comp.final_score, cadv_comp.final_score, leadership_score)
            biz_scale = self._business_scalability_score(bq_comp.final_score, innovation_score, margin_score)
            market_ldr = self._market_leadership_score(leadership_score, cadv_comp.final_score, bstab_comp.final_score)
            comp_sust = self._competitive_sustainability(moat_str, biz_longevity, biz_scale)
            
            # The synthetic Long-Term Wealth Creation Score
            lt_wealth_creation = self._long_term_wealth_creation_model(
                cap_compounding, reinvest_qual, moat_str, mgmt_exec, comp_sust
            )

            # 4. Conflict Detection
            self._detect_conflicts(
                gw_comp.final_score, cf_comp.final_score, roe_score, debt_risk, moat_comp.final_score,
                pr_comp.final_score, bq_comp.final_score, gov_score, inst_comp.final_score, fs_comp.final_score, margin_score
            )

            # 5. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 40.0) * 100.0)
            
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
                cf_comp.final_score, moat_comp.final_score, ca_comp.final_score, inst_comp.final_score
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (bq_comp.final_score * dynamic_weights["business_quality"]) +
                (fs_comp.final_score * dynamic_weights["financial_strength"]) +
                (gw_comp.final_score * dynamic_weights["growth_sustainability"]) +
                (pr_comp.final_score * dynamic_weights["profitability"]) +
                (cf_comp.final_score * dynamic_weights["cash_flow"]) +
                (ca_comp.final_score * dynamic_weights["capital_allocation"]) +
                (moat_comp.final_score * dynamic_weights["economic_moat"]) +
                (mg_comp.final_score * dynamic_weights["management"]) +
                (cadv_comp.final_score * dynamic_weights["competitive_advantage"]) +
                (bstab_comp.final_score * dynamic_weights["business_stability"]) +
                (inst_comp.final_score * dynamic_weights["institutional_confidence"]) +
                (lt_wealth_creation * dynamic_weights["wealth_creation"])
            )

            # Blend Advanced Models into Base Score (Institutional Smoothing)
            fused_compounder_score = (base_score * 0.40) + (lt_wealth_creation * 0.30) + (biz_longevity * 0.15) + (comp_sust * 0.15)

            # Apply Structural Penalties
            structural_penalty = 0.0
            if gov_score < 30:
                gov_pen = self.config.thresholds.get("governance_penalty", 20.0)
                structural_penalty += gov_pen
                self._score_reasons.append(f"Applied massive penalty of {gov_pen} due to poor corporate governance.")
            if cf_comp.final_score < 30:
                cf_pen = self.config.thresholds.get("cash_flow_penalty", 15.0)
                structural_penalty += cf_pen
                self._score_reasons.append(f"Applied penalty of {cf_pen} due to weak free cash flow restricting compounding.")
            if debt_risk > 75:
                debt_pen = self.config.thresholds.get("debt_penalty", 12.0)
                structural_penalty += debt_pen
                self._score_reasons.append(f"Applied penalty of {debt_pen} due to heavy debt load endangering longevity.")
            if gw_comp.final_score < 30:
                gw_pen = self.config.thresholds.get("growth_penalty", 10.0)
                structural_penalty += gw_pen
                self._score_reasons.append(f"Applied penalty of {gw_pen} due to severe growth stagnation.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            component_agreement = self._normalize(100.0 - abs(bq_comp.final_score - fs_comp.final_score))
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.20) + 
                (evidence_graph.evidence_coverage * 0.15) + 
                (min(total_evidence * 10, 100) * 0.15) +
                (bstab_comp.final_score * 0.15) +
                (inst_comp.final_score * 0.15) +
                (component_agreement * 0.20) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_overall_score = self._normalize((fused_compounder_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                bq_comp.final_score, gw_comp.final_score, cf_comp.final_score, ca_comp.final_score, 
                moat_comp.final_score, cadv_comp.final_score, inst_comp.final_score, lt_wealth_creation, gov_score
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_compounder_score": round(final_overall_score, 2),
                    "compounder_rating": self._determine_rating(final_overall_score),
                    "confidence": round(final_confidence, 2),
                    "business_quality_score": round(bq_comp.final_score, 2),
                    "financial_strength_score": round(fs_comp.final_score, 2),
                    "growth_sustainability_score": round(gw_comp.final_score, 2),
                    "profitability_score": round(pr_comp.final_score, 2),
                    "cash_flow_score": round(cf_comp.final_score, 2),
                    "capital_allocation_score": round(ca_comp.final_score, 2),
                    "economic_moat_score": round(moat_comp.final_score, 2),
                    "management_quality_score": round(mg_comp.final_score, 2),
                    "competitive_advantage_score": round(cadv_comp.final_score, 2),
                    "business_stability_score": round(bstab_comp.final_score, 2),
                    "institutional_confidence_score": round(inst_comp.final_score, 2),
                    "wealth_creation_score": round(lt_wealth_creation, 2),
                    "long_term_reliability": round(reliability, 2)
                },
                "components": {
                    "business_quality": asdict(bq_comp),
                    "financial_strength": asdict(fs_comp),
                    "growth_sustainability": asdict(gw_comp),
                    "profitability": asdict(pr_comp),
                    "cash_flow": asdict(cf_comp),
                    "capital_allocation": asdict(ca_comp),
                    "economic_moat": asdict(moat_comp),
                    "management": asdict(mg_comp),
                    "competitive_advantage": asdict(cadv_comp),
                    "business_stability": asdict(bstab_comp),
                    "institutional_confidence": asdict(inst_comp),
                    "wealth_creation": asdict(wc_comp)
                },
                "advanced_models": {
                    "business_longevity": round(biz_longevity, 2),
                    "growth_consistency": round(growth_consistency, 2),
                    "capital_compounding": round(cap_compounding, 2),
                    "reinvestment_quality": round(reinvest_qual, 2),
                    "management_execution": round(mgmt_exec, 2),
                    "economic_moat_strength": round(moat_str, 2),
                    "business_scalability": round(biz_scale, 2),
                    "market_leadership": round(market_ldr, 2),
                    "competitive_sustainability": round(comp_sust, 2)
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
    # ADVANCED COMPOUNDER MODELS
    # ---------------------------------------------------------
    
    def _business_longevity_model(self, bq: float, moat: float, stab: float) -> float:
        return self._normalize((bq * 0.4) + (moat * 0.4) + (stab * 0.2))

    def _growth_consistency_model(self, gw: float, margin: float, cf: float) -> float:
        return self._normalize((gw * 0.5) + (margin * 0.25) + (cf * 0.25))

    def _capital_compounding_score(self, roce: float, prof: float, cf: float) -> float:
        return self._normalize((roce * 0.5) + (prof * 0.25) + (cf * 0.25))

    def _reinvestment_quality_score(self, ca: float, gw: float, cf: float) -> float:
        return self._normalize((ca * 0.5) + (gw * 0.3) + (cf * 0.2))

    def _management_execution_score(self, mgmt: float, ca: float, gov: float) -> float:
        return self._normalize((mgmt * 0.4) + (ca * 0.4) + (gov * 0.2))

    def _economic_moat_strength(self, moat: float, cadv: float, ldr: float) -> float:
        return self._normalize((moat * 0.5) + (cadv * 0.3) + (ldr * 0.2))

    def _business_scalability_score(self, bq: float, innov: float, margin: float) -> float:
        return self._normalize((bq * 0.4) + (innov * 0.4) + (margin * 0.2))

    def _market_leadership_score(self, ldr: float, cadv: float, stab: float) -> float:
        return self._normalize((ldr * 0.5) + (cadv * 0.3) + (stab * 0.2))

    def _competitive_sustainability(self, moat_str: float, longev: float, scale: float) -> float:
        return self._normalize((moat_str * 0.5) + (longev * 0.3) + (scale * 0.2))

    def _long_term_wealth_creation_model(self, cap_comp: float, reinvest: float, moat_str: float, 
                                         mgmt_exec: float, comp_sust: float) -> float:
        """The ultimate synthesis of compounding models."""
        return self._normalize(
            (cap_comp * 0.25) + (reinvest * 0.20) + (moat_str * 0.20) + (mgmt_exec * 0.15) + (comp_sust * 0.20)
        )


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, gw: float, cf: float, roe: float, debt: float, moat: float, prof: float, 
                          bq: float, gov: float, inst: float, fs: float, margins: float) -> None:
        """Evaluates logical paradoxes preventing long-term compounding."""
        
        # 1. High Growth + Weak Cash Flow
        if gw > 75 and cf < 35:
            self._conflicts += 1
            warn = "Conflict: High growth driven without free cash flow backing (Unsustainable cash burn)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Excellent ROE + Heavy Debt
        if roe > 75 and debt > 70:
            self._conflicts += 1
            warn = "Conflict: Excellent ROE is heavily synthethic, driven by dangerous debt leverage."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Wide Moat + Weak Profitability
        if moat > 75 and prof < 35:
            self._conflicts += 1
            warn = "Conflict: Reported wide moat fails to translate into actual profitability."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Excellent Business + Poor Governance
        if bq > 80 and gov < 35:
            self._conflicts += 1
            warn = "Conflict: Excellent underlying business model compromised by poor corporate governance."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Strong Institutional + Weak Fundamentals
        if inst > 75 and fs < 35:
            self._conflicts += 1
            warn = "Conflict: Strong institutional holding persists despite severely weak fundamentals."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 6. High Profit + Negative Cash Flow
        if prof > 75 and cf < 30:
            self._conflicts += 1
            warn = "Conflict: Reported high profitability heavily misaligned with actual cash generation."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 7. Fast Growth + Declining Margins
        if gw > 75 and margins < 35:
            self._conflicts += 1
            warn = "Conflict: Fast growth is destroying shareholder value through rapidly declining margins."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, bq: float, roe: float, roce: float, gw: float, 
                                    cf: float, moat: float, ca: float, inst: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on exceptional compounding traits."""
        weights = dict(self.config.base_weights)

        # Shift 1: Exceptional Business Quality
        if bq > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Business Quality detected, increasing Business weight.")
            weights["business_quality"] += 0.05
            weights["management"] -= 0.05

        # Shift 2: Excellent ROE + ROCE confirms extreme financial compounding
        if roe > 80 and roce > 80:
            self._score_reasons.append("Adaptive Shift: Exceptional compounding returns (ROE/ROCE), scaling Financial weight.")
            weights["financial_strength"] += 0.05
            weights["business_stability"] -= 0.05

        # Shift 3: Consistent Growth momentum
        if gw > 80:
            self._score_reasons.append("Adaptive Shift: Exceptional long-term growth trajectory, increasing Growth weight.")
            weights["growth_sustainability"] += 0.05
            weights["institutional_confidence"] -= 0.05

        # Shift 4: Exceptional Free Cash Flow
        if cf > 85:
            self._score_reasons.append("Adaptive Shift: Massive free cash flow generation, scaling Cash Flow weight.")
            weights["cash_flow"] += 0.05
            weights["profitability"] -= 0.05
            
        # Shift 5: Wide Economic Moat
        if moat > 85:
            self._score_reasons.append("Adaptive Shift: Wide, impenetrable economic moat, increasing Moat weight.")
            weights["economic_moat"] += 0.05
            weights["competitive_advantage"] -= 0.05
            
        # Shift 6: Exceptional Capital Allocation
        if ca > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional capital reinvestment efficiency, increasing Capital Allocation weight.")
            weights["capital_allocation"] += 0.05
            weights["wealth_creation"] -= 0.05

        # Shift 7: Strong Institutional Ownership
        if inst > 85:
            self._score_reasons.append("Adaptive Shift: Massive institutional conviction, increasing Institutional weight.")
            weights["institutional_confidence"] += 0.04
            weights["growth_sustainability"] -= 0.04

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

    def _generate_explanations(self, bq: float, gw: float, cf: float, ca: float, 
                               moat: float, cadv: float, inst: float, wc: float, gov: float) -> None:
        """Synthesizes human-readable institutional logic for final compounding state."""
        if bq > 75:
            self._score_reasons.append("Business demonstrates exceptional long-term quality.")
        if gw > 75:
            self._score_reasons.append("Revenue and earnings remain consistently strong.")
            
        if cf > 75:
            self._score_reasons.append("Cash flow comfortably supports future expansion and compounding.")
        elif cf < 35:
            self._score_reasons.append("Cash flow weakness limits long-term compounding ability.")
            
        if ca > 75:
            self._score_reasons.append("Management allocates capital exceptionally efficiently.")
            
        if moat > 75:
            self._score_reasons.append("Economic moat remains wide and highly durable.")
            
        if cadv > 75:
            self._score_reasons.append("Competitive advantage appears highly sustainable.")
            
        if inst > 75:
            self._score_reasons.append("Institutional ownership significantly strengthens confidence.")
            
        if wc > 75:
            self._score_reasons.append("Business shows extremely strong wealth creation potential.")
            
        if gov < 35:
            self._score_reasons.append("Weak corporate governance severely reduces long-term confidence.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Conflicting business signals reduce long-term structural confidence.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "excellent", "wide", "healthy", "efficient", "consistent", "stable"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "poor", "none", "declining", "inefficient", "negative", "volatile"]):
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
                "overall_compounder_score": 50.0, 
                "compounder_rating": "Neutral", 
                "confidence": 0.0,
                "business_quality_score": 50.0,
                "financial_strength_score": 50.0,
                "growth_sustainability_score": 50.0,
                "profitability_score": 50.0,
                "cash_flow_score": 50.0,
                "capital_allocation_score": 50.0,
                "economic_moat_score": 50.0,
                "management_quality_score": 50.0,
                "competitive_advantage_score": 50.0,
                "business_stability_score": 50.0,
                "institutional_confidence_score": 50.0,
                "wealth_creation_score": 50.0,
                "long_term_reliability": 0.0
            },
            "components": {},
            "advanced_models": {},
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
def calculate_compounder_score(compounder_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional compounder score.
    
    Args:
        compounder_json: The dictionary payload from Layer-2 Fundamental/Institutional Analyzers.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = CompounderEngine()
    return engine.calculate(compounder_json)
