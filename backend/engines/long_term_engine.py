"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: long_term_engine.py

Institutional Long-Term Investment Scoring Engine.
Inherits from BaseEngine. This highest-level fusion engine integrates fundamental, 
institutional, trend, smart money, and market regime intelligence to generate a 
5-30 year Investment Quality Score. 

Unlike the Compounder Engine (which is business-centric), this engine is 
investment-centric—focusing on long-term stability, expected CAGR, institutional 
conviction, and overall investment reliability.
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
def get_long_term_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Long-Term Investment Scoring."""
    return EngineConfig(
        profile_name="Institutional_Long_Term",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Multi-Dimensional Long-Term Investment Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "business_quality": 0.18,
            "financial_strength": 0.15,
            "growth_sustainability": 0.15,
            "cash_flow": 0.10,
            "capital_allocation": 0.08,
            "economic_moat": 0.08,
            "management": 0.06,
            "institutional_ownership": 0.06,
            "trend_stability": 0.05,
            "market_regime": 0.04,
            "business_longevity": 0.03,
            "expected_cagr": 0.02
        },
        thresholds={
            "conflict_penalty": 15.0,
            "governance_penalty": 20.0,
            "cash_flow_penalty": 15.0,
            "debt_penalty": 12.0,
            "growth_penalty": 12.0
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
# LONG TERM ENGINE (Inherits BaseEngine)
# =====================================================================
class LongTermEngine(BaseEngine):
    """
    Institutional Long-Term Scoring Engine.
    Transforms massively varied Layer-2 JSON intelligence into a singular, 
    deterministic Layer-3 Long-Term Investment Score targeting 5-30 year horizons.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_long_term_profile())

    def calculate(self, long_term_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            long_term_json: The dictionary payload from ALL relevant Layer-2 Analyzers.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(long_term_json, start_time, "LTINV")

        if not isinstance(long_term_json, dict) or not long_term_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction across diverse analyzers
            flat_data = self._flatten_dict(long_term_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Sub-metrics for conflict detection and advanced models
            prof_raw = self._extract_metric(flat_data, ["profitability", "margin", "roe", "roce"], 50.0)
            gov_raw = self._extract_metric(flat_data, ["governance", "ethics", "board"], 50.0)
            debt_risk = self._extract_metric(flat_data, ["debt", "leverage", "gearing"], 20.0)
            
            # 2. Component Extractions
            bq_comp = self._compute_component(
                flat_data, ["business_quality", "model", "competitive"], "excellent", "weak", "Business Quality"
            )
            fs_comp = self._compute_component(
                flat_data, ["financial_strength", "balance_sheet"], "healthy", "weak", "Financial Strength"
            )
            gw_comp = self._compute_component(
                flat_data, ["growth", "revenue", "eps_growth"], "consistent", "declining", "Growth Sustainability"
            )
            cf_comp = self._compute_component(
                flat_data, ["cash_flow", "fcf", "operating_cash"], "strong", "negative", "Cash Flow"
            )
            ca_comp = self._compute_component(
                flat_data, ["capital_allocation", "reinvestment"], "efficient", "poor", "Capital Allocation"
            )
            moat_comp = self._compute_component(
                flat_data, ["moat", "economic_moat", "advantage"], "wide", "none", "Economic Moat"
            )
            mg_comp = self._compute_component(
                flat_data, ["management", "leadership", "promoter"], "excellent", "poor", "Management Quality"
            )
            inst_comp = self._compute_component(
                flat_data, ["institutional", "ownership", "fii", "dii"], "strong", "weak", "Institutional Ownership"
            )
            smc_comp = self._compute_component(
                flat_data, ["smart_money", "institutional_footprint"], "accumulation", "distribution", "Smart Money"
            )
            trend_comp = self._compute_component(
                flat_data, ["trend", "macro_trend", "secular_trend"], "bullish", "bearish", "Trend Stability"
            )
            regime_comp = self._compute_component(
                flat_data, ["regime", "market_regime", "environment"], "bullish", "bearish", "Market Regime"
            )
            longev_comp = self._compute_component(
                flat_data, ["longevity", "sustainability", "survival"], "high", "low", "Business Longevity"
            )
            cagr_comp = self._compute_component(
                flat_data, ["cagr", "expected_cagr", "return_potential"], "high", "low", "Expected CAGR"
            )

            # 3. Advanced Long-Term Models (Layer-3 Intelligence Fusion)
            cap_compounding = self._capital_compounding_model(gw_comp.final_score, ca_comp.final_score, cf_comp.final_score)
            moat_durability = self._economic_moat_durability_model(moat_comp.final_score, bq_comp.final_score, prof_raw)
            inst_conf = self._institutional_confidence_model(inst_comp.final_score, smc_comp.final_score)
            biz_stab = self._business_stability_model(bq_comp.final_score, fs_comp.final_score, longev_comp.final_score)
            cagr_qual = self._expected_cagr_quality_model(cagr_comp.final_score, ca_comp.final_score, moat_comp.final_score)
            lt_wealth = self._long_term_wealth_creation_model(biz_stab, cap_compounding, moat_durability, cagr_qual)

            # 4. Conflict Detection
            self._detect_conflicts(
                gw_comp.final_score, cf_comp.final_score, bq_comp.final_score, gov_raw,
                inst_comp.final_score, fs_comp.final_score, moat_comp.final_score, prof_raw,
                trend_comp.final_score, regime_comp.final_score, cagr_comp.final_score, ca_comp.final_score
            )

            # 5. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 45.0) * 100.0)
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 6. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(
                bq_comp.final_score, moat_comp.final_score, cf_comp.final_score,
                inst_comp.final_score, trend_comp.final_score, regime_comp.final_score, gov_raw
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (bq_comp.final_score * dynamic_weights["business_quality"]) +
                (fs_comp.final_score * dynamic_weights["financial_strength"]) +
                (gw_comp.final_score * dynamic_weights["growth_sustainability"]) +
                (cf_comp.final_score * dynamic_weights["cash_flow"]) +
                (ca_comp.final_score * dynamic_weights["capital_allocation"]) +
                (moat_comp.final_score * dynamic_weights["economic_moat"]) +
                (mg_comp.final_score * dynamic_weights["management"]) +
                (inst_comp.final_score * dynamic_weights["institutional_ownership"]) +
                (trend_comp.final_score * dynamic_weights["trend_stability"]) +
                (regime_comp.final_score * dynamic_weights["market_regime"]) +
                (longev_comp.final_score * dynamic_weights["business_longevity"]) +
                (cagr_comp.final_score * dynamic_weights["expected_cagr"])
            )

            # Blend Advanced Models into Base Score (Investment Reality Smoothing)
            fused_investment_score = (base_score * 0.40) + (lt_wealth * 0.30) + (biz_stab * 0.15) + (inst_conf * 0.15)

            # Apply Structural Penalties
            structural_penalty = 0.0
            if gov_raw < 30:
                gov_pen = self.config.thresholds.get("governance_penalty", 20.0)
                structural_penalty += gov_pen
                self._score_reasons.append(f"Applied severe penalty of {gov_pen} due to poor corporate governance/ethics.")
            if cf_comp.final_score < 30:
                cf_pen = self.config.thresholds.get("cash_flow_penalty", 15.0)
                structural_penalty += cf_pen
                self._score_reasons.append(f"Applied penalty of {cf_pen} due to chronically weak long-term cash flow.")
            if debt_risk > 80:
                debt_pen = self.config.thresholds.get("debt_penalty", 12.0)
                structural_penalty += debt_pen
                self._score_reasons.append(f"Applied penalty of {debt_pen} due to extreme leverage threatening solvency.")
            if gw_comp.final_score < 30:
                gw_pen = self.config.thresholds.get("growth_penalty", 12.0)
                structural_penalty += gw_pen
                self._score_reasons.append(f"Applied penalty of {gw_pen} due to structural growth stagnation.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            
            # Key institutional agreement vectors
            component_agreement = self._normalize(100.0 - abs(bq_comp.final_score - fs_comp.final_score))
            market_stability = self._normalize(100.0 - abs(trend_comp.final_score - regime_comp.final_score))
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.20) + 
                (evidence_graph.evidence_coverage * 0.10) + 
                (min(total_evidence * 10, 100) * 0.10) +
                (biz_stab * 0.20) +
                (inst_conf * 0.15) +
                (market_stability * 0.10) +
                (component_agreement * 0.15) - 
                (self._conflicts * 15)
            )
            
            # The ultimate reliability of this long-term assessment
            reliability = self._investment_reliability_model(final_confidence, biz_stab, conflict_ratio)

            # Execute final risk-adjusted normalization
            final_overall_score = self._normalize((fused_investment_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                bq_comp.final_score, fs_comp.final_score, cf_comp.final_score, moat_comp.final_score,
                ca_comp.final_score, inst_comp.final_score, cagr_comp.final_score, trend_comp.final_score,
                regime_comp.final_score, gov_raw
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_long_term_score": round(final_overall_score, 2),
                    "investment_score": round(fused_investment_score, 2),
                    "investment_rating": self._determine_rating(final_overall_score),
                    "confidence": round(final_confidence, 2),
                    "expected_cagr_score": round(cagr_comp.final_score, 2),
                    "business_quality_score": round(bq_comp.final_score, 2),
                    "financial_strength_score": round(fs_comp.final_score, 2),
                    "growth_sustainability_score": round(gw_comp.final_score, 2),
                    "cash_flow_sustainability_score": round(cf_comp.final_score, 2),
                    "capital_allocation_score": round(ca_comp.final_score, 2),
                    "economic_moat_score": round(moat_comp.final_score, 2),
                    "management_quality_score": round(mg_comp.final_score, 2),
                    "institutional_ownership_score": round(inst_comp.final_score, 2),
                    "smart_money_confidence_score": round(smc_comp.final_score, 2),
                    "trend_stability_score": round(trend_comp.final_score, 2),
                    "market_regime_score": round(regime_comp.final_score, 2),
                    "business_longevity_score": round(longev_comp.final_score, 2),
                    "investment_reliability": round(reliability, 2)
                },
                "components": {
                    "business_quality": asdict(bq_comp),
                    "financial_strength": asdict(fs_comp),
                    "growth_sustainability": asdict(gw_comp),
                    "cash_flow": asdict(cf_comp),
                    "capital_allocation": asdict(ca_comp),
                    "economic_moat": asdict(moat_comp),
                    "management": asdict(mg_comp),
                    "institutional_ownership": asdict(inst_comp),
                    "smart_money": asdict(smc_comp),
                    "trend_stability": asdict(trend_comp),
                    "market_regime": asdict(regime_comp),
                    "business_longevity": asdict(longev_comp),
                    "expected_cagr": asdict(cagr_comp)
                },
                "advanced_models": {
                    "business_longevity": round(biz_stab, 2),
                    "growth_sustainability": round(gw_comp.final_score, 2),
                    "capital_compounding": round(cap_compounding, 2),
                    "economic_moat_durability": round(moat_durability, 2),
                    "institutional_confidence": round(inst_conf, 2),
                    "business_stability": round(biz_stab, 2),
                    "expected_cagr_quality": round(cagr_qual, 2),
                    "long_term_wealth_creation": round(lt_wealth, 2),
                    "investment_reliability": round(reliability, 2)
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
    # ADVANCED LONG-TERM MODELS
    # ---------------------------------------------------------

    def _capital_compounding_model(self, gw: float, ca: float, cf: float) -> float:
        """Models the raw engine of long-term compounding."""
        return self._normalize((gw * 0.4) + (ca * 0.4) + (cf * 0.2))

    def _economic_moat_durability_model(self, moat: float, bq: float, prof: float) -> float:
        """Assesses if the moat translates to durable, long-term profitability."""
        return self._normalize((moat * 0.5) + (bq * 0.3) + (prof * 0.2))

    def _institutional_confidence_model(self, inst: float, smc: float) -> float:
        """Measures combined conviction of large-scale capital allocators."""
        return self._normalize((inst * 0.6) + (smc * 0.4))

    def _business_stability_model(self, bq: float, fs: float, longev: float) -> float:
        """Defines the core structural survivability over decades."""
        return self._normalize((bq * 0.4) + (fs * 0.4) + (longev * 0.2))

    def _expected_cagr_quality_model(self, cagr: float, ca: float, moat: float) -> float:
        """Qualifies raw expected CAGR against the mechanism that produces it."""
        return self._normalize((cagr * 0.5) + (ca * 0.3) + (moat * 0.2))
        
    def _long_term_wealth_creation_model(self, biz_stab: float, cap_comp: float, moat_dur: float, cagr_qual: float) -> float:
        """The master synthesis for long-term (5-30 yr) investment quality."""
        return self._normalize((biz_stab * 0.3) + (cap_comp * 0.3) + (moat_dur * 0.2) + (cagr_qual * 0.2))

    def _investment_reliability_model(self, conf: float, stab: float, conflict_ratio: float) -> float:
        """Determines how heavily an institution can rely on this specific score output."""
        return self._normalize((conf * 0.6) + (stab * 0.4) - (conflict_ratio * 0.5))


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, gw: float, cf: float, bq: float, gov: float, 
                          inst: float, fs: float, moat: float, prof: float, 
                          trend: float, regime: float, cagr: float, ca: float) -> None:
        """Evaluates logical paradoxes preventing 5-30 year viability."""
        
        # 1. High Growth + Weak Cash Flow
        if gw > 75 and cf < 35:
            self._conflicts += 1
            warn = "Conflict: High long-term growth projection contradicts weak cash flow reality."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Excellent Business + Weak Governance
        if bq > 80 and gov < 35:
            self._conflicts += 1
            warn = "Conflict: Excellent business model jeopardized by severe governance issues."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Strong Institutional Buying + Weak Fundamentals
        if inst > 75 and fs < 35:
            self._conflicts += 1
            warn = "Conflict: Strong institutional holding persists despite severely weak fundamentals."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Wide Moat + Weak Profitability
        if moat > 80 and prof < 35:
            self._conflicts += 1
            warn = "Conflict: Reported wide moat fails to translate into actual structural profitability."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Bullish Trend + Bearish Market Regime
        if trend > 75 and regime < 35:
            self._conflicts += 1
            warn = "Conflict: Local bullish trend fighting a severe bearish secular market regime."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 6. High Expected CAGR + Poor Capital Allocation
        if cagr > 75 and ca < 35:
            self._conflicts += 1
            warn = "Conflict: High CAGR expectations are illogical given historically poor capital allocation."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, bq: float, moat: float, cf: float, 
                                    inst: float, trend: float, regime: float, gov: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on extreme long-term traits."""
        weights = dict(self.config.base_weights)

        # Shift 1: Exceptional Business Quality
        if bq > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Business Quality detected, increasing Business weight.")
            weights["business_quality"] += 0.05
            weights["management"] -= 0.05

        # Shift 2: Wide Economic Moat guarantees long-term survival
        if moat > 85:
            self._score_reasons.append("Adaptive Shift: Wide economic moat ensures longevity, scaling Moat weight.")
            weights["economic_moat"] += 0.05
            weights["growth_sustainability"] -= 0.05

        # Shift 3: Exceptional Cash Flow allows compounding
        if cf > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional cash flow, increasing Cash Flow weight.")
            weights["cash_flow"] += 0.05
            weights["financial_strength"] -= 0.05

        # Shift 4: Strong Institutional Conviction
        if inst > 85:
            self._score_reasons.append("Adaptive Shift: Heavy institutional holding, scaling Institutional weight.")
            weights["institutional_ownership"] += 0.04
            weights["market_regime"] -= 0.04
            
        # Shift 5: Stable Long-Term Trend
        if trend > 85:
            self._score_reasons.append("Adaptive Shift: Highly stable macro trend, increasing Trend weight.")
            weights["trend_stability"] += 0.04
            weights["expected_cagr"] -= 0.04
            
        # Shift 6: Bullish Secular Market Regime
        if regime > 85:
            self._score_reasons.append("Adaptive Shift: Secular bull market regime, increasing Regime weight.")
            weights["market_regime"] += 0.04
            weights["business_longevity"] -= 0.04

        # Shift 7: Weak Governance overrides all
        if gov < 30:
            self._score_reasons.append("Adaptive Shift: Weak governance identified, heavily increasing fundamental scrutiny.")
            # We don't have a direct 'risk' weight, so we pull from growth and give to business quality
            # to emphasize structural survival over growth dreams.
            weights["business_quality"] += 0.10
            weights["growth_sustainability"] -= 0.10

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

    def _generate_explanations(self, bq: float, fs: float, cf: float, moat: float, 
                               ca: float, inst: float, cagr: float, trend: float, 
                               regime: float, gov: float) -> None:
        """Synthesizes human-readable institutional logic for final long-term investment state."""
        if bq > 75:
            self._score_reasons.append("Business quality structurally supports long-term wealth creation.")
            
        if fs > 75:
            self._score_reasons.append("Financial strength and balance sheet remain exceptional.")
            
        if cf > 75:
            self._score_reasons.append("Cash flow consistently supports expansion without dilution.")
            
        if moat > 75:
            self._score_reasons.append("Economic moat appears highly durable over the next decade.")
            
        if ca > 75:
            self._score_reasons.append("Management demonstrates disciplined, accretive capital allocation.")
            
        if inst > 75:
            self._score_reasons.append("Institutional ownership significantly strengthens long-term conviction.")
            
        if cagr > 75:
            self._score_reasons.append("Business shows sustainable competitive advantage and high Expected CAGR.")
            
        if trend > 75:
            self._score_reasons.append("Long-term macro trend remains structurally healthy.")
            
        if regime > 75:
            self._score_reasons.append("Secular market regime strongly supports this investment class.")
            
        if gov < 35:
            self._score_reasons.append("Governance risks severely compromise 5-30 year viability.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Conflicting structural signals reduce overall long-term investment confidence.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "excellent", "wide", "healthy", "efficient", "consistent", "stable", "bullish", "accumulation"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "poor", "none", "declining", "inefficient", "negative", "volatile", "bearish", "distribution"]):
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
                "overall_long_term_score": 50.0, 
                "investment_score": 50.0,
                "investment_rating": "Neutral", 
                "confidence": 0.0,
                "expected_cagr_score": 50.0,
                "business_quality_score": 50.0,
                "financial_strength_score": 50.0,
                "growth_sustainability_score": 50.0,
                "cash_flow_sustainability_score": 50.0,
                "capital_allocation_score": 50.0,
                "economic_moat_score": 50.0,
                "management_quality_score": 50.0,
                "institutional_ownership_score": 50.0,
                "smart_money_confidence_score": 50.0,
                "trend_stability_score": 50.0,
                "market_regime_score": 50.0,
                "business_longevity_score": 50.0,
                "investment_reliability": 0.0
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
def calculate_long_term_score(long_term_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional long-term investment score.
    
    Args:
        long_term_json: The dictionary payload compiled from relevant Layer-2 Analyzers.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional fusion scores.
    """
    engine = LongTermEngine()
    return engine.calculate(long_term_json)
