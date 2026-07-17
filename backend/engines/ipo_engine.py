"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: ipo_engine.py

Institutional IPO Scoring Implementation.
Inherits from BaseEngine. Converts IPO Analyzer intelligence (Grey Market Premium, 
Subscription data, Anchor Investors, Valuation) into deterministic, institutional-grade 
scores using advanced quantitative models applied strictly to intelligence vectors.
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
def get_ipo_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for IPO Scoring."""
    return EngineConfig(
        profile_name="Institutional_Conservative",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Adaptive Evidence-Weighted IPO Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "ipo_quality": 0.18,
            "listing_strength": 0.15,
            "subscription": 0.15,
            "institutional_participation": 0.12,
            "anchor_investors": 0.10,
            "valuation": 0.08,
            "demand": 0.08,
            "liquidity": 0.05,
            "business_quality": 0.05,
            "risk": 0.02,
            "opportunity": 0.02
        },
        thresholds={
            "conflict_penalty": 15.0,
            "listing_penalty": 12.0,
            "valuation_penalty": 10.0,
            "liquidity_penalty": 15.0
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
# IPO ENGINE (Inherits BaseEngine)
# =====================================================================
class IPOEngine(BaseEngine):
    """
    Institutional IPO Scoring Engine.
    Transforms Layer-2 IPO Analyzer JSON into deterministic Layer-3 scores
    using advanced quantitative models applied strictly to intelligence vectors.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_ipo_profile())

    def calculate(self, ipo_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            ipo_json: The dictionary payload from Layer-2 IPO Analyzer.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(ipo_json, start_time, "IPO")

        if not isinstance(ipo_json, dict) or not ipo_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof)
            flat_data = self._flatten_dict(ipo_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Raw Metric for inverted components
            risk_raw = self._extract_metric(flat_data, ["risk", "danger", "warning", "lock_in"], 20.0)
            valuation_premium_raw = self._extract_metric(flat_data, ["valuation", "premium", "pricing"], 50.0)
            
            # 2. Component Extractions
            ipo_qual_comp = self._compute_component(
                flat_data, ["ipo_quality", "issue_quality", "overall_quality"], "high", "low", "IPO Quality"
            )
            list_str_comp = self._compute_component(
                flat_data, ["listing", "gmp", "grey_market"], "strong", "weak", "Listing Strength"
            )
            subs_comp = self._compute_component(
                flat_data, ["subscription", "oversubscription", "bidding"], "high", "low", "Subscription"
            )
            inst_part_comp = self._compute_component(
                flat_data, ["institutional", "qib", "qib_subscription"], "strong", "weak", "Institutional Participation"
            )
            anchor_comp = self._compute_component(
                flat_data, ["anchor", "anchor_quality", "anchor_investors"], "excellent", "poor", "Anchor Investors"
            )
            demand_comp = self._compute_component(
                flat_data, ["demand", "retail_subscription", "nii_subscription"], "huge", "weak", "Demand Strength"
            )
            liq_comp = self._compute_component(
                flat_data, ["liquidity", "float", "tradable"], "high", "low", "Liquidity"
            )
            bq_comp = self._compute_component(
                flat_data, ["business", "financial", "promoter_quality"], "excellent", "poor", "Business Quality"
            )
            opp_comp = self._compute_component(
                flat_data, ["opportunity", "post_listing_stability", "upside"], "high", "low", "Opportunity"
            )
            
            # Valuation is generally treated as inverse for attractiveness (High premium = Expensive = Low Score)
            val_comp = self._compute_inverse_component(
                flat_data, ["valuation", "pricing_quality", "expensive"], "attractive", "expensive", "Valuation Attractiveness"
            )
            # Risk is an inverse metric (High Risk = Low Score/High Danger)
            risk_comp = self._compute_inverse_component(
                flat_data, ["risk", "factors", "lock_in_analysis"], "low", "high", "Risk Safety"
            )

            # 3. Advanced Quantitative Models (Layer-3 Intelligence)
            inst_demand_model = self._institutional_demand_model(inst_part_comp.final_score, anchor_comp.final_score, subs_comp.final_score)
            listing_sus_model = self._listing_sustainability_model(list_str_comp.final_score, liq_comp.final_score, val_comp.final_score)
            sub_qual_model = self._subscription_quality_model(demand_comp.final_score, subs_comp.final_score, inst_part_comp.final_score)
            business_str_model = self._business_strength_model(bq_comp.final_score, risk_comp.final_score, ipo_qual_comp.final_score)
            demand_stability_model = self._demand_stability_model(demand_comp.final_score, liq_comp.final_score)

            # 4. Conflict Detection
            self._detect_conflicts(
                subs_comp.final_score, list_str_comp.final_score, bq_comp.final_score, 
                anchor_comp.final_score, inst_part_comp.final_score, val_comp.final_score, 
                demand_comp.final_score, risk_raw, liq_comp.final_score
            )

            # 5. Build Evidence Graph
            pos_count = len(self._positive_log)
            neg_count = len(self._negative_log)
            total_evidence = pos_count + neg_count + len(self._warning_log)
            coverage = min(100.0, (len(flat_data) / 30.0) * 100.0)
            
            evidence_graph = EvidenceGraph(
                evidence_score=self._normalize(50 + (pos_count * 5) - (neg_count * 5) - (self._conflicts * 15)),
                positive_count=pos_count,
                negative_count=neg_count,
                conflict_count=self._conflicts,
                evidence_coverage=round(coverage, 2)
            )

            # 6. Adaptive Weight Engine
            dynamic_weights = self._calculate_adaptive_weights(
                inst_part_comp.final_score, list_str_comp.final_score, anchor_comp.final_score, 
                demand_comp.final_score, bq_comp.final_score, risk_raw
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (ipo_qual_comp.final_score * dynamic_weights["ipo_quality"]) +
                (list_str_comp.final_score * dynamic_weights["listing_strength"]) +
                (subs_comp.final_score * dynamic_weights["subscription"]) +
                (inst_part_comp.final_score * dynamic_weights["institutional_participation"]) +
                (anchor_comp.final_score * dynamic_weights["anchor_investors"]) +
                (val_comp.final_score * dynamic_weights["valuation"]) +
                (demand_comp.final_score * dynamic_weights["demand"]) +
                (liq_comp.final_score * dynamic_weights["liquidity"]) +
                (bq_comp.final_score * dynamic_weights["business_quality"]) +
                (risk_comp.final_score * dynamic_weights["risk"]) +
                (opp_comp.final_score * dynamic_weights["opportunity"])
            )

            # Blend Quantitative Models into Base Score
            fused_ipo_score = (base_score * 0.40) + (inst_demand_model * 0.20) + (listing_sus_model * 0.20) + (sub_qual_model * 0.20)

            # Apply Structural Penalties
            structural_penalty = 0.0
            if list_str_comp.final_score < 30 and subs_comp.final_score > 70:
                list_pen = self.config.thresholds.get("listing_penalty", 12.0)
                structural_penalty += list_pen
                self._score_reasons.append(f"Applied penalty of {list_pen} due to weak listing prospects despite high subscription.")
            if val_comp.final_score < 30 and demand_comp.final_score < 40:
                val_pen = self.config.thresholds.get("valuation_penalty", 10.0)
                structural_penalty += val_pen
                self._score_reasons.append(f"Applied penalty of {val_pen} due to premium valuation coupled with weak demand.")
            if liq_comp.final_score < 30:
                liq_pen = self.config.thresholds.get("liquidity_penalty", 15.0)
                structural_penalty += liq_pen
                self._score_reasons.append(f"Applied penalty of {liq_pen} due to severe post-listing liquidity constraints.")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            component_agreement = 100.0 - abs(inst_part_comp.final_score - anchor_comp.final_score)
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.25) + 
                (evidence_graph.evidence_coverage * 0.15) + 
                (min(total_evidence * 10, 100) * 0.15) +
                (inst_part_comp.final_score * 0.15) +
                (business_str_model * 0.15) +
                (component_agreement * 0.15) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_overall_score = self._normalize((fused_ipo_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                bq_comp.final_score, inst_part_comp.final_score, anchor_comp.final_score,
                list_str_comp.final_score, demand_comp.final_score, val_comp.final_score,
                liq_comp.final_score
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_ipo_score": round(final_overall_score, 2),
                    "ipo_rating": self._determine_rating(final_overall_score),
                    "confidence": round(final_confidence, 2),
                    "ipo_quality_score": round(ipo_qual_comp.final_score, 2),
                    "listing_strength_score": round(list_str_comp.final_score, 2),
                    "subscription_score": round(subs_comp.final_score, 2),
                    "institutional_participation_score": round(inst_part_comp.final_score, 2),
                    "anchor_investor_score": round(anchor_comp.final_score, 2),
                    "valuation_score": round(val_comp.final_score, 2),
                    "demand_strength_score": round(demand_comp.final_score, 2),
                    "liquidity_score": round(liq_comp.final_score, 2),
                    "business_quality_score": round(bq_comp.final_score, 2),
                    "risk_score": round(risk_raw, 2),
                    "opportunity_score": round(opp_comp.final_score, 2),
                    "ipo_reliability": round(reliability, 2)
                },
                "components": {
                    "ipo_quality": asdict(ipo_qual_comp),
                    "listing_strength": asdict(list_str_comp),
                    "subscription": asdict(subs_comp),
                    "institutional_participation": asdict(inst_part_comp),
                    "anchor_investors": asdict(anchor_comp),
                    "valuation": asdict(val_comp),
                    "demand": asdict(demand_comp),
                    "liquidity": asdict(liq_comp),
                    "business_quality": asdict(bq_comp),
                    "risk": asdict(risk_comp),
                    "opportunity": asdict(opp_comp)
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
    # ADVANCED IPO MODELS
    # ---------------------------------------------------------

    def _institutional_demand_model(self, inst_part: float, anchor: float, subs: float) -> float:
        """Models true professional demand aggregating QIB, Anchor, and overall sub rates."""
        return self._normalize((inst_part * 0.45) + (anchor * 0.35) + (subs * 0.20))
        
    def _listing_sustainability_model(self, listing_str: float, liq: float, val: float) -> float:
        """Models the probability of the listing premium holding post-open."""
        return self._normalize((listing_str * 0.50) + (liq * 0.25) + (val * 0.25))

    def _subscription_quality_model(self, demand: float, subs: float, inst_part: float) -> float:
        """Evaluates whether the oversubscription is driven by smart money vs retail frenzy."""
        return self._normalize((inst_part * 0.50) + (subs * 0.30) + (demand * 0.20))

    def _business_strength_model(self, bq: float, risk_safety: float, ipo_qual: float) -> float:
        """Core fundamental stability assessment for the IPO asset."""
        return self._normalize((bq * 0.40) + (ipo_qual * 0.40) + (risk_safety * 0.20))

    def _demand_stability_model(self, demand: float, liq: float) -> float:
        """Measures whether demand will crash into illiquidity post-listing."""
        return self._normalize((demand * 0.60) + (liq * 0.40))


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, subs: float, list_str: float, bq: float, anchor: float, 
                          inst_part: float, val: float, demand: float, risk_raw: float, 
                          liq: float) -> None:
        """Evaluates logical paradoxes in IPO states."""
        
        # 1. High Subscription + Weak Listing Strength
        if subs > 80 and list_str < 40:
            self._conflicts += 1
            warn = "Conflict: Massive oversubscription but weak grey market/listing strength indications."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Excellent GMP + Poor Fundamentals
        if list_str > 80 and bq < 40:
            self._conflicts += 1
            warn = "Conflict: High listing premium fueled by hype rather than fundamental business quality."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Strong Anchors + Weak QIB
        if anchor > 80 and inst_part < 40:
            self._conflicts += 1
            warn = "Conflict: Strong anchor book completely unsupported by broader institutional (QIB) bidding."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Premium Valuation + Weak Demand
        if val < 30 and demand < 40: # val < 30 means expensive
            self._conflicts += 1
            warn = "Conflict: Extremely premium valuation meeting weak market demand (Pricing mismatch)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Excellent Business + High Risk
        if bq > 80 and risk_raw > 75:
            self._conflicts += 1
            warn = "Conflict: High quality business facing severe structural/regulatory risk factors."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 6. Strong Listing + Poor Liquidity
        if list_str > 80 and liq < 30:
            self._conflicts += 1
            warn = "Conflict: Strong expected listing premium constrained by highly illiquid float (Trap risk)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, inst_part: float, list_str: float, anchor: float, 
                                    demand: float, bq: float, risk_raw: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on extreme IPO conditions."""
        weights = dict(self.config.base_weights)

        # Shift 1: Strong QIB Subscription dominates outcome
        if inst_part > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional QIB subscription, increasing Institutional Participation weight.")
            weights["institutional_participation"] += 0.08
            weights["demand"] -= 0.04
            weights["liquidity"] -= 0.04

        # Shift 2: Excellent Listing Strength (GMP)
        if list_str > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional listing premium indicated, scaling Listing Strength weight.")
            weights["listing_strength"] += 0.05
            weights["valuation"] -= 0.05

        # Shift 3: Very Strong Anchor Investors
        if anchor > 85:
            self._score_reasons.append("Adaptive Shift: Premier anchor investor book detected, increasing Anchor weight.")
            weights["anchor_investors"] += 0.05
            weights["risk"] -= 0.05

        # Shift 4: Huge Retail/NII Demand
        if demand > 85:
            self._score_reasons.append("Adaptive Shift: Massive overall demand detected, increasing Subscription weight.")
            weights["subscription"] += 0.05
            weights["opportunity"] -= 0.05
            
        # Shift 5: Strong Business Quality anchors long-term logic
        if bq > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional business quality, increasing Business weight.")
            weights["business_quality"] += 0.05
            weights["valuation"] -= 0.05
            
        # Shift 6: High Risk demands scrutiny
        if risk_raw > 80:
            self._score_reasons.append("Adaptive Shift: High risk factors present, increasing Risk penalty weight.")
            weights["risk"] += 0.08
            weights["ipo_quality"] -= 0.08

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

    def _generate_explanations(self, bq: float, inst_part: float, anchor: float, 
                               list_str: float, demand: float, val: float, liq: float) -> None:
        """Synthesizes human-readable logic for final execution state."""
        if bq > 75:
            self._score_reasons.append("IPO fundamentals and business quality remain strong.")
        if inst_part > 75:
            self._score_reasons.append("Institutional participation (QIB) is exceptional indicating professional demand.")
            
        if anchor > 75:
            self._score_reasons.append("Anchor investors profile significantly improves structural confidence.")
            
        if list_str > 75:
            self._score_reasons.append("Pre-market and listing strength signals appear robust.")
            
        if demand > 75:
            self._score_reasons.append("Overall demand significantly exceeds available supply.")
            
        if val > 75:
            self._score_reasons.append("Valuation remains highly attractive relative to peers.")
        elif val < 30:
            self._score_reasons.append("High valuation premium reduces listing day opportunity.")
            
        if liq > 75:
            self._score_reasons.append("Post-listing liquidity outlook is healthy.")
            
        if inst_part < 30:
            self._score_reasons.append("Weak institutional participation lowers overall confidence.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Conflicting IPO signals and mechanics reduce structural confidence.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "excellent", "huge"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "poor", "expensive"]):
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
        Extracts and evaluates an inversely correlated component (e.g., Risk, Valuation premium).
        High raw input means High Risk / Expensive. Component final score maps High Risk to Low Score (Safe = 100).
        """
        raw_risk = self._extract_metric(flat_data, keys, 20.0)
        inverted_raw = self._normalize(100.0 - raw_risk)
        
        bonus, penalty = 0.0, 0.0
        
        # Low risk -> Bonus
        if self._contains_keyword(flat_data, keys, [pos_kw, "false", "no", "low", "attractive", "cheap"]):
            bonus = 10.0
            self._positive_log.append(f"Favorable state regarding {name} validated.")
            
        # High risk -> Penalty
        if self._contains_keyword(flat_data, keys, [neg_kw, "true", "yes", "high", "expensive", "extreme"]):
            penalty = 15.0
            self._negative_log.append(f"High risk/Negative state regarding {name} detected.")
            
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
                "overall_ipo_score": 50.0, 
                "ipo_rating": "Neutral", 
                "confidence": 0.0,
                "ipo_quality_score": 50.0,
                "listing_strength_score": 50.0,
                "subscription_score": 50.0,
                "institutional_participation_score": 50.0,
                "anchor_investor_score": 50.0,
                "valuation_score": 50.0,
                "demand_strength_score": 50.0,
                "liquidity_score": 50.0,
                "business_quality_score": 50.0,
                "risk_score": 50.0,
                "opportunity_score": 50.0,
                "ipo_reliability": 0.0
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
def calculate_ipo_score(ipo_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional IPO score.
    
    Args:
        ipo_json: The dictionary payload from Layer-2 IPO Analyzer.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional scores.
    """
    engine = IPOEngine()
    return engine.calculate(ipo_json)
