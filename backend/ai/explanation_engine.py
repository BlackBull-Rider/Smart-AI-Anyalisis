"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: explanation_engine.py

Institutional Explanation Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the AI Explainability 
and Audit Trail Boundary. 

Key Responsibilities:
- Converts complex quantitative scores and absolute decisions into human-readable 
  institutional narratives.
- Maps the Decision Chain from Layer-1 Indicators up to Layer-5 Dashboards.
- Exposes the Evidence Graph (Why a decision was made).

Boundary Constraint: This engine DOES NOT calculate Indicators, Scores, Rankings, 
Recommendations, Entry/Exit signals, or Portfolio Actions. It only EXPLAINS them.
"""

import time
from enum import Enum
from typing import Any
from dataclasses import dataclass, asdict

from backend.decision.base_decision_engine import (
    BaseDecisionEngine,
    DecisionConfig,
    DecisionContext,
    DecisionStatusEnum,
    WarningSeverityEnum,
    DecisionTrace
)


# =====================================================================
# ENUMS (Strict Output Typing)
# =====================================================================
class ExplanationType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    AVOID = "AVOID"
    WATCHLIST = "WATCHLIST"
    PORTFOLIO = "PORTFOLIO"
    MARKET = "MARKET"
    GENERAL = "GENERAL"

class ExplanationPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

class ExplanationAudience(str, Enum):
    TRADER = "TRADER"
    INVESTOR = "INVESTOR"
    PORTFOLIO_MANAGER = "PORTFOLIO_MANAGER"
    INSTITUTION = "INSTITUTION"
    GENERAL = "GENERAL"


# =====================================================================
# STRUCTURED OBJECTS (For UI, Reports, and API Consumers)
# =====================================================================
@dataclass(frozen=True)
class ExplanationProfile:
    explanation_score: float
    clarity: float
    consistency: float
    confidence: float
    completeness: float

@dataclass(frozen=True)
class EvidenceItem:
    engine: str
    factor: str
    score: float
    weight: float
    impact: str

@dataclass(frozen=True)
class DecisionChain:
    market: str
    ranking: str
    recommendation: str
    watchlist: str
    portfolio: str
    dashboard: str

@dataclass(frozen=True)
class ExplanationReason:
    primary: str
    secondary: str
    supporting_points: list[str]
    warnings: list[str]

@dataclass(frozen=True)
class ExplanationSummary:
    headline: str
    executive_summary: str
    technical_summary: str
    risk_summary: str
    action_summary: str


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_explanation_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Explanation_V3",
        version="3.0.0",
        stage="Layer-5: Explainability & Audit",
        schema_version="3.0",
        api_version="v6",
        decision_method="Deterministic Narrative Synthesis and Evidence Tracing",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "completeness": 0.30,
            "consistency": 0.30,
            "clarity": 0.20,
            "evidence_density": 0.20
        },
        thresholds={
            "conflict_penalty": 15.0,
            "high_clarity_threshold": 80.0,
            "acceptable_completeness": 60.0
        }
    )


# =====================================================================
# EXPLANATION DECISION ENGINE
# =====================================================================
class ExplanationEngine(BaseDecisionEngine):
    """
    Institutional Explanation Engine.
    Translates the entire structural decision pipeline into natural language 
    executive narratives, ensuring absolute AI transparency and traceability.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_explanation_profile())

    # Injected evaluate alias for BaseDecisionEngine compatibility
    def evaluate(self, *args, **kwargs) -> Any:
        return self.evaluate_explanation(*args, **kwargs)

    def evaluate_explanation(
        self, 
        master: dict[str, Any] | None, 
        ranking: dict[str, Any] | None, 
        recommendation: dict[str, Any] | None, 
        watchlist: dict[str, Any] | None, 
        portfolio: dict[str, Any] | None, 
        dashboard: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Main execution pipeline for AI Narrative Generation and Evidence Tracing.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_EXPLANATION_ENGINE")
        
        # Consolidate all inputs for unified parsing
        aggregate_data = {
            "master": master or {},
            "ranking": ranking or {},
            "recommendation": recommendation or {},
            "watchlist": watchlist or {},
            "portfolio": portfolio or {},
            "dashboard": dashboard or {}
        }
        
        trace = self._build_trace(aggregate_data, start_time, ctx, 6)

        try:
            self._add_step(ctx, "EXTRACT_DECISION_NODES")
            flat = self._flatten_dict(aggregate_data)
            
            # 1. Core L5 Extractions
            rec_action = self._dynamic_lookup_string(flat, ["recommendation", "final_recommendation"]).upper()
            rank_tier = self._dynamic_lookup_string(flat, ["ranking_tier"]).upper()
            wl_status = self._dynamic_lookup_string(flat, ["watchlist_status"]).upper()
            ptf_action = self._dynamic_lookup_string(flat, ["portfolio_action", "rebalancing_action"]).upper()
            dash_health = self._dynamic_lookup_string(flat, ["dashboard_health"]).upper()

            # 2. Core L4 Extractions
            risk = self._dynamic_lookup(flat, ["overall_risk", "composite_risk"], 50.0)
            reward = self._dynamic_lookup(flat, ["reward_score", "reward_quality"], 50.0)
            conf = self._dynamic_lookup(flat, ["confidence_score", "reliability_score"], 50.0)
            conv = self._dynamic_lookup(flat, ["conviction_score", "actionability_score"], 50.0)
            alloc = self._dynamic_lookup(flat, ["initial_allocation_pct"], 0.0)
            target = self._dynamic_lookup_string(flat, ["target_type", "primary_target"])

            # 3. Core L3 Extractions
            trend = self._dynamic_lookup(flat, ["trend_score", "trend_strength"], 50.0)
            inst = self._dynamic_lookup(flat, ["institutional_score", "smart_money_score"], 50.0)
            fund = self._dynamic_lookup(flat, ["fundamental_score", "business_quality"], 50.0)
            regime = self._dynamic_lookup(flat, ["regime_score", "market_regime"], 50.0)

            # 4. Generate Explanations
            self._add_step(ctx, "GENERATE_NARRATIVES")
            exp_type = self._determine_explanation_type(rec_action, wl_status, ptf_action)
            headline, exec_summ, tech_summ, risk_summ, action_summ = self._generate_summaries(
                exp_type, rec_action, rank_tier, risk, reward, conf, conv, trend, inst, fund, regime, target
            )
            
            summary = ExplanationSummary(headline, exec_summ, tech_summ, risk_summ, action_summ)

            # 5. Build Evidence Graph
            self._add_step(ctx, "BUILD_EVIDENCE_GRAPH")
            evidence_items = self._build_evidence_items(trend, inst, fund, risk, reward, conf, conv)

            # 6. Build Decision Chain
            self._add_step(ctx, "BUILD_DECISION_CHAIN")
            d_chain = DecisionChain(
                market=f"Regime Score {round(regime,1)}",
                ranking=f"Tier {rank_tier}",
                recommendation=f"Action {rec_action}",
                watchlist=f"Status {wl_status}",
                portfolio=f"Action {ptf_action}",
                dashboard=f"Health {dash_health}"
            )

            # 7. Construct Reason Profiles
            supporting_pts = [f"Trend Quality: {round(trend,1)}%", f"Inst. Backing: {round(inst,1)}%", f"Actionability: {round(conv,1)}%"]
            warnings = []
            
            if risk > 70: warnings.append(f"Elevated structural risk ({round(risk,1)}%).")
            if conf > 80 and conv < 50: warnings.append("Reliable data but low execution conviction.")
            if rec_action == "STRONG_BUY" and regime < 40: warnings.append("Bullish recommendation fighting bearish macro regime.")

            reason = ExplanationReason(
                primary=exec_summ,
                secondary=action_summ,
                supporting_points=supporting_pts,
                warnings=warnings
            )

            # 8. Evaluate Explainability Quality
            self._add_step(ctx, "EVALUATE_EXPLAINABILITY_QUALITY")
            data_completeness = self._normalize(len([x for x in [trend, inst, fund, risk, reward, conf, conv] if x != 50.0]) / 7 * 100)
            clarity = self._normalize(100.0 - (len(warnings) * 10.0))
            consistency = self._normalize(100.0 - (abs(conf - conv) * 0.5))
            
            exp_score = self._normalize((data_completeness * 0.4) + (clarity * 0.3) + (consistency * 0.3))

            profile = ExplanationProfile(
                explanation_score=round(exp_score, 2),
                clarity=round(clarity, 2),
                consistency=round(consistency, 2),
                confidence=round(conf, 2),
                completeness=round(data_completeness, 2)
            )

            # 9. Conflict Detection (Paradoxes in Reasoning)
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(ctx, rec_action, risk, conf, conv, ptf_action, dash_health, data_completeness)

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "explanation_type": exp_type.value,
                "explanation_priority": ExplanationPriority.HIGH.value if exp_score > 60 else ExplanationPriority.MEDIUM.value,
                "explanation_audience": ExplanationAudience.PORTFOLIO_MANAGER.value,
                "explanation_profile": asdict(profile),
                "explanation_summary": asdict(summary),
                "explanation_reason": asdict(reason),
                "decision_chain": asdict(d_chain),
                "evidence_graph": [asdict(e) for e in evidence_items],
                "ai_narrative": f"{headline}\n\n{exec_summ}\n\n{tech_summ}\n\n{risk_summ}\n\n{action_summ}",
                "risk_flags": warnings
            }

            status_enum = DecisionStatusEnum.SUCCESS
            status_msg = "Audit Trail and Institutional Explanations generated."

            object.__setattr__(trace, 'steps_executed', ctx.steps)
            object.__setattr__(trace, 'inputs_parsed', len(flat))

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=conf,
                rating_score=exp_score, # Rating represents Explainability / Clarity
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Explanation Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # DECISION LOGIC & NARRATIVE GENERATORS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _determine_explanation_type(self, rec: str, wl: str, ptf: str) -> ExplanationType:
        if rec in ["STRONG_BUY", "BUY", "ACCUMULATE"]: return ExplanationType.BUY
        if rec in ["SELL", "REDUCE"]: return ExplanationType.SELL
        if rec == "HOLD": return ExplanationType.HOLD
        if rec == "AVOID": return ExplanationType.AVOID
        if wl in ["ACTIVE", "READY", "MONITOR"]: return ExplanationType.WATCHLIST
        if ptf in ["REBALANCE", "ADD", "REMOVE"]: return ExplanationType.PORTFOLIO
        return ExplanationType.GENERAL

    def _generate_summaries(self, e_type: ExplanationType, rec: str, rank: str, risk: float, 
                            reward: float, conf: float, conv: float, trend: float, 
                            inst: float, fund: float, regime: float, target: str):
        
        # Headline
        if rec in ["STRONG_BUY", "BUY"]:
            headline = f"High-Conviction {rec.replace('_', ' ').title()} Opportunity Identified."
        elif rec in ["SELL", "REDUCE"]:
            headline = f"Exposure Reduction Mandated: {rec.title()} Signal Activated."
        elif rec == "AVOID":
            headline = "Capital Preservation: Avoid Exposure."
        else:
            headline = "Asset Evaluation and Monitoring Update."

        # Executive Summary
        if e_type == ExplanationType.BUY:
            exec_summ = f"An exceptional alignment of structural metrics elevates this asset to a {rank} ranking. The system outputs a {rec} recommendation driven by a {round(conv,1)}% actionable conviction score."
        elif e_type == ExplanationType.SELL:
            exec_summ = f"Structural deterioration necessitates a {rec} action. Diminished conviction ({round(conv,1)}%) and elevated systemic pressures support liquidation or reduction."
        else:
            exec_summ = f"Current metrics dictate a {rec} stance. The opportunity holds a {rank} tier ranking, pending further systemic confirmation."

        # Technical/Fundamental Summary
        tech_summ = f"Technical trend alignment is scored at {round(trend,1)}%. Institutional flow participation supports the thesis at {round(inst,1)}%. Business fundamentals register a quality score of {round(fund,1)}%."

        # Risk/Reward Summary
        r_state = "highly favorable" if reward > 70 and risk < 40 else "balanced" if risk < 60 else "hostile"
        risk_summ = f"The risk-adjusted reward profile is {r_state}. Downside structural risk is quantified at {round(risk,1)}%, juxtaposed against an upside reward quality of {round(reward,1)}% targeting {target}."

        # Action Summary
        if conf > 75 and conv > 75:
            action_summ = "Multiple engines independently confirm high reliability and strong actionability. Execute according to standard allocation limits."
        elif conf > 75 and conv < 50:
            action_summ = "Data reliability is high, but actionability (conviction) is low due to poor risk/reward or timing. Observe until triggers align."
        else:
            action_summ = "Systematic action constrained by internal logic paradoxes or unconfirmed data sets. Proceed with strict capital defense."

        return headline, exec_summ, tech_summ, risk_summ, action_summ

    def _build_evidence_items(self, trend: float, inst: float, fund: float, 
                              risk: float, reward: float, conf: float, conv: float) -> list[EvidenceItem]:
        evidence = []
        evidence.append(EvidenceItem("TrendEngine", "Directional Structure", round(trend,1), 0.15, "Positive" if trend > 50 else "Negative"))
        evidence.append(EvidenceItem("InstitutionalEngine", "Smart Money Flow", round(inst,1), 0.15, "Positive" if inst > 50 else "Negative"))
        evidence.append(EvidenceItem("FundamentalEngine", "Business Quality", round(fund,1), 0.10, "Positive" if fund > 50 else "Negative"))
        evidence.append(EvidenceItem("RiskEngine", "Structural Risk", round(risk,1), 0.20, "Negative" if risk > 60 else "Positive"))
        evidence.append(EvidenceItem("RewardEngine", "Upside Potential", round(reward,1), 0.20, "Positive" if reward > 50 else "Negative"))
        evidence.append(EvidenceItem("ConfidenceEngine", "Data Reliability", round(conf,1), 0.10, "Positive" if conf > 50 else "Negative"))
        evidence.append(EvidenceItem("ConvictionEngine", "Actionability", round(conv,1), 0.10, "Positive" if conv > 50 else "Negative"))
        return evidence

    def _detect_conflicts(self, ctx: DecisionContext, rec: str, risk: float, conf: float, 
                          conv: float, ptf: str, dash: str, completeness: float) -> None:
        """Identifies paradoxes between different explanation nodes."""
        
        # 1. Buy + High Risk
        if rec in ["BUY", "STRONG_BUY"] and risk > 75.0:
            self._add_conflict(ctx, "Bullish recommendation accompanied by extremely high structural risk.", penalty=15.0)
            
        # 2. Strong Confidence + Low Conviction
        if conf > 80.0 and conv < 40.0:
            self._add_conflict(ctx, "High analytical confidence but severely poor execution actionability.", penalty=10.0)
            
        # 3. Excellent Portfolio + Critical Dashboard
        if ptf in ["REBALANCE", "REMOVE"] and dash == "EXCELLENT":
            self._add_conflict(ctx, "Critical portfolio rebalancing demanded despite Excellent dashboard health.", penalty=20.0)
            
        # 4. Recommendation + Evidence Missing
        if rec != "UNKNOWN" and completeness < 50.0:
            self._add_conflict(ctx, "Firm investment decision generated despite critically low systemic data completeness.", penalty=25.0)

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Explanation Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "explanation_type": ExplanationType.GENERAL.value,
            "explanation_priority": ExplanationPriority.CRITICAL.value,
            "explanation_audience": ExplanationAudience.GENERAL.value,
            "explanation_profile": asdict(ExplanationProfile(0.0, 0.0, 0.0, 0.0, 0.0)),
            "explanation_summary": asdict(ExplanationSummary("ERROR", "Engine Offline", "N/A", "N/A", "N/A")),
            "explanation_reason": asdict(ExplanationReason("SYSTEM_FAILURE", "Engine crash", [], [])),
            "decision_chain": asdict(DecisionChain("N/A", "N/A", "N/A", "N/A", "N/A", "N/A")),
            "evidence_graph": [],
            "ai_narrative": "SYSTEM FAILURE: Unable to generate institutional explanation due to engine crash.",
            "risk_flags": ["SYSTEM_FAILURE_EXPLANATION"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Explanation generation failed. Engine offline.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_explanation(
    master: dict[str, Any] | None, 
    ranking: dict[str, Any] | None, 
    recommendation: dict[str, Any] | None, 
    watchlist: dict[str, Any] | None, 
    portfolio: dict[str, Any] | None, 
    dashboard: dict[str, Any] | None
) -> dict[str, Any]:
    return ExplanationEngine().evaluate_explanation(master, ranking, recommendation, watchlist, portfolio, dashboard)
