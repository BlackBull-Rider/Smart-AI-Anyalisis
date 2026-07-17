"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: dashboard_engine.py

Institutional Dashboard Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Executive Presentation 
Layer. It aggregates, summarizes, prioritizes, and presents outputs from previous engines.

Key Responsibilities:
- Aggregates Market Regime, Portfolio Intelligence, Watchlists, and Recommendations.
- Generates Executive Summaries, Highlights, Alerts, and Dashboard Widgets.
- Surfaces Top Opportunities and Top Risks without recalculating new AI decisions.

Boundary Constraint: This engine DOES NOT calculate Indicators, Scores, Rankings, 
Recommendations, Entry/Exit signals, or Portfolio Actions. It is a Presentation Engine.
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
class DashboardHealth(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

class DashboardPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

class DashboardWidget(str, Enum):
    MARKET = "MARKET"
    PORTFOLIO = "PORTFOLIO"
    WATCHLIST = "WATCHLIST"
    RECOMMENDATION = "RECOMMENDATION"
    SECTOR = "SECTOR"
    RISK = "RISK"
    PERFORMANCE = "PERFORMANCE"
    ALERT = "ALERT"
    SUMMARY = "SUMMARY"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 Presentation / UI Rendering)
# =====================================================================
@dataclass(frozen=True)
class DashboardProfile:
    dashboard_score: float
    system_health: float
    market_health: float
    portfolio_health: float
    watchlist_health: float
    recommendation_health: float

@dataclass(frozen=True)
class DashboardCard:
    title: str
    subtitle: str
    score: float
    priority: str
    status: str

@dataclass(frozen=True)
class DashboardHighlight:
    title: str
    description: str
    importance: str

@dataclass(frozen=True)
class DashboardAlert:
    severity: str
    message: str
    source_engine: str

@dataclass(frozen=True)
class DashboardSummary:
    market: str
    portfolio: str
    watchlist: str
    recommendation: str
    overall: str


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_dashboard_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Dashboard_V3",
        version="3.0.0",
        stage="Layer-5: Executive Presentation",
        schema_version="3.0",
        api_version="v6",
        decision_method="Holistic Aggregation and Executive Prioritization",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "market_weight": 0.20,
            "portfolio_weight": 0.30,
            "recommendation_weight": 0.30,
            "watchlist_weight": 0.20
        },
        thresholds={
            "conflict_penalty": 15.0,
            "excellent_health_threshold": 80.0,
            "good_health_threshold": 60.0,
            "warning_health_threshold": 40.0,
            "critical_alert_threshold": 80.0 # Risk threshold to trigger critical alert
        }
    )


# =====================================================================
# DASHBOARD DECISION ENGINE
# =====================================================================
class DashboardEngine(BaseDecisionEngine):
    """
    Institutional Dashboard Engine.
    Aggregates macro intelligence, portfolio states, and opportunity pipelines 
    into a unified, human-readable executive presentation payload.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_dashboard_profile())

    def evaluate_dashboard(
        self, 
        market: dict[str, Any] | None, 
        portfolio: dict[str, Any] | None, 
        recommendations: list[dict[str, Any]], 
        watchlist: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Main execution pipeline for Executive Dashboard Generation.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_DASHBOARD_ENGINE")
        
        # Package inputs for deterministic tracing
        payload_envelope = {
            "market_provided": bool(market), 
            "portfolio_provided": bool(portfolio), 
            "rec_count": len(recommendations) if recommendations else 0,
            "watchlist_count": len(watchlist) if watchlist else 0
        }
        trace = self._build_trace(payload_envelope, start_time, ctx, sum(payload_envelope.values()))

        try:
            self._add_step(ctx, "EXTRACT_AND_AGGREGATE_DATA")

            # 1. Market Aggregation
            mkt_flat = self._flatten_dict(market) if market else {}
            market_regime = self._dynamic_lookup(mkt_flat, ["regime_score", "market_regime"], 50.0)
            market_state = self._dynamic_lookup_string(mkt_flat, ["regime_status", "market_state"]).upper()
            market_risk = self._dynamic_lookup(mkt_flat, ["systemic_risk", "market_risk"], 50.0)

            # 2. Portfolio Aggregation
            ptf_flat = self._flatten_dict(portfolio) if portfolio else {}
            ptf_score = self._dynamic_lookup(ptf_flat, ["portfolio_score", "portfolio_quality"], 50.0)
            ptf_risk = self._dynamic_lookup(ptf_flat, ["portfolio_risk"], 50.0)
            ptf_health = self._dynamic_lookup_string(ptf_flat, ["portfolio_health"]).upper()
            ptf_cash = self._dynamic_lookup(ptf_flat, ["cash_utilization_pct"], 0.0)
            rebalance_suggs = ptf_flat.get("rebalancing_suggestions", [])

            # 3. Recommendation Aggregation
            rec_summary = {"STRONG_BUY": 0, "BUY": 0, "ACCUMULATE": 0, "HOLD": 0, "REDUCE": 0, "SELL": 0, "AVOID": 0, "WATCH": 0}
            top_opportunities = []
            rec_health_agg = 0.0
            
            if recommendations:
                # Sort recommendations by master rank or recommendation score
                recs_sorted = sorted(
                    recommendations, 
                    key=lambda x: self._dynamic_lookup(self._flatten_dict(x), ["recommendation_score", "master_rank_score"], 0.0), 
                    reverse=True
                )
                
                for r in recs_sorted:
                    r_flat = self._flatten_dict(r)
                    r_action = self._dynamic_lookup_string(r_flat, ["recommendation", "final_recommendation"]).upper()
                    if r_action in rec_summary:
                        rec_summary[r_action] += 1
                    rec_health_agg += self._dynamic_lookup(r_flat, ["recommendation_score"], 50.0)

                # Extract Top 3 Opportunities
                for r in recs_sorted[:3]:
                    r_flat = self._flatten_dict(r)
                    sym = self._dynamic_lookup_string(r_flat, ["symbol", "asset"])
                    action = self._dynamic_lookup_string(r_flat, ["recommendation"])
                    score = self._dynamic_lookup(r_flat, ["recommendation_score"], 0.0)
                    top_opportunities.append(DashboardCard(f"Top Pick: {sym}", f"Action: {action}", round(score,2), "HIGH", "ACTIVE"))
                    
                rec_health_agg = rec_health_agg / len(recommendations) if len(recommendations) > 0 else 50.0
            else:
                rec_health_agg = 50.0

            # 4. Watchlist Aggregation
            wl_summary = {"READY": 0, "ACTIVE": 0, "MONITOR": 0, "ARCHIVED": 0, "TRIGGERED": 0}
            wl_health_agg = 0.0
            
            if watchlist:
                for w in watchlist:
                    w_flat = self._flatten_dict(w)
                    w_status = self._dynamic_lookup_string(w_flat, ["watchlist_status"]).upper()
                    if w_status in wl_summary:
                        wl_summary[w_status] += 1
                    wl_health_agg += self._dynamic_lookup(w_flat, ["watchlist_score", "trigger_probability"], 50.0)
                wl_health_agg = wl_health_agg / len(watchlist) if len(watchlist) > 0 else 50.0
            else:
                wl_health_agg = 50.0

            # 5. Core Dashboard Scoring Model
            self._add_step(ctx, "COMPUTE_DASHBOARD_HEALTH")
            w = self.config.base_weights
            dashboard_score = self._normalize(
                (market_regime * w.get("market_weight", 0.20)) +
                (ptf_score * w.get("portfolio_weight", 0.30)) +
                (rec_health_agg * w.get("recommendation_weight", 0.30)) +
                (wl_health_agg * w.get("watchlist_weight", 0.20))
            )
            
            # System Health Proxy
            system_health = self._normalize(100.0 - ((market_risk + ptf_risk) / 2.0))

            profile = DashboardProfile(
                dashboard_score=round(dashboard_score, 2),
                system_health=round(system_health, 2),
                market_health=round(market_regime, 2),
                portfolio_health=round(ptf_score, 2),
                watchlist_health=round(wl_health_agg, 2),
                recommendation_health=round(rec_health_agg, 2)
            )

            # 6. Categorize Overall Health
            self._add_step(ctx, "CATEGORIZE_HEALTH")
            dash_health, dash_priority = self._determine_dashboard_health(dashboard_score, system_health)

            # 7. Generate Alerts & Highlights
            self._add_step(ctx, "GENERATE_ALERTS_AND_HIGHLIGHTS")
            alerts = self._generate_alerts(market_risk, ptf_health, ptf_cash, market_regime, rebalance_suggs)
            highlights = self._generate_highlights(rec_summary, wl_summary, top_opportunities)

            # 8. Conflict Detection (Data Aggregation Paradoxes)
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, dash_health, ptf_health, market_regime, alerts, rec_summary, top_opportunities
            )

            final_dashboard_score = self._normalize(dashboard_score - (ctx.conflict_penalty * 0.5))

            # 9. Build Executive Summary
            self._add_step(ctx, "BUILD_EXECUTIVE_SUMMARY")
            summary = DashboardSummary(
                market=f"Market regime is {market_state} (Score: {round(market_regime,1)}). Macro risk is {round(market_risk,1)}%.",
                portfolio=f"Portfolio structural health is {ptf_health}. Available cash: {round(ptf_cash,1)}%.",
                watchlist=f"{wl_summary['READY']} assets READY for execution. {wl_summary['ACTIVE']} actively monitored.",
                recommendation=f"Pipeline reveals {rec_summary['STRONG_BUY']} Strong Buys and {rec_summary['BUY']} Buys.",
                overall=self._generate_executive_paragraph(dash_health, market_state, ptf_health, rec_summary['STRONG_BUY'], len(alerts))
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "dashboard_health": dash_health.value,
                "dashboard_priority": dash_priority.value,
                "dashboard_profile": asdict(profile),
                "market_summary": {"regime_score": round(market_regime,2), "state": market_state, "macro_risk": round(market_risk,2)},
                "portfolio_summary": {"health": ptf_health, "score": round(ptf_score,2), "risk": round(ptf_risk,2), "cash": round(ptf_cash,2)},
                "recommendation_summary": rec_summary,
                "watchlist_summary": wl_summary,
                "top_opportunities": [asdict(card) for card in top_opportunities],
                "alerts": [asdict(alert) for alert in alerts],
                "highlights": [asdict(h) for h in highlights],
                "executive_summary": asdict(summary)
            }

            status_enum = DecisionStatusEnum.SUCCESS
            status_msg = "Executive Dashboard aggregated successfully."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = payload_envelope["rec_count"] + payload_envelope["watchlist_count"]

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=round(system_health, 2), # System confidence proxy
                rating_score=final_dashboard_score, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Dashboard Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _determine_dashboard_health(self, score: float, sys_health: float):
        t = self.config.thresholds
        if score >= t.get("excellent_health_threshold", 80.0) and sys_health >= 60.0:
            return DashboardHealth.EXCELLENT, DashboardPriority.NONE
        if score >= t.get("good_health_threshold", 60.0):
            return DashboardHealth.GOOD, DashboardPriority.LOW
        if score >= t.get("warning_health_threshold", 40.0):
            return DashboardHealth.MODERATE, DashboardPriority.MEDIUM
        if score >= 20.0:
            return DashboardHealth.WARNING, DashboardPriority.HIGH
            
        return DashboardHealth.CRITICAL, DashboardPriority.CRITICAL

    def _generate_alerts(self, market_risk: float, ptf_health: str, ptf_cash: float, regime: float, rebal_suggs: list) -> list[DashboardAlert]:
        alerts = []
        t = self.config.thresholds
        
        if market_risk >= t.get("critical_alert_threshold", 80.0):
            alerts.append(DashboardAlert("CRITICAL", "Macro market risk is severely elevated.", "MarketRegimeEngine"))
            
        if ptf_health in ["CRITICAL", "WEAK"]:
            alerts.append(DashboardAlert("HIGH", f"Portfolio structural health deteriorated to {ptf_health}.", "PortfolioEngine"))
            
        if regime < 40.0 and ptf_cash < 15.0:
            alerts.append(DashboardAlert("HIGH", "Cash reserves dangerously low for current bearish market regime.", "PortfolioEngine"))
            
        if rebal_suggs:
            # Check if there are critical removals
            for sugg in rebal_suggs:
                if isinstance(sugg, dict) and sugg.get("action") == "REMOVE":
                    alerts.append(DashboardAlert("MEDIUM", f"Action Required: Remove holding {sugg.get('target_symbol', 'UNKNOWN')}.", "PortfolioEngine"))
                    
        return alerts

    def _generate_highlights(self, rec_sum: dict, wl_sum: dict, top_opps: list[DashboardCard]) -> list[DashboardHighlight]:
        highlights = []
        
        if rec_sum["STRONG_BUY"] > 0:
            highlights.append(DashboardHighlight("Rare Opportunities", f"Detected {rec_sum['STRONG_BUY']} Strong Buy configurations in the universe.", "HIGH"))
            
        if wl_sum["READY"] > 0:
            highlights.append(DashboardHighlight("Pending Execution", f"{wl_sum['READY']} watchlist items are marked READY for immediate trigger.", "CRITICAL"))
            
        if wl_sum["TRIGGERED"] > 0:
            highlights.append(DashboardHighlight("Triggers Fired", f"{wl_sum['TRIGGERED']} opportunity triggers activated today.", "HIGH"))
            
        return highlights

    def _generate_executive_paragraph(self, health: DashboardHealth, mkt_state: str, ptf_health: str, strong_buys: int, alerts_count: int) -> str:
        """Generates a human-readable synthesized paragraph for the C-suite/Fund Manager."""
        status_text = "Overall systemic health is robust." if health in [DashboardHealth.EXCELLENT, DashboardHealth.GOOD] else "Overall systemic health is under pressure."
        opportunity_text = f"The pipeline contains {strong_buys} premium (Strong Buy) opportunities." if strong_buys > 0 else "The opportunity pipeline is currently defensive."
        risk_text = f"Action required: {alerts_count} active alerts require immediate portfolio attention." if alerts_count > 0 else "No critical systemic alerts detected."
        
        return f"{status_text} Market regime indicates a {mkt_state} environment, while portfolio health registers as {ptf_health}. {opportunity_text} {risk_text}"

    def _detect_conflicts(self, ctx: DecisionContext, dash_health: DashboardHealth, 
                          ptf_health: str, regime: float, alerts: list, rec_sum: dict, top_opps: list) -> None:
        """Identifies paradoxes in data aggregation mapping."""
        
        # 1. Excellent Dashboard + Critical Portfolio
        if dash_health == DashboardHealth.EXCELLENT and ptf_health == "CRITICAL":
            self._add_conflict(ctx, "Dashboard health aggregated to Excellent despite Critical portfolio state (Data Paradox).", penalty=20.0)
            
        # 2. Bull Market + Bear Dashboard
        if regime > 80.0 and dash_health == DashboardHealth.CRITICAL:
            self._add_conflict(ctx, "Extreme bearish dashboard evaluation during a confirmed macro Bull Regime.", penalty=15.0)
            
        # 3. No Risks + Multiple Critical Alerts
        # (Assuming risk scores are low, but alerts populated via edge cases)
        critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        if dash_health == DashboardHealth.EXCELLENT and len(critical_alerts) > 0:
            self._add_conflict(ctx, "Systemic Excellence contradicted by active Critical severity alerts.", penalty=20.0)
            
        # 4. Strong Buy Count 0 + Top Opportunity claims Strong Buy
        if rec_sum.get("STRONG_BUY", 0) == 0:
            for opp in top_opps:
                if "STRONG_BUY" in opp.subtitle:
                    self._add_conflict(ctx, "Top opportunity widget claims Strong Buy while aggregated counts report zero.", penalty=10.0)

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Dashboard Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "dashboard_health": DashboardHealth.UNKNOWN.value,
            "dashboard_priority": DashboardPriority.NONE.value,
            "dashboard_profile": asdict(DashboardProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            "market_summary": {"regime_score": 0.0, "state": "UNKNOWN", "macro_risk": 100.0},
            "portfolio_summary": {"health": "UNKNOWN", "score": 0.0, "risk": 100.0, "cash": 0.0},
            "recommendation_summary": {},
            "watchlist_summary": {},
            "top_opportunities": [],
            "alerts": [asdict(DashboardAlert("CRITICAL", "Dashboard aggregation failed.", "DashboardEngine"))],
            "highlights": [],
            "executive_summary": asdict(DashboardSummary("ERROR", "ERROR", "ERROR", "ERROR", "SYSTEM FAILURE"))
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Dashboard aggregation failed. Engine offline.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_dashboard(
    market: dict[str, Any] | None, 
    portfolio: dict[str, Any] | None, 
    recommendations: list[dict[str, Any]], 
    watchlist: list[dict[str, Any]]
) -> dict[str, Any]:
    return DashboardEngine().evaluate_dashboard(market, portfolio, recommendations, watchlist)
