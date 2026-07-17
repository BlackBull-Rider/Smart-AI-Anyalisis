"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: watchlist_engine.py

Institutional Watchlist Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Opportunity Monitoring 
and Trigger Definition Boundary. 

Key Responsibilities:
- Converts Final Recommendations into trackable, actionable Watchlist states.
- Defines explicit Triggers (Price, Volume, Momentum) that must occur for execution.
- Establishes strict Monitoring Frequencies (Continuous, Daily, Weekly).

Boundary Constraint: This engine DOES NOT generate Recommendations, Rankings, 
Entry signals, Position Sizes, or Portfolio allocations. It manages pending opportunities.
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
class WatchlistStatus(str, Enum):
    ACTIVE = "ACTIVE"
    MONITOR = "MONITOR"
    READY = "READY"
    TRIGGERED = "TRIGGERED"
    ARCHIVED = "ARCHIVED"
    REMOVED = "REMOVED"

class WatchlistType(str, Enum):
    HIGH_CONVICTION = "HIGH_CONVICTION"
    BREAKOUT = "BREAKOUT"
    PULLBACK = "PULLBACK"
    COMPOUNDER = "COMPOUNDER"
    IPO = "IPO"
    INSTITUTIONAL = "INSTITUTIONAL"
    VALUE = "VALUE"
    DEFENSIVE = "DEFENSIVE"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    NONE = "NONE"

class WatchlistPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class WatchlistWindow(str, Enum):
    TODAY = "TODAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    MULTI_YEAR = "MULTI_YEAR"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    NONE = "NONE"

class MonitoringFrequency(str, Enum):
    EVERY_SCAN = "EVERY_SCAN"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    EVENT_TRIGGERED = "EVENT_TRIGGERED"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 Dashboard / Portfolio Execution)
# =====================================================================
@dataclass(frozen=True)
class WatchlistProfile:
    watchlist_score: float
    trigger_probability: float
    monitoring_quality: float
    execution_readiness: float
    watchlist_quality: float

@dataclass(frozen=True)
class WatchlistTrigger:
    trigger_condition: str
    entry_trigger: str
    price_trigger: str
    volume_trigger: str
    momentum_trigger: str
    risk_trigger: str
    confirmation_trigger: str

@dataclass(frozen=True)
class WatchlistComponent:
    name: str
    score: float
    weight: float
    confidence: float

@dataclass(frozen=True)
class WatchlistReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class WatchlistChecklist:
    recommendation_ok: bool
    ranking_ok: bool
    risk_ok: bool
    reward_ok: bool
    confidence_ok: bool
    conviction_ok: bool
    market_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_watchlist_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Watchlist_V3",
        version="3.0.0",
        stage="Layer-5: Opportunity Management",
        schema_version="3.0",
        api_version="v6",
        decision_method="Adaptive Continuous Trigger Definition",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "recommendation_strength": 0.25,
            "conviction": 0.20,
            "master_rank": 0.15,
            "execution_readiness": 0.15,
            "reward_quality": 0.15,
            "risk_safety": 0.10
        },
        thresholds={
            "conflict_penalty": 15.0,
            "ready_trigger_threshold": 80.0,
            "active_monitor_threshold": 60.0,
            "adaptive_rec_mult": 0.20,
            "adaptive_conviction_mult": 0.20,
            "adaptive_readiness_mult": 0.15
        }
    )


# =====================================================================
# WATCHLIST DECISION ENGINE
# =====================================================================
class WatchlistEngine(BaseDecisionEngine):
    """
    Institutional Watchlist Engine.
    Converts absolute Investment Recommendations into structured opportunity management 
    states. Defines precise execution triggers without directly executing trades.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_watchlist_profile())

    def evaluate_watchlist(self, stock_output: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main execution pipeline for Opportunity Monitoring and Trigger Definition.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_WATCHLIST_ENGINE")
        
        trace = self._build_trace(stock_output, start_time, ctx, 0)

        if not isinstance(stock_output, dict) or not stock_output:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            flat_data = self._flatten_dict(stock_output)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_UPSTREAM_DATA")

            # 1. Identity & Gatekeeper Data
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()

            # 2. Extract L5 Recommendations & Rankings
            rec_str = self._dynamic_lookup_string(flat_data, ["recommendation", "final_recommendation"]).upper()
            rec_strength = self._dynamic_lookup(flat_data, ["recommendation_strength", "recommendation_score"], 50.0)
            master_rank = self._dynamic_lookup(flat_data, ["master_rank_score", "master_score"], 50.0)
            
            # 3. Extract L4 Decisions
            conviction = self._dynamic_lookup(flat_data, ["conviction_score", "actionability_score"], 50.0)
            confidence = self._dynamic_lookup(flat_data, ["confidence_score", "reliability_score"], 50.0)
            readiness = self._dynamic_lookup(flat_data, ["execution_readiness", "entry_quality"], 50.0)
            overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], 50.0)
            reward = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score"], 50.0)
            risk_safety = self._normalize(100.0 - overall_risk)
            
            # 4. Extract Key L3 Characteristics
            fund = self._dynamic_lookup(flat_data, ["fundamental_score", "business_quality"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
            breakout = self._dynamic_lookup(flat_data, ["breakout_score"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
            ipo = self._dynamic_lookup(flat_data, ["ipo_score"], 0.0)

            # 5. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(rec_strength, conviction, readiness)

            # 6. Core Watchlist Models
            self._add_step(ctx, "COMPUTE_WATCHLIST_SCORE")
            
            base_watchlist_score = self._normalize(
                (rec_strength * dynamic_weights["recommendation_strength"]) +
                (conviction * dynamic_weights["conviction"]) +
                (master_rank * dynamic_weights["master_rank"]) +
                (readiness * dynamic_weights["execution_readiness"]) +
                (reward * dynamic_weights["reward_quality"]) +
                (risk_safety * dynamic_weights["risk_safety"])
            )

            # Calculate Trigger Probability (How close is it to firing?)
            trigger_prob = self._normalize((readiness * 0.6) + (conviction * 0.4))
            monitoring_quality = self._normalize((confidence * 0.5) + (risk_safety * 0.5))

            wl_profile = WatchlistProfile(
                watchlist_score=round(base_watchlist_score, 2),
                trigger_probability=round(trigger_prob, 2),
                monitoring_quality=round(monitoring_quality, 2),
                execution_readiness=round(readiness, 2),
                watchlist_quality=round((base_watchlist_score + monitoring_quality) / 2.0, 2)
            )

            components = [
                WatchlistComponent("Recommendation Power", round(rec_strength, 2), dynamic_weights.get("recommendation_strength", 0.25), round(confidence, 2)),
                WatchlistComponent("Actionable Conviction", round(conviction, 2), dynamic_weights.get("conviction", 0.20), round(confidence, 2)),
                WatchlistComponent("Execution Proximity", round(readiness, 2), dynamic_weights.get("execution_readiness", 0.15), round(confidence, 2))
            ]

            # 7. Determine Status, Type, and Triggers
            self._add_step(ctx, "DETERMINE_STATE_AND_TRIGGERS")
            wl_status = self._determine_status(rec_str, base_watchlist_score, readiness, is_permitted)
            wl_type = self._determine_type(rec_str, conviction, breakout, fund, inst, ipo, regime)
            wl_window, wl_priority = self._determine_window_and_priority(wl_status, wl_type, base_watchlist_score)
            wl_freq, next_review = self._determine_monitoring_policy(wl_window)
            
            trigger_obj = self._build_triggers(wl_type, readiness, wl_status)

            # 8. Checklists
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = WatchlistChecklist(
                recommendation_ok=(rec_str in ["STRONG_BUY", "BUY", "ACCUMULATE", "WATCH"]),
                ranking_ok=(master_rank >= 50.0),
                risk_ok=(overall_risk <= 60.0),
                reward_ok=(reward >= 50.0),
                confidence_ok=(confidence >= 50.0),
                conviction_ok=(conviction >= 50.0),
                market_ok=(regime >= 40.0)
            )

            # 9. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, wl_status, wl_type, conviction, overall_risk, fund, regime, confidence, is_permitted
            )

            # Adjust Score Post-Conflict
            final_watchlist_score = self._normalize(base_watchlist_score - (ctx.conflict_penalty * 0.4))
            final_wl_quality = self._normalize(wl_profile.watchlist_quality - (ctx.conflict_penalty * 0.4))

            # 10. Build Reason
            self._add_step(ctx, "BUILD_WATCHLIST_REASON")
            w_reason = WatchlistReason(
                primary=self._get_primary_reason(wl_type, wl_status, is_permitted),
                secondary=f"Trigger Prob: {round(trigger_prob,1)}% | Readiness: {round(readiness,1)}%",
                confidence=round(confidence, 2)
            )

            # 11. Explanations & Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, wl_status, wl_type, wl_freq, is_permitted, conviction, readiness, inst, fund
            )

            # 12. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "watchlist_status": wl_status.value,
                "watchlist_type": wl_type.value,
                "watchlist_score": round(final_watchlist_score, 2),
                "watchlist_priority": wl_priority.value,
                "watchlist_window": wl_window.value,
                "watchlist_trigger": asdict(trigger_obj),
                "watchlist_profile": asdict(wl_profile),
                "watchlist_components": [asdict(c) for c in components],
                "watchlist_reason": asdict(w_reason),
                "watchlist_checklist": asdict(checklist),
                "monitoring_frequency": wl_freq.value,
                "next_review": next_review,
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS
            status_msg = "Opportunity successfully mapped to monitoring engine."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=confidence,
                rating_score=final_wl_quality, # Engine rating based on overall watchlist opportunity quality
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Watchlist Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(stock_output, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, rec: float, conviction: float, readiness: float) -> dict[str, float]:
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["recommendation_strength"] += (rec / 100.0) * t.get("adaptive_rec_mult", 0.20)
        w["conviction"] += (conviction / 100.0) * t.get("adaptive_conviction_mult", 0.20)
        w["execution_readiness"] += (readiness / 100.0) * t.get("adaptive_readiness_mult", 0.15)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_status(self, rec_str: str, score: float, readiness: float, permitted: bool) -> WatchlistStatus:
        if rec_str in ["SELL", "REDUCE", "AVOID"]:
            return WatchlistStatus.REMOVED
        if not permitted:
            return WatchlistStatus.ARCHIVED
            
        if rec_str in ["STRONG_BUY", "BUY"]:
            if readiness >= 80.0:
                return WatchlistStatus.READY
            else:
                return WatchlistStatus.ACTIVE
                
        if rec_str == "ACCUMULATE":
            return WatchlistStatus.ACTIVE
            
        if score >= 50.0 or rec_str == "WATCH":
            return WatchlistStatus.MONITOR
            
        return WatchlistStatus.ARCHIVED

    def _determine_type(self, rec: str, conviction: float, breakout: float, fund: float, inst: float, ipo: float, regime: float) -> WatchlistType:
        if ipo > 50.0: return WatchlistType.IPO
        if rec in ["STRONG_BUY", "BUY"] and conviction > 80.0: return WatchlistType.HIGH_CONVICTION
        if fund > 80.0 and regime > 50.0: return WatchlistType.COMPOUNDER
        if breakout > 75.0: return WatchlistType.BREAKOUT
        if inst > 75.0: return WatchlistType.INSTITUTIONAL
        if fund > 70.0 and regime < 40.0: return WatchlistType.DEFENSIVE
        if rec == "ACCUMULATE": return WatchlistType.PULLBACK
        
        return WatchlistType.VALUE

    def _build_triggers(self, wl_type: WatchlistType, readiness: float, status: WatchlistStatus) -> WatchlistTrigger:
        """Constructs explicit, actionable conditions that must be met to convert Watchlist to Portfolio."""
        if status in [WatchlistStatus.REMOVED, WatchlistStatus.ARCHIVED]:
            return WatchlistTrigger("N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
            
        if wl_type == WatchlistType.BREAKOUT:
            return WatchlistTrigger(
                trigger_condition="Price breaks and holds above identified resistance.",
                entry_trigger="Execution on close above resistance zone.",
                price_trigger="> Resistance Level",
                volume_trigger="Relative Volume > 1.5x average.",
                momentum_trigger="RSI > 60 and MACD ascending.",
                risk_trigger="Stop Loss cleanly established below breakout candle.",
                confirmation_trigger="Hourly/Daily close confirmation."
            )
        elif wl_type == WatchlistType.PULLBACK or wl_type == WatchlistType.COMPOUNDER:
            return WatchlistTrigger(
                trigger_condition="Price retraces to logical support or value zone.",
                entry_trigger="Limit order at established support / discount zone.",
                price_trigger="Touch/Rejection at Support",
                volume_trigger="Volume dries up on pullback.",
                momentum_trigger="Oversold reading turning upward.",
                risk_trigger="Risk/Reward ratio > 2.5 on entry.",
                confirmation_trigger="Bullish reversal candlestick pattern."
            )
        elif wl_type == WatchlistType.INSTITUTIONAL:
            return WatchlistTrigger(
                trigger_condition="Institutional order block entry confirmed.",
                entry_trigger="Price tests and rejects order block floor.",
                price_trigger="Inside Order Block",
                volume_trigger="Smart Money Volume Spike.",
                momentum_trigger="Positive divergence on lower timeframes.",
                risk_trigger="Volatility contraction in zone.",
                confirmation_trigger="Institutional footprint detected via Level 2 / Delivery data."
            )
        else: # High Conviction / Default
            action = "Immediate execution upon opening alignment." if readiness > 80 else "Wait for final alignment."
            return WatchlistTrigger(
                trigger_condition="Awaiting final systemic alignment.",
                entry_trigger=action,
                price_trigger="Near current market price if conditions hold.",
                volume_trigger="Standard supporting volume.",
                momentum_trigger="Trend alignment sustained.",
                risk_trigger="Market regime remains supportive.",
                confirmation_trigger="All L4 Engines green."
            )

    def _determine_window_and_priority(self, status: WatchlistStatus, wl_type: WatchlistType, score: float):
        if status in [WatchlistStatus.REMOVED, WatchlistStatus.ARCHIVED]:
            return WatchlistWindow.NONE, WatchlistPriority.NONE
            
        if status == WatchlistStatus.READY:
            priority = WatchlistPriority.CRITICAL
            window = WatchlistWindow.TODAY
        elif status == WatchlistStatus.ACTIVE:
            priority = WatchlistPriority.HIGH
            window = WatchlistWindow.SHORT_TERM
        else:
            priority = WatchlistPriority.MEDIUM if score > 60 else WatchlistPriority.LOW
            window = WatchlistWindow.MEDIUM_TERM
            
        if wl_type == WatchlistType.COMPOUNDER: window = WatchlistWindow.LONG_TERM
        if wl_type == WatchlistType.IPO: window = WatchlistWindow.EVENT_DRIVEN
        
        return window, priority

    def _determine_monitoring_policy(self, window: WatchlistWindow) -> tuple[MonitoringFrequency, str]:
        if window == WatchlistWindow.TODAY:
            return MonitoringFrequency.EVERY_SCAN, "Continuous/Intraday"
        if window == WatchlistWindow.SHORT_TERM:
            return MonitoringFrequency.DAILY, "Next Daily Close"
        if window == WatchlistWindow.MEDIUM_TERM:
            return MonitoringFrequency.WEEKLY, "Next Weekly Close"
        if window in [WatchlistWindow.LONG_TERM, WatchlistWindow.MULTI_YEAR]:
            return MonitoringFrequency.MONTHLY, "Next Monthly Review"
        if window == WatchlistWindow.EVENT_DRIVEN:
            return MonitoringFrequency.EVENT_TRIGGERED, "Upon Event Execution"
            
        return MonitoringFrequency.NONE, "N/A"

    def _get_primary_reason(self, wl_type: WatchlistType, status: WatchlistStatus, permitted: bool) -> str:
        if not permitted: return "Market permission denied. Watchlist item archived."
        if status == WatchlistStatus.READY: return "High-conviction opportunity aligned. Awaiting exact entry trigger."
        
        if wl_type == WatchlistType.PULLBACK: return "Exceptional business/setup awaiting a favorable pullback entry."
        if wl_type == WatchlistType.INSTITUTIONAL: return "Institutional accumulation detected. Confirmation pending."
        if wl_type == WatchlistType.BREAKOUT: return "High-conviction opportunity pending structural breakout confirmation."
        if wl_type == WatchlistType.COMPOUNDER: return "Outstanding compounder identified for structured long-term monitoring."
        if wl_type == WatchlistType.IPO: return "New listing requires post-IPO stabilization prior to activation."
        
        return "Opportunity flagged for continuous tracking."

    def _detect_conflicts(self, ctx: DecisionContext, status: WatchlistStatus, wl_type: WatchlistType, 
                          conviction: float, risk: float, fund: float, regime: float, conf: float, permitted: bool) -> None:
        """Identifies paradoxes in the monitoring categorization."""
        
        # 1. ACTIVE + Market Blocked
        if status == WatchlistStatus.ACTIVE and not permitted:
            self._add_conflict(ctx, "Active monitoring proposed despite Market Gatekeeper blockade.", penalty=20.0)
            
        # 2. READY + Very High Risk
        if status == WatchlistStatus.READY and risk > 80.0:
            self._add_conflict(ctx, "Stock marked READY for execution amidst extremely high structural risk.", penalty=15.0)
            
        # 3. HIGH_CONVICTION + Low Confidence
        if wl_type == WatchlistType.HIGH_CONVICTION and conf < 50.0:
            self._add_conflict(ctx, "High Conviction watchlist assignment contradicts low underlying analytical confidence.", penalty=20.0)
            
        # 4. COMPOUNDER + Weak Fundamentals
        if wl_type == WatchlistType.COMPOUNDER and fund < 50.0:
            self._add_conflict(ctx, "Compounder watchlist assigned to asset with objectively weak business fundamentals.", penalty=25.0)
            
        # 5. BREAKOUT + Bear Market
        if wl_type == WatchlistType.BREAKOUT and regime < 35.0:
            self._add_conflict(ctx, "Breakout monitoring assigned during hostile bearish regime (High Fake-out Trap Risk).", penalty=15.0)

    def _generate_flags_and_explanations(self, ctx: DecisionContext, status: WatchlistStatus, 
                                         wl_type: WatchlistType, freq: MonitoringFrequency, permitted: bool, 
                                         conviction: float, readiness: float, inst: float, fund: float) -> list[str]:
        flags = []
        
        if not permitted:
            self._add_explanation(ctx, "Market execution blocked. Removing from active pipeline.")
            flags.append("GATEKEEPER_WATCHLIST_REMOVAL")
            return flags

        # Explanations
        if status == WatchlistStatus.READY:
            self._add_explanation(ctx, "All systemic preconditions met. Stock is locked and loaded for immediate trigger.")
        elif status == WatchlistStatus.ACTIVE:
            self._add_explanation(ctx, "Stock is actively stalked. Awaiting final technical/fundamental alignment.")
            
        if freq == MonitoringFrequency.EVERY_SCAN:
            self._add_explanation(ctx, "Opportunity severity requires continuous tick/intraday algorithmic scanning.")
            flags.append("ELEVATED_SCAN_FREQUENCY")
            
        if wl_type == WatchlistType.PULLBACK:
            self._add_explanation(ctx, "Monitoring for value-zone retracement to optimize risk/reward ratio.")
        elif wl_type == WatchlistType.INSTITUTIONAL:
            self._add_explanation(ctx, "Tracking smart money footprints to confirm definitive accumulation vector.")
            
        if readiness < 40.0 and status in [WatchlistStatus.ACTIVE, WatchlistStatus.READY]:
            self._add_explanation(ctx, "Warning: Execution readiness mathematically lags behind watchlist urgency.")
            flags.append("READINESS_LAG")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Logical paradoxes detected in monitoring assignment. Priority mathematically degraded.")
            flags.append("WATCHLIST_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK (Single Stock)
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Watchlist Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "watchlist_status": WatchlistStatus.REMOVED.value,
            "watchlist_type": WatchlistType.NONE.value,
            "watchlist_score": 0.0,
            "watchlist_priority": WatchlistPriority.NONE.value,
            "watchlist_window": WatchlistWindow.NONE.value,
            "watchlist_trigger": asdict(WatchlistTrigger("ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR")),
            "watchlist_profile": asdict(WatchlistProfile(0.0, 0.0, 0.0, 0.0, 0.0)),
            "watchlist_components": [],
            "watchlist_reason": asdict(WatchlistReason("SYSTEM_FAILURE", "Engine crash on this candidate", 0.0)),
            "watchlist_checklist": asdict(WatchlistChecklist(False, False, False, False, False, False, False)),
            "monitoring_frequency": MonitoringFrequency.NONE.value,
            "next_review": "N/A",
            "risk_flags": ["SYSTEM_FAILURE_WATCHLIST"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Watchlist evaluation failed. Candidate ignored.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_watchlist(stock_output: dict[str, Any] | None) -> dict[str, Any]:
    return WatchlistEngine().evaluate_watchlist(stock_output)
