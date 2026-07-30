"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: portfolio_engine.py

Institutional Portfolio Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates strictly as the Macro Portfolio 
Intelligence and Optimization Boundary.

Key Responsibilities:
- Assesses Holistic Portfolio Health, Diversification, Risk, and Expected CAGR.
- Evaluates Sector Concentration and Cash Utilization relative to Market Regime.
- Generates Portfolio-Level Rebalancing Suggestions (Add, Reduce, Remove, Promote).

Boundary Constraint: This engine DOES NOT calculate single-stock indicators, Entry, 
StopLoss, Targets, Position Sizing, or Individual Stock Recommendations.
"""

import time
import json
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
class PortfolioHealth(str, Enum):
    EXCELLENT = "EXCELLENT"
    STRONG = "STRONG"
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

class PortfolioAction(str, Enum):
    ADD = "ADD"
    REDUCE = "REDUCE"
    REMOVE = "REMOVE"
    HOLD = "HOLD"
    REBALANCE = "REBALANCE"
    WATCH = "WATCH"
    NONE = "NONE"

class PortfolioPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

class PortfolioWindow(str, Enum):
    TODAY = "TODAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    MULTI_YEAR = "MULTI_YEAR"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 Dashboard / Explanation Parsers)
# =====================================================================
@dataclass(frozen=True)
class PortfolioProfile:
    portfolio_score: float
    portfolio_quality: float
    portfolio_risk: float
    portfolio_reward: float
    portfolio_cagr: float
    portfolio_drawdown: float
    cash_utilization_pct: float
    diversification_score: float

@dataclass(frozen=True)
class PortfolioHolding:
    symbol: str
    allocation_pct: float
    sector: str
    risk: float
    reward: float
    recommendation: str
    conviction: float
    confidence: float

@dataclass(frozen=True)
class RebalancingSuggestion:
    action: str
    target_symbol: str
    sector: str
    priority: str
    reason: str

@dataclass(frozen=True)
class PortfolioComponent:
    name: str
    score: float
    weight: float
    confidence: float

@dataclass(frozen=True)
class PortfolioReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class PortfolioChecklist:
    diversified: bool
    risk_ok: bool
    cash_ok: bool
    allocation_ok: bool
    sector_ok: bool
    conviction_ok: bool
    quality_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_portfolio_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Portfolio_V3",
        version="3.0.0",
        stage="Layer-5: Portfolio Intelligence",
        schema_version="3.0",
        api_version="v6",
        decision_method="Holistic Risk-Parity & Regime-Adaptive Optimization",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "weighted_quality": 0.30,
            "diversification": 0.20,
            "regime_alignment": 0.20,
            "risk_efficiency": 0.15,
            "cash_optimization": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "max_sector_exposure": 25.0,     # Max 25% in one sector
            "max_single_stock_exposure": 15.0, # Max 15% in one stock
            "bear_regime_min_cash": 30.0,    # Minimum cash % in bear market
            "bull_regime_max_cash": 15.0,    # Max cash % in strong bull market (avoid cash drag)
            "critical_drawdown_risk": 25.0,
            "excellent_health_threshold": 80.0,
            "strong_health_threshold": 65.0,
            "moderate_health_threshold": 45.0
        }
    )


# =====================================================================
# PORTFOLIO DECISION ENGINE
# =====================================================================
class PortfolioEngine(BaseDecisionEngine):
    """
    Institutional Portfolio Engine.
    Aggregates active holdings, watchlist candidates, and cash positions to 
    evaluate holistic systemic health, correlation dangers, and rebalancing needs.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_portfolio_profile())

    # Injected evaluate alias for BaseDecisionEngine compatibility
    def evaluate(self, *args, **kwargs) -> Any:
        return self.evaluate_portfolio(*args, **kwargs)

    def evaluate_portfolio(self, portfolio: list[dict[str, Any]], watchlist: list[dict[str, Any]], cash_pct: float) -> dict[str, Any]:
        """
        Main execution pipeline for Portfolio Intelligence.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_PORTFOLIO_ENGINE")
        
        # Package inputs for deterministic tracing
        payload_envelope = {"portfolio_count": len(portfolio), "watchlist_count": len(watchlist), "cash": cash_pct}
        trace = self._build_trace(payload_envelope, start_time, ctx, len(portfolio) + len(watchlist))

        try:
            self._add_step(ctx, "EXTRACT_AND_AGGREGATE_HOLDINGS")
            
            # 1. Parse Holdings & Sectors
            parsed_holdings = []
            sector_allocations = {}
            total_invested_pct = 0.0
            
            # Extract Macro Regime from the first holding (Systemic state is global)
            global_regime = 50.0
            if portfolio:
                flat_first = self._flatten_dict(portfolio[0])
                global_regime = self._dynamic_lookup(flat_first, ["regime_score", "market_regime", "market_stability"], 50.0)

            # Process Each Holding
            for stock in portfolio:
                flat = self._flatten_dict(stock)
                
                sym = self._dynamic_lookup_string(flat, ["symbol", "ticker", "asset"]).upper()
                sec = self._dynamic_lookup_string(flat, ["sector", "industry", "segment"]).upper()
                if sec == "UNKNOWN": sec = "GENERAL"
                
                alloc = self._dynamic_lookup(flat, ["allocation_pct", "portfolio_exposure_pct", "weight"], 0.0)
                risk = self._dynamic_lookup(flat, ["overall_risk", "composite_risk"], 50.0)
                reward = self._dynamic_lookup(flat, ["reward_score", "expected_return_pct", "reward_quality"], 50.0)
                rec = self._dynamic_lookup_string(flat, ["recommendation", "final_recommendation"]).upper()
                conv = self._dynamic_lookup(flat, ["conviction_score", "actionability_score"], 50.0)
                conf = self._dynamic_lookup(flat, ["confidence_score", "reliability_score"], 50.0)
                
                holding_obj = PortfolioHolding(sym, alloc, sec, risk, reward, rec, conv, conf)
                parsed_holdings.append(holding_obj)
                
                total_invested_pct += alloc
                sector_allocations[sec] = sector_allocations.get(sec, 0.0) + alloc

            # Calculate Actual Cash if discrepancy exists
            actual_cash_pct = max(0.0, 100.0 - total_invested_pct) if total_invested_pct > 0 else cash_pct

            # 2. Compute Weighted Portfolio Metrics
            self._add_step(ctx, "COMPUTE_WEIGHTED_METRICS")
            w_quality, w_risk, w_reward, w_cagr, w_dd = self._compute_weighted_metrics(portfolio, total_invested_pct)
            
            # 3. Evaluate Diversification & Concentration
            self._add_step(ctx, "EVALUATE_DIVERSIFICATION")
            div_score, concentration_flags = self._evaluate_diversification(
                parsed_holdings, sector_allocations, actual_cash_pct, global_regime
            )

            # 4. Generate Rebalancing Suggestions
            self._add_step(ctx, "GENERATE_REBALANCING_PLAN")
            rebalancing_suggestions = self._generate_rebalancing(
                parsed_holdings, watchlist, sector_allocations, actual_cash_pct, global_regime
            )

            # 5. Core Portfolio Scoring Model
            self._add_step(ctx, "CALCULATE_PORTFOLIO_SCORE")
            cash_optimization = self._evaluate_cash_optimization(actual_cash_pct, global_regime)
            regime_alignment = self._normalize(global_regime)
            risk_efficiency = self._normalize(w_reward - w_risk + 50.0) # Reward vs Risk spread
            
            base_portfolio_score = self._normalize(
                (w_quality * self.config.base_weights["weighted_quality"]) +
                (div_score * self.config.base_weights["diversification"]) +
                (regime_alignment * self.config.base_weights["regime_alignment"]) +
                (risk_efficiency * self.config.base_weights["risk_efficiency"]) +
                (cash_optimization * self.config.base_weights["cash_optimization"])
            )

            # 6. Categorize Health & Priorities
            self._add_step(ctx, "CATEGORIZE_HEALTH")
            health, priority, window, main_action = self._determine_portfolio_health(
                base_portfolio_score, w_risk, actual_cash_pct, global_regime, len(rebalancing_suggestions)
            )

            # 7. Checklists & Conflicts
            self._add_step(ctx, "EVALUATE_CHECKLISTS_AND_CONFLICTS")
            t = self.config.thresholds
            
            checklist = PortfolioChecklist(
                diversified=(div_score >= 70.0),
                risk_ok=(w_risk <= 60.0),
                cash_ok=(cash_optimization >= 50.0),
                allocation_ok=True, # Assuming mathematical constraints hold
                sector_ok=(not any(v > t.get("max_sector_exposure", 25.0) for v in sector_allocations.values())),
                conviction_ok=(all(h.conviction >= 40.0 for h in parsed_holdings) if parsed_holdings else False),
                quality_ok=(w_quality >= 50.0)
            )

            self._detect_conflicts(
                ctx, health, div_score, sector_allocations, w_cagr, w_dd, actual_cash_pct, global_regime
            )

            final_portfolio_score = self._normalize(base_portfolio_score - (ctx.conflict_penalty * 0.5))

            # 8. Build Reason & Profile
            self._add_step(ctx, "BUILD_PROFILE_AND_REASONS")
            p_reason = PortfolioReason(
                primary=f"Portfolio structural health is {health.value}.",
                secondary=f"Diversification: {round(div_score,1)}% | Risk/Reward: {round(w_risk,1)}/{round(w_reward,1)}",
                confidence=round(self._normalize((w_quality + div_score) / 2.0), 2)
            )

            p_profile = PortfolioProfile(
                portfolio_score=round(final_portfolio_score, 2),
                portfolio_quality=round(w_quality, 2),
                portfolio_risk=round(w_risk, 2),
                portfolio_reward=round(w_reward, 2),
                portfolio_cagr=round(w_cagr, 2),
                portfolio_drawdown=round(w_dd, 2),
                cash_utilization_pct=round(actual_cash_pct, 2),
                diversification_score=round(div_score, 2)
            )

            # 9. Explanations & Flags
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, health, concentration_flags, actual_cash_pct, global_regime, w_cagr, w_dd, div_score, rebalancing_suggestions
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "portfolio_health": health.value,
                "portfolio_action": main_action.value,
                "portfolio_priority": priority.value,
                "portfolio_window": window.value,
                "portfolio_profile": asdict(p_profile),
                "sector_allocations": {k: round(v, 2) for k, v in sector_allocations.items()},
                "rebalancing_suggestions": [asdict(r) for r in rebalancing_suggestions],
                "portfolio_reason": asdict(p_reason),
                "portfolio_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS
            status_msg = "Holistic Portfolio Intelligence generated successfully."

            object.__setattr__(trace, 'steps_executed', ctx.steps)
            object.__setattr__(trace, 'inputs_parsed', len(portfolio) + len(watchlist))

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=round(self._normalize((w_quality + div_score) / 2.0), 2),
                rating_score=final_portfolio_score, 
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Portfolio Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(trace))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _compute_weighted_metrics(self, portfolio: list[dict[str, Any]], total_invested: float):
        """Calculates portfolio-level weighted averages across all active holdings."""
        if not portfolio or total_invested == 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
            
        t_qual, t_risk, t_reward, t_cagr, t_dd = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for stock in portfolio:
            flat = self._flatten_dict(stock)
            alloc = self._dynamic_lookup(flat, ["allocation_pct", "portfolio_exposure_pct", "weight"], 0.0)
            weight = alloc / total_invested
            
            t_qual += self._dynamic_lookup(flat, ["master_score", "overall_quality", "conviction_score"], 50.0) * weight
            t_risk += self._dynamic_lookup(flat, ["overall_risk", "composite_risk"], 50.0) * weight
            t_reward += self._dynamic_lookup(flat, ["reward_score", "reward_quality"], 50.0) * weight
            t_cagr += self._dynamic_lookup(flat, ["expected_cagr_pct", "cagr"], 0.0) * weight
            t_dd += self._dynamic_lookup(flat, ["drawdown_risk", "max_drawdown"], 0.0) * weight
            
        return t_qual, t_risk, t_reward, t_cagr, t_dd

    def _evaluate_diversification(self, holdings: list[PortfolioHolding], sectors: dict[str, float], cash: float, regime: float):
        """Calculates a robust diversification score penalizing extreme concentrations."""
        t = self.config.thresholds
        div_score = 100.0
        flags = []
        
        # Sector Penalty
        max_sec_limit = t.get("max_sector_exposure", 25.0)
        for sec, alloc in sectors.items():
            if alloc > max_sec_limit:
                penalty = (alloc - max_sec_limit) * 1.5
                div_score -= penalty
                flags.append(f"Sector Overweight: {sec} ({round(alloc,1)}%)")
                
        # Single Stock Penalty
        max_stock_limit = t.get("max_single_stock_exposure", 15.0)
        for h in holdings:
            if h.allocation_pct > max_stock_limit:
                penalty = (h.allocation_pct - max_stock_limit) * 2.0
                div_score -= penalty
                flags.append(f"Stock Overweight: {h.symbol} ({round(h.allocation_pct,1)}%)")
                
        # Correlation Proxy (Fewer sectors = High correlation risk)
        active_sectors = sum(1 for v in sectors.values() if v > 2.0)
        if active_sectors < 3 and len(holdings) > 3:
            div_score -= 15.0
            flags.append("High Correlation Risk: Insufficient sector breadth.")
            
        return self._clamp(div_score, 0.0, 100.0), flags

    def _evaluate_cash_optimization(self, cash: float, regime: float) -> float:
        """Determines if the cash level is appropriate for the macro market regime."""
        t = self.config.thresholds
        score = 100.0
        
        if regime < 40.0: # Bear Market
            min_cash = t.get("bear_regime_min_cash", 30.0)
            if cash < min_cash:
                score -= (min_cash - cash) * 2.0 # Penalty for being fully invested in a bear market
        elif regime > 70.0: # Strong Bull Market
            max_cash = t.get("bull_regime_max_cash", 15.0)
            if cash > max_cash:
                score -= (cash - max_cash) * 2.0 # Penalty for cash drag in a bull market
                
        return self._clamp(score, 0.0, 100.0)

    def _generate_rebalancing(self, holdings: list[PortfolioHolding], watchlist: list[dict[str, Any]], 
                              sectors: dict[str, float], cash: float, regime: float) -> list[RebalancingSuggestion]:
        suggestions = []
        t = self.config.thresholds
        
        # 1. Identify Reductions / Removals from Holdings
        for h in holdings:
            if h.recommendation in ["SELL", "AVOID"]:
                suggestions.append(RebalancingSuggestion("REMOVE", h.symbol, h.sector, "CRITICAL", "Exit recommendation generated by L5 Engine."))
            elif h.recommendation == "REDUCE":
                suggestions.append(RebalancingSuggestion("REDUCE", h.symbol, h.sector, "HIGH", "Risk mitigation reduction advised."))
            elif h.allocation_pct > t.get("max_single_stock_exposure", 15.0):
                suggestions.append(RebalancingSuggestion("REDUCE", h.symbol, h.sector, "MEDIUM", "Position exceeds maximum single-stock exposure limits."))
                
        # 2. Identify Sector Overweights
        max_sec_limit = t.get("max_sector_exposure", 25.0)
        overweight_sectors = [sec for sec, alloc in sectors.items() if alloc > max_sec_limit]
        
        # 3. Assess Watchlist Promotions
        if watchlist:
            # Sort watchlist by score
            wl_sorted = sorted(watchlist, key=lambda x: self._dynamic_lookup(self._flatten_dict(x), ["watchlist_score", "master_score"], 0.0), reverse=True)
            
            for w in wl_sorted:
                flat_w = self._flatten_dict(w)
                sym = self._dynamic_lookup_string(flat_w, ["symbol"]).upper()
                sec = self._dynamic_lookup_string(flat_w, ["sector"]).upper()
                status = self._dynamic_lookup_string(flat_w, ["watchlist_status", "status"]).upper()
                
                if status == "READY":
                    # Check if sector is full
                    if sec in overweight_sectors:
                        suggestions.append(RebalancingSuggestion("WATCH", sym, sec, "LOW", "Candidate is READY, but target sector is currently overweight."))
                    elif cash > 10.0:
                        suggestions.append(RebalancingSuggestion("ADD", sym, sec, "HIGH", "High-conviction watchlist candidate triggered. Cash available for deployment."))
                    else:
                        suggestions.append(RebalancingSuggestion("REBALANCE", sym, sec, "MEDIUM", "Candidate triggered, but requires capital reallocation (raise cash)."))

        # 4. Macro Cash Adjustments
        if regime < 40.0 and cash < t.get("bear_regime_min_cash", 30.0):
            suggestions.append(RebalancingSuggestion("RAISE_CASH", "PORTFOLIO", "MACRO", "HIGH", "Defensive cash reserves severely below bear market minimums."))
            
        return suggestions

    def _determine_portfolio_health(self, score: float, risk: float, cash: float, regime: float, rebalance_count: int):
        t = self.config.thresholds
        
        if score >= t.get("excellent_health_threshold", 80.0) and risk < 40.0:
            health = PortfolioHealth.EXCELLENT
        elif score >= t.get("strong_health_threshold", 65.0):
            health = PortfolioHealth.STRONG
        elif score >= t.get("moderate_health_threshold", 45.0):
            health = PortfolioHealth.GOOD if risk < 60.0 else PortfolioHealth.MODERATE
        elif score >= 30.0:
            health = PortfolioHealth.WEAK
        else:
            health = PortfolioHealth.CRITICAL

        # Main Action & Priority
        if rebalance_count > 3 or health in [PortfolioHealth.WEAK, PortfolioHealth.CRITICAL]:
            action = PortfolioAction.REBALANCE
            priority = PortfolioPriority.CRITICAL
            window = PortfolioWindow.TODAY
        elif rebalance_count > 0:
            action = PortfolioAction.REBALANCE
            priority = PortfolioPriority.HIGH
            window = PortfolioWindow.SHORT_TERM
        else:
            action = PortfolioAction.HOLD
            priority = PortfolioPriority.LOW
            window = PortfolioWindow.LONG_TERM

        return health, priority, window, action

    def _detect_conflicts(self, ctx: DecisionContext, health: PortfolioHealth, div_score: float, 
                          sectors: dict[str, float], cagr: float, dd: float, cash: float, regime: float) -> None:
        """Identifies systemic paradoxes at the macro portfolio level."""
        
        # 1. Excellent Portfolio + High Concentration
        if health in [PortfolioHealth.EXCELLENT, PortfolioHealth.STRONG] and div_score < 40.0:
            self._add_conflict(ctx, "High portfolio score flagged despite severe lack of sector/asset diversification.", penalty=15.0)
            
        # 2. Diversified + 80% One Sector
        if div_score > 80.0 and any(alloc > 60.0 for alloc in sectors.values()):
            self._add_conflict(ctx, "Mathematical diversification contradicts massive single-sector allocation.", penalty=20.0)
            
        # 3. High CAGR + Extreme Drawdown
        if cagr > 30.0 and dd > 40.0:
            self._add_conflict(ctx, "High expected CAGR structurally undermined by extreme portfolio drawdown risks.", penalty=15.0)
            
        # 4. Zero Cash + Bear Market
        if cash < 5.0 and regime < 35.0:
            self._add_conflict(ctx, "Zero cash utilization completely contradicts hostile bearish market realities.", penalty=20.0)

    def _generate_flags_and_explanations(self, ctx: DecisionContext, health: PortfolioHealth, 
                                         concentration_flags: list[str], cash: float, regime: float, 
                                         cagr: float, dd: float, div_score: float, rebal: list[RebalancingSuggestion]) -> list[str]:
        flags = []
        
        # Add concentration warnings directly
        for f in concentration_flags:
            self._add_warning(ctx, f, WarningSeverityEnum.RISK)
            flags.append("CONCENTRATION_RISK")

        # Explanations
        if concentration_flags:
            self._add_explanation(ctx, "Portfolio concentration exceeds optimal institutional limits in specified assets/sectors.")
        elif div_score > 75.0:
            self._add_explanation(ctx, "Portfolio diversification significantly improves long-term systemic resilience.")

        if regime < 40.0 and cash < 20.0:
            self._add_explanation(ctx, "Cash allocation is dangerously low for the current bearish market regime.")
            flags.append("INSUFFICIENT_CASH_BUFFER")
            
        if any(r.action == "ADD" for r in rebal):
            self._add_explanation(ctx, "High-conviction watchlist candidate identified for portfolio promotion.")
            
        if any(r.action == "REMOVE" for r in rebal):
            self._add_explanation(ctx, "Weak holding identified for immediate portfolio liquidation.")

        if cagr > 15.0 and dd < 20.0:
            self._add_explanation(ctx, "Risk-adjusted CAGR metrics remain exceptionally favorable relative to drawdown exposure.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Structural paradoxes identified within portfolio allocation logic.")
            flags.append("PORTFOLIO_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Portfolio Engine.", WarningSeverityEnum.CRITICAL)
        
        payload = {
            "portfolio_health": PortfolioHealth.UNKNOWN.value,
            "portfolio_action": PortfolioAction.NONE.value,
            "portfolio_priority": PortfolioPriority.NONE.value,
            "portfolio_window": PortfolioWindow.NONE.value,
            "portfolio_profile": asdict(PortfolioProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            "sector_allocations": {},
            "rebalancing_suggestions": [],
            "portfolio_reason": asdict(PortfolioReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "portfolio_checklist": asdict(PortfolioChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_PORTFOLIO"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Portfolio evaluation failed. Engine offline.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_portfolio(portfolio: list[dict[str, Any]], watchlist: list[dict[str, Any]], cash_pct: float) -> dict[str, Any]:
    return PortfolioEngine().evaluate_portfolio(portfolio, watchlist, cash_pct)
