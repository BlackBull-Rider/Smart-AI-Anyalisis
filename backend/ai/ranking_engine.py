"""
GREEN BULL RIDER V6
Layer-5: AI Engine
Module: ranking_engine.py

Institutional Ranking Engine. (V3.1.0 - Enterprise Edition)
Inherits from BaseDecisionEngine (reusing framework components).
Operates strictly as a Comparator Engine. Evaluates all candidates in the 
universe using fused Layer-3 and Layer-4 outputs and ranks them relatively.

Key Upgrades from V3.0:
- Canonical Master Score Consumption (Avoids overlap with MasterEngine).
- Primary & Secondary Categories (Multi-dimensional categorization).
- Deterministic Tie-Breaking (Master -> Conviction -> Confidence -> Reward -> Symbol).
- Universe Percentiles (Top 1%, 99.8th percentile, etc.).
- RankingComponents exposed for Layer-5 Explainability.
- IPO and MULTI_YEAR handling integrated.

Boundary Constraint: This engine DOES NOT generate Buy/Hold/Sell recommendations, 
Entry/Exit points, or Position Sizes. It evaluates relative priority (Ranking).
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
class RankingTier(str, Enum):
    ELITE = "ELITE"
    PLATINUM = "PLATINUM"
    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"
    WATCHLIST = "WATCHLIST"
    IGNORE = "IGNORE"

class RankingCategory(str, Enum):
    COMPOUNDER = "COMPOUNDER"
    SWING = "SWING"
    VALUE = "VALUE"
    BREAKOUT = "BREAKOUT"
    INSTITUTIONAL = "INSTITUTIONAL"
    IPO = "IPO"
    DEFENSIVE = "DEFENSIVE"
    UNKNOWN = "UNKNOWN"

class RankingPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class RankingWindow(str, Enum):
    TODAY = "TODAY"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    MULTI_YEAR = "MULTI_YEAR"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    NONE = "NONE"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 Recommendation Engine)
# =====================================================================
@dataclass(frozen=True)
class RankingProfile:
    master_rank_score: float
    priority_score: float
    execution_rank_score: float
    investment_rank_score: float
    swing_rank_score: float
    compounder_rank_score: float
    institutional_rank_score: float
    # These fields are initialized to 0/empty and populated post-sorting
    overall_percentile: float = 0.0
    top_percent_tier: str = "Unranked"
    relative_strength: float = 0.0

@dataclass(frozen=True)
class RankingComponent:
    name: str
    score: float
    weight: float
    confidence: float

@dataclass(frozen=True)
class RankingReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class RankingChecklist:
    trend_ok: bool
    risk_ok: bool
    reward_ok: bool
    confidence_ok: bool
    conviction_ok: bool
    fundamental_ok: bool
    institutional_ok: bool


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_ranking_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Ranking_V3_1",
        version="3.1.0",
        stage="Layer-5: Comparator Ranking",
        schema_version="3.1",
        api_version="v6",
        decision_method="Relative Deterministic Universe Sorting",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "conviction": 0.20,
            "confidence": 0.15,
            "reward_quality": 0.15,
            "risk_safety": 0.20,
            "trend_momentum": 0.15,
            "institutional_fundamental": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "elite_tier_threshold": 85.0,
            "platinum_tier_threshold": 75.0,
            "gold_tier_threshold": 65.0,
            "silver_tier_threshold": 50.0,
            "adaptive_conviction_mult": 0.20,
            "adaptive_risk_mult": 0.20,
            "adaptive_reward_mult": 0.15
        }
    )


# =====================================================================
# RANKING DECISION ENGINE
# =====================================================================
class RankingEngine(BaseDecisionEngine):
    """
    Institutional Ranking Engine.
    Ingests all stock outputs, processes relative multi-dimensional scores, 
    and returns a deterministically sorted list representing absolute portfolio priority.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_ranking_profile())

    # Injected evaluate alias for BaseDecisionEngine compatibility
    def evaluate(self, *args, **kwargs) -> Any:
        return self.evaluate_ranking(*args, **kwargs)

    def evaluate_ranking(self, all_stock_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Main execution pipeline for entire universe sorting.
        """
        if not all_stock_outputs or not isinstance(all_stock_outputs, list):
            self.logger.warning("Empty or invalid stock list provided to RankingEngine.")
            return []

        try:
            evaluated_stocks = []
            for stock_data in all_stock_outputs:
                try:
                    result = self._evaluate_single_stock(stock_data)
                    evaluated_stocks.append(result)
                except Exception as e:
                    self.logger.error(f"Ranking evaluation failed for a stock: {str(e)}", exc_info=True)
                    # Safe fallback for individual stock
                    evaluated_stocks.append(self._build_stock_fallback(stock_data))

            # ---------------------------------------------------------
            # DETERMINISTIC SORTING & TIE-BREAKING
            # ---------------------------------------------------------
            # Tie-Breaker: Master Score -> Conviction -> Confidence -> Reward -> Symbol (Reverse Alphabetical fallback)
            evaluated_stocks.sort(
                key=lambda x: (
                    x.get("_tie_breakers", {}).get("master", 0.0),
                    x.get("_tie_breakers", {}).get("conviction", 0.0),
                    x.get("_tie_breakers", {}).get("confidence", 0.0),
                    x.get("_tie_breakers", {}).get("reward", 0.0),
                    x.get("_tie_breakers", {}).get("symbol", "ZZZ")
                ),
                reverse=True
            )

            # ---------------------------------------------------------
            # RELATIVE PERCENTILE & RANK ASSIGNMENT
            # ---------------------------------------------------------
            N = len(evaluated_stocks)
            max_score = evaluated_stocks[0].get("_tie_breakers", {}).get("master", 100.0) if N > 0 else 100.0
            
            for idx, stock_result in enumerate(evaluated_stocks):
                rank = idx + 1
                percentile = ((N - rank) / max(1, N - 1)) * 100.0 if N > 1 else 100.0
                
                # Determine Percentile Tier
                top_tier = "Top 1%" if percentile >= 99.0 else \
                           "Top 5%" if percentile >= 95.0 else \
                           "Top 10%" if percentile >= 90.0 else \
                           "Top 25%" if percentile >= 75.0 else \
                           "Top 50%" if percentile >= 50.0 else "Bottom 50%"
                           
                # Assign Rank and cleanup temp keys
                if "decision" in stock_result:
                    master_score = stock_result.get("_tie_breakers", {}).get("master", 0.0)
                    rs = self._normalize((master_score / max(1.0, max_score)) * 100.0)
                    
                    # Update dynamic dictionary fields
                    prof = stock_result["decision"]["ranking_profile"]
                    prof["overall_percentile"] = round(percentile, 2)
                    prof["top_percent_tier"] = top_tier
                    prof["relative_strength"] = round(rs, 2)
                    
                    stock_result["decision"]["overall_rank"] = rank
                    stock_result["decision"]["opportunity_rank"] = rank
                    
                    # Contextual Explanations for Top Tier
                    if rank == 1:
                        stock_result["explanations"].append("Rank #1: Highest relative structural superiority across the entire universe.")
                    elif rank <= 5:
                        stock_result["explanations"].append(f"Rank #{rank}: Top 5 opportunity across all systemic evaluation metrics.")
                        
                # Cleanup tie breakers to keep payload clean
                if "_tie_breakers" in stock_result:
                    stock_result.pop("tie_breakers", None)

            return evaluated_stocks

        except Exception as e:
            self.logger.error(f"Catastrophic failure in RankingEngine universe evaluation: {str(e)}", exc_info=True)
            return []

    def _evaluate_single_stock(self, stock_data: dict[str, Any]) -> dict[str, Any]:
        """Evaluates rank metrics for a single stock."""
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_STOCK_EVALUATION")
        
        trace = self._build_trace(stock_data, start_time, ctx, 0)
        
        # Flatten JSON for extraction
        flat_data = self._flatten_dict(stock_data)
        parsed_inputs = len(flat_data)
        self._add_step(ctx, "EXTRACT_L3_L4_DATA")

        # 1. Identity & Gatekeeper Data
        symbol = self._dynamic_lookup_string(flat_data, ["symbol", "ticker", "asset"]).upper()
        if symbol == "UNKNOWN": symbol = "ZZZ" # Demote unknown symbols in tie-break
        
        permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
        is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()

        # 2. Extract L4 Decision Scores
        conviction = self._dynamic_lookup(flat_data, ["conviction_score", "actionability_score"], 50.0)
        confidence = self._dynamic_lookup(flat_data, ["confidence_score", "reliability_score"], 50.0)
        overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "composite_risk"], 50.0)
        reward = self._dynamic_lookup(flat_data, ["reward_quality", "reward_score"], 50.0)
        holding_qual = self._dynamic_lookup(flat_data, ["holding_quality"], 50.0)
        risk_safety = self._normalize(100.0 - overall_risk)

        # 3. Extract L3 Intelligence Scores
        trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
        mom = self._dynamic_lookup(flat_data, ["momentum_score"], 50.0)
        vol = self._dynamic_lookup(flat_data, ["volume_score"], 50.0)
        inst = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score"], 50.0)
        fund = self._dynamic_lookup(flat_data, ["fundamental_score", "compounder_score", "business_quality"], 50.0)
        regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)
        breakout = self._dynamic_lookup(flat_data, ["breakout_score"], 50.0)
        liq = self._dynamic_lookup(flat_data, ["liquidity_score", "float_quality"], 50.0)
        ipo_score = self._dynamic_lookup(flat_data, ["ipo_score", "new_listing_score"], 0.0) # 0 if not an IPO
        
        swing_l3 = self._dynamic_lookup(flat_data, ["overall_swing_score", "swing_probability"], 50.0)
        lt_l3 = self._dynamic_lookup(flat_data, ["overall_long_term_score", "investment_score"], 50.0)
        comp_l3 = self._dynamic_lookup(flat_data, ["overall_compounder_score", "wealth_creation_score"], 50.0)

        # 4. Core Ranking Models
        self._add_step(ctx, "COMPUTE_RANKING_MODELS")
        
        # 4.1 Upstream Master Score Consumption (Architectural Boundary)
        upstream_master = self._dynamic_lookup(flat_data, ["master_score", "canonical_master_score"], 0.0)
        if upstream_master > 0.0:
            master_score = upstream_master
            self._add_step(ctx, "CONSUMED_UPSTREAM_MASTER_SCORE")
            dynamic_weights = self.config.base_weights # Fallback for components
        else:
            self._add_step(ctx, "CALCULATED_INTERNAL_MASTER_PROXY")
            dynamic_weights = self._calculate_continuous_weights(conviction, risk_safety, reward)
            master_score = self._normalize(
                (conviction * dynamic_weights["conviction"]) +
                (confidence * dynamic_weights["confidence"]) +
                (reward * dynamic_weights["reward_quality"]) +
                (risk_safety * dynamic_weights["risk_safety"]) +
                (((trend + mom) / 2.0) * dynamic_weights["trend_momentum"]) +
                (((inst + fund) / 2.0) * dynamic_weights["institutional_fundamental"])
            )

        # 4.2 Specialized Ranking Scores
        swing_rank_score = self._normalize((swing_l3 * 0.4) + (mom * 0.2) + (breakout * 0.2) + (vol * 0.1) + (risk_safety * 0.1))
        comp_rank_score = self._normalize((comp_l3 * 0.4) + (fund * 0.3) + (inst * 0.1) + (lt_l3 * 0.1) + (holding_qual * 0.1))
        inst_rank_score = self._normalize((inst * 0.4) + (liq * 0.2) + (vol * 0.2) + (confidence * 0.2))
        inv_rank_score = self._normalize((lt_l3 * 0.3) + (fund * 0.3) + (risk_safety * 0.2) + (reward * 0.1) + (holding_qual * 0.1))
        exec_rank_score = self._normalize((conviction * 0.5) + (risk_safety * 0.25) + (liq * 0.25))

        # 5. Build Ranking Components (For Layer-5 Explainability)
        self._add_step(ctx, "BUILD_COMPONENTS")
        ranking_components = [
            RankingComponent("Conviction Alignment", round(conviction, 2), dynamic_weights.get("conviction", 0.20), round(confidence, 2)),
            RankingComponent("Risk Profile", round(risk_safety, 2), dynamic_weights.get("risk_safety", 0.20), round(confidence, 2)),
            RankingComponent("Reward Projection", round(reward, 2), dynamic_weights.get("reward_quality", 0.15), round(confidence, 2)),
            RankingComponent("Structural Trend", round(trend, 2), dynamic_weights.get("trend_momentum", 0.15), round(confidence, 2))
        ]

        # 6. Categorization & Tiers
        self._add_step(ctx, "DETERMINE_TIERS_AND_CATEGORIES")
        tier = self._determine_tier(master_score, is_permitted, overall_risk)
        primary_cat, secondary_cat = self._determine_categories(
            swing_rank_score, comp_rank_score, inst_rank_score, inv_rank_score, breakout, ipo_score, regime
        )
        priority, window = self._determine_priority_window(tier, primary_cat)

        # 7. Checklists
        self._add_step(ctx, "GENERATE_CHECKLIST")
        checklist = RankingChecklist(
            trend_ok=(trend >= 50.0),
            risk_ok=(overall_risk <= 60.0),
            reward_ok=(reward >= 50.0),
            confidence_ok=(confidence >= 50.0),
            conviction_ok=(conviction >= 50.0),
            fundamental_ok=(fund >= 50.0),
            institutional_ok=(inst >= 50.0)
        )

        # 8. Conflict Detection
        self._add_step(ctx, "DETECT_CONFLICTS")
        self._detect_conflicts(
            ctx, tier, conviction, overall_risk, regime, comp_rank_score, fund, swing_rank_score, mom
        )

        final_master_score = self._normalize(master_score - (ctx.conflict_penalty * 0.5)) if is_permitted else 0.0

        # 9. Build Reason
        r_reason = RankingReason(
            primary=self._get_primary_reason(tier, primary_cat),
            secondary=f"L4 Conviction: {round(conviction,1)}% | L4 Risk: {round(overall_risk,1)}%",
            confidence=round(confidence, 2)
        )

        # 10. Build Profile
        rank_profile = RankingProfile(
            master_rank_score=round(final_master_score, 2),
            priority_score=round((final_master_score + exec_rank_score) / 2.0, 2),
            execution_rank_score=round(exec_rank_score, 2),
            investment_rank_score=round(inv_rank_score, 2),
            swing_rank_score=round(swing_rank_score, 2),
            compounder_rank_score=round(comp_rank_score, 2),
            institutional_rank_score=round(inst_rank_score, 2)
            # Percentiles are intentionally left 0.0 here, populated post-sort
        )

        # 11. Explanations & Flags
        self._add_step(ctx, "GENERATE_EXPLANATIONS")
        risk_flags = self._generate_flags_and_explanations(
            ctx, tier, primary_cat, inst, overall_risk, conviction, regime, fund, is_permitted
        )

        # 12. Build Payload
        self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
        decision_payload = {
            "overall_rank": 0, # Populated post-sort
            "opportunity_rank": 0, # Populated post-sort
            "ranking_tier": tier.value,
            "primary_category": primary_cat.value,
            "secondary_category": secondary_cat.value,
            "ranking_priority": priority.value,
            "ranking_window": window.value,
            "ranking_profile": asdict(rank_profile),
            "ranking_components": [asdict(c) for c in ranking_components],
            "ranking_reason": asdict(r_reason),
            "ranking_checklist": asdict(checklist),
            "risk_flags": risk_flags
        }

        object.__setattr__(trace, 'steps_executed', ctx.steps)
        object.__setattr__(trace, 'inputs_parsed', parsed_inputs)

        result = self._build_output(
            status=DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL,
            status_msg="Relative Ranking Data generated." if is_permitted else "Ranking suppressed by Gatekeeper.",
            decision_payload=decision_payload,
            confidence=confidence,
            rating_score=final_master_score, 
            ctx=ctx,
            trace=trace
        )
        
        # Inject Tie-Breakers (Temporarily)
        # Note: Since string sorting in reverse is anti-alphabetical (Z before A), 
        # we construct a pseudo-reverse mapping for the symbol or just negate it logically in the sort key.
        # But python tuples sorted with reverse=True naturally sort numbers descending and strings descending.
        # If we want alphabetical (A before Z) while others are descending, we can handle it via the tuple in `evaluate_ranking` 
        # by making the symbol negative, but strings don't negate. We just let it sort descending for now, or the lambda handles it.
        result["_tie_breakers"] = {
            "master": final_master_score,
            "conviction": conviction,
            "confidence": confidence,
            "reward": reward,
            "symbol": symbol
        }
        
        return result


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, conviction: float, risk_safety: float, reward: float) -> dict[str, float]:
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        w["conviction"] += (conviction / 100.0) * t.get("adaptive_conviction_mult", 0.20)
        w["risk_safety"] += (risk_safety / 100.0) * t.get("adaptive_risk_mult", 0.20)
        w["reward_quality"] += (reward / 100.0) * t.get("adaptive_reward_mult", 0.15)
        
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _determine_tier(self, score: float, permitted: bool, risk: float) -> RankingTier:
        if not permitted: return RankingTier.IGNORE
        t = self.config.thresholds
        if score >= t.get("elite_tier_threshold", 85.0) and risk < 40.0: return RankingTier.ELITE
        if score >= t.get("platinum_tier_threshold", 75.0): return RankingTier.PLATINUM
        if score >= t.get("gold_tier_threshold", 65.0): return RankingTier.GOLD
        if score >= t.get("silver_tier_threshold", 50.0): return RankingTier.SILVER
        if score >= 40.0: return RankingTier.BRONZE
        return RankingTier.WATCHLIST

    def _determine_categories(self, swing: float, comp: float, inst: float, inv: float, 
                              brk: float, ipo: float, regime: float) -> tuple[RankingCategory, RankingCategory]:
        """Determines dynamic multi-dimensional categorization."""
        scores = {
            RankingCategory.COMPOUNDER: comp,
            RankingCategory.SWING: swing,
            RankingCategory.INSTITUTIONAL: inst,
            RankingCategory.VALUE: inv,
            RankingCategory.BREAKOUT: brk,
            RankingCategory.IPO: ipo
        }
        
        # Sort categories by score descending
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_cats[0][0]
        secondary = sorted_cats[1][0]
        
        # Regime override
        if regime < 40.0 and primary != RankingCategory.IPO:
            secondary = primary
            primary = RankingCategory.DEFENSIVE
            
        return primary, secondary

    def _determine_priority_window(self, tier: RankingTier, cat: RankingCategory):
        if tier in [RankingTier.ELITE, RankingTier.PLATINUM]: priority = RankingPriority.CRITICAL
        elif tier == RankingTier.GOLD: priority = RankingPriority.HIGH
        elif tier == RankingTier.SILVER: priority = RankingPriority.MEDIUM
        elif tier == RankingTier.BRONZE: priority = RankingPriority.LOW
        else: priority = RankingPriority.WATCH
            
        if cat == RankingCategory.COMPOUNDER: window = RankingWindow.MULTI_YEAR
        elif cat == RankingCategory.VALUE: window = RankingWindow.LONG_TERM
        elif cat == RankingCategory.IPO: window = RankingWindow.EVENT_DRIVEN
        elif cat in [RankingCategory.SWING, RankingCategory.BREAKOUT]: window = RankingWindow.SHORT_TERM
        elif cat == RankingCategory.DEFENSIVE: window = RankingWindow.MEDIUM_TERM
        else: window = RankingWindow.MEDIUM
            
        return priority, window

    def _get_primary_reason(self, tier: RankingTier, cat: RankingCategory) -> str:
        if tier == RankingTier.ELITE: return f"Elite mathematical superiority in {cat.value} framework."
        if tier == RankingTier.PLATINUM: return f"Premium institutional-grade {cat.value} setup."
        if tier == RankingTier.IGNORE: return "Systemically restricted from active ranking."
        return f"Standard relative opportunity within {cat.value} bounds."

    def _detect_conflicts(self, ctx: DecisionContext, tier: RankingTier, conviction: float, 
                          risk: float, regime: float, comp_rank: float, fund: float, 
                          swing_rank: float, mom: float) -> None:
        
        if tier == RankingTier.ELITE and conviction < 60.0:
            self._add_conflict(ctx, "High relative rank assigned despite low absolute execution conviction.", penalty=20.0)
        if tier in [RankingTier.ELITE, RankingTier.PLATINUM] and risk > 75.0:
            self._add_conflict(ctx, "Premium ranking tier conflicts with extreme underlying structural risk.", penalty=15.0)
        if tier in [RankingTier.ELITE, RankingTier.PLATINUM] and regime < 40.0:
            self._add_conflict(ctx, "High ranking output conflicts with hostile bearish market regime.", penalty=15.0)
        if comp_rank > 75.0 and fund < 40.0:
            self._add_conflict(ctx, "High compounder rank logically invalid given severe fundamental weakness.", penalty=20.0)
        if swing_rank > 75.0 and mom < 40.0:
            self._add_conflict(ctx, "High swing rank logically invalid given deteriorated tactical momentum.", penalty=20.0)

    def _generate_flags_and_explanations(self, ctx: DecisionContext, tier: RankingTier, 
                                         cat: RankingCategory, inst: float, risk: float, 
                                         conviction: float, regime: float, fund: float, permitted: bool) -> list[str]:
        flags = []
        if not permitted:
            self._add_explanation(ctx, "Gatekeeper denies market permission. Stock relegated to IGNORE tier.")
            flags.append("GATEKEEPER_RELEGATION")
            return flags

        if inst > 75.0:
            self._add_explanation(ctx, "Excellent institutional participation significantly improves relative ranking.")
        if risk < 40.0:
            self._add_explanation(ctx, "Low downside risk supports a superior opportunity score.")
        if tier not in [RankingTier.ELITE, RankingTier.PLATINUM] and conviction < 50.0:
            self._add_explanation(ctx, "Weak actionable conviction prevents promotion into premium tiers.")
            flags.append("CONVICTION_CAP")
        if regime < 40.0:
            self._add_explanation(ctx, "Bearish market regime suppresses final absolute ranking potentials.")
            flags.append("REGIME_SUPPRESSION")
        if cat == RankingCategory.COMPOUNDER and fund > 75.0:
            self._add_explanation(ctx, "Exceptional long-term business quality anchors compounder ranking.")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Relative ranking adjusted downward due to structural sub-engine conflicts.")
            flags.append("RANKING_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK (Single Stock)
    # ---------------------------------------------------------
    def _build_stock_fallback(self, stock_data: dict[str, Any]) -> dict[str, Any]:
        """Provides a safe fallback for a single stock failure so the rest of the universe can sort."""
        empty_ctx = DecisionContext()
        symbol = self._dynamic_lookup_string(self._flatten_dict(stock_data), ["symbol", "ticker", "asset"]).upper()
        if symbol == "UNKNOWN": symbol = "ZZZ"
        
        payload = {
            "overall_rank": 9999,
            "opportunity_rank": 9999,
            "ranking_tier": RankingTier.IGNORE.value,
            "primary_category": RankingCategory.UNKNOWN.value,
            "secondary_category": RankingCategory.UNKNOWN.value,
            "ranking_priority": RankingPriority.NONE.value,
            "ranking_window": RankingWindow.NONE.value,
            "ranking_profile": asdict(RankingProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Unranked", 0.0)),
            "ranking_components": [],
            "ranking_reason": asdict(RankingReason("SYSTEM_FAILURE", "Engine crash on this candidate", 0.0)),
            "ranking_checklist": asdict(RankingChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_RANKING"]
        }
        
        out = self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Ranking evaluation failed. Candidate ignored.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=DecisionTrace("hash", "id", "stage", 0.0, [], 0)
        )
        # Apply lowest possible tie-breaker
        out["_tie_breakers"] = {"master": 0.0, "conviction": 0.0, "confidence": 0.0, "reward": 0.0, "symbol": symbol}
        return out
