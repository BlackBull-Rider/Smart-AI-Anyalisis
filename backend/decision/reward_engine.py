"""
GREEN BULL RIDER V6
Layer-4: Decision Engine
Module: reward_engine.py

Institutional Reward Decision Engine. (V3.0.0 - Enterprise Edition)
Inherits from BaseDecisionEngine. Operates purely to quantify upside potential, 
expected returns (%), expected CAGR, and risk-adjusted reward probabilities.

Key Responsibilities:
- Assesses Expected Return and Expected CAGR based on holding horizons.
- Evaluates Risk/Reward Ratio (using upstream risk and target inputs).
- Generates structural Upside Probability and Base/Best/Worst case projections.

Boundary Constraint: This engine DOES NOT calculate Targets, Stop Loss, Entry, Exit, 
Position Size, or Capital Allocation. It evaluates "How much is the upside and is it realistic?".
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
    DecisionTrace,
    DecisionEvidence,
    DecisionStatus
)


# =====================================================================
# ENUMS (Strict Output Typing)
# =====================================================================
class RewardLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXCEPTIONAL = "EXCEPTIONAL"
    UNKNOWN = "UNKNOWN"

class RewardAction(str, Enum):
    IGNORE = "IGNORE"
    WATCH = "WATCH"
    ACCEPT = "ACCEPT"
    PRIORITIZE = "PRIORITIZE"
    HIGH_CONVICTION = "HIGH_CONVICTION"
    NONE = "NONE"

class RewardPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    WATCH = "WATCH"
    NONE = "NONE"

class RewardWindow(str, Enum):
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"
    OPEN = "OPEN"
    NONE = "NONE"

class RewardModel(str, Enum):
    SWING = "SWING"
    POSITIONAL = "POSITIONAL"
    COMPOUNDER = "COMPOUNDER"
    BREAKOUT = "BREAKOUT"
    TREND = "TREND"
    VALUE = "VALUE"
    UNKNOWN = "UNKNOWN"


# =====================================================================
# STRUCTURED OBJECTS (For Layer-5 AI / Execution Parsers)
# =====================================================================
@dataclass(frozen=True)
class RewardProfile:
    expected_return_pct: float
    expected_cagr_pct: float
    risk_reward_ratio: float
    upside_probability: float
    reward_quality: float
    reward_score: float

@dataclass(frozen=True)
class ProjectionCase:
    expected_return_pct: float
    expected_cagr_pct: float
    probability_pct: float

@dataclass(frozen=True)
class RewardProjection:
    best_case: ProjectionCase
    base_case: ProjectionCase
    worst_case: ProjectionCase
    confidence: float

@dataclass(frozen=True)
class RewardReason:
    primary: str
    secondary: str
    confidence: float

@dataclass(frozen=True)
class RewardChecklist:
    trend_supportive: bool
    momentum_supportive: bool
    fundamental_supportive: bool
    institutional_supportive: bool
    market_supportive: bool
    holding_aligned: bool
    risk_reward_favorable: bool

@dataclass(frozen=True)
class RewardComponent:
    name: str
    score: float
    weight: float
    confidence: float


# =====================================================================
# ENGINE PROFILE
# =====================================================================
def get_reward_profile() -> DecisionConfig:
    return DecisionConfig(
        profile_name="Institutional_Reward_V3",
        version="3.0.0",
        stage="Layer-4: Reward Assessment",
        schema_version="3.0",
        api_version="v6",
        decision_method="Adaptive Continuous Upside & CAGR Projection",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "target_quality": 0.20,
            "trend": 0.20,
            "momentum": 0.15,
            "fundamental": 0.15,
            "institutional": 0.15,
            "volatility_expansion": 0.15
        },
        thresholds={
            "conflict_penalty": 15.0,
            "exceptional_reward_threshold": 85.0,
            "very_high_reward_threshold": 75.0,
            "high_reward_threshold": 60.0,
            "acceptable_rr_ratio": 1.5,
            "high_conviction_rr_ratio": 3.0,
            "adaptive_fund_mult": 0.20,
            "adaptive_mom_mult": 0.15,
            "adaptive_trend_mult": 0.15,
            "adaptive_inst_mult": 0.10
        }
    )


# =====================================================================
# REWARD DECISION ENGINE
# =====================================================================
class RewardEngine(BaseDecisionEngine):
    """
    Institutional Reward Decision Engine.
    Quantifies the upside potential, estimating absolute returns and CAGRs based 
    on underlying structural scores and holding constraints. It strictly evaluates 
    the viability of the reward relative to assessed risks.
    """

    def __init__(self, config: DecisionConfig | None = None):
        super().__init__(config or get_reward_profile())

    def evaluate(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        return self.evaluate_reward(engine_outputs)

    def evaluate_reward(self, engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main decision execution pipeline for Reward Assessment.
        """
        start_time = time.perf_counter()
        ctx = DecisionContext()
        self._add_step(ctx, "INITIALIZE_REWARD_ENGINE")
        
        trace = self._build_trace(engine_outputs, start_time, ctx, 0)

        if not isinstance(engine_outputs, dict) or not engine_outputs:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction
            flat_data = self._flatten_dict(engine_outputs)
            parsed_inputs = len(flat_data)
            self._add_step(ctx, "EXTRACT_LAYER3_AND_LAYER4_DATA")

            # 1. Base Intelligence & Gatekeeper Permissions Extraction
            permission_str = self._dynamic_lookup_string(flat_data, ["market_permission", "trade_permission"])
            is_permitted = "BLOCK" not in permission_str.upper() and "RESTRICT" not in permission_str.upper()
            layer3_conf = self._dynamic_lookup(flat_data, ["confidence", "reliability"], 50.0)

            # Core Evaluated Inputs
            target_quality = self._dynamic_lookup(flat_data, ["target_quality_score", "target_quality"], 50.0)
            overall_risk = self._dynamic_lookup(flat_data, ["overall_risk", "risk_score"], 50.0)
            trend = self._dynamic_lookup(flat_data, ["trend_score", "trend_strength"], 50.0)
            mom = self._dynamic_lookup(flat_data, ["momentum_score", "momentum_strength"], 50.0)
            fund = self._dynamic_lookup(flat_data, ["fundamental_score", "business_quality", "compounder_score"], 50.0)
            inst = self._dynamic_lookup(flat_data, ["institutional_score", "smart_money_score", "accumulation"], 50.0)
            vol_expansion = self._dynamic_lookup(flat_data, ["volatility_expansion", "breakout_score"], 50.0)
            regime = self._dynamic_lookup(flat_data, ["regime_score", "market_regime"], 50.0)

            holding_type = self._dynamic_lookup_string(flat_data, ["holding_type", "holding_window"]).upper()

            # 2. Continuous Adaptive Weighting
            self._add_step(ctx, "CALCULATE_ADAPTIVE_WEIGHTS")
            dynamic_weights = self._calculate_continuous_weights(trend, mom, fund, inst, holding_type)

            # 3. Base Reward Score Calculation
            self._add_step(ctx, "CALCULATE_REWARD_SCORE")
            base_reward_score = self._normalize(
                (target_quality * dynamic_weights["target_quality"]) +
                (trend * dynamic_weights["trend"]) +
                (mom * dynamic_weights["momentum"]) +
                (fund * dynamic_weights["fundamental"]) +
                (inst * dynamic_weights["institutional"]) +
                (vol_expansion * dynamic_weights["volatility_expansion"])
            )

            # 4. Risk-Adjusted Calculations & Proxy Projections
            self._add_step(ctx, "CALCULATE_PROJECTIONS_AND_RR")
            # Calculate Risk/Reward Proxy (If Target vs Risk is high = High RR)
            risk_floor = max(10.0, overall_risk) # Prevent division by zero
            rr_ratio = round(self._clamp((base_reward_score / risk_floor) * 2.0, 0.1, 10.0), 2)
            
            upside_prob = self._calculate_upside_probability(base_reward_score, overall_risk, target_quality, regime)
            
            # Map structural scores to expected % returns based on holding profile
            expected_ret, expected_cagr = self._calculate_expected_returns(base_reward_score, fund, holding_type)
            projections = self._generate_projections(expected_ret, expected_cagr, upside_prob)

            # 5. Categorization and Prioritization
            self._add_step(ctx, "CATEGORIZE_REWARD")
            r_level, r_action, r_priority, r_model, r_window = self._determine_reward_categorization(
                base_reward_score, rr_ratio, upside_prob, holding_type, is_permitted
            )

            # 6. Generate Reward Checklist
            self._add_step(ctx, "GENERATE_CHECKLIST")
            checklist = RewardChecklist(
                trend_supportive=(trend >= 60.0),
                momentum_supportive=(mom >= 60.0),
                fundamental_supportive=(fund >= 60.0),
                institutional_supportive=(inst >= 60.0),
                market_supportive=(regime >= 50.0),
                holding_aligned=(holding_type not in ["NONE", "UNKNOWN"]),
                risk_reward_favorable=(rr_ratio >= self.config.thresholds.get("acceptable_rr_ratio", 1.5))
            )

            # 7. Conflict Detection
            self._add_step(ctx, "DETECT_CONFLICTS")
            self._detect_conflicts(
                ctx, base_reward_score, target_quality, expected_cagr, fund, overall_risk, regime, inst
            )

            # Adjust Quality by Conflicts and Risk (Reward Quality differs from raw score)
            reward_quality = self._clamp(base_reward_score - (ctx.conflict_penalty * 0.4) - (overall_risk * 0.15))

            # 8. Advanced Confidence Model
            self._add_step(ctx, "CALCULATE_ADVANCED_CONFIDENCE")
            reward_conf = self._calculate_advanced_confidence(
                ctx, layer3_conf, reward_quality, parsed_inputs, trend, mom, fund, inst, target_quality, overall_risk
            )
            
            r_reason = RewardReason(
                primary="Structural upside probability remains highly favorable." if reward_quality >= 60 else "Upside potential is structurally constrained.",
                secondary=f"RR Proxy: {rr_ratio} | Probability: {round(upside_prob,1)}%",
                confidence=round(reward_conf, 2)
            )

            # 9. Generate Flags and Institutional Explanations
            self._add_step(ctx, "GENERATE_EXPLANATIONS")
            risk_flags = self._generate_flags_and_explanations(
                ctx, r_level, rr_ratio, trend, inst, fund, mom, regime, overall_risk, is_permitted
            )

            # Build Profiles
            reward_profile = RewardProfile(
                expected_return_pct=round(expected_ret, 2),
                expected_cagr_pct=round(expected_cagr, 2),
                risk_reward_ratio=rr_ratio,
                upside_probability=round(upside_prob, 2),
                reward_quality=round(reward_quality, 2),
                reward_score=round(base_reward_score, 2)
            )

            # 10. Build Payload
            self._add_step(ctx, "BUILD_DECISION_PAYLOAD")
            decision_payload = {
                "reward_category": r_level.value,
                "reward_action": r_action.value,
                "reward_priority": r_priority.value,
                "reward_window": r_window.value,
                "reward_model": r_model.value,
                "reward_profile": asdict(reward_profile),
                "reward_projection": asdict(projections),
                "reward_confidence": round(reward_conf, 2),
                "reward_reason": asdict(r_reason),
                "reward_checklist": asdict(checklist),
                "risk_flags": risk_flags
            }

            status_enum = DecisionStatusEnum.SUCCESS if is_permitted else DecisionStatusEnum.PARTIAL
            status_msg = "Reward framework evaluated successfully." if is_permitted else "Reward evaluated under Gatekeeper restrictions."

            trace.steps_executed = ctx.steps
            trace.inputs_parsed = parsed_inputs

            return self._build_output(
                status=status_enum,
                status_msg=status_msg,
                decision_payload=decision_payload,
                confidence=reward_conf,
                rating_score=reward_quality, # Engine rating based on risk-adjusted quality
                ctx=ctx,
                trace=trace
            )

        except Exception as e:
            self.logger.error(f"Reward Engine execution failed: {str(e)}", exc_info=True)
            return self._sanitize_json(self._build_fallback(self._build_trace(engine_outputs, start_time, ctx, 0)))


    # ---------------------------------------------------------
    # DECISION LOGIC & MODELS
    # ---------------------------------------------------------

    def _dynamic_lookup_string(self, flat_data: dict[str, Any], keywords: list[str]) -> str:
        for key, value in flat_data.items():
            if any(kw in key for kw in keywords) and isinstance(value, str):
                return value
        return "UNKNOWN"

    def _calculate_continuous_weights(self, trend: float, mom: float, fund: float, inst: float, holding: str) -> dict[str, float]:
        """Dynamically adjusts component weights based on holding horizon and dominant traits."""
        w = dict(self.config.base_weights)
        t = self.config.thresholds
        
        if "COMPOUND" in holding or "LONG" in holding:
            w["fundamental"] += (fund / 100.0) * t.get("adaptive_fund_mult", 0.20)
            w["institutional"] += (inst / 100.0) * t.get("adaptive_inst_mult", 0.10)
            w["momentum"] -= 0.10
        elif "SWING" in holding or "INTRADAY" in holding:
            w["momentum"] += (mom / 100.0) * t.get("adaptive_mom_mult", 0.15)
            w["trend"] += (trend / 100.0) * t.get("adaptive_trend_mult", 0.15)
            w["fundamental"] -= 0.10
            
        # Normalize
        for k in w: w[k] = max(0.0, w[k])
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else self.config.base_weights

    def _calculate_upside_probability(self, r_score: float, risk: float, t_qual: float, regime: float) -> float:
        """Determines the mathematical likelihood of realizing the reward."""
        base_prob = (r_score * 0.4) + (t_qual * 0.3) + (regime * 0.3)
        # Higher risk structurally reduces the probability of a clean run
        risk_discount = self._clamp((risk - 50.0) * 0.5, 0.0, 30.0) if risk > 50.0 else 0.0
        return self._normalize(base_prob - risk_discount)

    def _calculate_expected_returns(self, r_score: float, fund: float, holding: str) -> tuple[float, float]:
        """Maps continuous structural scores to plausible institutional return percentages."""
        # Baseline proxies based on holding period and reward score 
        # (e.g., a score of 100 = max plausible structural move)
        
        if "INTRADAY" in holding:
            ret = self._clamp((r_score / 100.0) * 3.0, 0.1, 5.0) # 0.1% to 5.0%
            cagr = 0.0
        elif "SWING" in holding:
            ret = self._clamp((r_score / 100.0) * 15.0, 1.0, 25.0) # 1% to 25%
            cagr = 0.0
        elif "POSITIONAL" in holding:
            ret = self._clamp((r_score / 100.0) * 40.0, 5.0, 60.0)
            cagr = self._clamp((fund / 100.0) * 15.0, 0.0, 25.0)
        elif "COMPOUND" in holding or "LONG" in holding:
            ret = self._clamp((r_score / 100.0) * 150.0, 10.0, 500.0)
            cagr = self._clamp((fund / 100.0) * 35.0, 5.0, 45.0) # 5% to 45% CAGR
        else:
            ret = self._clamp((r_score / 100.0) * 10.0, 0.0, 20.0)
            cagr = 0.0
            
        return ret, cagr

    def _generate_projections(self, base_ret: float, base_cagr: float, base_prob: float) -> RewardProjection:
        """Constructs three-tier scenario projections (Best, Base, Worst)."""
        best_ret = base_ret * 1.5
        best_cagr = base_cagr * 1.3
        best_prob = self._clamp(base_prob - 20.0)
        
        worst_ret = base_ret * 0.4
        worst_cagr = base_cagr * 0.5
        worst_prob = self._clamp(base_prob + 15.0)
        
        return RewardProjection(
            best_case=ProjectionCase(round(best_ret, 2), round(best_cagr, 2), round(best_prob, 2)),
            base_case=ProjectionCase(round(base_ret, 2), round(base_cagr, 2), round(base_prob, 2)),
            worst_case=ProjectionCase(round(worst_ret, 2), round(worst_cagr, 2), round(worst_prob, 2)),
            confidence=round(base_prob, 2) # Projection confidence mirrors base probability
        )

    def _determine_reward_categorization(self, r_score: float, rr_ratio: float, prob: float, 
                                         holding: str, permitted: bool):
        t = self.config.thresholds
        
        if not permitted:
            return RewardLevel.UNKNOWN, RewardAction.IGNORE, RewardPriority.NONE, RewardModel.UNKNOWN, RewardWindow.NONE

        # Action & Level
        if r_score >= t.get("exceptional_reward_threshold", 85.0) and rr_ratio >= 2.0:
            level = RewardLevel.EXCEPTIONAL
            action = RewardAction.HIGH_CONVICTION
            priority = RewardPriority.CRITICAL
        elif r_score >= t.get("very_high_reward_threshold", 75.0) and rr_ratio >= 1.5:
            level = RewardLevel.VERY_HIGH
            action = RewardAction.PRIORITIZE
            priority = RewardPriority.HIGH
        elif r_score >= t.get("high_reward_threshold", 60.0):
            level = RewardLevel.HIGH
            action = RewardAction.ACCEPT
            priority = RewardPriority.MEDIUM
        elif r_score >= 40.0:
            level = RewardLevel.MODERATE
            action = RewardAction.WATCH
            priority = RewardPriority.LOW
        else:
            level = RewardLevel.LOW
            action = RewardAction.IGNORE
            priority = RewardPriority.WATCH
            
        # Model & Window Mapping
        h = holding.upper()
        if "INTRADAY" in h:
            model = RewardModel.MOMENTUM if hasattr(RewardModel, 'MOMENTUM') else RewardModel.UNKNOWN
            window = RewardWindow.SHORT_TERM
        elif "SWING" in h:
            model = RewardModel.SWING
            window = RewardWindow.SHORT_TERM
        elif "COMPOUND" in h or "LONG" in h:
            model = RewardModel.COMPOUNDER
            window = RewardWindow.LONG_TERM
        elif "POSITIONAL" in h:
            model = RewardModel.TREND
            window = RewardWindow.MEDIUM_TERM
        else:
            model = RewardModel.BREAKOUT
            window = RewardWindow.OPEN
            
        return level, action, priority, model, window

    def _detect_conflicts(self, ctx: DecisionContext, r_score: float, t_qual: float, cagr: float, 
                          fund: float, risk: float, regime: float, inst: float) -> None:
        """Identifies paradoxes between estimated reward and underlying reality."""
        
        # 1. High Reward + Low Target Quality
        if r_score > 75.0 and t_qual < 40.0:
            self._add_conflict(ctx, "High abstract reward conflicts with poor actionable target quality.", penalty=15.0)
            
        # 2. High CAGR + Weak Fundamentals
        if cagr > 15.0 and fund < 40.0:
            self._add_conflict(ctx, "High CAGR projection mathematically invalid due to severely weak fundamentals.", penalty=20.0)
            
        # 3. Strong Reward + Extreme Risk
        if r_score > 80.0 and risk > 80.0:
            self._add_conflict(ctx, "Strong upside reward entirely negated by extreme structural tail-risk.", penalty=15.0)
            
        # 4. Exceptional Reward + Bear Market
        if r_score > 85.0 and regime < 35.0:
            self._add_conflict(ctx, "Exceptional upside projections fundamentally conflict with bearish macro regime.", penalty=15.0)
            
        # 5. High Upside + Institutional Distribution
        if r_score > 75.0 and inst < 30.0:
            self._add_conflict(ctx, "High upside projection lacks required institutional/smart money backing.", penalty=10.0)

    def _calculate_advanced_confidence(self, ctx: DecisionContext, layer3_conf: float, r_qual: float, 
                                       parsed_count: int, trend: float, mom: float, fund: float, 
                                       inst: float, t_qual: float, risk: float) -> float:
        """Determines how structurally reliable the reward projection is."""
        engine_agreement = self._normalize(100.0 - abs(trend - mom) - abs(r_qual - t_qual))
        evidence_density = self._clamp((len(ctx.positive_evidence + ctx.negative_evidence) / max(1, parsed_count)) * 200.0)
        risk_alignment = self._normalize(100.0 - risk)
        
        structural_quality = self._normalize((trend + fund + inst) / 3.0)
        signal_stability = self._normalize(100.0 - (ctx.conflicts * 15.0))
        
        return self._normalize(
            (layer3_conf * 0.15) + (r_qual * 0.20) + 
            (engine_agreement * 0.15) + (evidence_density * 0.10) + 
            (structural_quality * 0.20) + (risk_alignment * 0.10) + (signal_stability * 0.10)
        )

    def _generate_flags_and_explanations(self, ctx: DecisionContext, r_level: RewardLevel, 
                                         rr_ratio: float, trend: float, inst: float, fund: float, 
                                         mom: float, regime: float, risk: float, permitted: bool) -> list[str]:
        flags = []
        
        if not permitted:
            self._add_explanation(ctx, "Reward modeling constrained: Market Regime denies deployment permission.")
            flags.append("GATEKEEPER_CONSTRAINT")
            return flags

        # Explanations
        if trend > 75.0:
            self._add_explanation(ctx, "Robust trend structure structurally supports above-average upside potential.")
        if inst > 75.0:
            self._add_explanation(ctx, "Institutional accumulation significantly increases long-term reward probability.")
            
        if rr_ratio >= self.config.thresholds.get("high_conviction_rr_ratio", 3.0):
            self._add_explanation(ctx, "Risk-adjusted reward metrics (R-Multiple) are exceptionally favorable.")
        elif rr_ratio < 1.0:
            self._add_explanation(ctx, "Inverse Risk/Reward ratio dictates extreme caution.")
            flags.append("POOR_RISK_REWARD")

        if fund > 75.0:
            self._add_explanation(ctx, "High fundamental quality structurally supports sustained CAGR compounding.")
            
        if mom < 40.0:
            self._add_explanation(ctx, "Current momentum deterioration strictly limits short-term reward realization.")
            flags.append("SHORT_TERM_REWARD_CAP")
            
        if regime < 40.0:
            self._add_explanation(ctx, "Bearish macro market regime severely reduces achievable absolute upside.")
            flags.append("REGIME_REWARD_COMPRESSION")

        if ctx.conflicts > 0:
            self._add_explanation(ctx, "Conflicting structural signals reduce statistical confidence in upside projections.")
            flags.append("PROJECTION_CONFLICTS")

        return flags

    # ---------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------
    def _build_fallback(self, trace: DecisionTrace) -> dict[str, Any]:
        empty_ctx = DecisionContext()
        self._add_warning(empty_ctx, "Catastrophic evaluation failure in Reward Engine.", WarningSeverityEnum.CRITICAL)
        
        null_case = ProjectionCase(0.0, 0.0, 0.0)
        
        payload = {
            "reward_category": RewardLevel.UNKNOWN.value,
            "reward_action": RewardAction.IGNORE.value,
            "reward_priority": RewardPriority.NONE.value,
            "reward_window": RewardWindow.NONE.value,
            "reward_model": RewardModel.UNKNOWN.value,
            "reward_profile": asdict(RewardProfile(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            "reward_projection": asdict(RewardProjection(null_case, null_case, null_case, 0.0)),
            "reward_confidence": 0.0,
            "reward_reason": asdict(RewardReason("SYSTEM_FAILURE", "Engine crash", 0.0)),
            "reward_checklist": asdict(RewardChecklist(False, False, False, False, False, False, False)),
            "risk_flags": ["SYSTEM_FAILURE_REWARD"]
        }
        
        return self._build_output(
            status=DecisionStatusEnum.FAILED,
            status_msg="Execution failed. Triggered failsafe.",
            decision_payload=payload,
            confidence=0.0,
            rating_score=0.0, 
            ctx=empty_ctx,
            trace=trace
        )

# =====================================================================
# PUBLIC API
# =====================================================================
def evaluate_reward(engine_outputs: dict[str, Any] | None) -> dict[str, Any]:
    return RewardEngine().evaluate_reward(engine_outputs)
