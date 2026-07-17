"""
GREEN BULL RIDER V6
Layer-3: Scoring Engine
Module: swing_engine.py

Institutional Swing Trading Fusion Engine.
Inherits from BaseEngine. This is a Master Fusion Engine that ingests intelligence 
from all Layer-2 Analyzers (Trend, Momentum, Volume, Volatility, Smart Money, 
Pattern, Support/Resistance, Institutional, Market Regime) and fuses them into 
a definitive, deterministic Swing Probability and Institutional Swing Score.
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
def get_swing_profile() -> EngineConfig:
    """Returns the primary EngineConfig profile for Swing Trading Fusion."""
    return EngineConfig(
        profile_name="Institutional_Swing",
        version="2.1.0",
        stage="Layer-3: Scoring",
        schema_version="1.0",
        api_version="v6",
        scoring_method="Multi-Dimensional Swing Fusion",
        normalization_method="Min-Max Clamp (0-100)",
        base_weights={
            "trend": 0.18,
            "momentum": 0.16,
            "volume": 0.12,
            "volatility": 0.10,
            "smart_money": 0.12,
            "pattern": 0.08,
            "support_resistance": 0.08,
            "institutional": 0.06,
            "market_regime": 0.05,
            "risk": 0.03,
            "reward": 0.02
        },
        thresholds={
            "conflict_penalty": 15.0,
            "volume_penalty": 12.0,
            "trend_penalty": 15.0,
            "momentum_penalty": 12.0,
            "risk_penalty": 20.0
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
# SWING FUSION ENGINE (Inherits BaseEngine)
# =====================================================================
class SwingEngine(BaseEngine):
    """
    Institutional Swing Scoring & Fusion Engine.
    Transforms massively varied Layer-2 JSON intelligence into a singular, 
    deterministic Layer-3 Swing Probability score utilizing advanced models.
    """

    def __init__(self, config: EngineConfig | None = None):
        super().__init__(config or get_swing_profile())

    def calculate(self, swing_json: dict[str, Any] | None) -> dict[str, Any]:
        """
        Main calculation execution pipeline.
        
        Args:
            swing_json: The dictionary payload from ALL Layer-2 Analyzers.
            
        Returns:
            A sanitized, deterministic JSON-serializable dictionary.
        """
        start_time = time.perf_counter()
        trace = self._generate_trace(swing_json, start_time, "SWING")

        if not isinstance(swing_json, dict) or not swing_json:
            return self._sanitize_json(self._build_fallback(trace))

        try:
            # Flatten JSON for dynamic keyword extraction (future-proof cross-analyzer parsing)
            flat_data = self._flatten_dict(swing_json)
            
            # 1. Base Intelligence Extraction
            analyzer_confidence = self._extract_metric(flat_data, ["confidence", "certainty", "probability"], 50.0)
            
            # Sub-metrics for Conflict Detection & Advanced Models
            breakout_raw = self._extract_metric(flat_data, ["breakout", "breakdown"], 50.0)
            dist_raw = self._extract_metric(flat_data, ["distribution", "selling_pressure"], 20.0)
            
            # 2. Component Extractions
            trend_comp = self._compute_component(
                flat_data, ["trend", "direction", "mtf", "alignment"], "bullish", "bearish", "Trend Alignment"
            )
            mom_comp = self._compute_component(
                flat_data, ["momentum", "strength", "rsi", "macd"], "strong", "weak", "Momentum"
            )
            vol_comp = self._compute_component(
                flat_data, ["volume", "explosion", "delivery"], "high", "low", "Volume Confirmation"
            )
            vola_comp = self._compute_component(
                flat_data, ["volatility_quality", "expansion", "squeeze", "stability"], "stable", "erratic", "Volatility Quality"
            )
            smc_comp = self._compute_component(
                flat_data, ["smart_money", "institutional_footprint", "bos"], "strong", "weak", "Smart Money"
            )
            pat_comp = self._compute_component(
                flat_data, ["pattern", "harmonic", "chart_pattern"], "bullish", "bearish", "Pattern Quality"
            )
            sr_comp = self._compute_component(
                flat_data, ["support", "resistance", "clearance"], "cleared", "rejected", "Support & Resistance"
            )
            inst_comp = self._compute_component(
                flat_data, ["institutional_buying", "fii", "dii", "accumulation"], "strong", "weak", "Institutional Participation"
            )
            regime_comp = self._compute_component(
                flat_data, ["regime", "environment", "market"], "bullish", "bearish", "Market Regime"
            )
            rew_comp = self._compute_component(
                flat_data, ["reward", "target", "upside", "rr_ratio"], "high", "low", "Reward Potential"
            )
            
            # Risk is inversely scored (High raw risk = Low Safety Score)
            risk_comp = self._compute_inverse_component(
                flat_data, ["risk", "danger", "stop_loss", "drawdown"], "low", "high", "Risk Safety"
            )

            # 3. Advanced Swing Models (Layer-3 Intelligence Fusion)
            trend_align_mod = self._trend_alignment_model(trend_comp.final_score, regime_comp.final_score)
            mom_cont_mod = self._momentum_continuation_model(mom_comp.final_score, vola_comp.final_score)
            vol_conf_mod = self._volume_confirmation_model(vol_comp.final_score, trend_comp.final_score, breakout_raw)
            brk_sust_mod = self._breakout_sustainability_model(pat_comp.final_score, vol_comp.final_score, mom_comp.final_score)
            smc_conf_mod = self._smart_money_confirmation_model(smc_comp.final_score, inst_comp.final_score, trend_comp.final_score)
            sr_clear_mod = self._support_resistance_model(sr_comp.final_score, vol_comp.final_score, mom_comp.final_score)
            rr_mod = self._risk_reward_model(rew_comp.final_score, risk_comp.final_score)
            
            # The Ultimate Swing Probability Model
            swing_prob = self._swing_probability_model(
                trend_align_mod, mom_cont_mod, vol_conf_mod, brk_sust_mod, smc_conf_mod, sr_clear_mod, rr_mod
            )

            # 4. Conflict Detection
            self._detect_conflicts(
                trend_comp.final_score, mom_comp.final_score, breakout_raw, vol_comp.final_score,
                pat_comp.final_score, smc_comp.final_score, sr_comp.final_score, inst_comp.final_score,
                dist_raw, regime_comp.final_score, risk_comp.final_score
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
                trend_comp.final_score, mom_comp.final_score, vol_comp.final_score,
                smc_comp.final_score, regime_comp.final_score, breakout_raw, risk_comp.final_score
            )

            # 7. Calculate Weighted Final Base Score
            base_score = (
                (trend_comp.final_score * dynamic_weights["trend"]) +
                (mom_comp.final_score * dynamic_weights["momentum"]) +
                (vol_comp.final_score * dynamic_weights["volume"]) +
                (vola_comp.final_score * dynamic_weights["volatility"]) +
                (smc_comp.final_score * dynamic_weights["smart_money"]) +
                (pat_comp.final_score * dynamic_weights["pattern"]) +
                (sr_comp.final_score * dynamic_weights["support_resistance"]) +
                (inst_comp.final_score * dynamic_weights["institutional"]) +
                (regime_comp.final_score * dynamic_weights["market_regime"]) +
                (risk_comp.final_score * dynamic_weights["risk"]) +
                (rew_comp.final_score * dynamic_weights["reward"])
            )

            # Blend Advanced Models into Base Score
            fused_swing_score = (base_score * 0.35) + (swing_prob * 0.40) + (trend_align_mod * 0.15) + (smc_conf_mod * 0.10)

            # Apply Structural Penalties
            structural_penalty = 0.0
            if risk_comp.final_score < 30:
                risk_pen = self.config.thresholds.get("risk_penalty", 20.0)
                structural_penalty += risk_pen
                self._score_reasons.append(f"Applied massive penalty of {risk_pen} due to extreme volatility/risk.")
            if trend_comp.final_score < 35:
                trend_pen = self.config.thresholds.get("trend_penalty", 15.0)
                structural_penalty += trend_pen
                self._score_reasons.append(f"Applied penalty of {trend_pen} due to poor trend alignment.")
            if vol_comp.final_score < 30 and breakout_raw > 70:
                vol_pen = self.config.thresholds.get("volume_penalty", 12.0)
                structural_penalty += vol_pen
                self._score_reasons.append(f"Applied penalty of {vol_pen} for unconfirmed breakout (Weak Volume).")

            # 8. Final Confidence Calculation
            conflict_ratio = (self._conflicts / max(total_evidence, 1)) * 100.0
            component_agreement = self._normalize(100.0 - abs(trend_comp.final_score - mom_comp.final_score))
            
            final_confidence = self._normalize(
                (analyzer_confidence * 0.20) + 
                (evidence_graph.evidence_coverage * 0.15) + 
                (min(total_evidence * 10, 100) * 0.15) +
                (regime_comp.final_score * 0.15) +
                (inst_comp.final_score * 0.15) +
                (component_agreement * 0.20) - 
                (self._conflicts * 15)
            )
            
            reliability = self._normalize(final_confidence - (conflict_ratio * 0.25))

            # Execute final risk-adjusted normalization
            final_overall_score = self._normalize((fused_swing_score * (0.5 + (final_confidence / 200.0))) - structural_penalty)
            
            # Synthesize contextual explanations
            self._generate_explanations(
                trend_comp.final_score, mom_comp.final_score, vol_comp.final_score, inst_comp.final_score,
                smc_comp.final_score, pat_comp.final_score, sr_comp.final_score, regime_comp.final_score, 
                risk_comp.final_score, breakout_raw
            )

            # Extract Upstream Textual Context
            self._extract_upstream_text(flat_data)

            # 9. Build Deterministic Result Payload
            result = {
                "status": asdict(OutputStatus(status="SUCCESS", quality="VALID")),
                "scores": {
                    "overall_swing_score": round(final_overall_score, 2),
                    "swing_rating": self._determine_rating(final_overall_score),
                    "confidence": round(final_confidence, 2),
                    "swing_probability": round(swing_prob, 2),
                    "trend_alignment_score": round(trend_align_mod, 2),
                    "momentum_score": round(mom_comp.final_score, 2),
                    "volume_confirmation_score": round(vol_conf_mod, 2),
                    "breakout_score": round(brk_sust_mod, 2),
                    "volatility_quality_score": round(vola_comp.final_score, 2),
                    "smart_money_score": round(smc_conf_mod, 2),
                    "pattern_quality_score": round(pat_comp.final_score, 2),
                    "support_resistance_score": round(sr_clear_mod, 2),
                    "institutional_support_score": round(inst_comp.final_score, 2),
                    "market_regime_score": round(regime_comp.final_score, 2),
                    "risk_score": round(100.0 - risk_comp.final_score, 2), # Exposing raw risk
                    "reward_potential_score": round(rew_comp.final_score, 2),
                    "swing_reliability": round(reliability, 2)
                },
                "components": {
                    "trend": asdict(trend_comp),
                    "momentum": asdict(mom_comp),
                    "volume": asdict(vol_comp),
                    "volatility": asdict(vola_comp),
                    "smart_money": asdict(smc_comp),
                    "pattern": asdict(pat_comp),
                    "support_resistance": asdict(sr_comp),
                    "institutional": asdict(inst_comp),
                    "market_regime": asdict(regime_comp),
                    "risk_safety": asdict(risk_comp),
                    "reward": asdict(rew_comp)
                },
                "advanced_models": {
                    "trend_alignment": round(trend_align_mod, 2),
                    "momentum_continuation": round(mom_cont_mod, 2),
                    "volume_confirmation": round(vol_conf_mod, 2),
                    "breakout_sustainability": round(brk_sust_mod, 2),
                    "smart_money_confirmation": round(smc_conf_mod, 2),
                    "support_resistance_clearance": round(sr_clear_mod, 2),
                    "risk_reward_quality": round(rr_mod, 2),
                    "opportunity_strength": round(swing_prob, 2)
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
    # ADVANCED SWING MODELS (Multi-Analyzer Fusion)
    # ---------------------------------------------------------
    
    def _trend_alignment_model(self, trend: float, regime: float) -> float:
        return self._normalize((trend * 0.6) + (regime * 0.4))

    def _momentum_continuation_model(self, mom: float, vola_qual: float) -> float:
        return self._normalize((mom * 0.7) + (vola_qual * 0.3))

    def _volume_confirmation_model(self, vol: float, trend: float, breakout: float) -> float:
        return self._normalize((vol * 0.5) + (trend * 0.3) + (breakout * 0.2))

    def _breakout_sustainability_model(self, pat: float, vol: float, mom: float) -> float:
        return self._normalize((pat * 0.4) + (vol * 0.4) + (mom * 0.2))

    def _smart_money_confirmation_model(self, smc: float, inst: float, trend: float) -> float:
        return self._normalize((smc * 0.5) + (inst * 0.3) + (trend * 0.2))

    def _support_resistance_model(self, sr: float, vol: float, mom: float) -> float:
        return self._normalize((sr * 0.6) + (vol * 0.2) + (mom * 0.2))

    def _risk_reward_model(self, reward: float, risk_safety: float) -> float:
        """High reward + High safety (low risk) = Excellent RR."""
        return self._normalize((reward * 0.6) + (risk_safety * 0.4))

    def _swing_probability_model(self, ta: float, mc: float, vc: float, bs: float, smc: float, sr: float, rr: float) -> float:
        """The Ultimate Layer-3 Swing Probability Fusion Formula."""
        return self._normalize(
            (ta * 0.20) + (mc * 0.15) + (vc * 0.15) + (bs * 0.10) + 
            (smc * 0.15) + (sr * 0.15) + (rr * 0.10)
        )


    # ---------------------------------------------------------
    # INTELLIGENCE LOGIC & ADAPTIVE WEIGHTS
    # ---------------------------------------------------------
    
    def _detect_conflicts(self, trend: float, mom: float, breakout: float, vol: float, 
                          pat: float, smc: float, sr: float, inst: float, dist_raw: float, 
                          regime: float, risk_safety: float) -> None:
        """Evaluates logical paradoxes preventing optimal swing execution."""
        
        # 1. Bullish Trend + Bearish Momentum
        if trend > 75 and mom < 35:
            self._conflicts += 1
            warn = "Conflict: Strong directional trend contradicts weakening underlying momentum."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 2. Breakout + Weak Volume
        if breakout > 75 and vol < 35:
            self._conflicts += 1
            warn = "Conflict: Pattern breakout lacks volume confirmation (High trap risk)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 3. Bullish Pattern + Bearish Smart Money
        if pat > 75 and smc < 35:
            self._conflicts += 1
            warn = "Conflict: Retail bullish pattern contradicted by smart money distribution/weakness."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 4. Strong Momentum + Heavy Resistance
        if mom > 75 and sr < 35:
            self._conflicts += 1
            warn = "Conflict: Strong momentum driving directly into heavy uncleared resistance."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 5. Institutional Buying + Distribution
        if inst > 75 and dist_raw > 75:
            self._conflicts += 1
            warn = "Conflict: Institutional activity heavily mixed (Buying amidst severe dumping)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 6. Bullish Market + High Risk
        if regime > 75 and risk_safety < 30:
            self._conflicts += 1
            warn = "Conflict: Broad bullish regime compromised by extreme asset-specific volatility risk."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)
            
        # 7. Strong Trend + Weak Pattern
        if trend > 75 and pat < 30:
            self._conflicts += 1
            warn = "Conflict: Strong macro trend but terrible micro pattern structure (Poor entry timing)."
            self._warning_log.append(warn)
            self._score_reasons.append(warn)

    def _calculate_adaptive_weights(self, trend: float, mom: float, vol: float, 
                                    smc: float, regime: float, breakout: float, risk_safety: float) -> dict[str, float]:
        """Dynamically redistributes component weights based on exceptional setup traits."""
        weights = dict(self.config.base_weights)

        # Shift 1: Strong Trend
        if trend > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Trend detected, scaling Trend weight.")
            weights["trend"] += 0.05
            weights["pattern"] -= 0.05

        # Shift 2: Strong Momentum
        if mom > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Momentum detected, scaling Momentum weight.")
            weights["momentum"] += 0.05
            weights["support_resistance"] -= 0.05

        # Shift 3: Volume Explosion
        if vol > 85:
            self._score_reasons.append("Adaptive Shift: Volume explosion confirmed, increasing Volume weight.")
            weights["volume"] += 0.05
            weights["volatility"] -= 0.05

        # Shift 4: Strong Smart Money
        if smc > 85:
            self._score_reasons.append("Adaptive Shift: Heavy Smart Money involvement, increasing Smart Money weight.")
            weights["smart_money"] += 0.06
            weights["institutional"] -= 0.06
            
        # Shift 5: Bullish Market Regime
        if regime > 85:
            self._score_reasons.append("Adaptive Shift: Highly bullish market regime, increasing Regime weight.")
            weights["market_regime"] += 0.05
            weights["risk"] -= 0.02
            weights["reward"] -= 0.03
            
        # Shift 6: Excellent Breakout
        if breakout > 85:
            self._score_reasons.append("Adaptive Shift: Exceptional Breakout mechanics, increasing Pattern weight.")
            weights["pattern"] += 0.06
            weights["trend"] -= 0.06

        # Shift 7: High Volatility Risk (Safety is low)
        if risk_safety < 30:
            self._score_reasons.append("Adaptive Shift: High volatility risk detected, heavily increasing Risk weight.")
            weights["risk"] += 0.10
            weights["trend"] -= 0.05
            weights["momentum"] -= 0.05

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

    def _generate_explanations(self, trend: float, mom: float, vol: float, inst: float, 
                               smc: float, pat: float, sr: float, regime: float, 
                               risk_safety: float, breakout: float) -> None:
        """Synthesizes human-readable institutional logic for final swing state."""
        if trend > 75 and mom > 75:
            self._score_reasons.append("Trend and momentum remain perfectly aligned for swing continuation.")
            
        if breakout > 75 and vol > 75:
            self._score_reasons.append("Volume strongly confirms the structural breakout.")
            
        if inst > 75:
            self._score_reasons.append("Institutional participation strongly supports the directional bias.")
            
        if smc > 75:
            self._score_reasons.append("Smart money activity confirms ongoing accumulation/markup.")
            
        if pat > 75:
            self._score_reasons.append("Pattern structure and quality is exceptionally high.")
            
        if sr > 75:
            self._score_reasons.append("Key support remains completely intact with clear upward space.")
            
        if regime > 75:
            self._score_reasons.append("Broader market regime heavily favors swing continuation.")
            
        if risk_safety < 35:
            self._score_reasons.append("High volatility and wide risk parameters severely reduce confidence.")
            
        if self._conflicts > 0:
            self._score_reasons.append("Conflicting inter-market signals reduce overall swing reliability.")


    # ---------------------------------------------------------
    # COMPONENT BUILDERS (Internal)
    # ---------------------------------------------------------
    
    def _compute_component(self, flat_data: dict[str, Any], keys: list[str], pos_kw: str, neg_kw: str, name: str) -> ScoreBreakdown:
        """Extracts and evaluates a standard sub-component using base framework tools."""
        raw = self._extract_metric(flat_data, keys, 50.0)
        bonus, penalty = 0.0, 0.0
        
        if self._contains_keyword(flat_data, keys, [pos_kw, "true", "yes", "high", "strong", "bullish", "cleared", "stable", "excellent"]):
            bonus = 10.0
            self._positive_log.append(f"Strong/Positive {name} validated.")
            
        if self._contains_keyword(flat_data, keys, [neg_kw, "false", "no", "low", "weak", "bearish", "rejected", "erratic", "poor"]):
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
        Extracts and evaluates an inversely correlated component (e.g., Risk).
        High raw input means High Risk. Component final score maps High Risk to Low Score (Safe = 100).
        """
        raw_risk = self._extract_metric(flat_data, keys, 20.0)
        inverted_raw = self._normalize(100.0 - raw_risk)
        
        bonus, penalty = 0.0, 0.0
        
        # Low risk -> Bonus
        if self._contains_keyword(flat_data, keys, [pos_kw, "false", "no", "low", "safe"]):
            bonus = 10.0
            self._positive_log.append(f"Favorable state regarding {name} validated.")
            
        # High risk -> Penalty
        if self._contains_keyword(flat_data, keys, [neg_kw, "true", "yes", "high", "extreme", "danger"]):
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
                "overall_swing_score": 50.0, 
                "swing_rating": "Neutral", 
                "confidence": 0.0,
                "swing_probability": 50.0,
                "trend_alignment_score": 50.0,
                "momentum_score": 50.0,
                "volume_confirmation_score": 50.0,
                "breakout_score": 50.0,
                "volatility_quality_score": 50.0,
                "smart_money_score": 50.0,
                "pattern_quality_score": 50.0,
                "support_resistance_score": 50.0,
                "institutional_support_score": 50.0,
                "market_regime_score": 50.0,
                "risk_score": 50.0,
                "reward_potential_score": 50.0,
                "swing_reliability": 0.0
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
def calculate_swing_score(swing_json: dict[str, Any] | None) -> dict[str, Any]:
    """
    Public API endpoint to calculate the institutional swing score.
    
    Args:
        swing_json: The dictionary payload compiled from ALL Layer-2 Analyzers.
        
    Returns:
        A JSON-compatible dictionary containing deterministic institutional fusion scores.
    """
    engine = SwingEngine()
    return engine.calculate(swing_json)
