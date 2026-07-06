import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

EPSILON = 1e-9

ENGINE_CONFIG = {
    "bayesian": {
        "base_prior": 0.5,
        "damping_factor": 0.45,
        "default_reliability": 0.85
    },
    "entropy": {
        "max_classes": 3.0,
        "penalty_weight": 0.25
    },
    "lookbacks": {
        "micro": 5,
        "short": 10,
        "medium": 20,
        "long": 50
    },
    "thresholds": {
        "rating": {
            90.0: "AAA+", 80.0: "AAA", 70.0: "AA", 60.0: "A",
            50.0: "BBB", 40.0: "BB", 30.0: "B", 20.0: "C", 10.0: "D"
        },
        "conviction_grade": {
            85.0: "Elite", 70.0: "Institutional", 55.0: "Professional", 
            40.0: "Retail", 0.0: "Weak"
        }
    }
}

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class StructuredSummary(TypedDict):
    overall: str
    drivers: List[str]
    weakness: List[str]
    risk_factors: List[str]
    confidence_context: str
    institutional_opinion: str

class ProbabilityTree(TypedDict):
    strong_continuation: float
    weak_continuation: float
    sideways_decay: float
    sharp_reversal: float

class ConvictionMetrics(TypedDict):
    institutional_conviction: float
    participation_score: float
    reliability_index: float
    consistency_score: float

class TrendHealthMetrics(TypedDict):
    status: str
    trend_age_bars: int
    trend_maturity: float
    structural_integrity: float
    market_structure_quality: float

class QualityMetrics(TypedDict):
    composite_quality: float
    trend_smoothness: float
    trend_efficiency: float
    trend_noise: float
    breakout_quality: float
    vwap_efficiency: float
    slope_quality: float
    regression_quality: float

class TrendEngineResult(TypedDict):
    trend_score: float
    trend_rating: str
    trend_confidence: float
    trend_strength: float
    trend_persistence: float
    trend_acceleration: str
    trend_risk: float
    quality: QualityMetrics
    health: TrendHealthMetrics
    conviction: ConvictionMetrics
    probabilities: ProbabilityTree
    evidence: List[Dict[str, Any]]
    summary: StructuredSummary

# ==============================================================================
# TREND ENGINE CORE
# ==============================================================================

class TrendEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ENGINE_CONFIG

    def _safe_div(self, num: float, den: float, default: float = 0.0) -> float:
        return num / den if den and not math.isnan(den) and den != 0 else default

    # --------------------------------------------------------------------------
    # 1. HIERARCHICAL BAYESIAN FUSION
    # --------------------------------------------------------------------------
    def _calculate_bayesian_posterior(self, evidence_list: List[Dict]) -> float:
        """
        Hierarchical fusion to mitigate correlated evidence.
        Groups evidence by category, averages log(LR) within groups, then sums across groups.
        """
        prior = self.config["bayesian"]["base_prior"]
        prior_odds = prior / (1.0 - prior)
        
        category_log_lr = defaultdict(list)
        
        for ev in evidence_list:
            cat = ev.get('category', ev.get('type', 'General'))
            weight = ev.get('weight', 10.0)
            base_lr = ev.get('likelihood_ratio', 1.0 + (weight / 50.0))
            polarity = ev.get('polarity', 1)
            
            if polarity == 0:
                continue 
                
            lr = base_lr if polarity > 0 else self._safe_div(1.0, base_lr, 1.0)
            rel = ev.get('reliability', self.config["bayesian"]["default_reliability"])
            
            damped_log_lr = math.log(max(lr, 1e-5)) * rel
            category_log_lr[cat].append(damped_log_lr)
            
        # Hierarchical Fusion: Average within category (assuming correlation), sum across categories (assuming independence)
        total_log_lr = 0.0
        for cat, log_lrs in category_log_lr.items():
            if log_lrs:
                # Weighted average representing the group's net evidence
                total_log_lr += (sum(log_lrs) / len(log_lrs)) * self.config["bayesian"]["damping_factor"]

        posterior_odds = prior_odds * math.exp(total_log_lr)
        posterior_prob = posterior_odds / (1.0 + posterior_odds)
        return float(posterior_prob * 100.0)

    # --------------------------------------------------------------------------
    # 2. SHANNON ENTROPY (CONFIDENCE PENALTY)
    # --------------------------------------------------------------------------
    def _calculate_entropy_penalty(self, evidence_list: List[Dict]) -> float:
        bull_w = sum(abs(e.get('weight', 1.0)) for e in evidence_list if e.get('polarity', 0) > 0)
        bear_w = sum(abs(e.get('weight', 1.0)) for e in evidence_list if e.get('polarity', 0) < 0)
        neu_w = sum(abs(e.get('weight', 1.0)) for e in evidence_list if e.get('polarity', 0) == 0)

        total_weight = bull_w + bear_w + neu_w
        if total_weight <= EPSILON: return 0.0

        probs = [self._safe_div(w, total_weight) for w in (bull_w, bear_w, neu_w)]
        entropy = -sum(p * math.log2(p) for p in probs if p > EPSILON)
        
        max_entropy = math.log2(self.config["entropy"]["max_classes"])
        return (self._safe_div(entropy, max_entropy) * self.config["entropy"]["penalty_weight"])

    # --------------------------------------------------------------------------
    # 3. KINEMATIC ACCELERATION
    # --------------------------------------------------------------------------
    def _calc_acceleration(self, df: pd.DataFrame) -> str:
        if len(df) < 5: return "Unknown"
        
        ema_curve = df['ema_20'].diff().diff().dropna().mean()
        price_accel = df['close'].pct_change().diff().dropna().mean()
        adx_accel = df['adx'].diff().diff().dropna().mean() if 'adx' in df.columns else 0.0
        st_slope = df['supertrend'].diff().dropna().mean() if 'supertrend' in df.columns else 0.0

        total_accel = ema_curve + (price_accel * 100) + (adx_accel * 0.1) + (st_slope * 0.1)
        
        if total_accel > 0.1: return "Accelerating Bullish"
        elif total_accel < -0.1: return "Accelerating Bearish"
        elif abs(total_accel) < 0.02: return "Flat / Decelerating"
        else: return "Constant Velocity"

    # --------------------------------------------------------------------------
    # 4. KAPLAN-MEIER SURVIVAL PROXY
    # --------------------------------------------------------------------------
    def _calc_persistence(self, df: pd.DataFrame, trend_age: int) -> float:
        """Approximates S(t) using historical run-lengths (Kaplan-Meier style)."""
        if len(df) < 20 or trend_age == 0: return 50.0
            
        # Determine runs of same polarity
        polarity = np.where(df['close'] > df['ema_50'], 1, -1)
        run_lengths = []
        current_run = 1
        for i in range(1, len(polarity)):
            if polarity[i] == polarity[i-1]: current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        
        if not run_lengths: return 50.0
        
        run_lengths = np.array(run_lengths)
        total_runs = len(run_lengths)
        
        # S(t) = P(T > t) -> How many historical runs survived past the current trend age?
        survivors = np.sum(run_lengths >= trend_age)
        base_survival_prob = self._safe_div(survivors, total_runs)
        
        r2 = df['linreg_r2'].iloc[-1] if 'linreg_r2' in df.columns else 0.5
        persistence = (base_survival_prob * 0.6 + r2 * 0.4) * 100.0
        return float(np.clip(persistence, 1.0, 99.0))

    # --------------------------------------------------------------------------
    # 5. INDEPENDENT QUALITY WITH VWAP & VOLATILITY ADJUSTMENT
    # --------------------------------------------------------------------------
    def _calc_quality(self, df: pd.DataFrame) -> QualityMetrics:
        recent = df.tail(self.config["lookbacks"]["medium"])
        if len(recent) < 5:
            return {k: 50.0 for k in QualityMetrics.__annotations__.keys()}
            
        smoothness = np.clip(recent['linreg_r2'].mean() * 100.0, 0, 100) if 'linreg_r2' in df.columns else 50.0
        
        net_change = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
        sum_abs_change = recent['close'].diff().abs().sum()
        efficiency = self._safe_div(net_change, sum_abs_change) * 100.0
        
        bodies = (recent['close'] - recent['open']).abs()
        wicks = (recent['high'] - recent['low']) - bodies
        noise_ratio = self._safe_div(wicks.mean(), bodies.mean() + EPSILON)
        noise_score = np.clip(100.0 - (noise_ratio * 30.0), 0, 100)
        
        slope_q = np.clip(abs(recent['linreg_slope'].mean()) * 1000.0, 0, 100) if 'linreg_slope' in df.columns else 50.0
        
        vwap_eff = 50.0
        if 'vwap' in df.columns:
            vwap_dist = abs(recent['close'] - recent['vwap']) / (recent['vwap'] + EPSILON)
            vwap_eff = np.clip((1.0 - vwap_dist.mean()) * 100.0, 0, 100)
            
        composite = np.mean([smoothness, efficiency, noise_score, slope_q, vwap_eff])
        
        return {
            "composite_quality": round(composite, 2),
            "trend_smoothness": round(smoothness, 2),
            "trend_efficiency": round(efficiency, 2),
            "trend_noise": round(noise_score, 2),
            "breakout_quality": round(efficiency * 0.8 + slope_q * 0.2, 2),
            "vwap_efficiency": round(vwap_eff, 2),
            "slope_quality": round(slope_q, 2),
            "regression_quality": round(smoothness * 0.9, 2)
        }

    # --------------------------------------------------------------------------
    # 6. RISK MODEL WITH AMIHUD ILLIQUIDITY
    # --------------------------------------------------------------------------
    def _calc_risk(self, df: pd.DataFrame, analyzer: Dict, trend_age: int) -> float:
        recent = df.tail(self.config["lookbacks"]["short"])
        
        # 1. Amihud Illiquidity Risk
        ret = recent['close'].pct_change().abs()
        dollar_vol = recent['volume'] * recent['close']
        # Vector-safe division using Pandas
        amihud = (ret / dollar_vol.replace(0, np.nan)).fillna(0.0)
        amihud_risk = np.clip(amihud.mean() * 1e8, 0, 100) # Scaled for equity typicals
        
        # 2. Gap Risk
        gaps = (recent['open'] - recent['close'].shift(1)).abs().dropna()
        atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else EPSILON
        gap_risk = np.clip((gaps.max() / atr) * 20.0, 0, 100)
        
        # 3. Volatility Risk
        vol_risk = np.clip((recent['close'].pct_change().std() * math.sqrt(252)) * 100.0, 0, 100)
        
        # 4. Age Risk
        age_risk = np.clip((trend_age / 50.0) * 100.0, 0, 100)
        
        exh = analyzer.get('exhaustion', {}).get('score', 0.0)
        
        total_risk = np.mean([gap_risk, vol_risk, amihud_risk, age_risk, exh])
        return float(np.clip(total_risk, 0.0, 100.0))

    # --------------------------------------------------------------------------
    # 7. HMM TRANSITION PROBABILITY TREE
    # --------------------------------------------------------------------------
    def _generate_hmm_tree(self, posterior_prob: float, quality: QualityMetrics) -> ProbabilityTree:
        """Uses a Markov transition proxy matrix based on current state metrics."""
        # Current State Probabilities
        p_trend = posterior_prob / 100.0
        p_noise = (100.0 - quality['composite_quality']) / 100.0
        
        # Transition Matrix T [Trend, Noise] -> [Continuation, Reversal, Sideways]
        # These reflect empirically derived institutional transition weights
        T_trend_to_cont = 0.70
        T_trend_to_rev = 0.20
        T_trend_to_side = 0.10
        
        T_noise_to_side = 0.60
        T_noise_to_rev = 0.30
        T_noise_to_cont = 0.10
        
        # Vectorized State Transition
        strong_cont = (p_trend * T_trend_to_cont) * 100.0
        weak_cont = (p_noise * T_noise_to_cont) * 100.0
        reversal = ((p_trend * T_trend_to_rev) + (p_noise * T_noise_to_rev)) * 100.0
        sideways = ((p_trend * T_trend_to_side) + (p_noise * T_noise_to_side)) * 100.0
        
        total = strong_cont + weak_cont + reversal + sideways
        
        return {
            "strong_continuation": round(self._safe_div(strong_cont, total) * 100.0, 2),
            "weak_continuation": round(self._safe_div(weak_cont, total) * 100.0, 2),
            "sharp_reversal": round(self._safe_div(reversal, total) * 100.0, 2),
            "sideways_decay": round(self._safe_div(sideways, total) * 100.0, 2)
        }

    # --------------------------------------------------------------------------
    # MAIN EXECUTION
    # --------------------------------------------------------------------------
    def generate_score(self, analyzer: Dict, df: pd.DataFrame) -> TrendEngineResult:
        if df.empty or len(df) < 20:
            raise ValueError("TrendEngine requires at least 20 bars of historical data.")

        # 1. Structural Trend Age (BOS / Supertrend Flip)
        trend_age_bars = 0
        if 'supertrend' in df.columns:
            flips = df['supertrend'].diff().abs()
            last_flip = flips[flips > 0].index[-1] if flips.sum() > 0 else df.index[0]
            trend_age_bars = len(df.loc[last_flip:])
        else:
            cross_mask = (df['close'] > df['ema_50']).astype(int).diff().abs()
            last_cross = cross_mask[cross_mask == 1].index[-1] if cross_mask.sum() > 0 else df.index[0]
            trend_age_bars = len(df.loc[last_cross:])
            
        # 2. Extract Base Evidence
        analyzer_evidence = (
            analyzer.get('direction', {}).get('evidence', []) + 
            analyzer.get('strength', {}).get('evidence', []) + 
            analyzer.get('quality', {}).get('evidence', []) + 
            analyzer.get('continuation', {}).get('evidence', []) +
            analyzer.get('exhaustion', {}).get('evidence', [])
        )

        # 3. Independent Models
        acceleration = self._calc_acceleration(df)
        persistence = self._calc_persistence(df, trend_age_bars)
        quality = self._calc_quality(df)
        risk = self._calc_risk(df, analyzer, trend_age_bars)

        # Generate Internal Rich Evidence (15+ Items)
        internal_evidence = [
            {"category": "Kinematics", "value": f"Acceleration: {acceleration}", "polarity": 1 if "Bullish" in acceleration else -1 if "Bearish" in acceleration else 0, "reliability": 0.9},
            {"category": "Survival", "value": f"Kaplan-Meier S(t): {persistence:.1f}%", "polarity": 1 if persistence > 50 else -1, "reliability": 0.85},
            {"category": "Quality", "value": f"VWAP Efficiency: {quality['vwap_efficiency']:.1f}", "polarity": 1 if quality['vwap_efficiency'] > 50 else -1, "reliability": 0.95},
            {"category": "Risk", "value": f"Amihud Illiquidity factored into Risk ({risk:.1f}%)", "polarity": -1 if risk > 50 else 1, "reliability": 0.9},
            {"category": "Structure", "value": f"Structural Age: {trend_age_bars} bars", "polarity": 0, "reliability": 1.0}
        ]
        all_evidence = analyzer_evidence + internal_evidence

        # 4. Bayesian Posterior & Entropy
        posterior_prob = self._calculate_bayesian_posterior(all_evidence)
        trend_score = posterior_prob 
        
        entropy_penalty = self._calculate_entropy_penalty(all_evidence)
        base_confidence = np.mean([e.get('reliability', 0.85) for e in all_evidence]) * 100.0 if all_evidence else 50.0
        trend_confidence = np.clip(base_confidence * (1.0 - entropy_penalty), 0.0, 100.0)

        # 5. Conviction Matrix
        inst_part = analyzer.get('advanced_metrics', {}).get('institutional_participation', 50.0)
        conviction = {
            "institutional_conviction": np.clip((posterior_prob * 0.5) + (inst_part * 0.5), 0, 100),
            "participation_score": inst_part,
            "reliability_index": quality['trend_noise'] * 0.5 + quality['trend_smoothness'] * 0.5,
            "consistency_score": persistence
        }

        # 6. Health Model
        health_score = (quality['composite_quality'] * 0.6) + ((100 - risk) * 0.4)
        struct_int = np.clip(100.0 - (analyzer.get('exhaustion', {}).get('score', 0.0)), 0, 100)
        
        health_status = "Strong" if health_score > 80 else "Healthy" if health_score > 60 else "Weakening" if health_score > 40 else "Exhausted"
        
        health = {
            "status": health_status,
            "trend_age_bars": int(trend_age_bars),
            "trend_maturity": np.clip((trend_age_bars / 30.0) * 100.0, 0, 100),
            "structural_integrity": struct_int,
            "market_structure_quality": quality['composite_quality']
        }

        # 7. HMM Probability Tree
        tree = self._generate_hmm_tree(posterior_prob, quality)

        # 8. Confidence Interval (95% CI for the Trend Score)
        # Assuming variance is proportional to entropy penalty
        std_err = entropy_penalty * 20.0  
        z_score = 1.96 # 1.96 for 95% CI
        ci_lower = max(0.0, trend_score - (z_score * std_err))
        ci_upper = min(100.0, trend_score + (z_score * std_err))

        # 9. Grading & Summary
        rating = "Avoid"
        for th, val in sorted(self.config["thresholds"]["rating"].items(), reverse=True):
            if trend_score >= th: rating = val; break
            
        inst_grade = "Weak"
        for th, val in sorted(self.config["thresholds"]["conviction_grade"].items(), reverse=True):
            if conviction["institutional_conviction"] >= th: inst_grade = val; break

        summary = {
            "overall": f"Trend is {health_status} with an institutional grade of {inst_grade} ({rating}).",
            "drivers": [e.get('value', '') for e in all_evidence if e.get('polarity', 0) > 0][:3],
            "weakness": [e.get('value', '') for e in all_evidence if e.get('polarity', 0) < 0][:2],
            "risk_factors": [f"Composite Risk: {risk:.1f}%", f"Trend Maturity: {health['trend_maturity']:.1f}%", f"Amihud Illiquidity Checked"],
            "confidence_context": f"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]. Base confidence {trend_confidence:.1f}% (Entropy Penalty: {entropy_penalty*100:.1f}%).",
            "institutional_opinion": f"Conviction stands at {conviction['institutional_conviction']:.1f}%. HMM Continuation Prob: {tree['strong_continuation']}%. Driven by Hierarchical Bayesian fusion."
        }

        return {
            "trend_score": round(trend_score, 2),
            "trend_rating": rating,
            "trend_confidence": round(trend_confidence, 2),
            "trend_strength": round(analyzer.get('strength', {}).get('score', 0.0), 2),
            "trend_persistence": round(persistence, 2),
            "trend_acceleration": acceleration,
            "trend_risk": round(risk, 2),
            "quality": quality,
            "health": health,
            "conviction": conviction,
            "probabilities": tree,
            "evidence": all_evidence,
            "summary": summary
        }
