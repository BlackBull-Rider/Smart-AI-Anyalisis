import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Any, Tuple
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
        "penalty_weight": 0.30 
    },
    "lookbacks": {
        "short": 5,
        "medium": 14,
        "long": 30
    },
    "thresholds": {
        "rating": {
            85.0: "Explosive", 70.0: "Strong", 55.0: "Active", 
            40.0: "Building", 25.0: "Fading", 10.0: "Weak", 0.0: "Dead"
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

class MarkovTree(TypedDict):
    ignition: float
    expansion: float
    climax: float
    decay: float
    mean_reversion: float

class PhysicsMetrics(TypedDict):
    norm_velocity: float
    norm_acceleration: float
    norm_jerk: float
    kinetic_energy: float
    momentum_impulse: float
    momentum_path_curvature: float  # DOCUMENTED: Dynamic path of the ROC momentum, not price.
    energy_dissipation_rate: float
    kinematic_state: str

class QualityMetrics(TypedDict):
    momentum_efficiency: float
    momentum_smoothness: float
    oscillator_agreement: float
    bullish_consensus: float       # ADDED: Tracking clear bullish momentum power
    structural_integrity: float
    feature_availability_score: float

class RiskMetrics(TypedDict):
    oscillator_conflict: float
    volatility_shock_risk: float
    liquidity_collapse_risk: float
    momentum_failure_prob: float
    false_breakout_prob: float
    volatility_clustering: float
    composite_risk: float

class HealthMetrics(TypedDict):
    status: str
    momentum_age_bars: int
    survival_probability: float
    exhaustion_level: float

class MomentumEngineResult(TypedDict):
    momentum_score: float
    momentum_rating: str
    momentum_confidence: float
    raw_posterior: float
    entropy_value: float
    effective_evidence_count: float
    physics: PhysicsMetrics
    quality: QualityMetrics
    risk: RiskMetrics
    health: HealthMetrics
    probabilities: MarkovTree
    evidence: List[Dict[str, Any]]
    summary: StructuredSummary

# ==============================================================================
# MOMENTUM ENGINE CORE
# ==============================================================================

class MomentumEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or ENGINE_CONFIG
        self.expected_schema = [
            'close', 'volume', 'roc', 'macd_line', 'macd_histogram', 
            'rsi', 'atr_14', 'mfi', 'cci', 'stoch_k'
        ]

    def _safe_div(self, num: float, den: float, default: float = 0.0) -> float:
        """Scalar-only division helper wrapped with scalar verification."""
        if isinstance(num, (list, tuple, np.ndarray, pd.Series)) or isinstance(den, (list, tuple, np.ndarray, pd.Series)):
            raise TypeError("MomentumEngine._safe_div accepts scalar inputs only.")
        return num / den if den and not math.isnan(den) and den != 0 else default

    def _validate_schema(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        working_df = df.copy()
        missing_count = 0
        total_cols = len(self.expected_schema)
        
        for col in self.expected_schema:
            if col not in working_df.columns or working_df[col].isna().all():
                missing_count += 1
                if col in ['rsi', 'mfi', 'stoch_k']: working_df[col] = 50.0
                elif col in ['roc', 'macd_line', 'macd_histogram', 'cci']: working_df[col] = 0.0
                elif col == 'atr_14': working_df[col] = working_df['close'] * 0.01 if 'close' in working_df else 1.0
                elif col == 'volume': working_df[col] = 1.0
                else: working_df[col] = 0.0
                
        avail_score = ((total_cols - missing_count) / total_cols) * 100.0
        return working_df, avail_score

    # --------------------------------------------------------------------------
    # 1. BAYESIAN FUSION & DYNAMIC SCALING
    # --------------------------------------------------------------------------
    def _calculate_bayesian_posterior(self, evidence_list: List[Dict]) -> Tuple[float, float]:
        """Returns (posterior_prob, dynamic_rho_avg)"""
        prior = self.config["bayesian"]["base_prior"]
        prior_odds = prior / (1.0 - prior)
        category_log_lr = defaultdict(list)
        
        for ev in evidence_list:
            cat = ev.get('category', ev.get('type', 'General'))
            weight = ev.get('weight', 10.0)
            base_lr = ev.get('likelihood_ratio', 1.0 + (weight / 50.0))
            polarity = ev.get('polarity', 1)
            
            if polarity == 0: continue 
                
            lr = base_lr if polarity > 0 else self._safe_div(1.0, base_lr, 1.0)
            rel = ev.get('reliability', self.config["bayesian"]["default_reliability"])
            freshness = ev.get('freshness', 1.0) 
            
            damped_log_lr = math.log(max(lr, 1e-5)) * rel * freshness
            category_log_lr[cat].append(damped_log_lr)
            
        total_log_lr = 0.0
        rho_sum = 0.0
        cat_count = len(category_log_lr)
        
        for cat, log_lrs in category_log_lr.items():
            if log_lrs:
                # DYNAMIC CORRELATION: Higher count within same category expands correlation factor
                n_items = len(log_lrs)
                dyn_rho = min(0.50, 0.10 + (n_items * 0.05))
                rho_sum += dyn_rho
                
                total_log_lr += (sum(log_lrs) / n_items) * self.config["bayesian"]["damping_factor"]

        # TECHNICAL OBSERVATION 2: Guarding math.exp against underflow/overflow bounds
        clamped_log_lr = max(-20.0, min(20.0, total_log_lr))
        posterior_odds = prior_odds * math.exp(clamped_log_lr)
        posterior_prob = posterior_odds / (1.0 + posterior_odds)
        
        avg_rho = self._safe_div(rho_sum, cat_count, 0.25)
        return float(posterior_prob * 100.0), avg_rho

    def _calculate_entropy(self, evidence_list: List[Dict]) -> float:
        bull_w = sum(abs(e.get('weight', 1.0)) for e in evidence_list if e.get('polarity', 0) > 0)
        bear_w = sum(abs(e.get('weight', 1.0)) for e in evidence_list if e.get('polarity', 0) < 0)
        neu_w = sum(abs(e.get('weight', 1.0)) for e in evidence_list if e.get('polarity', 0) == 0)

        total_weight = bull_w + bear_w + neu_w
        if total_weight <= EPSILON: return 0.0

        probs = [self._safe_div(w, total_weight) for w in (bull_w, bear_w, neu_w)]
        entropy = -sum(p * math.log2(p) for p in probs if p > EPSILON)
        max_entropy = math.log2(self.config["entropy"]["max_classes"])
        
        return self._safe_div(entropy, max_entropy)

    # --------------------------------------------------------------------------
    # 2. MOMENTUM PHYSICS (Calculus of the ROC Path)
    # --------------------------------------------------------------------------
    def _calc_physics(self, df: pd.DataFrame) -> PhysicsMetrics:
        roc = df['roc']
        vol = df['volume']
        
        velocity = roc.diff().fillna(0)
        acceleration = velocity.diff().fillna(0)
        jerk = acceleration.diff().fillna(0)
        
        v_norm = self._safe_div(velocity.iloc[-1] - velocity.mean(), velocity.std() + EPSILON)
        a_norm = self._safe_div(acceleration.iloc[-1] - acceleration.mean(), acceleration.std() + EPSILON)
        j_norm = self._safe_div(jerk.iloc[-1] - jerk.mean(), jerk.std() + EPSILON)
        
        mass = self._safe_div(vol.iloc[-1], vol.mean() + EPSILON)
        ke = 0.5 * mass * (v_norm ** 2) * np.sign(v_norm) 
        impulse = mass * a_norm
        
        # DOCUMENTED: Curvature parameters represent the rate of change of the ROC momentum vector path
        curvature = abs(v_norm * j_norm - a_norm**2) / (math.pow(1 + v_norm**2, 1.5) + EPSILON)
        dissipation = max(0.0, -1.0 * ke * a_norm) if v_norm > 0 else max(0.0, ke * a_norm)
        
        state = "Explosive" if (v_norm > 1.0 and a_norm > 0 and j_norm > 0) else \
                "Accelerating" if (v_norm > 0 and a_norm > 0) else \
                "Decelerating" if (v_norm > 0 and a_norm < 0) else \
                "Collapsing" if (v_norm < 0 and a_norm < 0) else "Neutral / Mean Reverting"

        return {
            "norm_velocity": round(v_norm, 3),
            "norm_acceleration": round(a_norm, 3),
            "norm_jerk": round(j_norm, 3),
            "kinetic_energy": round(ke, 3),
            "momentum_impulse": round(impulse, 3),
            "momentum_path_curvature": round(curvature, 3),
            "energy_dissipation_rate": round(dissipation, 3),
            "kinematic_state": state
        }

    # --------------------------------------------------------------------------
    # 3. ADVANCED QUALITY & RISK
    # --------------------------------------------------------------------------
    def _calc_quality(self, df: pd.DataFrame, avail_score: float) -> QualityMetrics:
        recent = df.tail(self.config["lookbacks"]["medium"])
        roc = recent['roc'].dropna()
        
        # TECHNICAL OBSERVATION 3: Safe guard against zero variance to prevent np.corrcoef warnings/NaN
        if len(roc) > 3 and roc.std() > EPSILON:
            x = np.arange(len(roc))
            smoothness = (np.corrcoef(x, roc)[0, 1] ** 2) * 100.0
        else:
            smoothness = 0.0
            
        net_roc = abs(roc.iloc[-1] - roc.iloc[0]) if len(roc) > 0 else 0
        abs_roc_sum = roc.diff().abs().sum()
        efficiency = self._safe_div(net_roc, abs_roc_sum) * 100.0
        
        # Weighted Oscillator Voting & Bullish Direction Separation
        weights = {'rsi': 0.25, 'macd': 0.25, 'roc': 0.20, 'cci': 0.15, 'stoch': 0.15}
        bull_votes = 0.0
        total_w = sum(weights.values())
        
        if recent['rsi'].iloc[-1] > 50: bull_votes += weights['rsi']
        if recent['macd_line'].iloc[-1] > 0: bull_votes += weights['macd']
        if recent['roc'].iloc[-1] > 0: bull_votes += weights['roc']
        if recent['cci'].iloc[-1] > 0: bull_votes += weights['cci']
        if recent['stoch_k'].iloc[-1] > 50: bull_votes += weights['stoch']
        
        # Bullish Consensus shows absolute directional force (0 to 100)
        bullish_consensus = (bull_votes / total_w) * 100.0
        
        # Agreement shows the tight clustering of oscillators regardless of trend side
        agreement_score = bullish_consensus if bullish_consensus >= 50.0 else (100.0 - bullish_consensus)

        return {
            "momentum_efficiency": round(np.clip(efficiency, 0.0, 100.0), 2),
            "momentum_smoothness": round(np.clip(np.nan_to_num(smoothness), 0.0, 100.0), 2),
            "oscillator_agreement": round(agreement_score, 2),
            "bullish_consensus": round(bullish_consensus, 2),
            "structural_integrity": round((efficiency * 0.4) + (smoothness * 0.4) + (avail_score * 0.2), 2),
            "feature_availability_score": round(avail_score, 2)
        }

    def _calc_risk(self, df: pd.DataFrame, physics: PhysicsMetrics, qual: QualityMetrics) -> RiskMetrics:
        conflict = 100.0 - qual['oscillator_agreement']
        atr = df['atr_14'].tail(14)
        
        vol_shock = np.clip((atr.iloc[-1] / (atr.mean() + EPSILON) - 1.0) * 100.0, 0.0, 100.0)
        vol_trend = df['volume'].tail(5).diff().mean()
        liq_risk = 100.0 if (vol_trend < 0 and physics['norm_velocity'] > 0) else 0.0
        
        fail_prob = 80.0 if (abs(physics['norm_velocity']) > 1.0 and physics['norm_acceleration'] < -0.5) else 20.0
        vol_cluster = np.clip((atr.std() / (atr.mean() + EPSILON)) * 100.0, 0.0, 100.0)
        
        fb_prob = 85.0 if (physics['kinetic_energy'] > 0 and physics['momentum_impulse'] < 0 and vol_shock < 20) else 15.0
        composite = np.mean([conflict, vol_shock, liq_risk, fail_prob, fb_prob, vol_cluster * 0.5])
        
        return {
            "oscillator_conflict": round(conflict, 2),
            "volatility_shock_risk": round(vol_shock, 2),
            "liquidity_collapse_risk": round(liq_risk, 2),
            "momentum_failure_prob": round(fail_prob, 2),
            "false_breakout_prob": round(fb_prob, 2),
            "volatility_clustering": round(vol_cluster, 2),
            "composite_risk": round(np.clip(composite, 0.0, 100.0), 2)
        }

    def _calc_temporal_health(self, df: pd.DataFrame) -> Tuple[int, float]:
        hist = df['macd_histogram']
        polarity = np.where(hist > 0, 1, -1)
        
        age = 0
        current_pol = polarity[-1]
        for i in range(len(polarity)-1, -1, -1):
            if polarity[i] == current_pol: age += 1
            else: break
            
        run_lengths = []
        curr_run = 1
        for i in range(1, len(polarity)):
            if polarity[i] == polarity[i-1]: curr_run += 1
            else:
                run_lengths.append(curr_run)
                curr_run = 1
        run_lengths.append(curr_run)
        
        total_runs = len(run_lengths)
        survivors = np.sum(np.array(run_lengths) > age)
        survival_prob = self._safe_div(survivors, total_runs) * 100.0 if total_runs > 0 else 50.0
        
        return age, round(np.clip(survival_prob, 0.0, 100.0), 2)

    # --------------------------------------------------------------------------
    # 4. TRUE MARKOV STATE TRANSITION (Bayesian Blended States)
    # --------------------------------------------------------------------------
    def _apply_markov_matrix(self, p_post: float, phys: PhysicsMetrics, risk: RiskMetrics) -> MarkovTree:
        """Vectorized Markov Transition. Weights the initial vector S_t with the posterior probability."""
        p_trend = max(p_post / 100.0, 0.01) 
        v, a = phys['norm_velocity'], phys['norm_acceleration']
        
        # Blending raw kinematic states with raw Bayesian likelihoods
        s_ign = (1.0 if (abs(v) < 0.5 and abs(a) > 1.0) else 0.1) * p_trend
        s_exp = (1.0 if (v > 1.0 and a > 0.5) else 0.1) * p_trend
        s_clm = (1.0 if (v > 2.0 and a < 0) else 0.1) * p_trend
        s_dec = (1.0 if (v > 0 and a < -1.0) else 0.1) * (1.0 - p_trend)
        s_rev = (1.0 if (abs(v) < 0.2 and abs(a) < 0.2) else 0.1) * (1.0 - p_trend)
        
        S_t = np.array([s_ign, s_exp, s_clm, s_dec, s_rev])
        S_t = S_t / np.sum(S_t) 
        
        # 5x5 Transition Probability Matrix (Learned Proxy)
        T = np.array([
            [0.20, 0.60, 0.05, 0.10, 0.05], 
            [0.05, 0.60, 0.20, 0.10, 0.05], 
            [0.00, 0.10, 0.30, 0.50, 0.10], 
            [0.00, 0.10, 0.05, 0.60, 0.25], 
            [0.30, 0.10, 0.00, 0.00, 0.60]  
        ])
        
        r_penalty = risk['composite_risk'] / 100.0
        T[:, 3] += (T[:, 1] * r_penalty)
        T[:, 1] *= (1.0 - r_penalty)
        
        T = T / T.sum(axis=1)[:, np.newaxis]
        S_next = np.dot(S_t, T)
        
        return {
            "ignition": round(S_next[0] * 100.0, 2),
            "expansion": round(S_next[1] * 100.0, 2),
            "climax": round(S_next[2] * 100.0, 2),
            "decay": round(S_next[3] * 100.0, 2),
            "mean_reversion": round(S_next[4] * 100.0, 2)
        }

    # --------------------------------------------------------------------------
    # MAIN PIPELINE EXECUTION
    # --------------------------------------------------------------------------
    def generate_score(self, analyzer: Dict, raw_df: pd.DataFrame) -> MomentumEngineResult:
        if raw_df.empty or len(raw_df) < 20:
            raise ValueError("MomentumEngine requires at least 20 bars of historical data.")

        # Validate Schema and extract availability penalties
        df, avail_score = self._validate_schema(raw_df)

        # Gather base evidences
        all_evidence = []
        for key in ["momentum_strength", "momentum_acceleration", "momentum_exhaustion", "momentum_shift", "momentum_ignition", "momentum_compression", "institutional_momentum", "rsi_analysis", "macd_analysis", "divergence", "swing_readiness"]:
            if key in analyzer and "evidence" in analyzer[key]:
                all_evidence.extend(analyzer[key]["evidence"])

        for ev in all_evidence:
            if "reliability" not in ev: ev["reliability"] = 0.85

        # Independent Models Processing
        physics = self._calc_physics(df)
        quality = self._calc_quality(df, avail_score)
        age, survival = self._calc_temporal_health(df)
        risk = self._calc_risk(df, physics, quality)
        
        all_evidence.extend([
            {"category": "Kinematics", "value": f"Kinetic Energy: {physics['kinetic_energy']}", "polarity": np.sign(physics['norm_velocity']), "reliability": 0.95},
            {"category": "Survival", "value": f"Momentum S(t): {survival}% at Age {age}", "polarity": 1 if survival > 50 else -1, "reliability": 0.9},
            {"category": "Quality", "value": f"Oscillator Consensus: {quality['bullish_consensus']}%", "polarity": 1 if quality['bullish_consensus'] > 50 else -1, "reliability": 0.9}
        ])

        # Bayesian Fusion with Dynamic Calibration
        raw_posterior, dynamic_rho = self._calculate_bayesian_posterior(all_evidence)
        entropy_val = self._calculate_entropy(all_evidence)
        entropy_penalty = entropy_val * self.config["entropy"]["penalty_weight"]
        
        base_confidence = np.mean([e.get('reliability', 0.85) for e in all_evidence]) * 100.0 if all_evidence else 50.0
        
        # Composite Institutional Score Generation with strict Schema Penalty
        inst_conviction = analyzer.get('institutional_momentum', {}).get('institutional_score', 0.0)
        raw_score = (raw_posterior * 0.35) + (quality['structural_integrity'] * 0.25) + (survival * 0.20) + (inst_conviction * 0.20)
        
        # Direct Feature Availability Multiplier on the Score
        momentum_score = np.clip(raw_score - (risk['composite_risk'] * 0.3), 0.0, 100.0)
        momentum_score = momentum_score * (avail_score / 100.0) # Direct Score Penalty for missing features
        
        momentum_confidence = np.clip(base_confidence * (1.0 - entropy_penalty) * (avail_score / 100.0), 0.0, 100.0)

        # Lifecycle Classification
        exh = analyzer.get('momentum_exhaustion', {}).get('score', 0.0)
        health_status = "Explosive" if (momentum_score > 75 and risk['composite_risk'] < 20) else \
                        "Accelerating" if physics['norm_acceleration'] > 0 else \
                        "Mature" if age > 10 else \
                        "Exhausted" if exh > 60 else \
                        "Collapsing" if physics['norm_jerk'] < 0 and physics['norm_acceleration'] < 0 else "Igniting"

        health: HealthMetrics = {
            "status": health_status,
            "momentum_age_bars": age,
            "survival_probability": survival,
            "exhaustion_level": round(exh, 2)
        }

        # True Markov Matrix Multiplication
        markov_tree = self._apply_markov_matrix(raw_posterior, physics, risk)

        # Correlated Sample Size Correction for Confidence Interval Bounds
        p = raw_posterior / 100.0
        n_raw = len(all_evidence) if len(all_evidence) > 0 else 1
        n_eff = max(1.0, n_raw / (1.0 + (n_raw - 1.0) * dynamic_rho))
        
        std_err_p = math.sqrt((p * (1.0 - p)) / n_eff) if p < 1.0 and p > 0.0 else 0.05
        scaled_se = std_err_p * (1.0 + entropy_penalty) * 100.0 
        
        z_score = 1.96 
        ci_lower = max(0.0, momentum_score - (z_score * scaled_se))
        ci_upper = min(100.0, momentum_score + (z_score * scaled_se))

        # Output Grading
        rating = "Dead"
        for th, val in sorted(self.config["thresholds"]["rating"].items(), reverse=True):
            if momentum_score >= th: rating = val; break
            
        summary: StructuredSummary = {
            "overall": f"Momentum is {rating} ({round(momentum_score, 1)}). Kinematic State: {physics['kinematic_state']}.",
            "drivers": [e.get('value', '') for e in all_evidence if e.get('polarity', 0) > 0][:3],
            "weakness": [e.get('value', '') for e in all_evidence if e.get('polarity', 0) < 0][:2],
            "risk_factors": [f"Composite Risk: {risk['composite_risk']}%", f"FB Prob: {risk['false_breakout_prob']}%", f"Curvature: {physics['momentum_path_curvature']}"],
            "confidence_context": f"95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]. Schema Score: {avail_score}%. n_eff: {round(n_eff, 1)} (Dynamic Rho: {round(dynamic_rho, 2)}).",
            "institutional_opinion": f"Markov Expansion Prob: {markov_tree['expansion']}%. Dissipation: {physics['energy_dissipation_rate']}%. Bullish Consensus: {quality['browse_consensus'] if 'browse_consensus' in quality else quality['bullish_consensus']}%."
        }

        return {
            "momentum_score": round(momentum_score, 2),
            "momentum_rating": rating,
            "momentum_confidence": round(momentum_confidence, 2),
            "raw_posterior": round(raw_posterior, 2),
            "entropy_value": round(entropy_val, 4),
            "effective_evidence_count": round(n_eff, 2),
            "physics": physics,
            "quality": quality,
            "risk": risk,
            "health": health,
            "probabilities": markov_tree,
            "evidence": all_evidence,
            "summary": summary
        }
