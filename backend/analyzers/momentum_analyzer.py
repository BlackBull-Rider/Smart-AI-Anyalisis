import numpy as np
import pandas as pd
from typing import Dict, Any

class MomentumAnalyzer:
    """
    Green Bull Rider V6 - L1 Momentum Analyzer (Institutional Grade)
    
    Operates strictly on the pipeline-provided exact momentum feature whitelist.
    Evaluates kinematic acceleration, directional dominance, and statistical extremity
    using continuous derivatives and Z-Scores. Zero retail crossover logic.
    """

    def __init__(self):
        # Statistical Standard Deviation Boundaries
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes dynamic momentum tracking. Expects the following categorized blocks:
        - 'oscillator_block': rsi_14, stoch_k, stoch_d, stoch_rsi_k, stoch_rsi_d, williams_r, ultimate_osc, cci_20
        - 'kinematic_block': macd, macd_signal, macd_hist, ppo, ppo_signal, ppo_hist, roc_12, mom_10, trix_18, dpo_20, rsi_slope
        - 'directional_block': adx_14, plus_di, minus_di, dx
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. MATHEMATICAL MICRO-TRACKERS
        kinematics = self._track_kinematic_momentum(feature_blocks.get("kinematic_block"))
        directional = self._track_directional_dominance(feature_blocks.get("directional_block"))
        extremity = self._track_statistical_extremity(feature_blocks.get("oscillator_block"))

        # 2. SYNTHESIS
        directional_bias = self._synthesize_bias(kinematics, directional)
        momentum_state = self._synthesize_momentum_state(kinematics, extremity, directional_bias)
        confidence = self._compute_statistical_confidence(kinematics, directional, extremity)

        # 3. EXACT LOCKED JSON CONTRACT
        return {
            "analyzer_name": "momentum_analyzer",
            "momentum_state": momentum_state,
            "directional_bias": directional_bias,
            "kinematics": {
                "velocity_z": kinematics["velocity_z"],
                "acceleration_z": kinematics["acceleration_z"],
                "is_diverging": kinematics["is_diverging"]
            },
            "directional_strength": {
                "adx_velocity_z": directional["adx_vel_z"],
                "di_spread_z": directional["di_spread_z"],
                "dominance": directional["dominance_status"]
            },
            "extremity": {
                "oscillator_z": extremity["osc_aggregate_z"],
                "is_overextended": extremity["is_overextended"]
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _track_kinematic_momentum(self, df: pd.DataFrame) -> dict:
        """
        Tracks unbounded momentum (MACD, ROC, PPO) by measuring their 1st (Velocity) 
        and 2nd (Acceleration) derivatives to detect early exhaustion or divergence.
        """
        res = {"velocity_z": 0.0, "acceleration_z": 0.0, "is_diverging": False}
        if df is None or df.empty:
            return res

        # Focus on the pure kinetic vectors (Histograms and Rates of Change)
        kinetic_cols = [c for c in ['macd_hist', 'ppo_hist', 'roc_12', 'mom_10', 'rsi_slope', 'trix_18'] if c in df.columns]
        
        if kinetic_cols:
            matrix = df[kinetic_cols].ffill().bfill().to_numpy()
            
            # Mean cross-sectional momentum value
            mean_momentum = np.nanmean(matrix, axis=1)
            
            velocity = np.gradient(mean_momentum)
            acceleration = np.gradient(velocity)
            
            res["velocity_z"] = self._calculate_z_score(velocity)
            res["acceleration_z"] = self._calculate_z_score(acceleration)
            
            # Internal Divergence/Exhaustion: High velocity but aggressively decaying acceleration
            if abs(res["velocity_z"]) > self.Z_SIGNIFICANT and np.sign(res["velocity_z"]) != np.sign(res["acceleration_z"]):
                if abs(res["acceleration_z"]) > self.Z_MODERATE:
                    res["is_diverging"] = True

        return res

    def _track_directional_dominance(self, df: pd.DataFrame) -> dict:
        """
        Measures the structural expansion of trend momentum using ADX and DI spreads.
        """
        res = {"adx_vel_z": 0.0, "di_spread_z": 0.0, "dominance_status": "neutral"}
        if df is None or df.empty:
            return res

        if 'adx_14' in df.columns:
            adx = df['adx_14'].ffill().bfill().to_numpy()
            adx_velocity = np.gradient(adx)
            res["adx_vel_z"] = self._calculate_z_score(adx_velocity)

        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            pdi = df['plus_di'].ffill().bfill().to_numpy()
            mdi = df['minus_di'].ffill().bfill().to_numpy()
            
            # Directional Spread
            di_spread = pdi - mdi
            res["di_spread_z"] = self._calculate_z_score(di_spread)
            
            if res["di_spread_z"] > self.Z_SIGNIFICANT:
                res["dominance_status"] = "bull_dominant"
            elif res["di_spread_z"] < -self.Z_SIGNIFICANT:
                res["dominance_status"] = "bear_dominant"

        return res

    def _track_statistical_extremity(self, df: pd.DataFrame) -> dict:
        """
        Replaces fixed overbought/oversold levels (e.g., RSI > 70) with 
        a continuous Z-Score of bounded oscillator cross-sectional mean.
        """
        res = {"osc_aggregate_z": 0.0, "is_overextended": False}
        if df is None or df.empty:
            return res

        matrix = df.ffill().bfill().to_numpy()
        
        # Calculate mean of all oscillators at each time step
        mean_osc = np.nanmean(matrix, axis=1)
        
        # Determine how far the current oscillator basket is from its historical mean
        res["osc_aggregate_z"] = self._calculate_z_score(mean_osc)
        
        # Statistical overextension (Beyond 2 Standard Deviations)
        if abs(res["osc_aggregate_z"]) > self.Z_EXTREME:
            res["is_overextended"] = True
            
        return res

    # ==========================================
    # SYNTHESIS & LOGIC MAPPING
    # ==========================================

    def _synthesize_bias(self, kin: dict, dir_block: dict) -> str:
        """Determines directional bias mathematically."""
        score = 0
        
        if kin["velocity_z"] > self.Z_MODERATE: score += 1
        elif kin["velocity_z"] < -self.Z_MODERATE: score -= 1
        
        if dir_block["di_spread_z"] > self.Z_MODERATE: score += 1
        elif dir_block["di_spread_z"] < -self.Z_MODERATE: score -= 1
        
        if score > 0: return "bullish"
        if score < 0: return "bearish"
        return "neutral"

    def _synthesize_momentum_state(self, kin: dict, ext: dict, bias: str) -> str:
        """Synthesizes the lifecycle phase of the momentum wave."""
        if bias == "neutral":
            return "flat"
            
        # Is the momentum accelerating in the direction of the bias?
        is_accelerating = (bias == "bullish" and kin["acceleration_z"] > self.Z_MODERATE) or \
                          (bias == "bearish" and kin["acceleration_z"] < -self.Z_MODERATE)
                          
        if ext["is_overextended"] and kin["is_diverging"]:
            return "exhaustion"
            
        if kin["is_diverging"]:
            return "diverging"
            
        if is_accelerating:
            return "accelerating"
            
        return "decelerating"

    def _compute_statistical_confidence(self, kin: dict, dir_block: dict, ext: dict) -> str:
        """Confidence is highest when momentum velocity and directional dominance align strongly."""
        score = 3.0
        
        # High velocity + High ADX expansion = High Confidence
        if abs(kin["velocity_z"]) > self.Z_SIGNIFICANT: score += 1.0
        if dir_block["adx_vel_z"] > self.Z_MODERATE: score += 1.0
        
        # Contradictions lower confidence
        if ext["is_overextended"] and kin["acceleration_z"] > 0 and kin["velocity_z"] < 0:
            score -= 1.0
            
        if score >= 4.5: return "very_high"
        if score >= 3.5: return "high"
        if score >= 2.5: return "medium"
        if score >= 1.5: return "low"
        return "very_low"

    def _calculate_z_score(self, series: np.ndarray) -> float:
        """Calculates standard deviation mapping continuously over the provided window."""
        if len(series) == 0:
            return 0.0
        current_val = series[-1]
        hist_mean = np.nanmean(series)
        hist_std = np.nanstd(series) + self.EPSILON
        return float((current_val - hist_mean) / hist_std)

    def _fallback_contract(self) -> Dict[str, Any]:
        return {
            "analyzer_name": "momentum_analyzer",
            "momentum_state": "unknown",
            "directional_bias": "neutral",
            "kinematics": {"velocity_z": 0.0, "acceleration_z": 0.0, "is_diverging": False},
            "directional_strength": {"adx_velocity_z": 0.0, "di_spread_z": 0.0, "dominance": "neutral"},
            "extremity": {"oscillator_z": 0.0, "is_overextended": False},
            "confidence": "very_low"
        }
