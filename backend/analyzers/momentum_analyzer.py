import numpy as np
import pandas as pd
from typing import Dict, Any

class MomentumAnalyzer:
    """
    Green Bull Rider V6 - L1 Momentum Analyzer (Institutional Grade)
    """

    def __init__(self):
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        if not feature_blocks:
            return self._fallback_contract()

        kinematics = self._track_kinematic_momentum(feature_blocks.get("kinematic_block"))
        directional = self._track_directional_dominance(feature_blocks.get("directional_block"))
        extremity = self._track_statistical_extremity(feature_blocks.get("oscillator_block"))

        directional_bias = self._synthesize_bias(kinematics, directional)
        momentum_state = self._synthesize_momentum_state(kinematics, extremity, directional_bias)
        confidence = self._compute_statistical_confidence(kinematics, directional, extremity)

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

    def _track_kinematic_momentum(self, df: pd.DataFrame) -> dict:
        res = {"velocity_z": 0.0, "acceleration_z": 0.0, "is_diverging": False}
        if df is None or df.empty:
            return res

        kinetic_cols = [c for c in ['macd_hist', 'ppo_hist', 'roc_12', 'mom_10', 'rsi_slope', 'trix_18'] if c in df.columns]
        if kinetic_cols:
            matrix = df[kinetic_cols].apply(pd.to_numeric, errors='coerce').ffill().bfill().to_numpy(dtype=float)
            if not np.all(np.isnan(matrix)):
                mean_momentum = np.nanmean(matrix, axis=1)
                velocity = np.gradient(mean_momentum)
                acceleration = np.gradient(velocity)
                
                res["velocity_z"] = self._calculate_z_score(velocity)
                res["acceleration_z"] = self._calculate_z_score(acceleration)
                
                if abs(res["velocity_z"]) > self.Z_SIGNIFICANT and np.sign(res["velocity_z"]) != np.sign(res["acceleration_z"]):
                    if abs(res["acceleration_z"]) > self.Z_MODERATE:
                        res["is_diverging"] = True

        return res

    def _track_directional_dominance(self, df: pd.DataFrame) -> dict:
        res = {"adx_vel_z": 0.0, "di_spread_z": 0.0, "dominance_status": "neutral"}
        if df is None or df.empty:
            return res

        if 'adx_14' in df.columns:
            adx = pd.to_numeric(df['adx_14'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
            if not np.all(np.isnan(adx)):
                adx_velocity = np.gradient(adx)
                res["adx_vel_z"] = self._calculate_z_score(adx_velocity)

        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            pdi = pd.to_numeric(df['plus_di'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
            mdi = pd.to_numeric(df['minus_di'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
            
            if not np.all(np.isnan(pdi)) and not np.all(np.isnan(mdi)):
                di_spread = pdi - mdi
                res["di_spread_z"] = self._calculate_z_score(di_spread)
                
                if res["di_spread_z"] > self.Z_SIGNIFICANT:
                    res["dominance_status"] = "bull_dominant"
                elif res["di_spread_z"] < -self.Z_SIGNIFICANT:
                    res["dominance_status"] = "bear_dominant"

        return res

    def _track_statistical_extremity(self, df: pd.DataFrame) -> dict:
        res = {"osc_aggregate_z": 0.0, "is_overextended": False}
        if df is None or df.empty:
            return res

        matrix = df.apply(pd.to_numeric, errors='coerce').ffill().bfill().to_numpy(dtype=float)
        if not np.all(np.isnan(matrix)):
            mean_osc = np.nanmean(matrix, axis=1)
            res["osc_aggregate_z"] = self._calculate_z_score(mean_osc)
            
            if abs(res["osc_aggregate_z"]) > self.Z_EXTREME:
                res["is_overextended"] = True
            
        return res

    def _synthesize_bias(self, kin: dict, dir_block: dict) -> str:
        score = 0
        if kin["velocity_z"] > self.Z_MODERATE: score += 1
        elif kin["velocity_z"] < -self.Z_MODERATE: score -= 1
        
        if dir_block["di_spread_z"] > self.Z_MODERATE: score += 1
        elif dir_block["di_spread_z"] < -self.Z_MODERATE: score -= 1
        
        if score > 0: return "bullish"
        if score < 0: return "bearish"
        return "neutral"

    def _synthesize_momentum_state(self, kin: dict, ext: dict, bias: str) -> str:
        if bias == "neutral": return "flat"
            
        is_accelerating = (bias == "bullish" and kin["acceleration_z"] > self.Z_MODERATE) or \
                          (bias == "bearish" and kin["acceleration_z"] < -self.Z_MODERATE)
                          
        if ext["is_overextended"] and kin["is_diverging"]: return "exhaustion"
        if kin["is_diverging"]: return "diverging"
        if is_accelerating: return "accelerating"
        return "decelerating"

    def _compute_statistical_confidence(self, kin: dict, dir_block: dict, ext: dict) -> str:
        score = 3.0
        if abs(kin["velocity_z"]) > self.Z_SIGNIFICANT: score += 1.0
        if dir_block["adx_vel_z"] > self.Z_MODERATE: score += 1.0
        if ext["is_overextended"] and kin["acceleration_z"] > 0 and kin["velocity_z"] < 0: score -= 1.0
            
        if score >= 4.5: return "very_high"
        if score >= 3.5: return "high"
        if score >= 2.5: return "medium"
        if score >= 1.5: return "low"
        return "very_low"

    def _calculate_z_score(self, series: np.ndarray) -> float:
        series = series[~np.isnan(series)]
        if len(series) == 0: return 0.0
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
