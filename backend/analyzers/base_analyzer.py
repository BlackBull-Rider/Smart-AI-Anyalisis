import numpy as np
import pandas as pd
from typing import Dict, Union, List

class InstitutionalQuantBase:
    """
    Final Base Analyzer for 5-20 Day Swing Trading.
    100% Math-backed. Bridges raw Data to Frozen JSON Vocabulary.
    """
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty or len(df) < 5:
            raise ValueError("Insufficient data. Minimum 5 rows required.")
        
        self.df = df.copy()
        self.current_row = self.df.iloc[-1]
        self.total_rows = len(self.df)

    # ==========================================
    # CORE MATH & KINEMATICS (PURE NUMPY)
    # ==========================================
    def get_series(self, column: str, lookback: int = 0) -> np.ndarray:
        if column not in self.df.columns:
            return np.array([])
        
        data = self.df[column].to_numpy(dtype=float)
        if lookback > 0:
            data = data[-lookback:]
            
        return data[~np.isnan(data)]

    def calc_robust_zscore(self, column: str, lookback: int = 50) -> float:
        """Outlier/Gap-Proof Z-score using Median Absolute Deviation (MAD)."""
        data = self.get_series(column, lookback)
        if len(data) < 3: return 0.0
            
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0: return 0.0
            
        current_val = data[-1]
        return float((current_val - median) / (1.4826 * mad))

    def calc_kinematics(self, column: str, lookback: int = 3) -> Dict[str, float]:
        """Velocity (Speed) and Acceleration (Momentum change)."""
        data = self.get_series(column, max(10, lookback + 3))
        if len(data) < lookback + 2:
            return {"velocity": 0.0, "acceleration": 0.0}
            
        current_val = data[-1]
        past_val = data[-(lookback + 1)]
        past_val_2 = data[-(lookback + 2)]
        
        velocity = (current_val - past_val) / lookback
        past_velocity = (past_val - past_val_2) / lookback
        acceleration = velocity - past_velocity
        
        return {
            "velocity": float(velocity),
            "acceleration": float(acceleration)
        }

    def calc_distance_pct(self, val1: float, val2: float) -> float:
        """শতকরা দূরত্ব (যেমন Price to VWAP বা Support)।"""
        if val2 == 0 or np.isnan(val2) or np.isnan(val1): return 0.0
        return float(((val1 - val2) / val2) * 100)

    # ==========================================
    # FROZEN VOCABULARY MAPPERS (Strict Contract)
    # ==========================================
    def map_strength(self, robust_zscore: float) -> str:
        abs_z = abs(robust_zscore)
        if abs_z >= 2.5: return "very_strong"
        if abs_z >= 1.5: return "strong"
        if abs_z >= 0.75: return "moderate"
        if abs_z >= 0.25: return "weak"
        return "very_weak"

    def map_confidence(self, confidence_score: float) -> str:
        if confidence_score >= 85: return "very_high"
        if confidence_score >= 70: return "high"
        if confidence_score >= 50: return "medium"
        if confidence_score >= 30: return "low"
        return "very_low"

    def map_direction(self, velocity: float, threshold: float = 0.01) -> str:
        if velocity > threshold: return "bullish"
        if velocity < -threshold: return "bearish"
        return "neutral"

    def map_kinematic_regime(self, velocity: float, acceleration: float) -> str:
        if velocity > 0 and acceleration > 0: return "accelerating"
        if velocity < 0 and acceleration < 0: return "accelerating"
        if velocity > 0 and acceleration < 0: return "decelerating"
        if velocity < 0 and acceleration > 0: return "decelerating"
        if velocity == 0 and acceleration == 0: return "stable"
        return "mixed"

    def map_relative_position(self, current_val: float, high: float, low: float) -> str:
        if np.isnan(current_val) or np.isnan(high) or np.isnan(low) or high == low:
            return "unknown"
        if current_val > high: return "above"
        if current_val < low: return "below"
        
        pos_pct = (current_val - low) / (high - low)
        if pos_pct >= 0.66: return "inside_upper"
        if pos_pct <= 0.33: return "inside_lower"
        return "inside_middle"

    def map_price_zone(self, current_val: float, high: float, low: float) -> str:
        """SMC এবং Market Context-এর জন্য Premium/Discount/Equilibrium ম্যাপিং।"""
        if np.isnan(current_val) or np.isnan(high) or np.isnan(low) or high == low:
            return "unknown"
        
        pos_pct = (current_val - low) / (high - low)
        if pos_pct >= 0.60: return "premium"
        if pos_pct <= 0.40: return "discount"
        return "equilibrium"

    def map_volatility_state(self, zscore: float) -> str:
        """Z-Score থেকে Volatility Compression/Expansion ম্যাপিং।"""
        if zscore <= -1.0: return "compression"
        if zscore >= 1.0: return "expansion"
        return "stable"
