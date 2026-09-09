import numpy as np
import pandas as pd
from typing import Dict, Any

class TrendAnalyzer:
    """
    Green Bull Rider V6 - L1 Trend Analyzer (Final Institutional Architecture)
    
    Operates strictly on pipeline-routed feature blocks. Evaluates temporal 
    persistence, structural coherence, and kinematic vectors using continuous 
    mathematical functions and statistical Z-Scores. 
    Zero retail crossover logic. Zero arbitrary classification thresholds.
    """

    def __init__(self):
        # Statistical Standard Deviation Boundaries for mathematical classification
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes structural trend analysis dynamically on the provided window.
        Expects Pipeline to provide explicit feature grouping:
        - 'adaptive': kama, alma, t3, vidya, mcginley, hma, dema, tema, trima, zlema
        - 'baseline': sma, ema, wma, vwma, smma, lsma
        - 'regression_fit': r_squared, adjusted_r_squared, regression_r2, linear_regression_r2
        - 'regression_slope': regression_slope, linear_regression_slope, rolling_slope, trend_angle, slope_percentage
        - 'regime': supertrend, supertrend_trend
        - 'momentum': adx_14, plus_di, minus_di, dx
        - 'price': close, regression_value, linear_regression_value, regression_intercept
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. INDIVIDUAL FEATURE BEHAVIOR TRACKING (Vectorized, Multi-Period)
        adaptive_behavior = self._track_adaptive_coherence(feature_blocks.get("adaptive"))
        baseline_behavior = self._track_spatial_distribution(feature_blocks.get("baseline"))
        regression_behavior = self._track_regression_persistence(
            feature_blocks.get("regression_fit"), 
            feature_blocks.get("regression_slope")
        )
        regime_behavior = self._track_regime_stability(feature_blocks.get("regime"))
        momentum_behavior = self._track_kinematic_momentum(feature_blocks.get("momentum"))

        # 2. STRUCTURAL COHERENCE & MULTI-HORIZON EVIDENCE
        direction = self._synthesize_direction(adaptive_behavior, regression_behavior, regime_behavior)
        regime = self._synthesize_regime(baseline_behavior, regression_behavior, direction)
        
        # 3. MATHEMATICAL STRENGTH & MOMENTUM CONFIRMATION
        strength = self._calculate_vector_strength(regression_behavior, adaptive_behavior, baseline_behavior)
        momentum_conf = self._confirm_kinematic_momentum(momentum_behavior, direction)
        
        # 4. CONTINUOUS TREND CHANGE CALCULATION
        tc_status, tc_dir, tc_level = self._calculate_structural_deterioration(
            regression_behavior, baseline_behavior, adaptive_behavior, direction
        )

        alignment = self._evaluate_cross_block_alignment(
            adaptive_behavior, regression_behavior, regime_behavior
        )
        
        confidence = self._compute_statistical_confidence(
            regression_behavior, adaptive_behavior, baseline_behavior
        )

        # 5. EXACT LOCKED JSON CONTRACT
        return {
            "direction": direction,
            "regime": regime,
            "strength": strength,
            "alignment": alignment,
            "momentum_confirmation": momentum_conf,
            "trend_change": {
                "status": tc_status,
                "direction": tc_dir,
                "level": tc_level
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _track_adaptive_coherence(self, df: pd.DataFrame) -> dict:
        """Measures cross-sectional agreement (coherence) among adaptive features via Z-Scores."""
        if df is None or df.empty or len(df.columns) < 2:
            return {"coherence_z": 0.0, "vector_velocity_z": 0.0, "acceleration_z": 0.0}

        matrix = df.ffill().bfill().to_numpy()
        
        # 1st and 2nd Derivatives (Velocity and Acceleration of the adaptive block)
        velocity = np.gradient(matrix, axis=0)
        acceleration = np.gradient(velocity, axis=0)
        
        cross_mean_vel = np.nanmean(velocity, axis=1)
        cross_mean_acc = np.nanmean(acceleration, axis=1)
        cross_variance = np.nanvar(velocity, axis=1)
        
        # Z-Scores relative to the current historical window
        vel_z = self._calculate_z_score(cross_mean_vel)
        acc_z = self._calculate_z_score(cross_mean_acc)
        
        # Coherence (Inverted Variance Z-Score): Negative variance Z-score means tighter than average
        var_z = self._calculate_z_score(cross_variance)
        coherence_z = -var_z 

        return {
            "coherence_z": coherence_z,
            "vector_velocity_z": vel_z,
            "acceleration_z": acc_z
        }

    def _track_spatial_distribution(self, df: pd.DataFrame) -> dict:
        """Tracks the spatial expansion/compression of baseline structural features."""
        if df is None or df.empty or len(df.columns) < 2:
            return {"expansion_velocity_z": 0.0, "current_variance": 0.0, "peak_variance": 0.0}

        matrix = df.ffill().bfill().to_numpy()
        spatial_variance_series = np.nanvar(matrix, axis=1)
        
        expansion_velocity = np.gradient(spatial_variance_series)
        expansion_velocity_z = self._calculate_z_score(expansion_velocity)
        
        return {
            "expansion_velocity_z": expansion_velocity_z,
            "current_variance": spatial_variance_series[-1] if len(spatial_variance_series) > 0 else 0.0,
            "peak_variance": np.nanmax(spatial_variance_series) if len(spatial_variance_series) > 0 else 0.0
        }

    def _track_regression_persistence(self, fit_df: pd.DataFrame, slope_df: pd.DataFrame) -> dict:
        """Evaluates the historical persistence of structural fit (R2) and trajectory of slopes."""
        res = {
            "fit_z": 0.0, "fit_trajectory_z": 0.0, "slope_z": 0.0, 
            "current_fit": 0.0, "peak_fit": 0.0
        }
        
        if fit_df is not None and not fit_df.empty:
            fit_matrix = fit_df.ffill().bfill().to_numpy()
            mean_fit_series = np.nanmean(fit_matrix, axis=1)
            fit_trajectory = np.gradient(mean_fit_series)
            
            res["fit_z"] = self._calculate_z_score(mean_fit_series)
            res["fit_trajectory_z"] = self._calculate_z_score(fit_trajectory)
            res["current_fit"] = mean_fit_series[-1] if len(mean_fit_series) > 0 else 0.0
            res["peak_fit"] = np.nanmax(mean_fit_series) if len(mean_fit_series) > 0 else 0.0
            
        if slope_df is not None and not slope_df.empty:
            slope_matrix = slope_df.ffill().bfill().to_numpy()
            mean_slope_series = np.nanmean(slope_matrix, axis=1)
            res["slope_z"] = self._calculate_z_score(mean_slope_series)
            
        return res

    def _track_regime_stability(self, df: pd.DataFrame) -> dict:
        """Evaluates regime continuous kinematics (e.g., gradient of supertrend)."""
        if df is None or df.empty:
            return {"regime_velocity_z": 0.0}
            
        matrix = df.ffill().bfill().to_numpy()
        mean_regime = np.nanmean(matrix, axis=1)
        
        regime_velocity = np.gradient(mean_regime)
        regime_velocity_z = self._calculate_z_score(regime_velocity)
        
        return {"regime_velocity_z": regime_velocity_z}

    def _track_kinematic_momentum(self, df: pd.DataFrame) -> dict:
        """Tracks continuous acceleration of momentum indicators, ignoring absolute thresholds."""
        if df is None or df.empty:
            return {"momentum_velocity_z": 0.0, "momentum_acceleration_z": 0.0}
            
        matrix = df.ffill().bfill().to_numpy()
        
        velocity = np.gradient(matrix, axis=0)
        acceleration = np.gradient(velocity, axis=0)
        
        mean_velocity = np.nanmean(velocity, axis=1)
        mean_acceleration = np.nanmean(acceleration, axis=1)
        
        return {
            "momentum_velocity_z": self._calculate_z_score(mean_velocity),
            "momentum_acceleration_z": self._calculate_z_score(mean_acceleration)
        }

    # ==========================================
    # SYNTHESIS & EXACT CONTRACT MAPPING
    # ==========================================

    def _synthesize_direction(self, adapt: dict, reg: dict, regime_b: dict) -> str:
        """Synthesizes direction using statistical significance (Z-Scores)."""
        st_z = adapt["vector_velocity_z"]
        mt_z = reg["slope_z"]
        lt_z = regime_b["regime_velocity_z"]
        
        vectors = [st_z, mt_z, lt_z]
        bullish_vectors = sum(1 for v in vectors if v > self.Z_MODERATE)
        bearish_vectors = sum(1 for v in vectors if v < -self.Z_MODERATE)
        
        if bullish_vectors >= 2 and bearish_vectors == 0:
            return "bullish"
        if bearish_vectors >= 2 and bullish_vectors == 0:
            return "bearish"
            
        if abs(mt_z) < self.Z_MODERATE and reg["fit_trajectory_z"] < -self.Z_MODERATE:
            return "sideways"
            
        if bullish_vectors > 0 and bearish_vectors > 0:
            return "mixed"
            
        return "transitional"

    def _synthesize_regime(self, baseline: dict, reg: dict, direction: str) -> str:
        """Synthesizes structural regime via continuous variance and fit tracking."""
        if direction in ["bullish", "bearish"]:
            if reg["fit_z"] > self.Z_MODERATE and baseline["expansion_velocity_z"] > self.Z_MODERATE:
                return "strong_trend"
            if reg["fit_z"] > -self.Z_MODERATE:
                return "weak_trend"
                
        if baseline["expansion_velocity_z"] < -self.Z_SIGNIFICANT and reg["fit_z"] < -self.Z_MODERATE:
            return "range"
            
        if reg["fit_trajectory_z"] < -self.Z_SIGNIFICANT:
            return "transition"
            
        return "unknown"

    def _calculate_vector_strength(self, reg: dict, adapt: dict, baseline: dict) -> str:
        """Calculates strength strictly as the aggregate magnitude of converging statistical evidence."""
        aggregate_z = abs(adapt["vector_velocity_z"]) + reg["fit_z"] + baseline["expansion_velocity_z"] + adapt["coherence_z"]
        
        if aggregate_z > (self.Z_EXTREME * 2): return "very_strong"
        if aggregate_z > (self.Z_SIGNIFICANT * 2): return "strong"
        if aggregate_z > self.Z_MODERATE: return "moderate"
        if aggregate_z > -self.Z_SIGNIFICANT: return "weak"
        return "very_weak"

    def _confirm_kinematic_momentum(self, mom: dict, direction: str) -> str:
        """Evaluates internal momentum based on kinematic acceleration of the momentum block."""
        if direction in ["sideways", "unknown", "mixed", "transitional"]:
            return "unconfirmed"
            
        vel_z = mom["momentum_velocity_z"]
        acc_z = mom["momentum_acceleration_z"]
        
        is_accelerating = (direction == "bullish" and vel_z > self.Z_MODERATE and acc_z > 0) or \
                          (direction == "bearish" and vel_z < -self.Z_MODERATE and acc_z < 0)
                          
        is_decelerating = (direction == "bullish" and vel_z > self.Z_MODERATE and acc_z < -self.Z_MODERATE) or \
                          (direction == "bearish" and vel_z < -self.Z_MODERATE and acc_z > self.Z_MODERATE)

        if is_accelerating:
            return "confirmed"
        if is_decelerating:
            return "opposed"
            
        return "partially_confirmed"

    def _calculate_structural_deterioration(self, reg: dict, baseline: dict, adapt: dict, direction: str) -> tuple:
        """
        Computes trend change mathematically based on continuous pure ratios:
        Decay = 1.0 - (Current State / Peak State)
        """
        if direction not in ["bullish", "bearish"]:
            return "none", "none", 0.0

        tc_dir = "bearish" if direction == "bullish" else "bullish"
        
        # 1. Structural Fit Decay (R2 dropping from local peak)
        fit_decay = 0.0
        if reg["peak_fit"] > self.EPSILON:
            fit_decay = max(0.0, 1.0 - (reg["current_fit"] / reg["peak_fit"]))
            
        # 2. Spatial Compression (Variance dropping from local peak)
        variance_decay = 0.0
        if baseline["peak_variance"] > self.EPSILON:
            variance_decay = max(0.0, 1.0 - (baseline["current_variance"] / baseline["peak_variance"]))
            
        # Aggregate pure mathematical mean of decays
        base_level = np.nanmean([fit_decay, variance_decay])
        
        # Kinematic Acceleration Penalty (Does acceleration oppose the trend?)
        if (direction == "bullish" and adapt["acceleration_z"] < -self.Z_MODERATE) or \
           (direction == "bearish" and adapt["acceleration_z"] > self.Z_MODERATE):
            # Scale the level by the magnitude of opposing acceleration
            base_level += min(0.5, abs(adapt["acceleration_z"]) / 10.0)
            
        tc_level = min(1.0, max(0.0, base_level))
        
        # Statistical boundary mapping for categorical output
        if tc_level >= 0.8: status = "confirmed"
        elif tc_level >= 0.5: status = "developing"
        elif tc_level >= 0.2: status = "candidate"
        else:
            status = "none"
            tc_level = 0.0
            tc_dir = "none"
            
        return status, tc_dir, round(float(tc_level), 2)

    def _evaluate_cross_block_alignment(self, adapt: dict, reg: dict, regime_b: dict) -> dict:
        """Determines alignment dynamically across the feature blocks using Z-scores."""
        def map_z_to_sign(z_val):
            if z_val > self.Z_MODERATE: return "bullish"
            if z_val < -self.Z_MODERATE: return "bearish"
            return "neutral"
            
        return {
            "short_term": map_z_to_sign(adapt["vector_velocity_z"]),
            "medium_term": map_z_to_sign(reg["slope_z"]),
            "long_term": map_z_to_sign(regime_b["regime_velocity_z"])
        }

    def _compute_statistical_confidence(self, reg: dict, adapt: dict, baseline: dict) -> str:
        """Confidence derived from Structural Fit Z-Scores and Coherence Z-Scores."""
        aggregate_conf_z = reg["fit_z"] + adapt["coherence_z"]
        
        if aggregate_conf_z > self.Z_EXTREME: return "very_high"
        if aggregate_conf_z > self.Z_SIGNIFICANT: return "high"
        if aggregate_conf_z > -self.Z_MODERATE: return "medium"
        if aggregate_conf_z > -self.Z_SIGNIFICANT: return "low"
        return "very_low"

    def _calculate_z_score(self, series: np.ndarray) -> float:
        """Helper to compute the Z-Score of the latest observation relative to the provided window."""
        if len(series) == 0:
            return 0.0
        current_val = series[-1]
        hist_mean = np.nanmean(series)
        hist_std = np.nanstd(series) + self.EPSILON
        return float((current_val - hist_mean) / hist_std)

    def _fallback_contract(self) -> Dict[str, Any]:
        return {
            "direction": "unknown", "regime": "unknown", "strength": "very_weak",
            "alignment": {"short_term": "unknown", "medium_term": "unknown", "long_term": "unknown"},
            "momentum_confirmation": "unknown",
            "trend_change": {"status": "none", "direction": "none", "level": 0.0},
            "confidence": "very_low"
        }
