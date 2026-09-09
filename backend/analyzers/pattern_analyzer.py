import numpy as np
import pandas as pd
from typing import Dict, Any

class PatternAnalyzer:
    """
    Green Bull Rider V6 - L1 Pattern Analyzer (Setup Finder)
    
    Translates geometric chart shapes into institutional kinetic energy setups.
    Evaluates accumulation/distribution footprints and compression constraints 
    using continuous geometric metrics (Apex Proximity, Breakout Pressure).
    """

    def __init__(self):
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Expects Pipeline to provide explicit geometric feature grouping:
        - 'compression_metrics': apex_distance, compression_pct, breakout_pressure, hs_neckline_slope
        - 'kinetic_patterns': triangle_detected, wedge_detected, triangle_type, wedge_type, channel_type
        - 'accumulation_patterns': double_bottom_detected, triple_bottom_detected, ihs_detected, cup_detected, handle_detected, rounding_bottom_detected
        - 'distribution_patterns': double_top_detected, triple_top_detected, hs_detected
        - 'geometry_boundaries': triangle_upper, triangle_lower, neckline
        - 'meta_metrics': pattern_confidence, pattern_family, hs_breakout_confirmed
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. EVALUATE CONTINUOUS GEOMETRIC PRESSURE (Vectorized)
        pressure_state = self._track_geometric_pressure(feature_blocks.get("compression_metrics"))
        
        # 2. EVALUATE STRUCTURAL REGIMES (Accumulation vs Distribution)
        accumulation_state = self._track_accumulation_structures(
            feature_blocks.get("accumulation_patterns"),
            feature_blocks.get("compression_metrics")
        )
        distribution_state = self._track_distribution_structures(
            feature_blocks.get("distribution_patterns"),
            feature_blocks.get("compression_metrics")
        )
        kinetic_state = self._track_kinetic_coils(feature_blocks.get("kinetic_patterns"))
        
        meta = feature_blocks.get("meta_metrics")

        # 3. SYNTHESIS (Determining the actionable setup)
        dominant_family = self._synthesize_dominant_family(
            accumulation_state, distribution_state, kinetic_state, meta
        )
        structural_bias = self._synthesize_bias(dominant_family, kinetic_state, accumulation_state, distribution_state)
        breakout_status = self._evaluate_breakout_readiness(pressure_state, meta, dominant_family)
        
        confidence = self._compute_confidence(pressure_state, meta, dominant_family)

        # 4. EXACT LOCKED JSON CONTRACT
        return {
            "analyzer_name": "pattern_analyzer",
            "structural_bias": structural_bias,
            "dominant_family": dominant_family,
            "compression_metrics": {
                "breakout_pressure_z": pressure_state["pressure_z"],
                "compression_velocity_z": pressure_state["compression_vel_z"],
                "apex_proximity_z": pressure_state["apex_prox_z"]
            },
            "setup_readiness": {
                "status": breakout_status,
                "is_critical_squeeze": pressure_state["is_critical_squeeze"]
            },
            "structural_details": {
                "neckline_slope_vector": pressure_state["neckline_slope_z"],
                "is_accumulation_base": accumulation_state["is_active"],
                "is_distribution_top": distribution_state["is_active"]
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _track_geometric_pressure(self, metrics_df: pd.DataFrame) -> dict:
        """
        Tracks the continuous mathematical constraints of the pattern.
        Evaluates how tightly price is coiling and the pressure building up.
        """
        res = {
            "pressure_z": 0.0, "compression_vel_z": 0.0, "apex_prox_z": 0.0,
            "neckline_slope_z": 0.0, "is_critical_squeeze": False
        }
        if metrics_df is None or metrics_df.empty:
            return res

        if 'breakout_pressure' in metrics_df.columns:
            pressure = metrics_df['breakout_pressure'].ffill().bfill().to_numpy()
            res["pressure_z"] = self._calculate_z_score(pressure)

        if 'compression_pct' in metrics_df.columns:
            comp = metrics_df['compression_pct'].ffill().bfill().to_numpy()
            comp_vel = np.gradient(comp)
            res["compression_vel_z"] = self._calculate_z_score(comp_vel)

        if 'apex_distance' in metrics_df.columns:
            apex_dist = metrics_df['apex_distance'].ffill().bfill().to_numpy()
            # Invert distance so higher Z-score = closer to apex (higher urgency)
            proximity = 1.0 / (apex_dist + self.EPSILON)
            res["apex_prox_z"] = self._calculate_z_score(proximity)

        if 'hs_neckline_slope' in metrics_df.columns:
            slope = metrics_df['hs_neckline_slope'].ffill().bfill().to_numpy()
            res["neckline_slope_z"] = self._calculate_z_score(slope)

        # Critical Squeeze: High mathematical pressure + high compression velocity + nearing apex
        if res["pressure_z"] > self.Z_SIGNIFICANT and res["apex_prox_z"] > self.Z_MODERATE:
            res["is_critical_squeeze"] = True

        return res

    def _track_accumulation_structures(self, acc_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
        """Evaluates institutional accumulation footprints (Bottoms, Cups, IHS)."""
        res = {"is_active": False, "score": 0.0, "has_handle": False}
        if acc_df is None or acc_df.empty:
            return res

        # Extract latest categorical state
        latest = acc_df.iloc[-1]
        
        active_patterns = []
        if latest.get('double_bottom_detected', 0): active_patterns.append(1.0)
        if latest.get('triple_bottom_detected', 0): active_patterns.append(1.5)
        if latest.get('rounding_bottom_detected', 0): active_patterns.append(1.2)
        
        is_ihs = latest.get('ihs_detected', 0)
        if is_ihs:
            # Ascending neckline in IHS is structurally stronger
            slope = metrics_df['hs_neckline_slope'].iloc[-1] if (metrics_df is not None and 'hs_neckline_slope' in metrics_df.columns) else 0
            weight = 2.0 if slope > 0 else 1.5
            active_patterns.append(weight)
            
        if latest.get('cup_detected', 0):
            weight = 1.5
            if latest.get('handle_detected', 0):
                weight = 2.5 # Cup & Handle is a premier accumulation setup
                res["has_handle"] = True
            active_patterns.append(weight)

        if active_patterns:
            res["is_active"] = True
            res["score"] = sum(active_patterns)

        return res

    def _track_distribution_structures(self, dist_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
        """Evaluates institutional distribution footprints (Tops, HS)."""
        res = {"is_active": False, "score": 0.0}
        if dist_df is None or dist_df.empty:
            return res

        latest = dist_df.iloc[-1]
        active_patterns = []
        
        if latest.get('double_top_detected', 0): active_patterns.append(1.0)
        if latest.get('triple_top_detected', 0): active_patterns.append(1.5)
        
        is_hs = latest.get('hs_detected', 0)
        if is_hs:
            # Descending neckline in HS is structurally weaker (bearish)
            slope = metrics_df['hs_neckline_slope'].iloc[-1] if (metrics_df is not None and 'hs_neckline_slope' in metrics_df.columns) else 0
            weight = 2.0 if slope < 0 else 1.5
            active_patterns.append(weight)

        if active_patterns:
            res["is_active"] = True
            res["score"] = sum(active_patterns)

        return res

    def _track_kinetic_coils(self, kin_df: pd.DataFrame) -> dict:
        """Evaluates symmetrical/directional coils (Triangles, Wedges, Channels)."""
        res = {"is_active": False, "type": "none", "bias": "neutral"}
        if kin_df is None or kin_df.empty:
            return res

        latest = kin_df.iloc[-1]
        
        if latest.get('wedge_detected', 0):
            res["is_active"] = True
            res["type"] = str(latest.get('wedge_type', 'wedge'))
            # Rising wedge = bearish bias, Falling wedge = bullish bias
            if 'falling' in res["type"].lower(): res["bias"] = "bullish"
            elif 'rising' in res["type"].lower(): res["bias"] = "bearish"
            
        elif latest.get('triangle_detected', 0):
            res["is_active"] = True
            res["type"] = str(latest.get('triangle_type', 'triangle'))
            if 'ascending' in res["type"].lower(): res["bias"] = "bullish"
            elif 'descending' in res["type"].lower(): res["bias"] = "bearish"
            
        elif pd.notna(latest.get('channel_type')):
            res["is_active"] = True
            res["type"] = str(latest.get('channel_type'))
            if 'up' in res["type"].lower(): res["bias"] = "bullish"
            elif 'down' in res["type"].lower(): res["bias"] = "bearish"

        return res

    # ==========================================
    # SYNTHESIS & LOGIC MAPPING
    # ==========================================

    def _synthesize_dominant_family(self, acc: dict, dist: dict, kin: dict, meta: pd.DataFrame) -> str:
        """Determines the overarching structural regime."""
        # Fallback to pipeline's explicitly calculated family if provided and confident
        if meta is not None and 'pattern_family' in meta.columns:
            family = meta['pattern_family'].iloc[-1]
            if pd.notna(family) and str(family).strip() != "":
                return str(family).lower()

        # Dynamic deduction based on structural tracking scores
        if acc["is_active"] and acc["score"] > dist["score"]:
            return "accumulation_base"
        if dist["is_active"] and dist["score"] > acc["score"]:
            return "distribution_top"
        if kin["is_active"]:
            return "kinetic_compression"
            
        return "none"

    def _synthesize_bias(self, family: str, kin: dict, acc: dict, dist: dict) -> str:
        """Extracts structural directional bias from geometries."""
        if family == "accumulation_base" or acc["is_active"]:
            return "bullish"
        if family == "distribution_top" or dist["is_active"]:
            return "bearish"
        if family == "kinetic_compression":
            return kin["bias"] # Ascending triangle = bullish, etc.
            
        return "neutral"

    def _evaluate_breakout_readiness(self, press: dict, meta: pd.DataFrame, family: str) -> str:
        """
        Determines trade setup readiness using mathematical pressure and apex proximity.
        """
        if family == "none":
            return "none"

        # Explicit confirmation from pipeline logic
        if meta is not None and 'hs_breakout_confirmed' in meta.columns:
            if meta['hs_breakout_confirmed'].iloc[-1]:
                return "confirmed_breakout"

        # Continuous mathematical evaluation of breakout probability
        if press["is_critical_squeeze"]:
            return "imminent_breakout"
            
        if press["pressure_z"] > self.Z_MODERATE or press["compression_vel_z"] > self.Z_MODERATE:
            return "developing_setup"
            
        return "forming"

    def _compute_confidence(self, press: dict, meta: pd.DataFrame, family: str) -> str:
        """Blends pipeline base confidence with internal structural pressure Z-scores."""
        if family == "none":
            return "very_low"
            
        base_confidence = 0.5 # Default starting point
        if meta is not None and 'pattern_confidence' in meta.columns:
            # Assuming pattern_confidence is a normalized 0.0 to 1.0 score
            base_val = meta['pattern_confidence'].iloc[-1]
            if pd.notna(base_val):
                base_confidence = float(base_val)

        # Enhance confidence if there is extreme mathematical pressure corroborating the visual pattern
        if press["pressure_z"] > self.Z_SIGNIFICANT: base_confidence += 0.2
        if press["apex_prox_z"] > self.Z_SIGNIFICANT: base_confidence += 0.2
        
        # Penalty for low geometric pressure during a supposed squeeze
        if family == "kinetic_compression" and press["pressure_z"] < -self.Z_MODERATE:
            base_confidence -= 0.3

        if base_confidence >= 0.8: return "very_high"
        if base_confidence >= 0.6: return "high"
        if base_confidence >= 0.4: return "medium"
        if base_confidence >= 0.2: return "low"
        return "very_low"

    def _calculate_z_score(self, series: np.ndarray) -> float:
        if len(series) == 0:
            return 0.0
        current_val = series[-1]
        hist_mean = np.nanmean(series)
        hist_std = np.nanstd(series) + self.EPSILON
        return float((current_val - hist_mean) / hist_std)

    def _fallback_contract(self) -> Dict[str, Any]:
        return {
            "analyzer_name": "pattern_analyzer",
            "structural_bias": "neutral",
            "dominant_family": "none",
            "compression_metrics": {"breakout_pressure_z": 0.0, "compression_velocity_z": 0.0, "apex_proximity_z": 0.0},
            "setup_readiness": {"status": "none", "is_critical_squeeze": False},
            "structural_details": {"neckline_slope_vector": 0.0, "is_accumulation_base": False, "is_distribution_top": False},
            "confidence": "very_low"
        }
