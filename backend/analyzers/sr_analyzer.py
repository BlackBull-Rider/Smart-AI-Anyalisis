import numpy as np
import pandas as pd
from typing import Dict, Any

class SupportResistanceAnalyzer:
    """
    Green Bull Rider V6 - L1 S/R & Execution Analyzer
    
    Acts as the 'Execution Manager'. Evaluates trade geometry, dynamic RRR, 
    and validates logical SL/Target zones using continuous mathematical distance, 
    cluster density, and institutional zone alignment.
    Zero arbitrary percentage-based SLs.
    """

    def __init__(self):
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes proximity and risk geometry evaluation based on exactly the 
        provided 96 S/R whitelist keys grouped by the pipeline.
        - 'meta_metrics': risk_distance, reward_distance, distance_to_support, distance_to_resistance, level_confidence...
        - 'dynamic_zones': demand_zone, supply_zone, congestion_zone, golden_zone...
        - 'cluster_metrics': cluster_density, cluster_strength, zone_width...
        - 'levels': nearest_support, nearest_resistance, avp_poc, avp_vah, avp_val...
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. MATHEMATICAL PROXIMITY & RRR TRACKING
        execution_geometry = self._evaluate_execution_geometry(
            feature_blocks.get("meta_metrics")
        )
        
        # 2. ZONE DENSITY & CONFLUENCE
        confluence_state = self._evaluate_zone_confluence(
            feature_blocks.get("cluster_metrics"),
            feature_blocks.get("meta_metrics")
        )
        
        # 3. INSTITUTIONAL POSITIONING
        trade_location = self._evaluate_trade_location(
            feature_blocks.get("meta_metrics"),
            feature_blocks.get("dynamic_zones"),
            feature_blocks.get("levels")
        )

        # 4. SYNTHESIS
        rrr_profile = self._synthesize_rrr_profile(execution_geometry["rrr_ratio"])
        confidence = self._compute_execution_confidence(execution_geometry, confluence_state)

        # 5. EXACT LOCKED JSON CONTRACT
        return {
            "analyzer_name": "sr_analyzer",
            "execution_geometry": {
                "dynamic_rrr": execution_geometry["rrr_ratio"],
                "rrr_profile": rrr_profile,
                "risk_proximity_z": execution_geometry["risk_distance_z"],
                "reward_proximity_z": execution_geometry["reward_distance_z"]
            },
            "zone_confluence": {
                "cluster_strength_z": confluence_state["strength_z"],
                "cluster_density_z": confluence_state["density_z"],
                "is_heavy_congestion": confluence_state["is_heavy_congestion"]
            },
            "trade_location": {
                "position_context": trade_location["context"],
                "nearest_support_type": trade_location["support_type"],
                "nearest_resistance_type": trade_location["resistance_type"]
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _evaluate_execution_geometry(self, meta_df: pd.DataFrame) -> dict:
        """
        Computes continuous mathematical risk-to-reward metrics based on 
        the distance to pipeline-calculated logical boundaries.
        """
        res = {
            "rrr_ratio": 0.0, 
            "risk_distance_z": 0.0, 
            "reward_distance_z": 0.0
        }
        
        if meta_df is None or meta_df.empty:
            return res

        # Evaluate risk/reward explicitly using pipeline-calculated spatial distances
        if 'risk_distance' in meta_df.columns and 'reward_distance' in meta_df.columns:
            risk_dist = meta_df['risk_distance'].ffill().bfill().to_numpy()
            reward_dist = meta_df['reward_distance'].ffill().bfill().to_numpy()
            
            res["risk_distance_z"] = self._calculate_z_score(risk_dist)
            res["reward_distance_z"] = self._calculate_z_score(reward_dist)
            
            current_risk = risk_dist[-1]
            current_reward = reward_dist[-1]
            
            # Pure mathematical Ratio
            res["rrr_ratio"] = float(current_reward / (current_risk + self.EPSILON))
            
        # Fallback to direct distance metrics if explicit risk/reward isn't defined
        elif 'distance_to_support' in meta_df.columns and 'distance_to_resistance' in meta_df.columns:
            dist_sup = meta_df['distance_to_support'].ffill().bfill().to_numpy()
            dist_res = meta_df['distance_to_resistance'].ffill().bfill().to_numpy()
            
            res["risk_distance_z"] = self._calculate_z_score(dist_sup)
            res["reward_distance_z"] = self._calculate_z_score(dist_res)
            
            # Calculates generic ratio, L2 engine will decide orientation (Long/Short)
            current_sup = dist_sup[-1]
            current_res = dist_res[-1]
            res["rrr_ratio"] = float(current_res / (current_sup + self.EPSILON))

        return res

    def _evaluate_zone_confluence(self, cluster_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict:
        """
        Measures structural rigidity using Cluster Density and S/R Level Strengths.
        High density + High strength = Impenetrable Zone.
        """
        res = {"strength_z": 0.0, "density_z": 0.0, "is_heavy_congestion": False}
        
        if cluster_df is not None and not cluster_df.empty:
            if 'cluster_strength' in cluster_df.columns:
                strength = cluster_df['cluster_strength'].ffill().bfill().to_numpy()
                res["strength_z"] = self._calculate_z_score(strength)
                
            if 'cluster_density' in cluster_df.columns:
                density = cluster_df['cluster_density'].ffill().bfill().to_numpy()
                res["density_z"] = self._calculate_z_score(density)
                
        # If no explicit cluster blocks, try meta level strengths
        elif meta_df is not None and 'level_strength' in meta_df.columns:
            strength = meta_df['level_strength'].ffill().bfill().to_numpy()
            res["strength_z"] = self._calculate_z_score(strength)
            
        # Heavy Congestion logic (Z-Score convergence)
        if res["density_z"] > self.Z_SIGNIFICANT and res["strength_z"] > self.Z_SIGNIFICANT:
            res["is_heavy_congestion"] = True
            
        return res

    def _evaluate_trade_location(self, meta_df: pd.DataFrame, zone_df: pd.DataFrame, levels_df: pd.DataFrame) -> dict:
        """
        Classifies where the price resides functionally without making buy/sell decisions.
        Detects if price is at demand/supply, golden zones, or institutional AVP levels.
        """
        res = {
            "context": "mid_range",
            "support_type": "standard",
            "resistance_type": "standard"
        }
        
        # Check explicit dynamic zones
        if zone_df is not None and not zone_df.empty:
            latest = zone_df.iloc[-1]
            if latest.get('demand_zone', 0) > 0 or latest.get('golden_zone_lower', 0) > 0:
                res["context"] = "at_demand"
                res["support_type"] = "institutional_demand"
            elif latest.get('supply_zone', 0) > 0 or latest.get('golden_zone_upper', 0) > 0:
                res["context"] = "at_supply"
                res["resistance_type"] = "institutional_supply"
            elif latest.get('congestion_zone', 0) > 0:
                res["context"] = "in_congestion"

        # Refine S/R Type from levels
        if levels_df is not None and not levels_df.empty:
            latest_lvl = levels_df.iloc[-1]
            if latest_lvl.get('avp_poc', 0) > 0 or latest_lvl.get('swing_low_vwap', 0) > 0:
                res["support_type"] = "volume_backed_support"
            if latest_lvl.get('avp_vah', 0) > 0 or latest_lvl.get('smart_money_level', 0) > 0:
                res["resistance_type"] = "smart_money_resistance"

        # Check raw distance to boundaries if context is still mid-range
        if res["context"] == "mid_range" and meta_df is not None:
            dist_s = meta_df['distance_to_support'].iloc[-1] if 'distance_to_support' in meta_df.columns else 9999
            dist_r = meta_df['distance_to_resistance'].iloc[-1] if 'distance_to_resistance' in meta_df.columns else 9999
            
            # Simple proportional comparison
            total_range = dist_s + dist_r + self.EPSILON
            if dist_s / total_range < 0.2:
                res["context"] = "near_support"
            elif dist_r / total_range < 0.2:
                res["context"] = "near_resistance"

        return res

    # ==========================================
    # SYNTHESIS & LOGIC MAPPING
    # ==========================================

    def _synthesize_rrr_profile(self, rrr: float) -> str:
        """
        Maps continuous Mathematical RRR to categorical profiles for L2 Engine logic.
        (Note: 1.5, 2.0, 3.0 are mathematically universal execution standards, not retail hacks).
        """
        if rrr >= 3.0: return "exceptional"
        if rrr >= 2.0: return "optimal"
        if rrr >= 1.5: return "acceptable"
        if rrr >= 1.0: return "marginal"
        return "skewed_negative"

    def _compute_execution_confidence(self, geom: dict, conf: dict) -> str:
        """
        Execution confidence is high when RRR is strong and the underlying S/R boundary
        has high mathematical cluster density (hard for price to break).
        """
        score = 3.0
        
        # Geometric Edge
        if geom["rrr_ratio"] >= 2.0: score += 1.0
        elif geom["rrr_ratio"] < 1.0: score -= 1.0
        
        # Structural Rigidity of the Stop-Loss boundary
        if conf["strength_z"] > self.Z_SIGNIFICANT: score += 1.0
        if conf["density_z"] > self.Z_SIGNIFICANT: score += 1.0
        
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
            "analyzer_name": "sr_analyzer",
            "execution_geometry": {"dynamic_rrr": 0.0, "rrr_profile": "unknown", "risk_proximity_z": 0.0, "reward_proximity_z": 0.0},
            "zone_confluence": {"cluster_strength_z": 0.0, "cluster_density_z": 0.0, "is_heavy_congestion": False},
            "trade_location": {"position_context": "unknown", "nearest_support_type": "unknown", "nearest_resistance_type": "unknown"},
            "confidence": "very_low"
        }
