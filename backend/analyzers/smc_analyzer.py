import numpy as np
import pandas as pd
from typing import Dict, Any

class SmartMoneyAnalyzer:
    """
    Green Bull Rider V6 - L1 Smart Money (SMC) Analyzer
    
    Evaluates institutional order flow, market structure shifts (MSS), 
    liquidity sweeps, and algorithmic pricing zones (Premium/Discount).
    Strictly utilizes pipeline-provided SMC feature blocks and 
    continuous statistical Z-scores for footprint validation.
    """

    def __init__(self):
        # Statistical Standard Deviation Boundaries for continuous mapping
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes SMC analysis dynamically. Expects EXACT pipeline groupings:
        - 'structure_block': sm_bos, sm_choch, sm_mss, bos_up, bos_down, choch_up, choch_down, smc_trend, mss_bullish, mss_bearish
        - 'liquidity_block': liquidity_sweep, buy_side_liquidity, sell_side_liquidity, equal_highs, equal_lows, turtle_soup, judas_swing, stop_hunt_zone
        - 'zone_block': fresh_ob, mitigated_ob, breaker_block, flip_zone, bullish_fvg, bearish_fvg, bpr, active_fvg
        - 'pricing_block': premium_zone, discount_zone, equilibrium, premium_level, discount_level
        - 'scoring_block': sm_bos_score, sm_choch_score, sm_liquidity_score, sm_fvg_score, sm_trend_score, sm_institutional_score, sm_smart_money_score
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. MATHEMATICAL MICRO-TRACKERS (Vectorized over historical window)
        structure_state = self._track_market_structure(
            feature_blocks.get("structure_block"), 
            feature_blocks.get("scoring_block")
        )
        liquidity_state = self._track_liquidity_dynamics(
            feature_blocks.get("liquidity_block"), 
            feature_blocks.get("scoring_block")
        )
        institutional_zones = self._track_institutional_zones(
            feature_blocks.get("zone_block"), 
            feature_blocks.get("scoring_block")
        )
        pricing_context = self._track_pricing_context(feature_blocks.get("pricing_block"))
        footprint_scores = self._track_institutional_footprint(feature_blocks.get("scoring_block"))

        # 2. SYNTHESIS
        institutional_bias = self._synthesize_bias(structure_state, liquidity_state, footprint_scores)
        smc_phase = self._synthesize_smc_phase(structure_state, liquidity_state, institutional_zones)
        confidence = self._compute_statistical_confidence(footprint_scores, structure_state, liquidity_state)

        # 3. EXACT LOCKED JSON CONTRACT
        return {
            "analyzer_name": "smart_money_analyzer",
            "institutional_bias": institutional_bias,
            "smc_phase": smc_phase,
            "market_structure": {
                "current_state": structure_state["status"],
                "structure_strength_z": structure_state["strength_z"]
            },
            "liquidity_profile": {
                "sweep_status": liquidity_state["sweep_status"],
                "liquidity_density_z": liquidity_state["density_z"]
            },
            "institutional_zones": {
                "pricing_context": pricing_context["context"],
                "active_ob_present": institutional_zones["has_active_ob"],
                "fvg_imbalance_z": institutional_zones["fvg_strength_z"]
            },
            "smart_money_footprint": {
                "institutional_activity_z": footprint_scores["inst_score_z"],
                "composite_score_z": footprint_scores["composite_z"]
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _track_market_structure(self, struct_df: pd.DataFrame, score_df: pd.DataFrame) -> dict:
        """Evaluates BOS/CHoCH continuity and calculates structural rigidity Z-Score."""
        res = {"status": "ranging", "strength_z": 0.0, "vector": 0}
        
        if struct_df is not None and not struct_df.empty:
            latest = struct_df.iloc[-1]
            
            # Determine explicit state
            if latest.get('mss_bullish', 0) > 0 or latest.get('choch_up', 0) > 0:
                res["status"] = "bullish_choch"
                res["vector"] = 1
            elif latest.get('mss_bearish', 0) > 0 or latest.get('choch_down', 0) > 0:
                res["status"] = "bearish_choch"
                res["vector"] = -1
            elif latest.get('bos_up', 0) > 0 or latest.get('sm_bos', 0) == 1:
                res["status"] = "bullish_bos"
                res["vector"] = 1
            elif latest.get('bos_down', 0) > 0 or latest.get('sm_bos', 0) == -1:
                res["status"] = "bearish_bos"
                res["vector"] = -1
            elif 'smc_trend' in struct_df.columns:
                trend_val = latest['smc_trend']
                res["status"] = "bullish_trend" if trend_val > 0 else "bearish_trend" if trend_val < 0 else "ranging"
                res["vector"] = np.sign(trend_val)

        # Mathematical strength of structure
        if score_df is not None:
            score_cols = [c for c in ['sm_bos_score', 'sm_choch_score', 'sm_trend_score'] if c in score_df.columns]
            if score_cols:
                struct_scores = score_df[score_cols].ffill().bfill().mean(axis=1).to_numpy()
                res["strength_z"] = self._calculate_z_score(struct_scores)

        return res

    def _track_liquidity_dynamics(self, liq_df: pd.DataFrame, score_df: pd.DataFrame) -> dict:
        """Evaluates Stop Hunts/Sweeps and calculates liquidity pool density Z-Score."""
        res = {"sweep_status": "none", "density_z": 0.0, "sweep_vector": 0}
        
        if liq_df is not None and not liq_df.empty:
            latest = liq_df.iloc[-1]
            
            # Detect institutional sweeps (Judas Swing, Turtle Soup, Raw Sweeps)
            if latest.get('judas_swing', 0) > 0 or latest.get('turtle_soup', 0) > 0 or latest.get('stop_hunt_zone', 0) > 0:
                res["sweep_status"] = "institutional_manipulation"
            else:
                sweep_val = latest.get('liquidity_sweep', 0)
                if sweep_val > 0:
                    res["sweep_status"] = "bsl_swept"
                    res["sweep_vector"] = -1 # Sweeping highs implies bearish reversal potential
                elif sweep_val < 0:
                    res["sweep_status"] = "ssl_swept"
                    res["sweep_vector"] = 1 # Sweeping lows implies bullish reversal potential

        # Calculate historical liquidity density magnitude
        if score_df is not None and 'sm_liquidity_score' in score_df.columns:
            liq_score = score_df['sm_liquidity_score'].ffill().bfill().to_numpy()
            res["density_z"] = self._calculate_z_score(liq_score)

        return res

    def _track_institutional_zones(self, zone_df: pd.DataFrame, score_df: pd.DataFrame) -> dict:
        """Evaluates presence of active OB/FVG and their continuous imbalance Z-score."""
        res = {"has_active_ob": False, "fvg_strength_z": 0.0}
        
        if zone_df is not None and not zone_df.empty:
            latest = zone_df.iloc[-1]
            if latest.get('fresh_ob', 0) > 0 or latest.get('sm_fresh_ob', 0) > 0 or latest.get('breaker_block', 0) > 0:
                res["has_active_ob"] = True
                
        if score_df is not None and 'sm_fvg_score' in score_df.columns:
            fvg_score = score_df['sm_fvg_score'].ffill().bfill().to_numpy()
            res["fvg_strength_z"] = self._calculate_z_score(fvg_score)
            
        return res

    def _track_pricing_context(self, price_df: pd.DataFrame) -> dict:
        """Translates current price location into algorithmic Premium/Discount mapping."""
        res = {"context": "equilibrium"}
        if price_df is not None and not price_df.empty:
            latest = price_df.iloc[-1]
            if latest.get('premium_zone', 0) > 0:
                res["context"] = "premium"
            elif latest.get('discount_zone', 0) > 0:
                res["context"] = "discount"
        return res

    def _track_institutional_footprint(self, score_df: pd.DataFrame) -> dict:
        """Aggregates all institutional proxy scores into a statistical activity baseline."""
        res = {"inst_score_z": 0.0, "composite_z": 0.0}
        
        if score_df is not None and not score_df.empty:
            if 'sm_institutional_score' in score_df.columns:
                inst_score = score_df['sm_institutional_score'].ffill().bfill().to_numpy()
                res["inst_score_z"] = self._calculate_z_score(inst_score)
                
            if 'sm_smart_money_score' in score_df.columns:
                comp_score = score_df['sm_smart_money_score'].ffill().bfill().to_numpy()
                res["composite_z"] = self._calculate_z_score(comp_score)
                
        return res

    # ==========================================
    # SYNTHESIS & LOGIC MAPPING
    # ==========================================

    def _synthesize_bias(self, struct: dict, liq: dict, foot: dict) -> str:
        """
        Synthesizes directional bias. High institutional activity + Liquidity Sweep 
        creates a high-probability bias override.
        """
        # If there is a massive institutional footprint sweeping liquidity, bias shifts
        if foot["inst_score_z"] > self.Z_SIGNIFICANT and liq["sweep_vector"] != 0:
            return "bullish" if liq["sweep_vector"] > 0 else "bearish"
            
        # Otherwise respect mathematical structural momentum
        if struct["vector"] > 0 and struct["strength_z"] > -self.Z_MODERATE:
            return "bullish"
        if struct["vector"] < 0 and struct["strength_z"] > -self.Z_MODERATE:
            return "bearish"
            
        return "neutral"

    def _synthesize_smc_phase(self, struct: dict, liq: dict, zones: dict) -> str:
        """Defines the current Wyckoff/SMC lifecycle phase."""
        if struct["status"] in ["bullish_choch", "bearish_choch"]:
            return "reversal_phase"
            
        if struct["status"] in ["bullish_bos", "bullish_trend"]:
            if liq["sweep_status"] == "ssl_swept": return "reaccumulation"
            return "markup"
            
        if struct["status"] in ["bearish_bos", "bearish_trend"]:
            if liq["sweep_status"] == "bsl_swept": return "redistribution"
            return "markdown"
            
        if liq["sweep_status"] == "institutional_manipulation":
            return "manipulation"
            
        return "consolidation"

    def _compute_statistical_confidence(self, foot: dict, struct: dict, liq: dict) -> str:
        """Confidence is highest when Smart Money composite score spikes alongside structural shifts."""
        score = 3.0
        
        # High institutional volume/footprint validates the setup
        if foot["composite_z"] > self.Z_SIGNIFICANT: score += 1.0
        if foot["inst_score_z"] > self.Z_SIGNIFICANT: score += 1.0
        
        # Divergence: Structure breaking without institutional volume
        if struct["strength_z"] > self.Z_MODERATE and foot["composite_z"] < -self.Z_MODERATE:
            score -= 1.5
            
        if score >= 4.5: return "very_high"
        if score >= 3.5: return "high"
        if score >= 2.5: return "medium"
        if score >= 1.5: return "low"
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
            "analyzer_name": "smart_money_analyzer",
            "institutional_bias": "unknown", "smc_phase": "unknown",
            "market_structure": {"current_state": "unknown", "structure_strength_z": 0.0},
            "liquidity_profile": {"sweep_status": "none", "liquidity_density_z": 0.0},
            "institutional_zones": {"pricing_context": "unknown", "active_ob_present": False, "fvg_imbalance_z": 0.0},
            "smart_money_footprint": {"institutional_activity_z": 0.0, "composite_score_z": 0.0},
            "confidence": "very_low"
        }
