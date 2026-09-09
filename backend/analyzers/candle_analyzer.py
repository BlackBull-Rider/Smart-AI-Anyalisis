import numpy as np
import pandas as pd
from typing import Dict, Any

class CandleAnalyzer:
    """
    Green Bull Rider V6 - L1 Candle & Execution Analyzer
    
    Acts as the 'Execution Trigger'. Translates raw OHLC geometry into continuous 
    kinetic pressure, tracking expansion footprints and institutional rejection/absorption.
    Zero retail candlestick pattern dependency. Evaluates statistical Z-Scores of 
    intraday dominance and close location values.
    """

    def __init__(self):
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes final entry validation based on EXACT pipeline feature groupings:
        - 'pressure_block': dominance_score, pressure_score, balance_score, clv, candle_strength, bull_power, bear_power, direction_strength
        - 'volatility_block': candle_range, body_size, rolling_body_zscore, range_percentile, rolling_range_percentile, expansion_candle, compression_candle, impulse_candle
        - 'rejection_block': wick_strength, wick_balance, wick_ratio, absorption_candle, rejection_candle, liquidity_sweep_candle, smart_money_candle, indecision_candle
        - 'gap_block': gap_up, gap_down, breakaway_gap_up, breakaway_gap_down, gap_percent, gap_fill, inside_gap
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. MATHEMATICAL MICRO-TRACKERS
        pressure_state = self._track_kinetic_pressure(feature_blocks.get("pressure_block"))
        volatility_state = self._track_expansion_kinematics(feature_blocks.get("volatility_block"))
        rejection_state = self._track_institutional_rejection(feature_blocks.get("rejection_block"))
        gap_state = self._track_structural_gaps(feature_blocks.get("gap_block"))

        # 2. SYNTHESIS
        micro_trend = self._synthesize_micro_trend(pressure_state, volatility_state, rejection_state)
        trigger_status = self._evaluate_execution_trigger(pressure_state, volatility_state, rejection_state, gap_state)
        confidence = self._compute_execution_confidence(pressure_state, volatility_state, rejection_state)

        # 3. EXACT LOCKED JSON CONTRACT
        return {
            "analyzer_name": "candle_analyzer",
            "trigger_status": trigger_status,
            "micro_trend": micro_trend,
            "pressure_metrics": {
                "dominance_z_score": pressure_state["dominance_z"],
                "close_location_z": pressure_state["clv_z"],
                "is_pressure_divergent": pressure_state["is_divergent"]
            },
            "volatility_metrics": {
                "body_expansion_z": volatility_state["body_expansion_z"],
                "range_percentile": volatility_state["range_percentile"]
            },
            "institutional_footprint": {
                "is_rejection": rejection_state["is_rejection"],
                "is_absorption": rejection_state["is_absorption"],
                "is_smart_money_imbalance": rejection_state["is_smart_money"]
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _track_kinetic_pressure(self, df: pd.DataFrame) -> dict:
        """
        Evaluates who won the candle session by tracking dominance and CLV 
        (Close Location Value) dynamically against their historical distributions.
        """
        res = {"dominance_z": 0.0, "clv_z": 0.0, "is_divergent": False}
        if df is None or df.empty:
            return res

        # Dominance: (Buying Pressure - Selling Pressure) / Candle Range
        if 'dominance_score' in df.columns:
            dom = df['dominance_score'].ffill().bfill().to_numpy()
            res["dominance_z"] = self._calculate_z_score(dom)
            
        # CLV: Normalizes the close position relative to the High/Low range
        if 'clv' in df.columns:
            clv = df['clv'].ffill().bfill().to_numpy()
            res["clv_z"] = self._calculate_z_score(clv)

        # Divergence: If Dominance is highly positive but Close Location is dropping
        if abs(res["dominance_z"]) > self.Z_SIGNIFICANT and np.sign(res["dominance_z"]) != np.sign(res["clv_z"]):
            if abs(res["clv_z"]) > self.Z_MODERATE:
                res["is_divergent"] = True

        return res

    def _track_expansion_kinematics(self, df: pd.DataFrame) -> dict:
        """Tracks impulsive expansion vs compression using body/range Z-Scores."""
        res = {"body_expansion_z": 0.0, "range_percentile": 50.0, "is_impulsive": False}
        if df is None or df.empty:
            return res

        # Direct pipeline-provided Z-Score ensures standard mathematical consistency
        if 'rolling_body_zscore' in df.columns:
            res["body_expansion_z"] = df['rolling_body_zscore'].ffill().bfill().iloc[-1]
        elif 'body_size' in df.columns:
            b_size = df['body_size'].ffill().bfill().to_numpy()
            res["body_expansion_z"] = self._calculate_z_score(b_size)

        if 'range_percentile' in df.columns:
            res["range_percentile"] = df['range_percentile'].ffill().bfill().iloc[-1]
        elif 'rolling_range_percentile' in df.columns:
            res["range_percentile"] = df['rolling_range_percentile'].ffill().bfill().iloc[-1]

        # Explicit pipeline confirmations
        latest = df.iloc[-1]
        if latest.get('impulse_candle', 0) > 0 or latest.get('expansion_candle', 0) > 0:
            res["is_impulsive"] = True

        return res

    def _track_institutional_rejection(self, df: pd.DataFrame) -> dict:
        """Evaluates wick kinetics and pipeline structural anomalies."""
        res = {
            "wick_strength_z": 0.0, "is_rejection": False, 
            "is_absorption": False, "is_smart_money": False
        }
        if df is None or df.empty:
            return res

        if 'wick_strength' in df.columns:
            w_str = df['wick_strength'].ffill().bfill().to_numpy()
            res["wick_strength_z"] = self._calculate_z_score(w_str)

        latest = df.iloc[-1]
        if latest.get('rejection_candle', 0) > 0 or latest.get('liquidity_sweep_candle', 0) > 0:
            res["is_rejection"] = True
            
        if latest.get('absorption_candle', 0) > 0 or latest.get('indecision_candle', 0) > 0:
            res["is_absorption"] = True
            
        if latest.get('smart_money_candle', 0) > 0:
            res["is_smart_money"] = True

        return res

    def _track_structural_gaps(self, df: pd.DataFrame) -> dict:
        """Evaluates immediate repricing vectors (Gaps)."""
        res = {"has_gap": False, "is_breakaway": False, "gap_vector": 0}
        if df is None or df.empty:
            return res

        latest = df.iloc[-1]
        
        if latest.get('breakaway_gap_up', 0) > 0:
            res.update({"has_gap": True, "is_breakaway": True, "gap_vector": 1})
        elif latest.get('breakaway_gap_down', 0) > 0:
            res.update({"has_gap": True, "is_breakaway": True, "gap_vector": -1})
        elif latest.get('gap_up', 0) > 0 and not latest.get('gap_fill', 0):
            res.update({"has_gap": True, "is_breakaway": False, "gap_vector": 1})
        elif latest.get('gap_down', 0) > 0 and not latest.get('gap_fill', 0):
            res.update({"has_gap": True, "is_breakaway": False, "gap_vector": -1})
            
        return res

    # ==========================================
    # SYNTHESIS & LOGIC MAPPING
    # ==========================================

    def _synthesize_micro_trend(self, press: dict, vol: dict, rej: dict) -> str:
        """Determines the immediate kinetic state of the current bar."""
        if rej["is_rejection"] or rej["is_absorption"]:
            return "exhaustion"
            
        if vol["is_impulsive"] or abs(vol["body_expansion_z"]) > self.Z_SIGNIFICANT:
            if press["clv_z"] > self.Z_MODERATE:
                return "bullish_impulse"
            if press["clv_z"] < -self.Z_MODERATE:
                return "bearish_impulse"
                
        if press["is_divergent"]:
            return "indecision"
            
        return "neutral"

    def _evaluate_execution_trigger(self, press: dict, vol: dict, rej: dict, gap: dict) -> str:
        """
        Determines if the current candle geometry validates an execution.
        Returns categorical execution logic for L2 Engine mapping.
        """
        # 1. Breakaway gap confirms extreme urgency -> Execute immediately
        if gap["is_breakaway"]:
            return "valid_entry"

        # 2. Institutional footprint with highly deviant kinetic pressure -> High probability trigger
        if rej["is_smart_money"] or (rej["is_rejection"] and abs(press["dominance_z"]) > self.Z_SIGNIFICANT):
            return "valid_entry"
            
        # 3. Clean impulsive expansion without divergent pressure -> Valid continuation trigger
        if vol["is_impulsive"] and not press["is_divergent"] and abs(press["clv_z"]) > self.Z_SIGNIFICANT:
            return "valid_entry"

        # 4. Volatility is dead or pressure is highly divergent -> Do not execute
        if vol["body_expansion_z"] < -self.Z_MODERATE or press["is_divergent"]:
            return "no_trigger"

        # Default fallback for neutral forming candles
        return "wait_for_close"

    def _compute_execution_confidence(self, press: dict, vol: dict, rej: dict) -> str:
        """Confidence based on the alignment of volatility expansion and internal pressure."""
        score = 3.0
        
        # High structural expansion increases trigger validity
        if vol["body_expansion_z"] > self.Z_SIGNIFICANT: score += 1.0
        if vol["range_percentile"] > 80.0: score += 1.0
        
        # Institutional footprint adds massive confidence
        if rej["is_smart_money"] or rej["is_rejection"]: score += 1.5
        
        # Pressure divergence destroys confidence
        if press["is_divergent"]: score -= 2.0
        
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
            "analyzer_name": "candle_analyzer",
            "trigger_status": "no_trigger",
            "micro_trend": "unknown",
            "pressure_metrics": {"dominance_z_score": 0.0, "close_location_z": 0.0, "is_pressure_divergent": False},
            "volatility_metrics": {"body_expansion_z": 0.0, "range_percentile": 50.0},
            "institutional_footprint": {"is_rejection": False, "is_absorption": False, "is_smart_money_imbalance": False},
            "confidence": "very_low"
        }
