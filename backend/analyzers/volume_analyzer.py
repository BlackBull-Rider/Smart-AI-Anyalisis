import numpy as np
import pandas as pd
from typing import Dict, Any

class VolumeAnalyzer:
    """
    Green Bull Rider V6 - L1 Volume & Order Flow Analyzer
    
    Acts as the 'Truth Serum' of the architecture. Evaluates institutional 
    participation, delta flow, and smart money anomalies without relying on 
    arbitrary volume thresholds. Utilizes continuous statistical Z-Scores 
    and kinetic derivatives of flow estimators (OBV, ADL, Delta, VPIN).
    """

    def __init__(self):
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Executes volume kinematics based on EXACT pipeline feature groupings:
        - 'relative_metrics': volume, avg_volume_20, rolling_volume_20, rvol_20, volume_ratio, vol_ema_20, vol_zscore, vol_percentile, vol_roc
        - 'flow_metrics': obv, obv_ema, obv_roc, adl, adosc, cmf_20, mfi_14, buy_volume, sell_volume, delta_volume, volume_trend, pvt
        - 'kinetic_metrics': pvo, pvo_signal, pvo_hist, vol_macd, vol_macd_signal, vol_macd_hist, klinger, klinger_signal, klinger_hist, volume_osc, force_index, efi_13, eom, smoothed_eom
        - 'vwap_metrics': vwap, rolling_vwap, vwema, evwma
        - 'institutional_metrics': nvi, pvi, smart_money_index, effort_result, stopping_volume, no_demand, no_supply, amihud, vpin
        """
        if not feature_blocks:
            return self._fallback_contract()

        # 1. MATHEMATICAL MICRO-TRACKERS
        participation = self._track_relative_participation(feature_blocks.get("relative_metrics"))
        directional_flow = self._track_directional_flow(feature_blocks.get("flow_metrics"))
        kinetic_pressure = self._track_kinetic_pressure(feature_blocks.get("kinetic_metrics"))
        anomalies = self._track_institutional_anomalies(feature_blocks.get("institutional_metrics"))

        # 2. SYNTHESIS
        flow_bias = self._synthesize_flow_bias(directional_flow, kinetic_pressure, anomalies)
        volume_regime = self._synthesize_volume_regime(participation, directional_flow, anomalies, flow_bias)
        confidence = self._compute_confidence(participation, directional_flow, anomalies)

        # 3. EXACT LOCKED JSON CONTRACT
        return {
            "analyzer_name": "volume_analyzer",
            "volume_regime": volume_regime,
            "flow_bias": flow_bias,
            "participation": {
                "relative_activity_z": participation["activity_z"],
                "is_high_participation": participation["is_significant"]
            },
            "order_flow": {
                "accumulation_z": directional_flow["accumulation_z"],
                "delta_velocity_z": directional_flow["delta_vel_z"],
                "flow_acceleration_z": kinetic_pressure["pressure_accel_z"]
            },
            "institutional_anomalies": {
                "smart_money_activity_z": anomalies["smi_vel_z"],
                "toxicity_vpin_z": anomalies["toxicity_z"],
                "exhaustion_signature": anomalies["exhaustion_present"]
            },
            "confidence": confidence
        }

    # ==========================================
    # BEHAVIORAL TRACKING (Mathematical Core)
    # ==========================================

    def _track_relative_participation(self, df: pd.DataFrame) -> dict:
        """Evaluates absolute and relative volume intensity."""
        res = {"activity_z": 0.0, "is_significant": False}
        if df is None or df.empty:
            return res

        # Trust pipeline provided z-score if available, otherwise compute it mathematically
        if 'vol_zscore' in df.columns:
            res["activity_z"] = df['vol_zscore'].ffill().bfill().iloc[-1]
        elif 'rvol_20' in df.columns:
            rvol = df['rvol_20'].ffill().bfill().to_numpy()
            res["activity_z"] = self._calculate_z_score(rvol)
        elif 'volume' in df.columns:
            vol = df['volume'].ffill().bfill().to_numpy()
            res["activity_z"] = self._calculate_z_score(vol)

        if res["activity_z"] > self.Z_SIGNIFICANT:
            res["is_significant"] = True

        return res

    def _track_directional_flow(self, df: pd.DataFrame) -> dict:
        """
        Evaluates cumulative buying/selling pressure via OBV, ADL, and Delta.
        Measures the 1st derivative (velocity) to see who is actively seizing control.
        """
        res = {"accumulation_z": 0.0, "delta_vel_z": 0.0}
        if df is None or df.empty:
            return res

        # Cumulative Flow (OBV, ADL, PVT) - Are they trending up or down?
        flow_cols = [c for c in ['obv', 'adl', 'pvt'] if c in df.columns]
        if flow_cols:
            flow_matrix = df[flow_cols].ffill().bfill().to_numpy()
            flow_vel = np.gradient(flow_matrix, axis=0)
            mean_flow_vel = np.nanmean(flow_vel, axis=1)
            res["accumulation_z"] = self._calculate_z_score(mean_flow_vel)

        # Immediate Order Flow (Delta) - Aggressive buying vs selling
        if 'delta_volume' in df.columns:
            delta = df['delta_volume'].ffill().bfill().to_numpy()
            delta_vel = np.gradient(delta)
            res["delta_vel_z"] = self._calculate_z_score(delta_vel)
        elif 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            bv = df['buy_volume'].ffill().bfill().to_numpy()
            sv = df['sell_volume'].ffill().bfill().to_numpy()
            delta = bv - sv
            delta_vel = np.gradient(delta)
            res["delta_vel_z"] = self._calculate_z_score(delta_vel)

        return res

    def _track_kinetic_pressure(self, df: pd.DataFrame) -> dict:
        """
        Evaluates the acceleration of volume using Oscillators (PVO, Klinger).
        """
        res = {"pressure_accel_z": 0.0}
        if df is None or df.empty:
            return res

        hist_cols = [c for c in ['pvo_hist', 'vol_macd_hist', 'klinger_hist', 'force_index'] if c in df.columns]
        if hist_cols:
            hist_matrix = df[hist_cols].ffill().bfill().to_numpy()
            
            # The histogram is already a velocity proxy. Its gradient represents acceleration.
            acceleration = np.gradient(hist_matrix, axis=0)
            mean_accel = np.nanmean(acceleration, axis=1)
            res["pressure_accel_z"] = self._calculate_z_score(mean_accel)

        return res

    def _track_institutional_anomalies(self, df: pd.DataFrame) -> dict:
        """
        Identifies structural footprints (Smart Money Index, Exhaustion, Toxicity).
        """
        res = {"smi_vel_z": 0.0, "toxicity_z": 0.0, "exhaustion_present": "none"}
        if df is None or df.empty:
            return res

        # Smart Money vs Dumb Money (NVI/PVI/SMI)
        smi_cols = [c for c in ['smart_money_index', 'nvi'] if c in df.columns]
        if smi_cols:
            smi_matrix = df[smi_cols].ffill().bfill().to_numpy()
            smi_vel = np.gradient(smi_matrix, axis=0)
            mean_smi_vel = np.nanmean(smi_vel, axis=1)
            res["smi_vel_z"] = self._calculate_z_score(mean_smi_vel)

        # Toxicity & Illiquidity (VPIN / Amihud)
        tox_cols = [c for c in ['vpin', 'amihud'] if c in df.columns]
        if tox_cols:
            tox_matrix = df[tox_cols].ffill().bfill().to_numpy()
            mean_tox = np.nanmean(tox_matrix, axis=1)
            res["toxicity_z"] = self._calculate_z_score(mean_tox)

        # Exhaustion & Effort Anomalies
        latest = df.iloc[-1]
        if latest.get('stopping_volume', 0) > 0:
            res["exhaustion_present"] = "stopping_volume"
        elif 'effort_result' in df.columns:
            # Extreme effort vs result indicates exhaustion/absorption
            evr = df['effort_result'].ffill().bfill().to_numpy()
            evr_z = self._calculate_z_score(evr)
            if evr_z > self.Z_EXTREME:
                res["exhaustion_present"] = "effort_absorption"
        
        # Lack of participation triggers
        if latest.get('no_demand', 0) > 0:
            res["exhaustion_present"] = "no_demand"
        elif latest.get('no_supply', 0) > 0:
            res["exhaustion_present"] = "no_supply"

        return res

    # ==========================================
    # SYNTHESIS & LOGIC MAPPING
    # ==========================================

    def _synthesize_flow_bias(self, flow: dict, kin: dict, anom: dict) -> str:
        """Determines the true underlying order flow bias."""
        score = 0
        
        # Core cumulative flow (OBV/ADL)
        if flow["accumulation_z"] > self.Z_MODERATE: score += 1
        elif flow["accumulation_z"] < -self.Z_MODERATE: score -= 1
        
        # Immediate delta urgency
        if flow["delta_vel_z"] > self.Z_MODERATE: score += 1
        elif flow["delta_vel_z"] < -self.Z_MODERATE: score -= 1
        
        # Smart money footprint divergence
        if anom["smi_vel_z"] > self.Z_SIGNIFICANT: score += 2
        elif anom["smi_vel_z"] < -self.Z_SIGNIFICANT: score -= 2
        
        # Reversal signatures (Anomalies)
        if anom["exhaustion_present"] == "no_supply": score += 1
        elif anom["exhaustion_present"] == "no_demand": score -= 1
        
        if score > 0: return "bullish"
        if score < 0: return "bearish"
        return "neutral"

    def _synthesize_volume_regime(self, part: dict, flow: dict, anom: dict, bias: str) -> str:
        """Determines the lifecycle phase based on volume participation and flow."""
        # 1. Exhaustion / Capitulation (Extreme activity + Stopping Volume)
        if part["activity_z"] > self.Z_EXTREME or anom["exhaustion_present"] in ["stopping_volume", "effort_absorption"]:
            if flow["accumulation_z"] < -self.Z_SIGNIFICANT:
                return "capitulation"
            return "climactic_exhaustion"

        # 2. Dry Up (No activity + toxicity/illiquidity)
        if part["activity_z"] < -self.Z_SIGNIFICANT or anom["exhaustion_present"] in ["no_demand", "no_supply"]:
            return "dry_up"

        # 3. Markup / Markdown (Active trend participation)
        if part["is_significant"]:
            if bias == "bullish" and flow["accumulation_z"] > self.Z_MODERATE: return "markup"
            if bias == "bearish" and flow["accumulation_z"] < -self.Z_MODERATE: return "markdown"

        # 4. Stealth Phases (Divergence between price and smart money)
        if bias == "bullish" and flow["accumulation_z"] > self.Z_SIGNIFICANT:
            return "accumulation"
        if bias == "bearish" and flow["accumulation_z"] < -self.Z_SIGNIFICANT:
            return "distribution"

        return "neutral_churn"

    def _compute_confidence(self, part: dict, flow: dict, anom: dict) -> str:
        """Confidence is highest when high participation validates the directional flow."""
        score = 3.0
        
        if part["is_significant"]: score += 1.0
        if part["activity_z"] < -self.Z_MODERATE: score -= 1.0
        
        # If Delta and Cumulative Flow align, confidence increases
        if np.sign(flow["accumulation_z"]) == np.sign(flow["delta_vel_z"]) and abs(flow["delta_vel_z"]) > self.Z_MODERATE:
            score += 1.0
            
        # Toxicity/Illiquidity drops confidence due to erratic order flow
        if anom["toxicity_z"] > self.Z_SIGNIFICANT:
            score -= 1.0

        if score >= 4.5: return "very_high"
        if score >= 3.5: return "high"
        if score >= 2.5: return "medium"
        if score >= 1.5: return "low"
        return "very_low"

    def _calculate_z_score(self, series: np.ndarray) -> float:
        """Calculates historical standard deviation mapping."""
        if len(series) == 0:
            return 0.0
        current_val = series[-1]
        hist_mean = np.nanmean(series)
        hist_std = np.nanstd(series) + self.EPSILON
        return float((current_val - hist_mean) / hist_std)

    def _fallback_contract(self) -> Dict[str, Any]:
        return {
            "analyzer_name": "volume_analyzer",
            "volume_regime": "unknown", "flow_bias": "neutral",
            "participation": {"relative_activity_z": 0.0, "is_high_participation": False},
            "order_flow": {"accumulation_z": 0.0, "delta_velocity_z": 0.0, "flow_acceleration_z": 0.0},
            "institutional_anomalies": {"smart_money_activity_z": 0.0, "toxicity_vpin_z": 0.0, "exhaustion_signature": "none"},
            "confidence": "very_low"
        }
