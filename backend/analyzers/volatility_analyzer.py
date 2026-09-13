import numpy as np
import pandas as pd
from typing import Dict, Any

class VolatilityAnalyzer:
    """
    Green Bull Rider V6 - L1 Volatility Analyzer (Institutional Grade)
    """

    def __init__(self):
        self.Z_MODERATE = 0.5
        self.Z_SIGNIFICANT = 1.0
        self.Z_EXTREME = 2.0
        self.EPSILON = 1e-8

    def analyze(self, feature_blocks: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        if not feature_blocks:
            return self._fallback_contract()

        compression = self._track_energy_compression(
            feature_blocks.get("band_block"), 
            feature_blocks.get("atr_block")
        )
        expansion = self._track_kinetic_expansion(
            feature_blocks.get("band_block"), 
            feature_blocks.get("osc_block"),
            feature_blocks.get("dispersion_block")
        )
        excursion = self._track_band_excursion(
            feature_blocks.get("band_block"),
            feature_blocks.get("dispersion_block")
        )
        noise = self._track_structural_noise(feature_blocks.get("noise_block"))

        vol_state = self._synthesize_volatility_regime(compression, expansion, excursion, noise)
        energy_level = self._calculate_kinetic_energy_level(compression, expansion)
        confidence = self._compute_statistical_confidence(noise, compression, expansion)

        return {
            "analyzer_name": "volatility_analyzer",
            "volatility_state": vol_state,
            "energy_level": energy_level,
            "squeeze_metrics": {
                "is_squeezing": compression["is_active_squeeze"],
                "compression_z_score": compression["width_z"],
                "volatility_percentile": compression["atr_percentile_val"]
            },
            "expansion_metrics": {
                "expansion_velocity_z": expansion["expansion_vel_z"],
                "dispersion_coherence_z": expansion["dispersion_z"]
            },
            "excursion_metrics": {
                "percent_b_z": excursion["percent_b_z"],
                "mean_reversion_risk": excursion["risk_level"]
            },
            "structural_noise": {
                "chop_z_score": noise["chop_z"],
                "vhf_z_score": noise["vhf_z"],
                "status": noise["status"]
            },
            "confidence": confidence
        }

    def _track_energy_compression(self, band_df: pd.DataFrame, atr_df: pd.DataFrame) -> dict:
        res = {"width_z": 0.0, "atr_percentile_val": 0.0, "is_active_squeeze": False}
        
        if band_df is not None and not band_df.empty:
            if 'bb_width' in band_df.columns:
                bb_width = pd.to_numeric(band_df['bb_width'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
                res["width_z"] = self._calculate_z_score(bb_width)
                
            if 'bb_squeeze' in band_df.columns:
                sqz = pd.to_numeric(band_df['bb_squeeze'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
                if len(sqz) > 0 and sqz[-1] > 0:
                    res["is_active_squeeze"] = True

        if atr_df is not None and 'atr_percentile' in atr_df.columns:
            res["atr_percentile_val"] = float(pd.to_numeric(atr_df['atr_percentile'], errors='coerce').ffill().bfill().iloc[-1])
            
        if res["width_z"] < -self.Z_SIGNIFICANT:
            res["is_active_squeeze"] = True

        return res

    def _track_kinetic_expansion(self, band_df: pd.DataFrame, osc_df: pd.DataFrame, disp_df: pd.DataFrame) -> dict:
        res = {"expansion_vel_z": 0.0, "dispersion_z": 0.0}
        expansion_signals = []
        
        if band_df is not None and 'bb_width' in band_df.columns:
            bb_w_arr = pd.to_numeric(band_df['bb_width'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
            if not np.all(np.isnan(bb_w_arr)):
                bb_w_vel = np.gradient(bb_w_arr)
                expansion_signals.append(self._calculate_z_score(bb_w_vel))
            
        if osc_df is not None:
            for col in ['expansion_index', 'volatility_osc', 'chaikin_vol']:
                if col in osc_df.columns:
                    col_arr = pd.to_numeric(osc_df[col], errors='coerce').ffill().bfill().to_numpy(dtype=float)
                    if not np.all(np.isnan(col_arr)):
                        vel = np.gradient(col_arr)
                        expansion_signals.append(self._calculate_z_score(vel))
                    
        if expansion_signals:
            res["expansion_vel_z"] = np.nanmean(expansion_signals)

        if disp_df is not None:
            adv_cols = [c for c in ['parkinson_vol', 'garman_klass', 'rogers_satchell', 'yang_zhang', 'hv_21'] if c in disp_df.columns]
            if adv_cols:
                disp_matrix = disp_df[adv_cols].apply(pd.to_numeric, errors='coerce').ffill().bfill().to_numpy(dtype=float)
                if not np.all(np.isnan(disp_matrix)):
                    mean_dispersion = np.nanmean(disp_matrix, axis=1)
                    res["dispersion_z"] = self._calculate_z_score(mean_dispersion)

        return res

    def _track_band_excursion(self, band_df: pd.DataFrame, disp_df: pd.DataFrame) -> dict:
        res = {"percent_b_z": 0.0, "risk_level": "unknown"}
        
        if band_df is not None and 'bb_percent_b' in band_df.columns:
            pb = pd.to_numeric(band_df['bb_percent_b'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
            res["percent_b_z"] = self._calculate_z_score(pb)
            
        std_err_z = 0.0
        if disp_df is not None and 'standard_error' in disp_df.columns:
            se = pd.to_numeric(disp_df['standard_error'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
            std_err_z = self._calculate_z_score(se)
            
        excursion_magnitude = abs(res["percent_b_z"])
        
        if excursion_magnitude > self.Z_EXTREME and std_err_z > self.Z_SIGNIFICANT:
            res["risk_level"] = "high"
        elif excursion_magnitude > self.Z_SIGNIFICANT:
            res["risk_level"] = "elevated"
        else:
            res["risk_level"] = "low"
            
        return res

    def _track_structural_noise(self, noise_df: pd.DataFrame) -> dict:
        res = {"chop_z": 0.0, "vhf_z": 0.0, "status": "unknown"}
        
        if noise_df is not None:
            if 'choppiness' in noise_df.columns:
                chop = pd.to_numeric(noise_df['choppiness'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
                res["chop_z"] = self._calculate_z_score(chop)
                
            if 'vhf' in noise_df.columns:
                vhf = pd.to_numeric(noise_df['vhf'], errors='coerce').ffill().bfill().to_numpy(dtype=float)
                res["vhf_z"] = self._calculate_z_score(vhf)
                
        if res["chop_z"] > self.Z_SIGNIFICANT and res["vhf_z"] < -self.Z_MODERATE:
            res["status"] = "elevated_noise"
        elif res["chop_z"] < -self.Z_SIGNIFICANT and res["vhf_z"] > self.Z_MODERATE:
            res["status"] = "directional_clarity"
        else:
            res["status"] = "baseline_noise"
            
        return res

    def _synthesize_volatility_regime(self, comp: dict, exp: dict, exc: dict, noise: dict) -> str:
        if comp["is_active_squeeze"] or comp["width_z"] < -self.Z_SIGNIFICANT:
            return "compression"
        if exp["expansion_vel_z"] > self.Z_SIGNIFICANT and exp["dispersion_z"] > self.Z_MODERATE:
            if exc["risk_level"] == "high":
                return "exhaustion"
            return "expansion"
        if noise["status"] == "elevated_noise":
            return "choppy"
        return "normal"

    def _calculate_kinetic_energy_level(self, comp: dict, exp: dict) -> str:
        comp_mag = abs(min(0.0, comp["width_z"]))
        exp_mag = max(0.0, exp["expansion_vel_z"])
        dominant = max(comp_mag, exp_mag)
        
        if dominant > self.Z_EXTREME: return "extreme"
        if dominant > self.Z_SIGNIFICANT: return "high"
        if dominant > self.Z_MODERATE: return "moderate"
        return "low"

    def _compute_statistical_confidence(self, noise: dict, comp: dict, exp: dict) -> str:
        conf_score = 3.0
        if noise["status"] == "directional_clarity": conf_score += 1.0
        if noise["status"] == "elevated_noise": conf_score -= 1.0
        if comp["is_active_squeeze"] or abs(exp["expansion_vel_z"]) > self.Z_EXTREME:
            conf_score += 1.0
            
        if conf_score >= 4.5: return "very_high"
        if conf_score >= 3.5: return "high"
        if conf_score >= 2.5: return "medium"
        if conf_score >= 1.5: return "low"
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
            "analyzer_name": "volatility_analyzer",
            "volatility_state": "unknown", "energy_level": "low",
            "squeeze_metrics": {"is_squeezing": False, "compression_z_score": 0.0, "volatility_percentile": 0.0},
            "expansion_metrics": {"expansion_velocity_z": 0.0, "dispersion_coherence_z": 0.0},
            "excursion_metrics": {"percent_b_z": 0.0, "mean_reversion_risk": "unknown"},
            "structural_noise": {"chop_z_score": 0.0, "vhf_z_score": 0.0, "status": "unknown"},
            "confidence": "very_low"
        }
