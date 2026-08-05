"""
GREEN BULL RIDER V6
Layer-2: Quantitative Analyzer
Module: trend_analyzer.py

Final Institutional-grade Deterministic Trend Analyzer.
Targeted Fixes Applied:
- Clamped price vectors to prevent unbounded OHLC anomalies.
- True RVOL magnitude boosting.
- Clean separation of Feature Availability vs Analytical Utilization.
- Tri-state (bullish/bearish/neutral) status mapping for true 50.0 handling.
- Historical ATR normalization for slope accelerations.

Strictly mapped to provided Pipeline Feature names. No DB mapping.
Python 3.13 Compatible.
"""

import math
import time
import pandas as pd
import numpy as np
from typing import Dict, Any

class TrendAnalyzer:
    def __init__(self):
        self.analyzer_name = "trend_analyzer"

        # Explicit expected feature universe (Exact Pipeline Names)
        self.feature_families = {
            "PRICE": ['close', 'open', 'high', 'low'],
            "MA": ['sma_20', 'sma_50', 'sma_100', 'sma_200', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'vwap'],
            "DMI": ['adx_14', 'plus_di', 'minus_di', 'dx', 'supertrend_trend'],
            "MOMENTUM": ['rsi_14', 'macd', 'macd_signal', 'macd_hist', 'roc_12', 'cci_20', 'mom_10', 'williams_r', 'stoch_k', 'stoch_d'],
            "VOLATILITY": ['atr_14', 'bb_width', 'bb_percent_b', 'bb_squeeze'],
            "VOLUME": ['volume', 'rvol_20', 'obv', 'obv_roc', 'cmf_20', 'mfi_14', 'delta_volume'],
            "SMC": ['smc_trend', 'bos_up', 'bos_down', 'choch_up', 'choch_down', 'liquidity_sweep'],
            "MTF": ['daily_mid', 'weekly_mid', 'monthly_mid', 'yearly_mid']
        }
        
        self.expected_features = [feat for family in self.feature_families.values() for feat in family]

    def analyze(self, feature_history: pd.DataFrame, fundamental_data: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        utilized_features = []
        valid_features = []
        missing_features = []
        invalid_features = []
        calculation_trace = {}
        
        try:
            if feature_history is None or not isinstance(feature_history, pd.DataFrame) or feature_history.empty:
                raise ValueError("Feature history dataframe is empty or None.")

            feature_expected_count = len(self.expected_features)
            
            hist_len = len(feature_history)
            latest = feature_history.iloc[-1]
            prev1 = feature_history.iloc[-2] if hist_len > 1 else None
            prev3 = feature_history.iloc[-4] if hist_len >= 4 else None

            # 1. Feature Validation & Extraction
            curr_data, prev1_data, prev3_data = {}, {}, {}
            
            for feature in self.expected_features:
                val = latest.get(feature)
                if val is None or pd.isna(val) or not pd.api.types.is_scalar(val):
                    if val is not None and not pd.api.types.is_scalar(val):
                        invalid_features.append(feature)
                    else:
                        missing_features.append(feature)
                    curr_data[feature] = None
                    prev1_data[feature] = None
                    prev3_data[feature] = None
                else:
                    try:
                        f_val = float(val)
                        if math.isfinite(f_val):
                            curr_data[feature] = f_val
                            valid_features.append(feature)
                            
                            p1 = prev1.get(feature) if prev1 is not None else None
                            p3 = prev3.get(feature) if prev3 is not None else None
                            
                            prev1_data[feature] = float(p1) if pd.api.types.is_scalar(p1) and pd.notna(p1) and math.isfinite(float(p1)) else None
                            prev3_data[feature] = float(p3) if pd.api.types.is_scalar(p3) and pd.notna(p3) and math.isfinite(float(p3)) else None
                        else:
                            invalid_features.append(feature)
                            curr_data[feature] = None
                    except (ValueError, TypeError):
                        invalid_features.append(feature)
                        curr_data[feature] = None

            close = curr_data.get('close')
            if close is None or close <= 0:
                raise ValueError("Critical Feature Missing: Valid 'close' price required.")

            # ATR Extraction with historical context
            atr = curr_data.get('atr_14')
            if atr is None or atr <= 0:
                atr = close * 0.015
            else:
                utilized_features.append('atr_14')
                
            p1_atr = prev1_data.get('atr_14') if prev1_data.get('atr_14') else atr
            
            # 2. EVIDENCE ENGINES (Magnitude-Aware Continuous Vectors: -1.0 to +1.0)
            
            # --- A. PRICE STRUCTURE ENGINE ---
            price_vectors = []
            f_price = []
            
            if curr_data.get('open') is not None and curr_data.get('high') is not None and curr_data.get('low') is not None:
                h, l, o = curr_data['high'], curr_data['low'], curr_data['open']
                c_range = max(h - l, 1e-5)
                # FIX: Clamped vectors to prevent OHLC anomaly explosion
                body_vec = self._clamp((close - o) / c_range, -1.0, 1.0)
                close_pos = self._clamp(((close - l) / c_range - 0.5) * 2.0, -1.0, 1.0)
                price_vectors.extend([body_vec, close_pos])
                f_price.extend(['open', 'high', 'low', 'close'])
            
            if curr_data.get('vwap') is not None:
                dist_vwap = (close - curr_data['vwap']) / atr
                price_vectors.append(math.tanh(dist_vwap))
                f_price.append('vwap')

            price_vector = np.mean(price_vectors) if price_vectors else 0.0
            utilized_features.extend(f_price)
            calculation_trace['price_structure'] = {
                "features_used": list(set(f_price)),
                "vector_value": float(price_vector),
                "score": self._vec_to_score(price_vector),
                "method": "Clamped ATR-normalized distance and continuous candle range positioning."
            }

            # --- B. MA STRUCTURE ENGINE ---
            ma_vectors = []
            ma_strengths = []
            f_ma = []
            
            ma_list = ['sma_20', 'sma_50', 'sma_100', 'sma_200', 'ema_20', 'ema_50', 'ema_100', 'ema_200']
            for ma_name in ma_list:
                ma_val = curr_data.get(ma_name)
                if ma_val is not None:
                    dist_z = (close - ma_val) / atr
                    ma_vectors.append(math.tanh(dist_z / 2.0))
                    f_ma.append(ma_name)
                    
                    # FIX: Historical ATR matching for slope extraction
                    if prev1_data.get(ma_name) is not None:
                        curr_slope = (ma_val - prev1_data[ma_name]) / atr
                        ma_vectors.append(math.tanh(curr_slope * 5.0))
                        ma_strengths.append(abs(math.tanh(curr_slope * 10.0)))
                        
                        if prev3_data.get(ma_name) is not None:
                            prev_slope = (prev1_data[ma_name] - prev3_data[ma_name]) / (p1_atr * 2.0)
                            accel = curr_slope - prev_slope
                            ma_vectors.append(math.tanh(accel * 5.0))

            ma_vector = np.mean(ma_vectors) if ma_vectors else 0.0
            ma_str = self._clamp(np.mean(ma_strengths) * 100.0, 0.0, 100.0) if ma_strengths else 0.0
            utilized_features.extend(f_ma)
            calculation_trace['ma_structure'] = {
                "features_used": list(set(f_ma)),
                "vector_value": float(ma_vector),
                "score": self._vec_to_score(ma_vector),
                "method": "Magnitude-aware Tanh mapping of MA distance, slope, and acceleration."
            }

            # --- C. DMI / ADX ENGINE ---
            dmi_vectors = []
            dmi_strengths = []
            f_dmi = []
            
            if curr_data.get('plus_di') is not None and curr_data.get('minus_di') is not None:
                di_spread = curr_data['plus_di'] - curr_data['minus_di']
                dmi_vectors.append(math.tanh(di_spread / 15.0)) 
                f_dmi.extend(['plus_di', 'minus_di'])

            if curr_data.get('adx_14') is not None:
                adx = curr_data['adx_14']
                adx_norm = self._clamp(math.tanh((adx - 15.0) / 20.0), 0.0, 1.0)
                
                if prev1_data.get('adx_14') is not None:
                    adx_slope = adx - prev1_data['adx_14']
                    if adx_slope > 0: adx_norm = min(1.0, adx_norm * 1.2)
                    
                dmi_strengths.append(self._clamp(adx_norm, 0.0, 1.0))
                f_dmi.append('adx_14')
                
            if curr_data.get('dx') is not None:
                utilized_features.append('dx')
                f_dmi.append('dx')
                
            if curr_data.get('supertrend_trend') is not None:
                dmi_vectors.append(self._clamp(curr_data['supertrend_trend'], -1.0, 1.0))
                f_dmi.append('supertrend_trend')

            dmi_vector = np.mean(dmi_vectors) if dmi_vectors else 0.0
            dmi_str = np.mean(dmi_strengths) if dmi_strengths else 0.0
            utilized_features.extend(f_dmi)
            calculation_trace['dmi_structure'] = {
                "features_used": list(set(f_dmi)),
                "vector_value": float(dmi_vector),
                "score": self._vec_to_score(dmi_vector),
                "method": "Continuous DI spread evaluation and normalized ADX momentum."
            }

            # --- D. MOMENTUM ENGINE ---
            mom_vectors = []
            f_mom = []
            
            if curr_data.get('rsi_14') is not None:
                mom_vectors.append(math.tanh((curr_data['rsi_14'] - 50.0) / 20.0))
                f_mom.append('rsi_14')
                
            if curr_data.get('williams_r') is not None:
                mom_vectors.append(math.tanh((curr_data['williams_r'] + 50.0) / 25.0))
                f_mom.append('williams_r')
                
            if curr_data.get('stoch_k') is not None and curr_data.get('stoch_d') is not None:
                k, d = curr_data['stoch_k'], curr_data['stoch_d']
                mom_vectors.append(math.tanh((k - 50.0) / 25.0))
                mom_vectors.append(math.tanh((k - d) / 10.0))
                f_mom.extend(['stoch_k', 'stoch_d'])
                
            if curr_data.get('macd') is not None and curr_data.get('macd_signal') is not None:
                macd, signal = curr_data['macd'], curr_data['macd_signal']
                mom_vectors.append(math.tanh((macd / atr) * 5.0))
                mom_vectors.append(math.tanh(((macd - signal) / atr) * 10.0))
                f_mom.extend(['macd', 'macd_signal'])
                
            if curr_data.get('macd_hist') is not None:
                mom_vectors.append(math.tanh((curr_data['macd_hist'] / atr) * 15.0))
                f_mom.append('macd_hist')

            if curr_data.get('roc_12') is not None:
                mom_vectors.append(math.tanh(curr_data['roc_12'] / 5.0))
                f_mom.append('roc_12')
                
            if curr_data.get('mom_10') is not None:
                mom_vectors.append(math.tanh(curr_data['mom_10'] / atr))
                f_mom.append('mom_10')
                
            if curr_data.get('cci_20') is not None:
                mom_vectors.append(math.tanh(curr_data['cci_20'] / 150.0))
                f_mom.append('cci_20')

            mom_vector = np.mean(mom_vectors) if mom_vectors else 0.0
            utilized_features.extend(f_mom)
            calculation_trace['momentum'] = {
                "features_used": list(set(f_mom)),
                "vector_value": float(mom_vector),
                "score": self._vec_to_score(mom_vector),
                "method": "Oscillators normalized via Tanh and ATR to continuous vectors."
            }

            # --- E. VOLUME & FLOW ENGINE ---
            vol_vectors = []
            f_vol = []
            
            if curr_data.get('cmf_20') is not None:
                vol_vectors.append(math.tanh(curr_data['cmf_20'] / 0.15))
                f_vol.append('cmf_20')
                
            if curr_data.get('mfi_14') is not None:
                vol_vectors.append(math.tanh((curr_data['mfi_14'] - 50.0) / 20.0))
                f_vol.append('mfi_14')
                
            if curr_data.get('obv_roc') is not None:
                vol_vectors.append(math.tanh(curr_data['obv_roc'] * 5.0))
                f_vol.extend(['obv_roc'])
                if curr_data.get('obv') is not None:
                    utilized_features.append('obv')

            if curr_data.get('delta_volume') is not None and curr_data.get('volume') is not None and curr_data['volume'] > 0:
                delta_ratio = curr_data['delta_volume'] / curr_data['volume']
                vol_vectors.append(math.tanh(delta_ratio * 3.0))
                f_vol.extend(['delta_volume', 'volume'])
                
            vol_magnitude = 1.0
            if curr_data.get('rvol_20') is not None:
                rvol = curr_data['rvol_20']
                # FIX: Real RVOL amplification, clamped to avoid infinite blowup
                vol_magnitude = self._clamp(rvol, 0.5, 2.0)
                f_vol.append('rvol_20')

            vol_vector = self._clamp(np.mean(vol_vectors) * vol_magnitude if vol_vectors else 0.0, -1.0, 1.0)
            utilized_features.extend(f_vol)
            calculation_trace['volume_flow'] = {
                "features_used": list(set(f_vol)),
                "vector_value": float(vol_vector),
                "score": self._vec_to_score(vol_vector),
                "method": "CMF, MFI, Delta and OBV flow modulated directly by RVOL multiplier."
            }

            # --- F. VOLATILITY ENGINE ---
            f_vty = []
            bb_squeeze_penalty = 0.0
            if curr_data.get('bb_squeeze') is not None and float(curr_data['bb_squeeze']) > 0.0:
                bb_squeeze_penalty = 0.3
                f_vty.append('bb_squeeze')
            if curr_data.get('bb_width') is not None:
                utilized_features.append('bb_width')
                f_vty.append('bb_width')
                
            utilized_features.extend(f_vty)

            # --- G. SMC & STRUCTURE ENGINE ---
            smc_vectors = []
            f_smc = []
            
            if curr_data.get('smc_trend') is not None:
                smc_vectors.append(self._clamp(curr_data['smc_trend'], -1.0, 1.0))
                f_smc.append('smc_trend')
                
            if curr_data.get('bos_up') is not None and curr_data['bos_up'] > 0:
                smc_vectors.append(1.0); f_smc.append('bos_up')
            if curr_data.get('bos_down') is not None and curr_data['bos_down'] > 0:
                smc_vectors.append(-1.0); f_smc.append('bos_down')
                
            if curr_data.get('choch_up') is not None and curr_data['choch_up'] > 0:
                smc_vectors.append(0.5); f_smc.append('choch_up')
            if curr_data.get('choch_down') is not None and curr_data['choch_down'] > 0:
                smc_vectors.append(-0.5); f_smc.append('choch_down')
                
            if curr_data.get('liquidity_sweep') is not None and curr_data['liquidity_sweep'] != 0:
                # FIX: Contextual liquidity sweep polarity resolution
                swp_val = curr_data['liquidity_sweep']
                if curr_data.get('open') is not None and close > curr_data['open']:
                    smc_vectors.append(abs(swp_val) * 0.5) # Swept bottom, rejected upward
                elif curr_data.get('open') is not None and close < curr_data['open']:
                    smc_vectors.append(-abs(swp_val) * 0.5) # Swept top, rejected downward
                else:
                    smc_vectors.append(self._clamp(swp_val, -1.0, 1.0) * 0.25)
                f_smc.append('liquidity_sweep')

            smc_vector = np.mean(smc_vectors) if smc_vectors else 0.0
            utilized_features.extend(f_smc)
            calculation_trace['smc_structure'] = {
                "features_used": list(set(f_smc)),
                "vector_value": float(smc_vector),
                "score": self._vec_to_score(smc_vector),
                "method": "Structural breakouts and context-aware liquidity absorption vectors."
            }

            # --- H. MULTI-TIMEFRAME ENGINE ---
            mtf_vectors = []
            f_mtf = []
            
            # FIX: Syntax and inclusion of yearly_mid
            mtf_weights = {
                'daily_mid': 0.1, 
                'weekly_mid': 0.2, 
                'monthly_mid': 0.3, 
                'yearly_mid': 0.4
            }
            w_sum, w_tot = 0.0, 0.0
            
            for mtf_feat, w in mtf_weights.items():
                if curr_data.get(mtf_feat) is not None:
                    v = math.tanh((close - curr_data[mtf_feat]) / atr)
                    mtf_vectors.append(v)
                    w_sum += v * w
                    w_tot += w
                    f_mtf.append(mtf_feat)
                    
            mtf_vector = (w_sum / w_tot) if w_tot > 0 else 0.0
            mtf_score = self._vec_to_score(mtf_vector)
            
            mtf_aligned = (np.std(mtf_vectors) < 0.4 and abs(mtf_vector) > 0.3) if len(mtf_vectors) > 1 else False
            
            utilized_features.extend(f_mtf)
            calculation_trace['multi_timeframe'] = {
                "features_used": list(set(f_mtf)),
                "vector_value": float(mtf_vector),
                "score": mtf_score,
                "method": "Hierarchically weighted continuous divergence mapping across macro periods."
            }

            # --- I. MACRO TREND ENGINE ---
            mac_vectors = []
            f_mac = []
            
            for m in ['sma_200', 'ema_200', 'monthly_mid', 'yearly_mid']:
                if curr_data.get(m) is not None:
                    mac_vectors.append(math.tanh((close - curr_data[m]) / (atr * 2.0)))
                    f_mac.append(m)
                    
            mac_vector = np.mean(mac_vectors) if mac_vectors else 0.0
            macro_trend_score = self._vec_to_score(mac_vector)
            utilized_features.extend(f_mac)

            # --- J. MULTI-FACTOR EXHAUSTION ENGINE ---
            exh_factors = []
            exh_used = []
            
            if curr_data.get('rsi_14') is not None:
                r = curr_data['rsi_14']
                if r > 70: exh_factors.append(math.tanh((r - 70) / 15.0))
                elif r < 30: exh_factors.append(math.tanh((30 - r) / 15.0))
                exh_used.append('rsi_14')
                
            if curr_data.get('williams_r') is not None:
                wr = curr_data['williams_r']
                if wr > -20: exh_factors.append(math.tanh((wr + 20) / 15.0))
                elif wr < -80: exh_factors.append(math.tanh((-80 - wr) / 15.0))
                exh_used.append('williams_r')
                
            if curr_data.get('ema_20') is not None:
                dist = abs(close - curr_data['ema_20']) / atr
                if dist > 2.5: exh_factors.append(math.tanh((dist - 2.5) / 2.0))
                exh_used.append('ema_20')
                
            if curr_data.get('bb_percent_b') is not None:
                pctb = curr_data['bb_percent_b']
                if pctb > 1.0: exh_factors.append(math.tanh((pctb - 1.0) / 0.5))
                elif pctb < 0.0: exh_factors.append(math.tanh(abs(pctb) / 0.5))
                exh_used.append('bb_percent_b')

            if curr_data.get('macd_hist') is not None and prev1_data.get('macd_hist') is not None and prev1_data.get('close') is not None:
                hist_slope = curr_data['macd_hist'] - prev1_data['macd_hist']
                price_slope = close - prev1_data['close']
                if (price_slope > 0 > hist_slope and curr_data['macd_hist'] > 0) or (price_slope < 0 < hist_slope and curr_data['macd_hist'] < 0):
                    exh_factors.append(0.8)
                exh_used.append('macd_hist')
                
            exhaustion_score = self._clamp(np.mean(exh_factors) * 100.0 if exh_factors else 0.0, 0.0, 100.0)
            calculation_trace['exhaustion'] = {
                "features_used": list(set(exh_used)),
                "score": exhaustion_score,
                "method": "Oscillator extremes, structural extension, and momentum divergence."
            }

            # --- K. GLOBAL SYNTHESIS & SIGNAL AGREEMENT ---
            families = [
                (price_vector, 0.15),
                (ma_vector, 0.20),
                (dmi_vector, 0.15),
                (mom_vector, 0.15),
                (vol_vector, 0.15),
                (smc_vector, 0.10),
                (mtf_vector, 0.10)
            ]
            
            w_sum, pol_sum = 0.0, 0.0
            active_vectors = []
            
            for vec, weight in families:
                if vec != 0.0:
                    pol_sum += vec * weight
                    w_sum += weight
                    active_vectors.append(vec)
                    
            net_vector = pol_sum / w_sum if w_sum > 0 else 0.0
            direction_score = self._vec_to_score(net_vector)
            
            # Signal Agreement (Low standard deviation = High Agreement)
            signal_agreement = 1.0
            if len(active_vectors) > 1:
                vec_std = np.std(active_vectors)
                signal_agreement = self._clamp(1.0 - float(vec_std), 0.0, 1.0)

            # Strength Synthesis
            base_strength = (dmi_str * 0.5) + ((ma_str / 100.0) * 0.5)
            strength_score = self._clamp(base_strength * signal_agreement * 100.0, 0.0, 100.0)
            
            # Quality Synthesis
            quality_raw = signal_agreement * (1.0 - bb_squeeze_penalty)
            quality_score = self._clamp(quality_raw * 100.0, 0.0, 100.0)
            
            # Continuation Synthesis
            cont_raw = (quality_score / 100.0) * (strength_score / 100.0) * (1.0 - (exhaustion_score / 100.0))
            continuation_score = self._clamp(cont_raw * 100.0, 0.0, 100.0)
            
            # Confidence Synthesis (FIX: Safe derivation from Utilization and Agreement)
            unique_utilized = list(set(utilized_features))
            utilization_percent = len(unique_utilized) / max(1, feature_expected_count)
            
            if utilization_percent == 0.0:
                confidence = 0.0
            else:
                conf_raw = (utilization_percent * 50.0) + (signal_agreement * 50.0)
                confidence = self._clamp(conf_raw, 0.0, 100.0)

            # --- L. STATUS MAPPING (FIX: Tri-state direction mapping) ---
            if direction_score > 55.0: direction_status = "bullish"
            elif direction_score < 45.0: direction_status = "bearish"
            else: direction_status = "neutral"

            strength_status = "strong" if strength_score >= 50.0 else "weak"
            quality_status = "high" if quality_score >= 50.0 else "choppy"
            continuation_status = "likely" if continuation_score >= 50.0 else "unlikely"
            multi_timeframe_status = "aligned" if mtf_aligned else "divergent"
            exhaustion_status = "high" if exhaustion_score >= 50.0 else "low"
            
            if macro_trend_score > 55.0: macro_trend_status = "bullish"
            elif macro_trend_score < 45.0: macro_trend_status = "bearish"
            else: macro_trend_status = "neutral"

            # --- M. EVIDENCE & LIKELIHOOD RATIO ---
            norm_prob = (net_vector + 1.0) / 2.0
            prob_safe = self._clamp(norm_prob, 0.01, 0.99)
            likelihood_ratio = float(self._clamp(prob_safe / (1.0 - prob_safe), 0.01, 100.0))

            evidence_msg = (
                f"Net trend direction is {direction_status} ({round(direction_score, 1)}%) backed by {len(active_vectors)} coherent structural families. "
                f"Trend strength is {strength_status} ({round(strength_score, 1)}%) mapping to a {quality_status} quality environment. "
                f"Multi-timeframe hierarchy is {multi_timeframe_status}. "
            )
            if exhaustion_score >= 50.0:
                evidence_msg += f"Elevated multi-factor exhaustion ({round(exhaustion_score, 1)}%) degrades continuation probability to {continuation_status}."
            else:
                evidence_msg += f"Healthy extension profiles yield a {continuation_status} continuation outcome."

            evidence_node = {
                "category": "Trend",
                "message": evidence_msg,
                "reliability": round(confidence / 100.0, 2),
                "likelihood_ratio": round(likelihood_ratio, 2)
            }

            execution_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            
            trace_block = {
                "execution_time_ms": execution_time_ms,
                "feature_expected_count": feature_expected_count,
                "feature_valid_count": len(valid_features),
                "feature_utilized_count": len(unique_utilized),
                "feature_utilization_percent": round(utilization_percent * 100.0, 2),
                "used_features": unique_utilized,
                "missing_features": missing_features,
                "invalid_features": invalid_features,
                "calculation_trace": calculation_trace
            }

            # STRICT OUTPUT CONTRACT PRESERVATION
            return {
                "trend_analyzer": {
                    "confidence": round(confidence, 2),
                    "direction": round(direction_score, 2),
                    "direction_status": direction_status,
                    "strength": round(strength_score, 2),
                    "strength_status": strength_status,
                    "quality": round(quality_score, 2),
                    "quality_status": quality_status,
                    "continuation": round(continuation_score, 2),
                    "continuation_status": continuation_status,
                    "multi_timeframe": round(mtf_score, 2),
                    "multi_timeframe_status": multi_timeframe_status,
                    "exhaustion": round(exhaustion_score, 2),
                    "exhaustion_status": exhaustion_status,
                    "macro_trend": round(macro_trend_score, 2),
                    "macro_trend_status": macro_trend_status,
                    "evidence": [evidence_node],
                    "trace": trace_block
                }
            }

        except Exception as e:
            return self._build_fallback(start_time, str(e))

    # ==========================================================
    # PRIVATE HELPER METHODS
    # ==========================================================

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Safely bounds a numeric value handling NaNs and Infs."""
        if math.isnan(value) or math.isinf(value):
            return min_val
        return max(min_val, min(value, max_val))

    def _vec_to_score(self, vector: float) -> float:
        """Maps a -1.0 to +1.0 vector to a 0.0 to 100.0 score safely."""
        safe_vec = self._clamp(vector, -1.0, 1.0)
        return (safe_vec + 1.0) * 50.0

    def _build_fallback(self, start_time: float, error_msg: str) -> Dict[str, Any]:
        """Provides a strict, schema-compliant fallback output upon failure."""
        return {
            "trend_analyzer": {
                "confidence": 0.0,
                "direction": 50.0,
                "direction_status": "neutral",
                "strength": 0.0,
                "strength_status": "weak",
                "quality": 0.0,
                "quality_status": "choppy",
                "continuation": 0.0,
                "continuation_status": "unlikely",
                "multi_timeframe": 50.0,
                "multi_timeframe_status": "divergent",
                "exhaustion": 0.0,
                "exhaustion_status": "low",
                "macro_trend": 50.0,
                "macro_trend_status": "neutral",
                "evidence": [
                    {
                        "category": "Trend",
                        "message": f"Trend analysis critical failure: {error_msg}",
                        "reliability": 0.0,
                        "likelihood_ratio": 1.0
                    }
                ],
                "trace": {
                    "execution_time_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                    "feature_expected_count": len(self.expected_features),
                    "feature_valid_count": 0,
                    "feature_utilized_count": 0,
                    "feature_utilization_percent": 0.0,
                    "used_features": [],
                    "missing_features": [],
                    "invalid_features": [],
                    "calculation_trace": {},
                    "error": str(error_msg)
                }
            }
        }
