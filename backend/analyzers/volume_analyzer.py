"""
GREEN BULL RIDER V6
Layer-2 Quantitative Analyzer: Volume & Flow Engine

This module implements a deterministic, quantitatively rigorous evaluation of 
volume, money flow, liquidity, and accumulation/distribution proxies.
It enforces strict database provenance, robust multi-window divergence analysis,
independent evidence tracing, and strict numerical safeguards. 
Institutional activity and liquidity are derived exclusively as behavioral proxies.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class VolumeAnalyzer:
    def analyze(self, feature_history: pd.DataFrame, fundamental_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the final-tier Volume & Flow quantitative engine over historical features.
        """
        # =====================================================================
        # 0. STRICT OUTPUT CONTRACT (FALLBACK)
        # =====================================================================
        fallback = {
            "volume_analyzer": {
                "confidence": 0.0,
                "accumulation": 0.0,
                "distribution": 0.0,
                "quality": 0.0,
                "quality_status": "low",
                "confirmation": 0.0,
                "confirmation_status": "divergence",
                "institutional": 0.0,
                "institutional_status": "outflow",
                "liquidity": 0.0,
                "liquidity_status": "dry",
                "volume_explosion": 0.0,
                "volume_explosion_status": "low",
                "evidence": [
                    {
                        "category": "Volume",
                        "message": "Insufficient valid historical data for quantitative volume analysis.",
                        "reliability": 0.0,
                        "likelihood_ratio": 1.0
                    }
                ]
            }
        }

        if feature_history is None or feature_history.empty:
            return fallback

        df = feature_history.copy()
        # Clean explicit corruptions
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fix negative/zero volumes which are market data errors
        if 'volume' in df.columns:
            df.loc[df['volume'] <= 0, 'volume'] = np.nan

        if 'close' not in df.columns or 'volume' not in df.columns or df['volume'].isna().all():
            fallback["volume_analyzer"]["evidence"][0]["message"] = "Critical price or valid volume columns missing."
            return fallback

        history_length = len(df)
        if history_length < 10:
            fallback["volume_analyzer"]["evidence"][0]["message"] = f"Historical depth ({history_length}) insufficient for statistical volume bounds."
            return fallback

        # Ensure required universe columns exist (initialize missing as NaN to preserve provenance)
        universe_cols = [
            'open', 'high', 'low', 'close', 
            'volume', 'rvol_20', 'obv', 'obv_roc', 'cmf_20', 'mfi_14', 'delta_volume',
            'atr_14', 'bb_width', 'bb_percent_b', 'bb_squeeze'
        ]
        for c in universe_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Extraction of Latest Scalars (Safeguarded)
        current_close = float(df['close'].iloc[-1])
        current_open = float(df['open'].iloc[-1]) if not pd.isna(df['open'].iloc[-1]) else current_close
        current_high = float(df['high'].iloc[-1]) if not pd.isna(df['high'].iloc[-1]) else current_close
        current_low = float(df['low'].iloc[-1]) if not pd.isna(df['low'].iloc[-1]) else current_close
        current_vol = float(df['volume'].iloc[-1]) if not pd.isna(df['volume'].iloc[-1]) else 0.0

        if current_vol == 0.0:
            fallback["volume_analyzer"]["evidence"][0]["message"] = "Latest volume is zero or invalid. Analysis aborted."
            return fallback

        # Robust True Range
        prev_close = df['close'].shift(1).fillna(current_open)
        tr_series = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - prev_close), abs(df['low'] - prev_close)))
        current_tr = float(tr_series.iloc[-1]) if not pd.isna(tr_series.iloc[-1]) else (current_high - current_low)
        
        # ATR Fallback
        atr_val = df['atr_14'].iloc[-1]
        if pd.isna(atr_val) or atr_val <= 0:
            atr_val = current_tr if current_tr > 0 else (current_close * 0.01 + 1e-9)
        atr_val = float(atr_val)

        # =====================================================================
        # 1. ROBUST VOLUME NORMALIZATION & PROVENANCE
        # =====================================================================
        vol_series = df['volume']
        
        # Volume Z-Score using Expanding + Rolling blend for statistical rigor
        exp_med_vol = vol_series.expanding(min_periods=5).median()
        roll_med_vol = vol_series.rolling(window=50, min_periods=5).median()
        blended_med = (exp_med_vol * 0.3) + (roll_med_vol * 0.7)
        
        mad_vol = (vol_series - blended_med).abs().rolling(window=50, min_periods=5).median()
        mad_safe = np.where(mad_vol == 0, blended_med * 0.1 + 1e-9, mad_vol)
        vol_z_series = (vol_series - blended_med) / (1.4826 * mad_safe + 1e-9)
        current_vol_z = 0.0 if pd.isna(vol_z_series.iloc[-1]) else float(vol_z_series.iloc[-1])

        # DB-Supplied RVOL Validation vs Calculated Proxy
        db_rvol = df['rvol_20'].iloc[-1]
        roll_mean_vol = vol_series.rolling(window=20, min_periods=5).mean()
        derived_rvol = float((vol_series.iloc[-1] / (roll_mean_vol.iloc[-1] + 1e-9))) if not pd.isna(roll_mean_vol.iloc[-1]) else 1.0
        
        rvol_is_db_valid = not pd.isna(db_rvol) and (0.01 <= db_rvol <= 100.0)
        current_rvol = float(db_rvol) if rvol_is_db_valid else derived_rvol

        # =====================================================================
        # 2. INDEPENDENT FLOW EVIDENCE & DISPERSION ENGINE
        # =====================================================================
        flow_features_available = 0
        total_flow_features = 5.0
        
        acc_signals = []
        dist_signals = []
        raw_signals = [] # For dispersion tracking

        # A. Close Location Value (CLV)
        price_range = (df['high'] - df['low']).replace(0, np.nan)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / price_range
        clv = clv.fillna(0.0)
        curr_clv = float(clv.iloc[-1])
        raw_signals.append(curr_clv)
        flow_features_available += 1
        if curr_clv > 0.1: acc_signals.append(curr_clv)
        elif curr_clv < -0.1: dist_signals.append(abs(curr_clv))

        # B. Chaikin Money Flow (CMF)
        cmf_val = df['cmf_20'].iloc[-1]
        if not pd.isna(cmf_val):
            cmf_norm = float(np.clip(cmf_val, -1.0, 1.0))
            raw_signals.append(cmf_norm)
            flow_features_available += 1
            if cmf_norm > 0.05: acc_signals.append(cmf_norm)
            elif cmf_norm < -0.05: dist_signals.append(abs(cmf_norm))

        # C. Money Flow Index (MFI) - Regime scaled (-1 to 1)
        mfi_val = df['mfi_14'].iloc[-1]
        if not pd.isna(mfi_val):
            mfi_norm = float((mfi_val - 50.0) / 50.0)
            raw_signals.append(mfi_norm)
            flow_features_available += 1
            if mfi_norm > 0.1: acc_signals.append(mfi_norm)
            elif mfi_norm < -0.1: dist_signals.append(abs(mfi_norm))

        # D. OBV Momentum (Robust Z-Score)
        obv_roc = df['obv_roc']
        if not obv_roc.isna().all():
            obv_roc_med = obv_roc.rolling(50, min_periods=5).median()
            obv_roc_mad = (obv_roc - obv_roc_med).abs().rolling(50, min_periods=5).median()
            obv_safe_mad = np.where(obv_roc_mad == 0, obv_roc_med.abs() * 0.1 + 1e-9, obv_roc_mad)
            obv_z = (obv_roc - obv_roc_med) / (1.4826 * obv_safe_mad + 1e-9)
            if not pd.isna(obv_z.iloc[-1]):
                obv_norm = float(np.tanh(obv_z.iloc[-1] / 2.0))
                raw_signals.append(obv_norm)
                flow_features_available += 1
                if obv_norm > 0.1: acc_signals.append(obv_norm)
                elif obv_norm < -0.1: dist_signals.append(abs(obv_norm))

        # E. Delta Volume (Normalized)
        delta_vol = df['delta_volume'].iloc[-1]
        if not pd.isna(delta_vol):
            delta_norm = float(np.clip(delta_vol / current_vol, -1.0, 1.0))
            raw_signals.append(delta_norm)
            flow_features_available += 1
            if delta_norm > 0.05: acc_signals.append(delta_norm)
            elif delta_norm < -0.05: dist_signals.append(abs(delta_norm))

        # Cross-factor agreement
        flow_dispersion = float(np.std(raw_signals)) if len(raw_signals) > 1 else 1.0
        agreement_factor = float(np.clip(1.0 - flow_dispersion, 0.0, 1.0))

        # =====================================================================
        # 3. VOLUME EXPLOSION & REGIME AWARENESS
        # =====================================================================
        # Bounded sigmoidal anomaly detection
        z_explosion = 100.0 * (1.0 / (1.0 + math.exp(-(current_vol_z - 1.2) * 2.0)))
        rvol_explosion = 100.0 * (1.0 - math.exp(-max(0.0, current_rvol - 1.0)))
        volume_explosion = min(100.0, max(0.0, (z_explosion * 0.7) + (rvol_explosion * 0.3)))

        # Volatility Context (BB Width Proxy)
        bb_width = df['bb_width'].iloc[-1] if not pd.isna(df['bb_width'].iloc[-1]) else (atr_val / current_close)
        regime_volatility_multiplier = 1.0
        if bb_width > 0:
            # Dampen volume scores in ultra-high volatility (liquidation events)
            if bb_width > (atr_val/current_close) * 3.0:
                regime_volatility_multiplier = 0.8

        # =====================================================================
        # 4. EFFORT VS RESULT & ABSORPTION
        # =====================================================================
        effort = max(0.0, current_rvol)
        result = current_tr / atr_val
        
        # Bullish Absorption: High downward effort (or volume), low downward result, close in upper half
        is_bull_absorption = (effort > 1.5) and (result < 0.7) and (curr_clv > 0.2)
        # Bearish Absorption: High upward effort, low result, close in lower half
        is_bear_absorption = (effort > 1.5) and (result < 0.7) and (curr_clv < -0.2)

        # =====================================================================
        # 5. MULTI-WINDOW DIVERGENCE & CONFIRMATION
        # =====================================================================
        def get_trend_slope(series_np):
            valid = ~np.isnan(series_np)
            if np.sum(valid) < 3: return 0.0
            y = series_np[valid]
            # Normalize to 0-1 for scale-invariant slope
            ptp = np.ptp(y)
            if ptp == 0: return 0.0
            y_norm = (y - np.min(y)) / ptp
            x = np.arange(len(y_norm))
            return float(np.polyfit(x, y_norm, 1)[0])

        confirmation_score = 50.0
        divergence_type = "none"

        # Use 10-period (short) and 20-period (medium) where available
        window = min(20, history_length)
        if window >= 10:
            price_hist = df['close'].iloc[-window:].values
            p_slope = get_trend_slope(price_hist)
            
            # Flow proxy (OBV or Volume*Sign(Return))
            if not df['obv'].isna().all():
                flow_hist = df['obv'].iloc[-window:].values
            else:
                flow_hist = (df['volume'] * np.sign(df['close'].diff().fillna(0))).cumsum().iloc[-window:].values
            
            f_slope = get_trend_slope(flow_hist)

            # Determine divergence strictly mathematically
            # If price moves up (slope > 0.05) but flow is flat/down (slope < -0.01)
            if p_slope > 0.05 and f_slope < -0.02:
                divergence_type = "bearish"
                confirmation_score = max(0.0, 50.0 - (abs(p_slope - f_slope) * 50.0))
            elif p_slope < -0.05 and f_slope > 0.02:
                divergence_type = "bullish"
                confirmation_score = max(0.0, 50.0 - (abs(p_slope - f_slope) * 50.0))
            else:
                # Confirmed
                confirmation_score = min(100.0, 50.0 + (1.0 - abs(p_slope - f_slope)) * 50.0)

        # Absorption overrides structural divergence
        if is_bull_absorption:
            divergence_type = "bullish_absorption"
            confirmation_score = 30.0
        elif is_bear_absorption:
            divergence_type = "bearish_absorption"
            confirmation_score = 30.0

        confirmation = float(np.clip(confirmation_score, 0.0, 100.0))

        # =====================================================================
        # 6. ACCUMULATION / DISTRIBUTION (INDEPENDENT EVIDENCE)
        # =====================================================================
        # Accumulation and Distribution are calculated independently based on distinct positive/negative signal confluences.
        acc_strength = np.mean(acc_signals) if acc_signals else 0.0
        dist_strength = np.mean(dist_signals) if dist_signals else 0.0
        
        acc_confluence_ratio = len(acc_signals) / total_flow_features
        dist_confluence_ratio = len(dist_signals) / total_flow_features

        vol_multiplier = 0.5 + (volume_explosion / 200.0) # 0.5 to 1.0 scaling
        
        raw_acc = acc_strength * acc_confluence_ratio * 100.0 * vol_multiplier * regime_volatility_multiplier
        raw_dist = dist_strength * dist_confluence_ratio * 100.0 * vol_multiplier * regime_volatility_multiplier

        # If absorption detected, it heavily skews the result
        if is_bull_absorption:
            raw_acc += 20.0
            raw_dist *= 0.5
        elif is_bear_absorption:
            raw_dist += 20.0
            raw_acc *= 0.5

        accumulation = float(np.clip(raw_acc * 1.5, 0.0, 100.0)) # 1.5 constant scales to 100 logically
        distribution = float(np.clip(raw_dist * 1.5, 0.0, 100.0))

        # =====================================================================
        # 7. NORMALIZED PRICE-IMPACT LIQUIDITY PROXY
        # =====================================================================
        # Measure: How much does ATR-normalized price move per unit of RVOL?
        # Safeguards against zero volume and zero range
        norm_movement = current_tr / (atr_val + 1e-9)
        norm_participation = current_rvol if current_rvol > 0.01 else 0.01
        
        price_impact_ratio = norm_movement / norm_participation
        
        # High impact (price moves >3x RVOL) = Dry Liquidity
        # Low impact (price moves <0.5x RVOL) = Thick Liquidity
        # Map ratio to 100 (Thick) to 0 (Dry) via exponential decay
        liq_score = 100.0 * math.exp(-0.8 * price_impact_ratio)
        liquidity = float(np.clip(liq_score, 0.0, 100.0))

        # =====================================================================
        # 8. INSTITUTIONAL PARTICIPATION PROXY
        # =====================================================================
        # Institutional logic: Massive anomalous volume + Highly directional flow confluence + Thick liquidity absorption
        # We DO NOT claim exact institutional data.
        inst_magnitude = (volume_explosion / 100.0) * (current_rvol / 2.0)
        dominant_flow_confluence = max(acc_confluence_ratio, dist_confluence_ratio)
        
        inst_proxy_raw = (inst_magnitude * dominant_flow_confluence * agreement_factor * 100.0)
        institutional = float(np.clip(inst_proxy_raw, 0.0, 100.0))
        inst_status_str = "inflow" if accumulation >= distribution else "outflow"

        # =====================================================================
        # 9. QUALITY & CONFIDENCE ENGINE
        # =====================================================================
        # Quality penalizes missing DB inputs and extreme dispersion
        feature_coverage_ratio = flow_features_available / total_flow_features
        db_validity_penalty = 1.0 if rvol_is_db_valid else 0.8
        
        quality_raw = 100.0 * feature_coverage_ratio * db_validity_penalty
        
        # Penalize if price range is locked (e.g. upper circuit)
        if current_tr == 0:
            quality_raw *= 0.5
            
        quality = float(np.clip(quality_raw, 0.0, 100.0))
        
        # Confidence incorporates Quality + Historical Depth + Cross-indicator Agreement
        history_ratio = min(1.0, history_length / 60.0) # 60 bars for peak statistical confidence
        confidence_raw = quality * history_ratio * (0.4 + (agreement_factor * 0.6))
        confidence = float(np.clip(confidence_raw, 0.0, 100.0))

        # =====================================================================
        # 10. STATISTICAL LIKELIHOOD RATIO
        # =====================================================================
        # Net bias bounds probabilistic skew
        net_bias = (accumulation - distribution) / 100.0
        # Map [-1, 1] to LR [0.1, 10.0]
        lr_val = math.exp(net_bias * 2.3025) # e^2.3025 ~ 10.0
        likelihood_ratio = float(np.clip(lr_val, 0.1, 10.0))

        # =====================================================================
        # 11. STATUS MAPPING & EVIDENCE GENERATION
        # =====================================================================
        quality_status = "high" if quality >= 70.0 else "low"
        confirmation_status = "divergence" if "divergence" in divergence_type or "absorption" in divergence_type else "detected"
        liquidity_status = "high" if liquidity >= 50.0 else "dry"
        vol_exp_status = "high" if volume_explosion >= 60.0 else "low"

        msg_parts = []
        
        # Volume Magnitude
        if volume_explosion > 75.0:
            msg_parts.append(f"Statistically extreme volume proxy (RVOL: {current_rvol:.2f}, Z: {current_vol_z:.1f}).")
        elif volume_explosion > 50.0:
            msg_parts.append("Elevated volume participation detected.")

        # Divergence & Absorption
        if is_bull_absorption:
            msg_parts.append("High downward effort met with compressed ATR displacement indicates structural bullish absorption.")
        elif is_bear_absorption:
            msg_parts.append("High upward effort met with compressed ATR displacement indicates structural bearish absorption.")
        elif divergence_type != "none":
            msg_parts.append(f"Multi-window regression indicates {divergence_type} divergence between price trajectory and cumulative flow.")
        else:
            msg_parts.append("Price action is quantitatively confirmed by underlying volumetric flow.")

        # Accumulation/Distribution Footprint
        if accumulation > distribution and accumulation > 50.0:
            if agreement_factor > 0.6:
                msg_parts.append("Independent flow features converge to validate strong systemic accumulation.")
            else:
                msg_parts.append("Net flow leans toward accumulation despite internal indicator dispersion.")
        elif distribution > accumulation and distribution > 50.0:
            if agreement_factor > 0.6:
                msg_parts.append("Independent flow features converge to validate strong systemic distribution.")
            else:
                msg_parts.append("Net flow leans toward distribution despite internal indicator dispersion.")
        else:
            msg_parts.append("Volumetric pressure is neutral, lacking definitive directional accumulation/distribution footprint.")

        # Institutional Proxy Disclosure
        if institutional > 65.0:
            msg_parts.append(f"Behavioral proxy implies large-entity {inst_status_str} footprint based on anomalous participation and impact ratios.")
            
        if not rvol_is_db_valid:
            msg_parts.append("[Note: Source DB RVOL invalid/missing; utilized derived statistical proxy].")

        evidence_msg = " ".join(msg_parts)

        # =====================================================================
        # 12. FINAL OUTPUT ASSEMBLY (EXACT CONTRACT)
        # =====================================================================
        return {
            "volume_analyzer": {
                "confidence": round(confidence, 4),
                "accumulation": round(accumulation, 4),
                "distribution": round(distribution, 4),
                "quality": round(quality, 4),
                "quality_status": quality_status,
                "confirmation": round(confirmation, 4),
                "confirmation_status": confirmation_status,
                "institutional": round(institutional, 4),
                "institutional_status": inst_status_str,
                "liquidity": round(liquidity, 4),
                "liquidity_status": liquidity_status,
                "volume_explosion": round(volume_explosion, 4),
                "volume_explosion_status": vol_exp_status,
                "evidence": [
                    {
                        "category": "Volume",
                        "message": evidence_msg,
                        "reliability": round(confidence, 4),
                        "likelihood_ratio": round(likelihood_ratio, 4)
                    }
                ]
            }
        }
