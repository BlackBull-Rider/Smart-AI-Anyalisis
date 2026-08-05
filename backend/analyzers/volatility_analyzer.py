"""
GREEN BULL RIDER V6
Layer-2 Quantitative Analyzer: Volatility & Regime Engine

This module implements institutional-grade quantitative volatility intelligence.
It respects exact Layer-1 database provenance, avoids look-ahead bias, and extracts
multi-dimensional volatility intelligence (expansion, compression, risk, breakouts)
solely from explicitly provided feature snapshots.

No database calls. No history synthesis. No look-ahead bias.
"""

import math
from typing import Dict, Any, Optional

class VolatilityAnalyzer:
    """
    Evaluates market volatility conditions, breakout probability, and regime stability
    using a strictly provided dictionary of Layer-1 statistical features.
    """
    
    def analyze(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the quantitative volatility analysis engine based on a feature snapshot.
        """
        # =====================================================================
        # 0. STRICT OUTPUT CONTRACT FALLBACK
        # =====================================================================
        fallback = {
            "volatility_analyzer": {
                "confidence": 0.0,
                "risk": 0.0,
                "risk_status": "low",
                "atr": 0.0,
                "atr_status": "contraction",
                "expansion": 0.0,
                "expansion_status": "low",
                "compression": 0.0,
                "compression_status": "weak",
                "breakout": 0.0,
                "breakout_status": "false",
                "historical": 0.0,
                "historical_status": "stable",
                "volatility_quality": 0.0,
                "volatility_quality_status": "erratic",
                "evidence": [
                    {
                        "category": "Volatility",
                        "message": "Insufficient valid feature data for quantitative volatility analysis.",
                        "reliability": 0.0,
                        "likelihood_ratio": 1.0
                    }
                ]
            }
        }

        if not features or not isinstance(features, dict):
            return fallback

        # =====================================================================
        # 1. FEATURE UNIVERSE DEFINITION & PROVENANCE TRACING
        # =====================================================================
        UNIVERSE = [
            'atr', 'atr_14', 'tr', 'true_range', 'natr', 'natr_14', 'historical_volatility', 
            'hv', 'hv_21', 'std', 'std_20', 'variance', 'var_20', 'volatility_ratio',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent_b', 'bb_squeeze', 
            'bb_sqz_mom', 'keltner_upper', 'keltner_middle', 'keltner_lower', 'donchian_upper', 
            'donchian_middle', 'donchian_lower', 'atr_percentile', 'expansion_index', 
            'volatility_osc', 'adaptive_atr', 'atr_stop_dist', 'volatility_regime', 
            'compression', 'expansion', 'parkinson_vol', 'garman_klass', 'rogers_satchell', 
            'yang_zhang', 'choppiness', 'vhf', 'standard_error', 'rei', 'ulcer_index', 
            'chaikin_volatility', 'chaikin_vol', 'adx', 'di_plus', 'di_minus', 'supertrend', 
            'close', 'high', 'low', 'open'
        ]

        ALLOW_NEG = {
            'bb_percent_b', 'bb_sqz_mom', 'expansion_index', 'volatility_osc',
            'rei', 'chaikin_volatility', 'chaikin_vol', 'di_minus', 'di_plus'
        }

        trace = {
            "used": 0,
            "unavailable": 0,
            "invalid": 0,
            "total": len(UNIVERSE)
        }
        
        validated = {}

        # Phase 1: Sanitize, Validate, and Trace Coverage
        for key in UNIVERSE:
            if key not in features:
                trace["unavailable"] += 1
                continue
            
            val = features.get(key)
            if val is None:
                trace["unavailable"] += 1
                continue
                
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    trace["invalid"] += 1
                    continue
                if f < 0 and key not in ALLOW_NEG:
                    trace["invalid"] += 1
                    continue
                    
                validated[key] = f
                trace["used"] += 1
            except (ValueError, TypeError):
                trace["invalid"] += 1

        if trace["used"] == 0:
            return fallback

        def get_val(key: str) -> Optional[float]:
            return validated.get(key)

        def bounded_map(val: float, min_in: float, max_in: float) -> float:
            if val <= min_in: return 0.0
            if val >= max_in: return 100.0
            if max_in == min_in: return 0.0
            return ((val - min_in) / (max_in - min_in)) * 100.0

        def safe_tanh_map(val: float, scale: float = 1.0) -> float:
            return math.tanh(val * scale) * 100.0

        # =====================================================================
        # 2. FEATURE EXTRACTION
        # =====================================================================
        # Scalars
        c_close = get_val('close')
        
        # Volatility core
        atr_pct = get_val('atr_percentile')
        vol_ratio = get_val('volatility_ratio')
        natr = get_val('natr') or get_val('natr_14')
        
        # Bands & Channels
        bb_u = get_val('bb_upper')
        bb_l = get_val('bb_lower')
        kc_u = get_val('keltner_upper')
        kc_l = get_val('keltner_lower')
        dc_u = get_val('donchian_upper')
        dc_l = get_val('donchian_lower')
        bb_width = get_val('bb_width')
        
        # Market context
        bb_sqz_mom = get_val('bb_sqz_mom')
        exp_idx = get_val('expansion_index')
        vol_osc = get_val('volatility_osc')
        adx = get_val('adx')
        chop = get_val('choppiness')
        ulcer = get_val('ulcer_index')
        
        # Explicit classifications
        metric_comp = get_val('compression')
        metric_exp = get_val('expansion')

        # =====================================================================
        # 3. REGIME ENGINE (COMPRESSION / EXPANSION)
        # =====================================================================
        comp_signals = []
        exp_signals = []

        # A. BB / Keltner Relationship (Structural Squeeze)
        if bb_u is not None and bb_l is not None and kc_u is not None and kc_l is not None:
            # Squeeze is active if BB is entirely inside Keltner
            if bb_u < kc_u and bb_l > kc_l:
                comp_signals.append(100.0)
                exp_signals.append(0.0)
            else:
                comp_signals.append(0.0)
                # Expansion magnitude based on how far BB has breached Keltner
                bb_range = bb_u - bb_l
                kc_range = kc_u - kc_l
                if kc_range > 0 and bb_range > kc_range:
                    exp_signals.append(bounded_map(bb_range / kc_range, 1.0, 1.5))

        # B. ATR Percentile & Ratio (Distribution state)
        if atr_pct is not None:
            # Safeguard scale: 0-100 or 0-1
            ap_val = atr_pct * 100.0 if atr_pct <= 1.0 and atr_pct > 0.0 else atr_pct
            ap_val = max(0.0, min(100.0, ap_val))
            comp_signals.append(100.0 - ap_val)
            exp_signals.append(ap_val)

        if vol_ratio is not None:
            comp_signals.append(bounded_map(1.0 - vol_ratio, 0.0, 0.5))
            exp_signals.append(bounded_map(vol_ratio, 1.0, 1.5))

        # C. Explicit Features
        if exp_idx is not None:
            exp_signals.append(max(0.0, safe_tanh_map(exp_idx, 0.1)))
        if vol_osc is not None:
            if vol_osc > 0:
                exp_signals.append(bounded_map(vol_osc, 0.0, 50.0))
            else:
                comp_signals.append(bounded_map(abs(vol_osc), 0.0, 50.0))

        if metric_comp is not None:
            comp_signals.append(bounded_map(metric_comp, 0.0, 100.0 if metric_comp > 1.0 else 1.0))
        if metric_exp is not None:
            exp_signals.append(bounded_map(metric_exp, 0.0, 100.0 if metric_exp > 1.0 else 1.0))

        comp_score = sum(comp_signals) / len(comp_signals) if comp_signals else 0.0
        exp_score = sum(exp_signals) / len(exp_signals) if exp_signals else 0.0

        # =====================================================================
        # 4. CROSS-ESTIMATOR CONSISTENCY ENGINE (QUALITY/STABILITY)
        # =====================================================================
        est_keys = [
            'parkinson_vol', 'garman_klass', 'rogers_satchell', 'yang_zhang', 
            'historical_volatility', 'hv', 'hv_21', 'std', 'std_20', 'atr_14'
        ]
        
        estimators = [get_val(k) for k in est_keys if get_val(k) is not None and get_val(k) > 0]
        
        cv = 0.0  # Coefficient of Variation
        if len(estimators) >= 2:
            mean_e = sum(estimators) / len(estimators)
            var_e = sum((e - mean_e)**2 for e in estimators) / len(estimators)
            if mean_e > 0:
                cv = math.sqrt(var_e) / mean_e

        # Historical Stability is a function of estimator convergence + choppiness
        hist_signals = []
        if len(estimators) >= 2:
            # High dispersion (CV > 0.4) = erratic
            hist_signals.append(100.0 - bounded_map(cv, 0.0, 0.4))
            
        if chop is not None:
            # Choppiness > 61.8 indicates erratic trendless state
            hist_signals.append(100.0 - bounded_map(chop, 38.2, 61.8))
            
        historical = sum(hist_signals) / len(hist_signals) if hist_signals else 50.0

        # =====================================================================
        # 5. BREAKOUT / EXPANSION FUSION ENGINE
        # =====================================================================
        brk_signals = []
        
        # Donchian Channel Pressure (Price location relative to extremes)
        if c_close is not None and dc_u is not None and dc_l is not None:
            dc_range = dc_u - dc_l
            if dc_range > 0:
                dc_loc = (c_close - dc_l) / dc_range
                # Extremes (near 1 or near 0) imply structural breakout pressure
                if dc_loc > 0.9 or dc_loc < 0.1:
                    brk_signals.append(100.0)
                else:
                    brk_signals.append(bounded_map(abs(dc_loc - 0.5), 0.25, 0.45))

        # ADX trend strength confirmation
        if adx is not None:
            brk_signals.append(bounded_map(adx, 20.0, 40.0))

        # Bollinger Squeeze Momentum Release
        if bb_sqz_mom is not None:
            brk_signals.append(bounded_map(abs(bb_sqz_mom), 0.0, 0.5))
            
        # Base expansion contribution
        brk_signals.append(exp_score)

        breakout = sum(brk_signals) / len(brk_signals) if brk_signals else 0.0

        # =====================================================================
        # 6. VOLATILITY RISK ENGINE
        # =====================================================================
        risk_signals = []
        
        # Market-volatility absolute risk via NATR (e.g. > 4% is very high risk)
        if natr is not None:
            risk_signals.append(bounded_map(natr, 0.0, 5.0))
            
        # Downside/stress risk
        if ulcer is not None:
            risk_signals.append(bounded_map(ulcer, 0.0, 10.0))
            
        # Regime risk via expansion magnitude
        risk_signals.append(exp_score)
        
        # Uncertainty risk via estimator dispersion
        if len(estimators) >= 2:
            risk_signals.append(bounded_map(cv, 0.1, 0.5))

        risk = sum(risk_signals) / len(risk_signals) if risk_signals else 50.0
        
        # Isolated ATR status
        atr_score = atr_pct if atr_pct is not None else exp_score

        # =====================================================================
        # 7. QUALITY & CONFIDENCE ENGINE
        # =====================================================================
        # Coverage is proportional to total defined universe
        coverage_ratio = trace["used"] / trace["total"]
        
        # Quality: Reflects data density and structural agreement (lack of high CV)
        quality_raw = (coverage_ratio * 100.0)
        if len(estimators) >= 2:
            quality_raw *= (1.0 - bounded_map(cv, 0.2, 1.0)/100.0)
        quality = min(100.0, max(0.0, quality_raw))
        
        # Confidence: Scales up if we have sufficient orthogonal feature groups
        # We don't demand 100% of 50 features. Having ~15-20 valid orthogonal ones gives 100% confidence.
        EXPECTED_ROBUST_COUNT = 15.0
        conf_ratio = min(trace["used"] / EXPECTED_ROBUST_COUNT, 1.0)
        
        confidence = conf_ratio * 100.0
        if cv > 0.5:
            confidence -= 15.0 # Contradiction penalty
        confidence = min(100.0, max(0.0, confidence))

        # =====================================================================
        # 8. STATISTICAL LIKELIHOOD RATIO
        # =====================================================================
        # Heuristic probabilistic bounds [0.1, 10.0] derived from overall signal strength
        # Measures the quantitative intensity of the current state vs neutral.
        intensity = (exp_score + comp_score + breakout) / 300.0
        lr_mapped = 1.0
        if intensity > 0.6:
            lr_mapped = 1.0 + ((intensity - 0.6) / 0.4) * 9.0 
        elif intensity < 0.4:
            lr_mapped = 1.0 - ((0.4 - intensity) / 0.4) * 0.9
            
        likelihood_ratio = max(0.1, min(10.0, lr_mapped))

        # =====================================================================
        # 9. STATUS MAPPING
        # =====================================================================
        risk_status = "high" if risk >= 60.0 else "low"
        atr_status = "expansion" if atr_score >= 50.0 else "contraction"
        expansion_status = "high" if exp_score >= 60.0 else "low"
        compression_status = "strong" if comp_score >= 60.0 else "weak"
        breakout_status = "high" if breakout >= 60.0 else "false"
        historical_status = "stable" if historical >= 50.0 else "erratic"
        volatility_quality_status = "stable" if quality >= 50.0 else "erratic"

        # =====================================================================
        # 10. EVIDENCE GENERATION
        # =====================================================================
        evidence = []
        
        # Strictly formatted coverage string
        coverage_msg = f"Used {trace['used']}/{trace['total']} declared volatility features; {trace['unavailable']} unavailable; {trace['invalid']} invalid."
        evidence.append({
            "category": "FeatureCoverage",
            "message": coverage_msg,
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": 1.0
        })
        
        # Market Intelligence Analysis String
        analysis = []
        if exp_score >= 60.0:
            analysis.append(f"Volatility regime exhibits structural expansion (Exp Score: {exp_score:.1f}).")
        elif comp_score >= 60.0:
            analysis.append(f"Statistically significant structural compression detected (Comp Score: {comp_score:.1f}).")
        else:
            analysis.append("Volatility regime is structurally neutral.")
            
        if breakout >= 60.0:
            analysis.append("Band/channel proximity and momentum metrics support developing breakout volatility.")
            
        if historical < 40.0:
            analysis.append("Cross-estimator convergence is erratic, indicating transitional or unstable historical pricing.")
        elif len(estimators) >= 2:
            analysis.append("Cross-estimator convergence validates historical regime stability.")
            
        evidence.append({
            "category": "Volatility",
            "message": " ".join(analysis),
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": round(likelihood_ratio, 4)
        })

        # =====================================================================
        # 11. FINAL OUTPUT ASSEMBLY (EXACT CONTRACT)
        # =====================================================================
        return {
            "volatility_analyzer": {
                "confidence": round(float(confidence), 4),
                "risk": round(float(risk), 4),
                "risk_status": risk_status,
                "atr": round(float(atr_score), 4),
                "atr_status": atr_status,
                "expansion": round(float(exp_score), 4),
                "expansion_status": expansion_status,
                "compression": round(float(comp_score), 4),
                "compression_status": compression_status,
                "breakout": round(float(breakout), 4),
                "breakout_status": breakout_status,
                "historical": round(float(historical), 4),
                "historical_status": historical_status,
                "volatility_quality": round(float(quality), 4),
                "volatility_quality_status": volatility_quality_status,
                "evidence": evidence
            }
        }
