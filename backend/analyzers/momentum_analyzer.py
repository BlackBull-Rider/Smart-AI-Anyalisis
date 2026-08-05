"""
GREEN BULL RIDER V6
Layer-2 Quantitative Analyzer: Momentum Intelligence Engine

This module implements institutional-grade quantitative momentum intelligence.
It respects exact Layer-1 database provenance, avoids look-ahead bias, and extracts
multi-dimensional momentum insights solely from explicitly provided feature snapshots.

Architectural Guarantees:
- Strict Exact-Key Provenance: No aliases, no 'or' fallbacks, no missing-feature substitution.
- Independent Evidence: Each valid feature contributes orthogonally to its respective group.
- Deterministic Confidence: Derived from orthogonal group coverage, data validity, and signal agreement.
- Zero History Synthesis: Operates exclusively on the provided scalar snapshot.
"""

import math
from typing import Dict, Any, Optional, List

class MomentumAnalyzer:
    """
    Evaluates market momentum conditions, directional strength, divergence, and 
    acceleration strictly using a provided dictionary of Layer-1 features.
    """

    def analyze(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the quantitative momentum analysis engine based on a feature snapshot.
        """
        # =====================================================================
        # 0. STRICT OUTPUT CONTRACT FALLBACK
        # =====================================================================
        fallback = {
            "momentum_analyzer": {
                "confidence": 0.0,
                "strength": 0.0,
                "strength_status": "weak",
                "acceleration": 0.0,
                "acceleration_status": "low",
                "rsi": 0.0,
                "rsi_status": "bearish",
                "macd": 0.0,
                "macd_status": "bearish",
                "slowdown": 0.0,
                "slowdown_status": "high",
                "divergence": 0.0,
                "divergence_status": "none",
                "evidence": [
                    {
                        "category": "Momentum",
                        "message": "Insufficient valid Layer-1 momentum features for quantitative analysis.",
                        "reliability": 0.0,
                        "likelihood_ratio": 1.0
                    }
                ]
            }
        }

        if not features or not isinstance(features, dict):
            return fallback

        # =====================================================================
        # 1. FEATURE UNIVERSE DEFINITION & EXACT PROVENANCE TRACING
        # =====================================================================
        UNIVERSE = [
            'momentum', 'roc', 'roc_10', 'roc_20', 'rsi', 'rsi_14', 'rsi_slope', 
            'macd', 'macd_signal', 'macd_histogram', 'macd_slope', 'adx', 'di_plus', 'di_minus',
            'cci', 'trix', 'ppo', 'ppo_signal', 'ppo_histogram', 'dpo', 
            'stochastic', 'stochastic_k', 'stochastic_d', 'stoch_rsi', 'stoch_rsi_k', 'stoch_rsi_d', 
            'williams_r', 'ultimate_oscillator',
            'momentum_slope', 'roc_slope', 'trix_slope', 'ppo_slope',
            'rsi_divergence', 'macd_divergence', 'momentum_divergence', 'roc_divergence', 
            'stochastic_divergence', 'bullish_divergence', 'bearish_divergence',
            'momentum_acceleration', 'rsi_acceleration', 'macd_acceleration', 'roc_acceleration', 
            'momentum_zscore', 'roc_zscore', 'rsi_percentile', 'momentum_percentile', 
            'momentum_volatility', 'momentum_std', 'momentum_variance',
            'close', 'open', 'high', 'low'
        ]

        trace = {
            "declared": len(UNIVERSE),
            "unavailable": 0,
            "invalid": 0,
            "valid": 0,
            "used": 0
        }

        validated = {}
        used_keys = set()

        # Strict Sanitization and Validation
        for key in UNIVERSE:
            if key not in features:
                trace["unavailable"] += 1
                continue
            
            val = features.get(key)
            if val is None:
                trace["unavailable"] += 1
                continue
            
            try:
                if isinstance(val, (bool, str)):
                    trace["invalid"] += 1
                    continue
                    
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    trace["invalid"] += 1
                    continue
                
                validated[key] = f
                trace["valid"] += 1
            except (ValueError, TypeError):
                trace["invalid"] += 1

        if not validated:
            return fallback

        # Single unified accessor that strictly traces usage
        def get_val(key: str) -> Optional[float]:
            if key in validated:
                used_keys.add(key)
                return validated[key]
            return None

        # Bounded normalization helpers
        def safe_norm(val: float, min_in: float, max_in: float) -> float:
            if val <= min_in: return 0.0
            if val >= max_in: return 100.0
            if max_in == min_in: return 0.0
            return ((val - min_in) / (max_in - min_in)) * 100.0

        def safe_tanh_map(val: float, scale: float = 1.0) -> float:
            return math.tanh(val * scale) * 100.0

        # =====================================================================
        # 2. ORTHOGONAL EVIDENCE GROUPS (STRENGTH & BIAS)
        # =====================================================================
        directional_votes: List[float] = []
        intensity_votes: List[float] = []
        groups_present = 0

        # --- Group 1: RSI Engine ---
        rsi_vals = []
        v_rsi = get_val('rsi')
        v_rsi_14 = get_val('rsi_14')
        
        if v_rsi is not None: rsi_vals.append(v_rsi)
        if v_rsi_14 is not None: rsi_vals.append(v_rsi_14)

        rsi_score = 0.0
        rsi_status = "bearish"

        if rsi_vals:
            groups_present += 1
            avg_rsi = sum(rsi_vals) / len(rsi_vals)
            rsi_score = max(0.0, min(100.0, avg_rsi))
            rsi_status = "bullish" if rsi_score > 50.0 else "bearish"
            
            directional_votes.append(1.0 if rsi_score > 50.0 else -1.0)
            intensity_votes.append(abs(rsi_score - 50.0) * 2.0)

        # --- Group 2: MACD Engine ---
        v_macd = get_val('macd')
        v_macd_sig = get_val('macd_signal')
        v_macd_hist = get_val('macd_histogram')
        
        macd_score = 0.0
        macd_status = "bearish"
        macd_signals_used = 0
        macd_dir_sum = 0.0
        macd_int_sum = 0.0

        if v_macd_hist is not None:
            macd_signals_used += 1
            macd_dir_sum += (1.0 if v_macd_hist > 0 else -1.0)
            macd_int_sum += abs(safe_tanh_map(v_macd_hist, 10.0))
            macd_status = "bullish" if v_macd_hist > 0 else "bearish"
            
        if v_macd is not None and v_macd_sig is not None:
            macd_signals_used += 1
            macd_dir_sum += (1.0 if v_macd > v_macd_sig else -1.0)
            macd_int_sum += abs(safe_tanh_map(v_macd - v_macd_sig, 5.0))
            
        if v_macd is not None:
            macd_signals_used += 1
            macd_dir_sum += (1.0 if v_macd > 0 else -1.0)
            macd_int_sum += abs(safe_tanh_map(v_macd, 5.0))

        if macd_signals_used > 0:
            groups_present += 1
            macd_score = macd_int_sum / macd_signals_used
            net_macd_dir = macd_dir_sum / macd_signals_used
            
            directional_votes.append(1.0 if net_macd_dir > 0 else -1.0)
            intensity_votes.append(macd_score)

        # --- Group 3: Raw Momentum / ROC ---
        roc_signals = []
        for k in ['momentum', 'roc', 'roc_10', 'roc_20']:
            v = get_val(k)
            if v is not None:
                roc_signals.append((1.0 if v > 0 else -1.0, abs(safe_tanh_map(v, 0.5))))
                
        for k in ['momentum_zscore', 'roc_zscore']:
            v = get_val(k)
            if v is not None:
                roc_signals.append((1.0 if v > 0 else -1.0, abs(safe_tanh_map(v, 0.5))))

        if roc_signals:
            groups_present += 1
            avg_dir = sum(d for d, _ in roc_signals) / len(roc_signals)
            avg_int = sum(i for _, i in roc_signals) / len(roc_signals)
            directional_votes.append(1.0 if avg_dir > 0 else -1.0)
            intensity_votes.append(avg_int)

        # --- Group 4: Trend Directional Strength (ADX/DI) ---
        v_adx = get_val('adx')
        v_di_plus = get_val('di_plus')
        v_di_minus = get_val('di_minus')
        
        if v_di_plus is not None and v_di_minus is not None:
            groups_present += 1
            di_dir = 1.0 if v_di_plus > v_di_minus else -1.0
            directional_votes.append(di_dir)
            
            di_diff = abs(v_di_plus - v_di_minus)
            adx_val = v_adx if v_adx is not None else 0.0
            adx_intensity = safe_norm(adx_val, 15.0, 40.0)
            intensity_votes.append((adx_intensity * 0.7) + (safe_norm(di_diff, 0.0, 30.0) * 0.3))

        # --- Group 5: Core Oscillators ---
        osc_signals = []
        
        v_stoch_k = get_val('stochastic_k')
        v_stoch = get_val('stochastic')
        # Independent processing without 'or' aliases
        if v_stoch_k is not None: osc_signals.append((1.0 if v_stoch_k > 50.0 else -1.0, abs(v_stoch_k - 50.0) * 2.0))
        if v_stoch is not None: osc_signals.append((1.0 if v_stoch > 50.0 else -1.0, abs(v_stoch - 50.0) * 2.0))
            
        v_cci = get_val('cci')
        if v_cci is not None: osc_signals.append((1.0 if v_cci > 0 else -1.0, abs(safe_tanh_map(v_cci, 0.01))))
            
        v_will = get_val('williams_r')
        if v_will is not None: osc_signals.append((1.0 if v_will > -50.0 else -1.0, abs(v_will + 50.0) * 2.0))
            
        v_ult = get_val('ultimate_oscillator')
        if v_ult is not None: osc_signals.append((1.0 if v_ult > 50.0 else -1.0, abs(v_ult - 50.0) * 2.0))

        if osc_signals:
            groups_present += 1
            avg_dir = sum(d for d, _ in osc_signals) / len(osc_signals)
            avg_int = sum(i for _, i in osc_signals) / len(osc_signals)
            directional_votes.append(1.0 if avg_dir > 0 else -1.0)
            intensity_votes.append(avg_int)

        # --- Base Net Bias & Agreement ---
        net_bias = sum(directional_votes) / len(directional_votes) if directional_votes else 0.0
        base_intensity = sum(intensity_votes) / len(intensity_votes) if intensity_votes else 0.0
        
        # Agreement factor penalizes the final intensity if orthogonal groups contradict
        agreement_factor = abs(net_bias)
        
        strength_score = base_intensity * (0.4 + (agreement_factor * 0.6))
        strength = max(0.0, min(100.0, strength_score))
        strength_status = "strong" if strength >= 60.0 else "weak"

        # =====================================================================
        # 3. ACCELERATION ENGINE
        # =====================================================================
        accel_signals = []
        accel_dirs = []
        
        # Explicitly declared scale mapping for independent acceleration keys
        acc_maps = {
            'rsi_slope': 0.2, 'rsi_acceleration': 0.2,
            'macd_slope': 2.0, 'macd_acceleration': 2.0,
            'momentum_slope': 0.5, 'roc_slope': 0.5, 'momentum_acceleration': 0.5,
            'ppo_slope': 5.0, 'trix_slope': 5.0, 'roc_acceleration': 0.5
        }
        
        for k, scale in acc_maps.items():
            val = get_val(k)
            if val is not None:
                accel_dirs.append(1.0 if val > 0 else -1.0)
                accel_signals.append(abs(safe_tanh_map(val, scale)))

        if accel_signals:
            groups_present += 1

        accel_intensity = sum(accel_signals) / len(accel_signals) if accel_signals else 0.0
        accel_bias = sum(accel_dirs) / len(accel_dirs) if accel_dirs else 0.0
        
        acceleration = max(0.0, min(100.0, accel_intensity))
        acceleration_status = "high" if acceleration >= 50.0 else "low"

        # =====================================================================
        # 4. SLOWDOWN ENGINE (Momentum Divergence from Trajectory)
        # =====================================================================
        slowdown = 0.0
        
        # Structural Slowdown: Primary momentum is strong, but acceleration firmly opposes it
        if abs(net_bias) > 0.3 and len(accel_dirs) > 0:
            if (net_bias > 0 and accel_bias < -0.2) or (net_bias < 0 and accel_bias > 0.2):
                slowdown_base = (abs(net_bias) * 50.0) + (abs(accel_bias) * 50.0)
                slowdown = max(0.0, min(100.0, slowdown_base))
                
        # Oscillator Extreme Slowdown (MACD histogram decaying in overbought/oversold)
        if v_macd_hist is not None and v_macd is not None and rsi_vals:
            avg_r = sum(rsi_vals) / len(rsi_vals)
            if (avg_r > 70 and v_macd_hist < 0 and v_macd > 0) or (avg_r < 30 and v_macd_hist > 0 and v_macd < 0):
                slowdown = max(slowdown, 75.0)

        slowdown_status = "high" if slowdown >= 50.0 else "low"

        # =====================================================================
        # 5. DIVERGENCE ENGINE
        # =====================================================================
        div_signals = []
        div_flags = [
            'rsi_divergence', 'macd_divergence', 'momentum_divergence', 'roc_divergence',
            'stochastic_divergence', 'bullish_divergence', 'bearish_divergence'
        ]
        
        div_bias = 0.0
        for df in div_flags:
            d_val = get_val(df)
            if d_val is not None and abs(d_val) > 0:
                div_signals.append(abs(d_val))
                if 'bullish' in df or d_val > 0: div_bias += 1.0
                if 'bearish' in df or d_val < 0: div_bias -= 1.0
                
        divergence = 0.0
        if div_signals:
            groups_present += 1
            div_raw = sum(div_signals) * 20.0 
            divergence = max(0.0, min(100.0, div_raw))
            
        divergence_status = "strong" if divergence >= 50.0 else "none"

        # =====================================================================
        # 6. QUALITY & CONFIDENCE ENGINE
        # =====================================================================
        trace["used"] = len(used_keys)
        
        # 7 total possible orthogonal structural groups
        EXPECTED_GROUPS = 6.0 
        coverage_ratio = min(groups_present / EXPECTED_GROUPS, 1.0)
        
        # Confidence incorporates orthogonal coverage ratio + internal agreement
        confidence_raw = (coverage_ratio * 100.0) * (0.6 + (agreement_factor * 0.4))
        
        # Contradiction penalty if we have solid coverage but complete directional chaos
        if groups_present >= 3 and agreement_factor < 0.2:
            confidence_raw -= 20.0
            
        confidence = max(0.0, min(100.0, confidence_raw))

        # =====================================================================
        # 7. STATISTICAL LIKELIHOOD RATIO
        # =====================================================================
        # Bounded probabilistic skew [0.1, 10.0] derived from normalized consensus
        lr_exponent = net_bias * (coverage_ratio * agreement_factor * 2.3025) 
        likelihood_ratio = math.exp(lr_exponent)
        likelihood_ratio = max(0.1, min(10.0, likelihood_ratio))

        # =====================================================================
        # 8. EVIDENCE GENERATION
        # =====================================================================
        evidence = []
        
        coverage_msg = f"Declared {trace['declared']} momentum features; {trace['valid']} valid, {trace['used']} utilized across {groups_present} orthogonal groups ({trace['unavailable']} unavailable, {trace['invalid']} invalid)."
        evidence.append({
            "category": "FeatureCoverage",
            "message": coverage_msg,
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": 1.0
        })
        
        analysis = []
        
        bias_str = "bullish" if net_bias > 0 else "bearish"
        if strength >= 60.0:
            analysis.append(f"Institutional quantitative momentum indicates strong {bias_str} pressure (Strength: {strength:.1f}).")
        elif strength >= 40.0:
            analysis.append(f"Momentum exhibits moderate {bias_str} bias.")
        else:
            analysis.append("Momentum structure is currently weak or deeply contradicted across independent indicators.")
            
        if slowdown >= 50.0:
            analysis.append(f"Momentum acceleration contradicts primary trajectory, indicating significant structural slowdown (Score: {slowdown:.1f}).")
        elif acceleration >= 60.0:
            acc_dir_str = "increasing" if accel_bias * net_bias > 0 else "reversing"
            analysis.append(f"Momentum velocity is {acc_dir_str} rapidly.")
            
        if divergence >= 50.0:
            div_dir_str = "bullish" if div_bias > 0 else ("bearish" if div_bias < 0 else "structural")
            analysis.append(f"Cross-indicator analysis detects {div_dir_str} divergence from recent price extremes.")

        if groups_present >= 3:
            if agreement_factor > 0.8:
                analysis.append("Orthogonal momentum sub-groups show high systemic convergence.")
            elif agreement_factor < 0.3:
                analysis.append("Significant contradiction detected between trend strength and oscillator momentum groups.")

        evidence.append({
            "category": "Momentum",
            "message": " ".join(analysis),
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": round(likelihood_ratio, 4)
        })

        # =====================================================================
        # 9. FINAL OUTPUT ASSEMBLY (EXACT CONTRACT)
        # =====================================================================
        return {
            "momentum_analyzer": {
                "confidence": round(float(confidence), 4),
                "strength": round(float(strength), 4),
                "strength_status": strength_status,
                "acceleration": round(float(acceleration), 4),
                "acceleration_status": acceleration_status,
                "rsi": round(float(rsi_score), 4),
                "rsi_status": rsi_status,
                "macd": round(float(macd_score), 4),
                "macd_status": macd_status,
                "slowdown": round(float(slowdown), 4),
                "slowdown_status": slowdown_status,
                "divergence": round(float(divergence), 4),
                "divergence_status": divergence_status,
                "evidence": evidence
            }
        }
