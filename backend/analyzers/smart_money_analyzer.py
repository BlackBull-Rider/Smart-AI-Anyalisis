"""
GREEN BULL RIDER V7 (ULTIMATE)
Layer-2 Quantitative Analyzer: Smart Money / SMC Engine

Architectural Upgrades in V7:
1. Dynamic Weighting: Structure shifts are weighted by their actual quantitative scores.
2. Exponential Time Decay: Older OBs and FVGs naturally lose strength via Half-life math.
3. Non-Linear Synergy: Aligned orthogonal elements create a "Supernova" multiplier effect.
4. Conflict/Noise Filter: Strict footprint/structure divergence penalization.
"""

import math
from typing import Dict, Any, Optional, List

class SmartMoneyAnalyzer:
    def analyze(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # =====================================================================
        # 0. STRICT OUTPUT CONTRACT FALLBACK (Crash-Proof Net)
        # =====================================================================
        fallback = {
            "smart_money_analyzer": {
                "confidence": 0.0,
                "sweep": 0.0,
                "efficiency": 0.0,
                "structure": 0.0,
                "structure_status": "bearish",
                "liquidity": 0.0,
                "liquidity_status": "bearish",
                "order_block": 0.0,
                "order_block_status": "bearish",
                "fair_value_gap": 0.0,
                "fair_value_gap_status": "bearish",
                "footprint": 0.0,
                "footprint_status": "outflow",
                "zone": 0.0,
                "zone_status": "premium",
                "evidence": [{
                    "category": "SMC",
                    "message": "Insufficient valid Layer-1 SMC features for quantitative analysis.",
                    "reliability": 0.0,
                    "likelihood_ratio": 1.0
                }]
            }
        }

        if not features or not isinstance(features, dict):
            return fallback

        # =====================================================================
        # 1. EXACT DB FEATURE UNIVERSE DEFINITION
        # =====================================================================
        UNIVERSE = [
            'bos_up', 'bos_down', 'choch_up', 'choch_down', 'smc_trend', 
            'mss_bullish', 'mss_bearish', 'sm_bos', 'sm_choch', 'sm_mss', 
            'sm_bos_score', 'sm_choch_score', 'sm_trend_score', 
            
            'buy_side_liquidity', 'sell_side_liquidity', 'equal_highs', 'equal_lows', 
            'sm_buy_side_liquidity', 'sm_sell_side_liquidity', 'sm_liquidity_score', 
            
            'liquidity_sweep', 'turtle_soup', 'judas_swing', 'liquidity_sweep_candle',
            
            'fresh_ob', 'mitigated_ob', 'invalidated_ob', 'ob_high', 'ob_low', 
            'breaker_block', 'sm_fresh_ob', 'sm_mitigated_ob', 'order_block_age', 'order_block_score',
            
            'bullish_fvg', 'bearish_fvg', 'fvg_mitigated', 'fvg_active', 'bpr', 
            'sm_active_fvg', 'sm_mitigated_fvg', 'sm_fvg_score', 'fvg_age',
            
            'buy_volume', 'sell_volume', 'delta_volume', 'stopping_volume', 
            'buying_pressure', 'selling_pressure', 'institutional_body', 'institutional_wick', 
            'institutional_imbalance', 'institutional_pressure', 'effort_result',
            'smart_money_index', 'vpin', 'sm_institutional_score', 'sm_smart_money_score',
            
            'premium_level', 'equilibrium', 'discount_level', 'sm_premium_zone', 
            'sm_discount_zone', 'sm_equilibrium',
            
            'close', 'open', 'high', 'low'
        ]

        trace = {"declared": len(UNIVERSE), "unavailable": 0, "invalid": 0, "valid": 0, "used": 0}
        validated = {}
        used_keys = set()

        for key in UNIVERSE:
            if key not in features or features[key] is None:
                trace["unavailable"] += 1
                continue
            try:
                if isinstance(features[key], (bool, str)):
                    trace["invalid"] += 1
                    continue
                f = float(features[key])
                if math.isnan(f) or math.isinf(f):
                    trace["invalid"] += 1
                    continue
                validated[key] = f
                trace["valid"] += 1
            except (ValueError, TypeError):
                trace["invalid"] += 1

        if trace["valid"] == 0:
            return fallback

        def get_val(key: str, default: float = None) -> Optional[float]:
            if key in validated:
                used_keys.add(key)
                return validated[key]
            return default

        def safe_norm(val: float, min_in: float, max_in: float) -> float:
            if val <= min_in: return 0.0
            if val >= max_in: return 100.0
            if max_in == min_in: return 0.0
            return ((val - min_in) / (max_in - min_in)) * 100.0

        def time_decay_multiplier(age: float, half_life: float = 15.0) -> float:
            """Exponential decay: older structures lose quantitative power."""
            if age <= 0: return 1.0
            return math.exp(-0.693 * (age / half_life))

        c_close = get_val('close', 0.0)
        active_modules = set()
        directional_votes = []

        # =====================================================================
        # 2. ORTHOGONAL MODULES (WITH V7 UPGRADES)
        # =====================================================================

        # --- A. STRUCTURE ENGINE (Dynamic Weighting) ---
        struct_votes = []
        struct_intensity = []
        
        # Upgrades: Base shift gets multiplied by its actual score
        bos_score = get_val('sm_bos_score', 50.0) / 100.0
        choch_score = get_val('sm_choch_score', 50.0) / 100.0
        
        if get_val('bos_up', 0) > 0: struct_votes.append(1.0 * bos_score)
        if get_val('choch_up', 0) > 0: struct_votes.append(1.0 * choch_score)
        if get_val('bos_down', 0) > 0: struct_votes.append(-1.0 * bos_score)
        if get_val('choch_down', 0) > 0: struct_votes.append(-1.0 * choch_score)

        t_trend = get_val('smc_trend', 0)
        if t_trend != 0: struct_votes.append(1.0 if t_trend > 0 else -1.0)

        struct_intensity.append(safe_norm(get_val('sm_trend_score', 50.0), 0.0, 100.0))

        net_struct = sum(struct_votes) / max(1, len(struct_votes)) if struct_votes else 0.0
        structure_status = "bullish" if net_struct > 0 else "bearish"
        
        if struct_votes or struct_intensity:
            active_modules.add("structure")
            if net_struct != 0: directional_votes.append(net_struct)
            
        base_struct = sum(struct_intensity) / max(1, len(struct_intensity))
        structure = max(0.0, min(100.0, base_struct * (0.3 + 0.7 * abs(net_struct))))

        # --- B. LIQUIDITY ENGINE ---
        liq_votes = []
        
        v_bsl = get_val('buy_side_liquidity')
        v_ssl = get_val('sell_side_liquidity')
        if v_bsl is not None and v_ssl is not None and c_close > 0:
            if abs(v_bsl - c_close) < abs(v_ssl - c_close): liq_votes.append(1.0)
            else: liq_votes.append(-1.0)

        if get_val('equal_highs', 0) > 0: liq_votes.append(1.0)
        if get_val('equal_lows', 0) > 0: liq_votes.append(-1.0)

        liq_intensity = safe_norm(get_val('sm_liquidity_score', 50.0), 0.0, 100.0)
        net_liq = sum(liq_votes) / max(1, len(liq_votes)) if liq_votes else 0.0
        liquidity_status = "bullish" if net_liq > 0 else "bearish"

        if liq_votes or liq_intensity > 0:
            active_modules.add("liquidity")
            if net_liq != 0: directional_votes.append(net_liq)
            
        liquidity = max(0.0, min(100.0, liq_intensity * (0.5 + 0.5 * abs(net_liq))))

        # --- C. SWEEP ENGINE ---
        sweep_val = get_val('liquidity_sweep', 0)
        sweep = 0.0
        if sweep_val != 0:
            active_modules.add("sweep")
            sweep = 100.0 # Strict binary event magnitude
            # Note: A buy-side sweep (often negative logic in DB) is structurally bearish for price, and vice versa.

        # --- D. ORDER BLOCK ENGINE (Time Decay Upgrade) ---
        ob_votes = []
        ob_intensity = 0.0
        
        # Upgrade: Apply Exponential Decay based on OB Age
        ob_age = get_val('order_block_age', 0.0)
        decay_multi = time_decay_multiplier(ob_age, 15.0) # 15 candles half-life
        
        if get_val('fresh_ob', 0) > 0 or get_val('sm_fresh_ob', 0) > 0:
            ob_intensity = 100.0 * decay_multi
        elif get_val('mitigated_ob', 0) > 0:
            ob_intensity = 50.0 * decay_multi
            
        if get_val('invalidated_ob', 0) > 0: ob_intensity = 0.0

        raw_ob_score = get_val('order_block_score')
        if raw_ob_score is not None:
            ob_intensity = (ob_intensity + safe_norm(raw_ob_score, 0, 100)) / 2 * decay_multi

        # Direction inherits from structure if no direct reaction, else standalone
        net_ob = 0.0
        if ob_intensity > 0:
            active_modules.add("order_block")
            net_ob = net_struct if net_struct != 0 else 1.0 # Proxy directional vote
            directional_votes.append(net_ob)
            
        order_block_status = "bullish" if net_ob >= 0 else "bearish"
        order_block = max(0.0, min(100.0, ob_intensity))

        # --- E. FAIR VALUE GAP ENGINE (Time Decay Upgrade) ---
        fvg_votes = []
        if get_val('bullish_fvg', 0) > 0: fvg_votes.append(1.0)
        if get_val('bearish_fvg', 0) > 0: fvg_votes.append(-1.0)
        
        fvg_age = get_val('fvg_age', 0.0)
        fvg_decay = time_decay_multiplier(fvg_age, 10.0) # FVG closes faster, 10 candle half-life

        fvg_intensity = 0.0
        if get_val('fvg_active', 0) > 0: fvg_intensity = 100.0 * fvg_decay
        elif get_val('fvg_mitigated', 0) > 0: fvg_intensity = 50.0 * fvg_decay
        
        raw_fvg_score = get_val('sm_fvg_score')
        if raw_fvg_score is not None:
            fvg_intensity = (fvg_intensity + safe_norm(raw_fvg_score, 0, 100)) / 2 * fvg_decay

        net_fvg = sum(fvg_votes) / max(1, len(fvg_votes)) if fvg_votes else 0.0
        fair_value_gap_status = "bullish" if net_fvg > 0 else "bearish"
        
        if fvg_votes or fvg_intensity > 0:
            active_modules.add("fair_value_gap")
            if net_fvg != 0: directional_votes.append(net_fvg)
            
        fair_value_gap = max(0.0, min(100.0, fvg_intensity))

        # --- F. FOOTPRINT ENGINE (Weighting) ---
        fp_votes = []
        smi = get_val('smart_money_index', 0.0)
        if smi != 0: fp_votes.append(1.0 if smi > 0 else -1.0)
        
        delta = get_val('delta_volume', 0.0)
        if delta != 0: fp_votes.append(1.0 if delta > 0 else -1.0)
        
        net_fp = sum(fp_votes) / max(1, len(fp_votes)) if fp_votes else 0.0
        
        fp_intensity = safe_norm(get_val('sm_institutional_score', 50.0), 0, 100)
        if get_val('institutional_displacement_detected', 0) > 0: fp_intensity = max(fp_intensity, 90.0)
        
        footprint_status = "strong" if net_fp > 0 else "outflow"
        if fp_votes or fp_intensity > 0:
            active_modules.add("footprint")
            if net_fp != 0: directional_votes.append(net_fp)
            
        footprint = max(0.0, min(100.0, fp_intensity * (0.5 + 0.5 * abs(net_fp))))

        # --- G. ZONE ENGINE ---
        zone_status = "equilibrium"
        zone_score = 0.0
        
        if get_val('sm_discount_zone', 0) > 0: 
            zone_status = "discount"
            zone_score = 100.0
            directional_votes.append(1.0) # Discount highly favors buying
            active_modules.add("zone")
        elif get_val('sm_premium_zone', 0) > 0: 
            zone_status = "premium"
            zone_score = 100.0
            directional_votes.append(-1.0) # Premium highly favors selling
            active_modules.add("zone")

        # =====================================================================
        # 3. EFFICIENCY, SYNERGY & CONFLICT ENGINE (V7 UPGRADES)
        # =====================================================================
        trace["used"] = len(used_keys)
        module_coverage = min(len(active_modules) / 7.0, 1.0)
        
        net_direction = sum(directional_votes) / max(1, len(directional_votes)) if directional_votes else 0.0
        agreement_factor = abs(net_direction)
        
        # Upgrade: Non-Linear Synergy ("The Supernova Effect")
        # If coverage is high and agreement is nearly perfect, confidence scales exponentially.
        synergy_multiplier = 1.0 + (module_coverage * (agreement_factor ** 2) * 0.5) 
        
        efficiency = max(0.0, min(100.0, (module_coverage * 100.0) * agreement_factor * synergy_multiplier))
        
        raw_master_score = get_val('sm_smart_money_score', 50.0)
        confidence_raw = (efficiency * 0.6) + (raw_master_score * 0.4)
        
        # Upgrade: Conflict / Noise Filter (Strict Penalization)
        # If structure is saying BUY (+), but institutional footprint is saying SELL (-), murder the confidence.
        conflict_msg = None
        if abs(net_struct) > 0.2 and abs(net_fp) > 0.2:
            if (net_struct > 0 and net_fp < 0) or (net_struct < 0 and net_fp > 0):
                conflict_msg = "SEVERE CONFLICT: Market structure and Institutional footprint are inversely correlated. High Trap Probability."
                confidence_raw *= 0.4 # 60% penalty

        confidence = max(0.0, min(100.0, confidence_raw))

        # =====================================================================
        # 4. LIKELIHOOD RATIO
        # =====================================================================
        lr_exponent = net_direction * (synergy_multiplier * agreement_factor * 2.3025) 
        likelihood_ratio = max(0.1, min(10.0, math.exp(lr_exponent)))

        # =====================================================================
        # 5. EVIDENCE ASSEMBLY
        # =====================================================================
        evidence = [{
            "category": "FeatureCoverage",
            "message": f"SMC V7 Logic: {trace['declared']} keys mapped; {trace['valid']} valid, {trace['used']} utilized. Synergy Multiplier: {round(synergy_multiplier, 2)}x",
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": 1.0
        }]
        
        analysis = []
        if structure > 50: analysis.append(f"Structure definitively {structure_status}.")
        if footprint > 50: analysis.append(f"Institutional Flow strongly {footprint_status}.")
        if zone > 50: analysis.append(f"Execution edge present in {zone_status} zone.")
        if ob_intensity > 50: analysis.append(f"Interacting with high-potency, low-decay {order_block_status} OB.")
        
        if conflict_msg:
            analysis.append(conflict_msg)
        elif synergy_multiplier > 1.2:
            analysis.append("SUPERNOVA ALIGNMENT: Structure, Liquidity, and Footprint are harmonically converged.")

        evidence.append({
            "category": "SMC_Intelligence",
            "message": " ".join(analysis) if analysis else "Awaiting clear institutional footprint.",
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": round(likelihood_ratio, 4)
        })

        # =====================================================================
        # 6. FINAL JSON OUTPUT
        # =====================================================================
        return {
            "smart_money_analyzer": {
                "confidence": round(float(confidence), 4),
                "sweep": round(float(sweep), 4),
                "efficiency": round(float(efficiency), 4),
                "structure": round(float(structure), 4),
                "structure_status": structure_status,
                "liquidity": round(float(liquidity), 4),
                "liquidity_status": liquidity_status,
                "order_block": round(float(order_block), 4),
                "order_block_status": order_block_status,
                "fair_value_gap": round(float(fair_value_gap), 4),
                "fair_value_gap_status": fair_value_gap_status,
                "footprint": round(float(footprint), 4),
                "footprint_status": footprint_status,
                "zone": round(float(zone), 4),
                "zone_status": zone_status,
                "evidence": evidence
            }
        }
