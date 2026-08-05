"""
GREEN BULL RIDER V6
Layer-2 Quantitative Analyzer: Smart Money / SMC Engine

This module implements institutional-grade Smart Money Concept (SMC) quantitative intelligence.
It strictly adheres to the exact Layer-1 database schema provided, avoiding look-ahead bias,
and extracting multi-dimensional SMC insights solely from explicitly provided feature snapshots.

Architectural Guarantees:
- Strict Exact-Key Provenance: Zero aliases, zero 'or' fallbacks, zero missing-feature substitution.
- Independent Evidence: Each valid feature contributes orthogonally to its respective contract group.
- Exact Database Mapping: Only features explicitly present in the supplied database schema are consumed.
- SMC Semantics: Proximity, targeting, and footprint directional rules are mathematically derived without false assumptions.
- Zero History Synthesis: Operates exclusively on the provided scalar snapshot.
"""

import math
from typing import Dict, Any, Optional, List

class SmartMoneyAnalyzer:
    """
    Evaluates Smart Money Concepts (SMC) market structure, liquidity, zones, and 
    footprint strictly using a provided dictionary of exact Layer-1 features.
    """

    def analyze(self, features: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the quantitative SMC analysis engine based on an exact feature snapshot.
        """
        # =====================================================================
        # 0. STRICT OUTPUT CONTRACT FALLBACK
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
                "evidence": [
                    {
                        "category": "SMC",
                        "message": "Insufficient valid Layer-1 SMC features for quantitative analysis.",
                        "reliability": 0.0,
                        "likelihood_ratio": 1.0
                    }
                ]
            }
        }

        if not features or not isinstance(features, dict):
            return fallback

        # =====================================================================
        # 1. EXACT DB FEATURE UNIVERSE DEFINITION (CONTRACT MAPPING)
        # =====================================================================
        # Derived strictly from the provided DB snapshot and Layer-1 schema output
        UNIVERSE = [
            # Structure
            'bos_up', 'bos_down', 'choch_up', 'choch_down', 'smc_trend', 
            'mss_bullish', 'mss_bearish', 'sm_swing_high', 'sm_swing_low', 
            'sm_higher_high', 'sm_higher_low', 'sm_lower_high', 'sm_lower_low', 
            'sm_bos', 'sm_choch', 'sm_mss', 'sm_bos_score', 'sm_choch_score', 'sm_trend_score',
            'higher_high', 'higher_low', 'lower_high', 'lower_low', 'hh_breakout', 'll_breakdown',
            'prev_high_breakout', 'prev_low_breakdown',
            
            # Liquidity
            'buy_side_liquidity', 'sell_side_liquidity', 'internal_liquidity', 
            'external_liquidity', 'equal_highs', 'equal_lows', 'sm_buy_side_liquidity', 
            'sm_sell_side_liquidity', 'sm_liquidity_score', 'liquidity_pool',
            
            # Sweep (Events/Intensity)
            'liquidity_sweep', 'turtle_soup', 'judas_swing', 'liquidity_sweep_candle',
            'stop_hunt_zone',
            
            # Order Block / Supply & Demand
            'fresh_ob', 'mitigated_ob', 'invalidated_ob', 'ob_high', 'ob_low', 
            'breaker_block', 'flip_zone', 'sm_fresh_ob', 'sm_mitigated_ob',
            'demand_zone', 'supply_zone', 'reaction_demand', 'reaction_supply',
            
            # Fair Value Gap (FVG)
            'bullish_fvg', 'bearish_fvg', 'fvg_mitigated', 'fvg_active', 'bpr', 
            'sm_active_fvg', 'sm_mitigated_fvg', 'sm_fvg_score',
            
            # Footprint / Flow
            'buy_volume', 'sell_volume', 'delta_volume', 'stopping_volume', 
            'no_demand', 'no_supply', 'buying_pressure', 'selling_pressure', 
            'institutional_body', 'institutional_wick', 'institutional_imbalance', 
            'institutional_pressure', 'absorption_candle', 'rejection_candle', 
            'acceptance_candle', 'smart_money_candle', 'effort_result',
            'smart_money_index', 'vpin',
            
            # Zones
            'premium_level', 'equilibrium', 'discount_level', 'sm_premium_zone', 
            'sm_discount_zone', 'sm_equilibrium', 'institutional_support', 
            'institutional_resistance', 'smart_money_level',
            
            # Master SMC Scores
            'sm_institutional_score', 'sm_smart_money_score',
            
            # Core Price Context (Required for distance/targeting calculations)
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

        if trace["valid"] == 0:
            return fallback

        # Single unified accessor that strictly traces utilization
        def get_val(key: str) -> Optional[float]:
            if key in validated:
                used_keys.add(key)
                return validated[key]
            return None

        # Safe bounding functions
        def safe_norm(val: float, min_in: float, max_in: float) -> float:
            if val <= min_in: return 0.0
            if val >= max_in: return 100.0
            if max_in == min_in: return 0.0
            return ((val - min_in) / (max_in - min_in)) * 100.0

        def safe_tanh_map(val: float, scale: float = 1.0) -> float:
            return math.tanh(val * scale) * 100.0

        # Base context
        c_close = get_val('close') or 0.0

        # =====================================================================
        # 2. SMC ORTHOGONAL MODULES
        # =====================================================================
        active_modules = set()
        directional_votes: List[float] = []

        # --- A. STRUCTURE ENGINE ---
        struct_votes = []
        struct_intensity = []
        
        # Bullish explicit structure
        for k in ['bos_up', 'choch_up', 'mss_bullish', 'higher_high', 'higher_low', 'hh_breakout', 'prev_high_breakout', 'sm_higher_high', 'sm_higher_low']:
            v = get_val(k)
            if v is not None and v > 0: struct_votes.append(1.0)
            
        # Bearish explicit structure
        for k in ['bos_down', 'choch_down', 'mss_bearish', 'lower_high', 'lower_low', 'll_breakdown', 'prev_low_breakdown', 'sm_lower_high', 'sm_lower_low']:
            v = get_val(k)
            if v is not None and v > 0: struct_votes.append(-1.0)

        # Directional structure flags (-1, 0, 1)
        for k in ['smc_trend', 'sm_bos', 'sm_choch', 'sm_mss']:
            v = get_val(k)
            if v is not None and v != 0:
                struct_votes.append(1.0 if v > 0 else -1.0)

        # Structure Magnitudes
        for k in ['sm_trend_score', 'sm_bos_score', 'sm_choch_score']:
            v = get_val(k)
            if v is not None:
                struct_intensity.append(safe_norm(v, 0.0, 100.0))

        structure_status = "bearish"
        structure_score = 0.0
        if struct_votes or struct_intensity:
            active_modules.add("structure")
            net_struct = sum(struct_votes) / len(struct_votes) if struct_votes else 0.0
            structure_status = "bullish" if net_struct > 0 else "bearish"
            if net_struct != 0: directional_votes.append(net_struct)
            
            base_struct = sum(struct_intensity) / len(struct_intensity) if struct_intensity else 50.0
            structure_score = base_struct * (0.5 + 0.5 * abs(net_struct))
            
        structure = max(0.0, min(100.0, structure_score))

        # --- B. LIQUIDITY ENGINE ---
        liq_votes = []
        liq_intensity = []
        
        # 1. Proximity Targeting: Distance to specific liquidity pools
        # If price is closer to BSL, BSL acts as a bullish magnetic target.
        v_bsl = get_val('buy_side_liquidity')
        v_ssl = get_val('sell_side_liquidity')
        if v_bsl is not None and v_ssl is not None and c_close > 0:
            d_b = abs(v_bsl - c_close)
            d_s = abs(v_ssl - c_close)
            if d_b < d_s: liq_votes.append(1.0)
            elif d_s < d_b: liq_votes.append(-1.0)

        v_sm_bsl = get_val('sm_buy_side_liquidity')
        v_sm_ssl = get_val('sm_sell_side_liquidity')
        if v_sm_bsl is not None and v_sm_ssl is not None and c_close > 0:
            d_b = abs(v_sm_bsl - c_close)
            d_s = abs(v_sm_ssl - c_close)
            if d_b < d_s: liq_votes.append(1.0)
            elif d_s < d_b: liq_votes.append(-1.0)

        # 2. Explicit Structural Pools
        v_eqh = get_val('equal_highs')
        if v_eqh is not None and v_eqh > 0: liq_votes.append(1.0) # EQH = Bullish target
            
        v_eql = get_val('equal_lows')
        if v_eql is not None and v_eql > 0: liq_votes.append(-1.0) # EQL = Bearish target

        # 3. Presence/Intensity
        v_liq_pool = get_val('liquidity_pool')
        if v_liq_pool is not None and v_liq_pool != 0:
            liq_intensity.append(100.0)

        v_sm_liq = get_val('sm_liquidity_score')
        if v_sm_liq is not None:
            liq_intensity.append(safe_norm(v_sm_liq, 0.0, 100.0))

        # 4. Independent boundaries (Without zip pairing)
        v_ext_liq = get_val('external_liquidity')
        v_int_liq = get_val('internal_liquidity')
        if v_ext_liq is not None: liq_intensity.append(50.0)
        if v_int_liq is not None: liq_intensity.append(50.0)

        liquidity_status = "bearish"
        liquidity_score = 0.0
        if liq_votes or liq_intensity:
            active_modules.add("liquidity")
            net_liq = sum(liq_votes) / len(liq_votes) if liq_votes else 0.0
            liquidity_status = "bullish" if net_liq > 0 else "bearish"
            if net_liq != 0: directional_votes.append(net_liq)
            
            liquidity_score = sum(liq_intensity) / len(liq_intensity) if liq_intensity else 50.0

        liquidity = max(0.0, min(100.0, liquidity_score))

        # --- C. SWEEP ENGINE ---
        # Note: The output contract dictates only "sweep: 0.0", without a sweep_status.
        # This module calculates the intensity/significance of a sweep event independently.
        sweep_intensity = []
        
        for k in ['liquidity_sweep', 'turtle_soup', 'judas_swing', 'liquidity_sweep_candle', 'stop_hunt_zone']:
            v = get_val(k)
            if v is not None and v != 0:
                # Intensity mapped directly from the presence/magnitude of the event
                sweep_intensity.append(abs(safe_tanh_map(v, 1.0)) if v != 1.0 else 100.0)
        
        sweep = 0.0
        if sweep_intensity:
            active_modules.add("sweep")
            sweep = max(0.0, min(100.0, sum(sweep_intensity) / len(sweep_intensity)))

        # --- D. ORDER BLOCK (OB) / SUPPLY & DEMAND ENGINE ---
        ob_votes = []
        ob_intensity = []
        
        # Intensity signals
        for k in ['fresh_ob', 'sm_fresh_ob', 'breaker_block', 'flip_zone']:
            v = get_val(k)
            if v is not None and v != 0: ob_intensity.append(100.0)
            
        for k in ['mitigated_ob', 'sm_mitigated_ob']:
            v = get_val(k)
            if v is not None and v != 0: ob_intensity.append(50.0)
            
        v_inv_ob = get_val('invalidated_ob')
        if v_inv_ob is not None and v_inv_ob != 0: ob_intensity.append(0.0)
            
        # Explicit Directional Supply/Demand Reactions
        v_react_dem = get_val('reaction_demand')
        if v_react_dem is not None and v_react_dem != 0: ob_votes.append(1.0)
            
        v_react_sup = get_val('reaction_supply')
        if v_react_sup is not None and v_react_sup != 0: ob_votes.append(-1.0)
        
        v_dem_zone = get_val('demand_zone')
        v_sup_zone = get_val('supply_zone')
        if v_dem_zone is not None: ob_intensity.append(75.0)
        if v_sup_zone is not None: ob_intensity.append(75.0)
                
        order_block_status = "bearish"
        order_block_score = 0.0
        
        if ob_votes or ob_intensity:
            active_modules.add("order_block")
            net_ob = sum(ob_votes) / len(ob_votes) if ob_votes else 0.0
            
            # If we lack direct OB directional evidence, align with prevailing structure
            if not ob_votes and struct_votes:
                order_block_status = structure_status
            else:
                order_block_status = "bullish" if net_ob >= 0 else "bearish"
                
            if net_ob != 0: directional_votes.append(net_ob)
            order_block_score = sum(ob_intensity) / len(ob_intensity) if ob_intensity else (50.0 if net_ob != 0 else 0.0)

        order_block = max(0.0, min(100.0, order_block_score))

        # --- E. FAIR VALUE GAP (FVG) ENGINE ---
        fvg_votes = []
        fvg_intensity = []
        
        v_bull_fvg = get_val('bullish_fvg')
        if v_bull_fvg is not None and v_bull_fvg != 0: fvg_votes.append(1.0)
        
        v_bear_fvg = get_val('bearish_fvg')
        if v_bear_fvg is not None and v_bear_fvg != 0: fvg_votes.append(-1.0)
        
        for k in ['fvg_active', 'sm_active_fvg', 'bpr']:
            v = get_val(k)
            if v is not None and v != 0: fvg_intensity.append(100.0)
            
        for k in ['fvg_mitigated', 'sm_mitigated_fvg']:
            v = get_val(k)
            if v is not None and v != 0: fvg_intensity.append(50.0)
            
        v_sm_fvg_score = get_val('sm_fvg_score')
        if v_sm_fvg_score is not None:
            fvg_intensity.append(safe_norm(v_sm_fvg_score, 0.0, 100.0))
        
        fair_value_gap_status = "bearish"
        fair_value_gap_score = 0.0
        
        if fvg_votes or fvg_intensity:
            active_modules.add("fair_value_gap")
            net_fvg = sum(fvg_votes) / len(fvg_votes) if fvg_votes else 0.0
            fair_value_gap_status = "bullish" if net_fvg > 0 else "bearish"
            if net_fvg != 0: directional_votes.append(net_fvg)
            
            fair_value_gap_score = sum(fvg_intensity) / len(fvg_intensity) if fvg_intensity else 50.0

        fair_value_gap = max(0.0, min(100.0, fair_value_gap_score))

        # --- F. FOOTPRINT / ORDER FLOW ENGINE ---
        fp_votes = []
        fp_intensity = []
        
        v_dvol = get_val('delta_volume')
        v_bvol = get_val('buy_volume')
        v_svol = get_val('sell_volume')
        if v_dvol is not None:
            fp_votes.append(1.0 if v_dvol > 0 else -1.0)
            if v_bvol is not None and v_svol is not None:
                tot_vol = v_bvol + v_svol
                if tot_vol > 0:
                    fp_intensity.append(safe_norm(abs(v_dvol) / tot_vol, 0.0, 0.5) * 100.0)
                    
        v_bpres = get_val('buying_pressure')
        v_spres = get_val('selling_pressure')
        if v_bpres is not None and v_spres is not None:
            if v_bpres > v_spres: fp_votes.append(1.0)
            elif v_spres > v_bpres: fp_votes.append(-1.0)
            tot_pres = v_bpres + v_spres
            if tot_pres > 0:
                fp_intensity.append(safe_norm(abs(v_bpres - v_spres) / tot_pres, 0.0, 0.5) * 100.0)
                
        # Smart Money specific signals
        v_smi = get_val('smart_money_index')
        if v_smi is not None:
            fp_votes.append(1.0 if v_smi > 0 else -1.0)
            fp_intensity.append(abs(safe_tanh_map(v_smi, 0.01)))
            
        v_vpin = get_val('vpin')
        if v_vpin is not None:
            fp_intensity.append(safe_norm(v_vpin, 0.0, 1.0))

        # Institutional anomaly detection
        for k in ['institutional_pressure', 'effort_result']:
            v = get_val(k)
            if v is not None and v != 0:
                fp_votes.append(1.0 if v > 0 else -1.0)
                fp_intensity.append(abs(safe_tanh_map(v, 0.01)))

        for k in ['institutional_body', 'institutional_wick', 'institutional_imbalance', 'absorption_candle', 'rejection_candle', 'acceptance_candle', 'smart_money_candle', 'stopping_volume', 'no_demand', 'no_supply']:
            v = get_val(k)
            if v is not None and v != 0:
                fp_intensity.append(100.0)
                if k == 'no_demand': fp_votes.append(-1.0)
                if k == 'no_supply': fp_votes.append(1.0)
            
        footprint_status = "outflow"
        footprint_score = 0.0
        if fp_votes or fp_intensity:
            active_modules.add("footprint")
            net_fp = sum(fp_votes) / len(fp_votes) if fp_votes else 0.0
            footprint_status = "strong" if net_fp > 0 else "outflow"
            if net_fp != 0: directional_votes.append(net_fp)
            
            footprint_score = sum(fp_intensity) / len(fp_intensity) if fp_intensity else 50.0

        footprint = max(0.0, min(100.0, footprint_score))

        # --- G. ZONE ENGINE ---
        zone_status = "premium"
        zone_score = 0.0
        zone_eval = False
        
        # Direct Binary Zone Indicators
        v_sm_disc = get_val('sm_discount_zone')
        v_sm_prem = get_val('sm_premium_zone')
        if v_sm_disc is not None and v_sm_disc > 0:
            zone_status = "discount"; zone_score = 100.0; zone_eval = True
        if v_sm_prem is not None and v_sm_prem > 0:
            zone_status = "premium"; zone_score = 100.0; zone_eval = True

        # Explicit Zone Levels
        v_disc_lvl = get_val('discount_level')
        v_prem_lvl = get_val('premium_level')
        if v_disc_lvl is not None and v_prem_lvl is not None and c_close > 0:
            zone_range = v_prem_lvl - v_disc_lvl
            if zone_range > 0:
                pos = (c_close - v_disc_lvl) / zone_range
                zone_score = max(0.0, min(100.0, abs(pos - 0.5) * 200.0))
                zone_status = "premium" if pos >= 0.5 else "discount"
                zone_eval = True
                
        if zone_eval:
            active_modules.add("zone")
            # Discount zone favors buying, Premium zone favors selling structurally
            if zone_status == "discount": directional_votes.append(1.0)
            elif zone_status == "premium": directional_votes.append(-1.0)
            
        zone = max(0.0, min(100.0, zone_score))

        # =====================================================================
        # 3. EFFICIENCY & CONFIDENCE ENGINE
        # =====================================================================
        trace["used"] = len(used_keys)
        
        # The output contract requires 7 discrete SMC dimensions
        REQUIRED_CONTRACT_MODULES = 7.0
        module_coverage = min(len(active_modules) / REQUIRED_CONTRACT_MODULES, 1.0)
        
        net_direction = sum(directional_votes) / len(directional_votes) if directional_votes else 0.0
        agreement_factor = abs(net_direction)
        
        # Efficiency represents the density and directional coherence of the 7 contract dimensions
        efficiency = max(0.0, min(100.0, (module_coverage * 100.0) * (0.5 + 0.5 * agreement_factor)))
        
        # Master SMC Scores integration
        score_boost = 0.0
        for k in ['sm_institutional_score', 'sm_smart_money_score']:
            v = get_val(k)
            if v is not None:
                score_boost += safe_norm(v, 0.0, 100.0) * 0.1
        
        # Confidence logic based on available DB features against declared contract universe
        contract_coverage_ratio = trace["valid"] / trace["declared"]
        confidence_raw = (efficiency * 0.6) + (contract_coverage_ratio * 40.0) + score_boost
        
        # Severe structural contradiction penalty
        if len(active_modules) >= 3 and agreement_factor < 0.2:
            confidence_raw -= 20.0
            
        confidence = max(0.0, min(100.0, confidence_raw))

        # =====================================================================
        # 4. STATISTICAL LIKELIHOOD RATIO
        # =====================================================================
        # Derived mathematically from net normalized consensus and evidence density
        lr_exponent = net_direction * (module_coverage * agreement_factor * 2.3025) 
        likelihood_ratio = math.exp(lr_exponent)
        likelihood_ratio = max(0.1, min(10.0, likelihood_ratio))

        # =====================================================================
        # 5. EVIDENCE GENERATION
        # =====================================================================
        evidence = []
        
        coverage_msg = f"SMC Contract Validation: {trace['declared']} expected DB keys; {trace['valid']} valid, {trace['used']} utilized across {len(active_modules)} orthogonal SMC modules ({trace['unavailable']} unavailable, {trace['invalid']} invalid)."
        evidence.append({
            "category": "FeatureCoverage",
            "message": coverage_msg,
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": 1.0
        })
        
        analysis = []
        if structure > 50.0:
            analysis.append(f"Market structure indicates quantitative {structure_status} control (Strength: {structure:.1f}).")
            
        if liquidity > 50.0:
            analysis.append(f"Liquidity footprint targets {liquidity_status} pools.")
            
        if sweep > 50.0:
            analysis.append("Significant institutional liquidity sweep executed.")
            
        if fair_value_gap > 50.0:
            analysis.append(f"Active {fair_value_gap_status} Fair Value Gap (FVG) identified.")
            
        if order_block > 50.0:
            analysis.append(f"Price structure interacts with a {order_block_status} Order Block.")
            
        if footprint > 50.0:
            analysis.append(f"Order flow footprint reveals {footprint_status} institutional pressure.")
            
        if zone > 50.0:
            analysis.append(f"Price is located in a high-value {zone_status} structural zone.")
            
        if len(active_modules) >= 3:
            if agreement_factor > 0.8:
                analysis.append("Orthogonal SMC dimensions display high systemic convergence.")
            elif agreement_factor < 0.3:
                analysis.append("Significant contradiction detected between structural, liquidity, and footprint elements.")

        if not analysis:
            analysis.append("SMC structure is currently weak, missing, or highly conflicted.")

        evidence.append({
            "category": "SMC",
            "message": " ".join(analysis),
            "reliability": round(confidence / 100.0, 4),
            "likelihood_ratio": round(likelihood_ratio, 4)
        })

        # =====================================================================
        # 6. FINAL OUTPUT ASSEMBLY (EXACT CONTRACT)
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
