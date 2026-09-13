import numpy as np
import pandas as pd
from typing import Dict, Any

class L2MasterEngine:
    """
    Green Bull Rider V6 - L2 Master Engine (Setup Finder & Execution Manager)
    
    Phase 1: Correlates SMC, Pattern, Volatility, and S/R to discover institutional setups.
    Phase 2: Validates trigger via Candle, Volume, and Trend, mapping dynamic RRR.
    Strictly parses locked L1 JSON contracts. Zero retail shortcuts.
    """

    def __init__(self):
        self.MIN_RRR = 2.0
        self.EPSILON = 1e-8

    def evaluate_market(self, l1_contracts: Dict[str, dict], current_price: float) -> Dict[str, Any]:
        """
        Main entry point for the L2 Engine. Consumes all 8 L1 analyzer contracts.
        """
        if not l1_contracts:
            return self._fallback_contract()

        # Step 1: Discover High-Probability Setups
        setup_context = self._find_setup(l1_contracts)
        
        # Step 2: Validate Execution & Calculate Risk Geometry
        execution_context = self._validate_execution(setup_context, l1_contracts, current_price)

        return {
            "master_status": execution_context["status"],
            "setup": setup_context,
            "execution": execution_context
        }

    # ==========================================
    # PHASE 1: SETUP FINDER
    # ==========================================

    def _find_setup(self, contracts: Dict[str, dict]) -> dict:
        """
        Scans strictly for 3 institutional configurations using Exact L1 Keys.
        """
        smc = contracts.get("smart_money_analyzer", {})
        sr = contracts.get("sr_analyzer", {})
        pattern = contracts.get("pattern_analyzer", {})
        vol = contracts.get("volatility_analyzer", {})
        mom = contracts.get("momentum_analyzer", {})

        setup_name = "none"
        direction = "neutral"
        readiness = "forming"
        score = 0

        # Extract Deep Institutional Fields
        sweep_status = smc.get("liquidity_profile", {}).get("sweep_status", "none")
        sr_context = sr.get("trade_location", {}).get("position_context", "mid_range")
        smc_phase = smc.get("smc_phase", "consolidation")
        vol_state = vol.get("volatility_state", "normal")
        pat_family = pattern.get("dominant_family", "none")

        # ---------------------------------------------------------
        # SETUP 1: Institutional Liquidity Reversal
        # ---------------------------------------------------------
        if sweep_status in ["bsl_swept", "ssl_swept", "institutional_manipulation"]:
            if sr_context in ["at_demand", "at_supply"]:
                setup_name = "liquidity_sweep_reversal"
                direction = smc.get("institutional_bias", "neutral")
                score += 3
                
                # Check for structural accumulation/distribution corroboration
                pat_struct = pattern.get("structural_details", {})
                if pat_struct.get("is_accumulation_base") or pat_struct.get("is_distribution_top"):
                    score += 2
                
                if mom.get("momentum_state", "") == "diverging":
                    readiness = "prime"
                else:
                    readiness = "developing"

        # ---------------------------------------------------------
        # SETUP 2: Kinetic Volatility Breakout (Squeeze)
        # ---------------------------------------------------------
        elif vol_state == "compression" and vol.get("squeeze_metrics", {}).get("is_squeezing", False):
            if pat_family == "kinetic_compression":
                setup_name = "volatility_squeeze_breakout"
                direction = pattern.get("structural_bias", "neutral")
                score += 3
                
                # Check critical squeeze pressure
                if pattern.get("setup_readiness", {}).get("is_critical_squeeze", False):
                    score += 2
                    readiness = "imminent"
                else:
                    readiness = "developing"
                
                # Smart money accumulating before breakout
                if smc.get("smart_money_footprint", {}).get("composite_score_z", 0.0) > 1.0:
                    score += 1

        # ---------------------------------------------------------
        # SETUP 3: Smart Money Trend Continuation (Order Block)
        # ---------------------------------------------------------
        elif smc_phase in ["markup", "markdown", "reaccumulation", "redistribution"]:
            if smc.get("institutional_zones", {}).get("active_ob_present", False):
                # Validate that the OB is structurally dense
                if sr.get("zone_confluence", {}).get("cluster_strength_z", 0.0) > 1.0 or sr.get("zone_confluence", {}).get("is_heavy_congestion", False):
                    setup_name = "smart_money_continuation"
                    direction = smc.get("institutional_bias", "neutral")
                    score += 3
                    
                    if mom.get("momentum_state", "") == "accelerating":
                        readiness = "developing"
                    else:
                        readiness = "forming"

        return {
            "setup_name": setup_name,
            "direction": direction,
            "readiness": readiness,
            "confluence_score": score
        }

    # ==========================================
    # PHASE 2: EXECUTION MANAGER
    # ==========================================

    def _validate_execution(self, setup: dict, contracts: Dict[str, dict], current_price: float) -> dict:
        """
        Validates the discovered setup against execution engines: Trend, Volume, Candle, and S/R.
        """
        if setup["setup_name"] == "none" or setup["direction"] == "neutral":
            return self._empty_execution()

        trend = contracts.get("trend_analyzer", {})
        volume = contracts.get("volume_analyzer", {})
        candle = contracts.get("candle_analyzer", {})
        sr = contracts.get("sr_analyzer", {})

        # 1. Macro Trend Filter (Fight the trend only with extreme institutional footprint)
        trend_dir = trend.get("direction", "neutral")
        if trend_dir != "neutral" and trend_dir != setup["direction"]:
            inst_act_z = volume.get("institutional_anomalies", {}).get("smart_money_activity_z", 0.0)
            if inst_act_z < 1.5:
                return self._rejected_execution("trend_conflict")

        # 2. Volume & Order Flow Validator
        if volume.get("participation", {}).get("relative_activity_z", 0.0) < -1.0:
            return self._rejected_execution("dry_volume")
            
        vol_bias = volume.get("flow_bias", "neutral")
        if vol_bias != "neutral" and vol_bias != setup["direction"]:
            return self._rejected_execution("order_flow_conflict")

        # 3. Pinpoint Candle Trigger
        trigger = candle.get("trigger_status", "no_trigger")
        if trigger not in ["valid_entry", "wait_for_close"]:
            return self._rejected_execution("candle_trigger_failed")

        # 4. Risk to Reward (RRR) Validation via S/R Engine
        geom = sr.get("execution_geometry", {})
        rrr = geom.get("dynamic_rrr", 0.0)
        rrr_profile = geom.get("rrr_profile", "unknown")

        if rrr < self.MIN_RRR:
            return self._rejected_execution(f"poor_rrr_{rrr_profile}")

        # Final Approval
        status = "execute_now" if trigger == "valid_entry" else "pending_close"
        
        trade_loc = sr.get("trade_location", {})
        sl_type = trade_loc.get("nearest_support_type", "structural") if setup["direction"] == "bullish" else trade_loc.get("nearest_resistance_type", "structural")

        return {
            "status": status,
            "entry_price_ref": round(current_price, 2),
            "validated_rrr": round(rrr, 2),
            "rrr_profile": rrr_profile,
            "sl_logic": sl_type,
            "confluence_zone": trade_loc.get("position_context", "unknown")
        }

    # ==========================================
    # FALLBACKS
    # ==========================================

    def _empty_execution(self) -> dict:
        return {"status": "no_setup", "entry_price_ref": 0.0, "validated_rrr": 0.0, "rrr_profile": "none", "sl_logic": "none", "confluence_zone": "none"}

    def _rejected_execution(self, reason: str) -> dict:
        return {"status": f"rejected_{reason}", "entry_price_ref": 0.0, "validated_rrr": 0.0, "rrr_profile": "none", "sl_logic": "none", "confluence_zone": "none"}

    def _fallback_contract(self) -> dict:
        return {"master_status": "data_missing", "setup": {}, "execution": {}}
