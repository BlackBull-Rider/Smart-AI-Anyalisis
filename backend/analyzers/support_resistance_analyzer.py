import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
EPSILON = 1e-9

SR_CONFIG = {
    "default_proximity_pct": 0.015,
    "breakout_clearance_pct": 0.005,
    "weights": {
        "base_strength": 0.35,
        "volume_conf": 0.25,
        "smc_confluence": 0.25,
        "trend_alignment": 0.15
    },
    "institutional_bonus": {
        "liquidity_sweep": 15.0,
        "order_block": 20.0,
        "fvg": 15.0,
        "choch": 25.0,
        "bos": 20.0,
        "volume_expansion": 30.0,
        "low_volume_penalty": -20.0,
        "retest_success": 15.0,
        "retest_failure": -20.0
    },
    "thresholds": {
        "strong_trend": 60.0,
        "moderate_trend": 40.0,
        "high_quality": 75.0,
        "confirmation_min": 60.0,
        "volume_expansion": 1.2,
        "max_probability": 98.5  # Institutional cap (Never 100%)
    },
    "regime_modifiers": {
        "Trending": 1.2,
        "Mean Reversion": 0.8,
        "Volatile": 0.9,
        "Accumulation": 1.1,
        "Distribution": 0.7,
        "Neutral": 1.0
    }
}

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class EvidenceItem(TypedDict):
    category: str
    feature: str
    impact: float
    explanation: str

class SupportResult(TypedDict):
    support_available: bool
    support_type: str
    support_confidence: float
    support_quality: float
    nearest_support: float
    distance_to_support_pct: float
    support_holding: bool
    institutional_support: bool
    evidence: List[EvidenceItem]

class ResistanceResult(TypedDict):
    resistance_available: bool
    resistance_type: str
    resistance_confidence: float
    resistance_quality: float
    nearest_resistance: float
    distance_to_resistance_pct: float
    institutional_resistance: bool
    evidence: List[EvidenceItem]

class BreakoutResult(TypedDict):
    bullish_breakout: bool
    breakout_type: str
    breakout_quality: float
    breakout_confirmation: bool
    breakout_strength: float
    volume_confirmation: float
    false_breakout_risk: float
    institutional_breakout: bool
    expected_continuation: bool
    evidence: List[EvidenceItem]

class BreakdownResult(TypedDict):
    support_breakdown: bool
    breakdown_type: str
    breakdown_quality: float
    breakdown_confirmation: bool
    breakdown_strength: float
    volume_confirmation: float
    false_breakdown_risk: float
    institutional_selling: bool
    expected_continuation: bool
    evidence: List[EvidenceItem]

class RetestResult(TypedDict):
    bullish_retest: bool
    bearish_retest: bool
    retest_successful: bool
    retest_failed: bool
    retest_confidence: float
    retest_strength: float
    retest_quality: float
    evidence: List[EvidenceItem]

class SummaryResult(TypedDict):
    overall_structural_bias: str
    support_status: str
    resistance_status: str
    breakout_probability: float
    breakdown_probability: float
    retest_status: str
    institutional_view: str
    risk_level: str
    action: str

class SRAnalysisResult(TypedDict):
    strong_support: SupportResult
    strong_resistance: ResistanceResult
    breakout: BreakoutResult
    breakdown: BreakdownResult
    retest: RetestResult
    summary: SummaryResult

# ==============================================================================
# LAYER-2 ANALYZER ENGINE
# ==============================================================================

class SupportResistanceAnalyzer:

    L1_FEATURES = [
        'high', 'low', 'close', 'volume', 'swing_high', 'swing_low',
        'support_strength', 'resistance_strength', 'triangle_upper', 'triangle_lower',
        'channel_upper', 'channel_lower', 'channel_width', 'rectangle_upper', 'rectangle_lower',
        'rectangle_width', 'neckline', 'market_phase', 'breakout_pressure', 'compression_pct',
        'pattern_family', 'pattern_confidence', 'trend_direction', 'trend_strength',
        'volume_confirmation', 'market_regime', 'bos', 'choch', 'order_block', 'fvg', 'liquidity_sweep'
    ]

    def __init__(self):
        self._missing_features: List[str] = []

    def _extract_l1_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Safely extracts latest Layer-1 features, applying safe casting for bools/NaNs."""
        self._missing_features.clear()
        extracted = {}
        latest = df.iloc[-1]

        for feat in self.L1_FEATURES:
            if feat in df.columns:
                val = latest[feat]
                if pd.isna(val):
                    extracted[feat] = "Neutral" if feat in ['market_phase', 'pattern_family', 'market_regime'] else 0.0
                elif isinstance(val, (bool, np.bool_)):
                    extracted[feat] = 1.0 if val else 0.0
                else:
                    extracted[feat] = str(val) if isinstance(val, str) else float(val)
            else:
                extracted[feat] = "Neutral" if feat in ['market_phase', 'pattern_family', 'market_regime'] else 0.0
                self._missing_features.append(feat)

        if self._missing_features:
            logger.debug(f"S&R Analyzer: Missing L1 features defaulted: {self._missing_features}")

        return extracted

    def _find_nearest_level(self, current_price: float, levels_dict: Dict[str, float], direction: str) -> Tuple[float, str]:
        """Finds nearest valid structural boundary, safely bypassing NaNs."""
        valid_levels = {}
        for name, price in levels_dict.items():
            if pd.isna(price) or price <= EPSILON:
                continue
            if direction == 'down' and price < current_price:
                valid_levels[name] = price
            elif direction == 'up' and price > current_price:
                valid_levels[name] = price

        if not valid_levels:
            return 0.0, "None"

        nearest_name = max(valid_levels, key=valid_levels.get) if direction == 'down' else min(valid_levels, key=valid_levels.get)
        return valid_levels[nearest_name], nearest_name

    def _get_adaptive_tolerance(self, l1: Dict[str, Any], current_close: float) -> float:
        """Determines dynamic proximity tolerance with strict NaN protection."""
        tri_upper = l1.get('triangle_upper', 0.0)
        tri_lower = l1.get('triangle_lower', 0.0)

        tri_width = 0.0
        if pd.notna(tri_upper) and pd.notna(tri_lower):
            tri_width = abs(tri_upper - tri_lower)

        width = max([
            l1.get('channel_width', 0.0) if pd.notna(l1.get('channel_width')) else 0.0,
            l1.get('rectangle_width', 0.0) if pd.notna(l1.get('rectangle_width')) else 0.0,
            tri_width
        ])

        if width > EPSILON and not pd.isna(width):
            return width * 0.15  # 15% of active structural width
        return current_close * SR_CONFIG["default_proximity_pct"]

    def _analyze_support(self, l1: Dict[str, Any], current_close: float) -> SupportResult:
        evidence: List[EvidenceItem] = []

        candidates = {
            "Swing Low": l1['swing_low'], "Channel Lower": l1['channel_lower'],
            "Triangle Lower": l1['triangle_lower'], "Rectangle Lower": l1['rectangle_lower'],
            "Neckline": l1['neckline']
        }

        nearest_sup, sup_type = self._find_nearest_level(current_close, candidates, 'down')
        available = nearest_sup > 0
        dist_pct = ((current_close - nearest_sup) / current_close) * 100.0 if available else 0.0

        base_str = l1['support_strength']
        vol_conf = np.clip(l1['volume_confirmation'] * 20.0, 0.0, 100.0)

        institutional_score = 0.0
        inst_support = False

        if l1['liquidity_sweep'] == 1.0:
            impact = SR_CONFIG["institutional_bonus"]["liquidity_sweep"]
            institutional_score += impact
            inst_support = True
            evidence.append({"category": "SMC", "feature": "Liquidity Sweep", "impact": impact, "explanation": "Sell-side liquidity swept below support."})

        if l1['order_block'] == 1.0:
            impact = SR_CONFIG["institutional_bonus"]["order_block"]
            institutional_score += impact
            inst_support = True
            evidence.append({"category": "SMC", "feature": "Order Block", "impact": impact, "explanation": "Price resting in unmitigated bullish OB."})

        regime_mult = SR_CONFIG["regime_modifiers"].get(l1['market_regime'], 1.0)
        t_dir = float(l1.get('trend_direction', 0.0))
        if l1['market_regime'] == "Trending" and t_dir < -EPSILON:
            regime_mult = 0.8

        raw_conf = ((base_str * SR_CONFIG["weights"]["base_strength"]) +
                   (vol_conf * SR_CONFIG["weights"]["volume_conf"]) +
                   (institutional_score * SR_CONFIG["weights"]["smc_confluence"])) * regime_mult

        conf = np.clip(raw_conf, 0.0, SR_CONFIG["thresholds"]["max_probability"]) if available else 0.0
        quality = np.clip(conf * (1.0 if dist_pct < 2.0 else 0.8), 0.0, 100.0)

        req_trend = SR_CONFIG["thresholds"]["moderate_trend"] * (0.8 if l1['market_regime'] in ["Mean Reversion", "Accumulation"] else 1.0)
        holding = available and (dist_pct < (self._get_adaptive_tolerance(l1, current_close) / current_close * 100)) and (l1['trend_strength'] > req_trend or inst_support)

        if available:
            evidence.append({"category": "Structure", "feature": sup_type, "impact": base_str, "explanation": f"Nearest support at {nearest_sup:.2f}."})

        return {
            "support_available": available, "support_type": sup_type, "support_confidence": round(conf, 2),
            "support_quality": round(quality, 2), "nearest_support": round(nearest_sup, 2),
            "distance_to_support_pct": round(dist_pct, 2), "support_holding": holding,
            "institutional_support": inst_support, "evidence": evidence
        }

    def _analyze_resistance(self, l1: Dict[str, Any], current_close: float) -> ResistanceResult:
        evidence: List[EvidenceItem] = []

        candidates = {
            "Swing High": l1['swing_high'], "Channel Upper": l1['channel_upper'],
            "Triangle Upper": l1['triangle_upper'], "Rectangle Upper": l1['rectangle_upper'],
            "Neckline": l1['neckline']
        }

        nearest_res, res_type = self._find_nearest_level(current_close, candidates, 'up')
        available = nearest_res > 0
        dist_pct = ((nearest_res - current_close) / current_close) * 100.0 if available else 0.0

        base_str = l1['resistance_strength']
        vol_conf = np.clip(l1['volume_confirmation'] * 20.0, 0.0, 100.0)

        institutional_score = 0.0
        inst_resistance = False

        if l1['liquidity_sweep'] == -1.0:
            impact = SR_CONFIG["institutional_bonus"]["liquidity_sweep"]
            institutional_score += impact
            inst_resistance = True
            evidence.append({"category": "SMC", "feature": "Liquidity Sweep", "impact": impact, "explanation": "Buy-side liquidity swept above resistance."})

        if l1['order_block'] == -1.0:
            impact = SR_CONFIG["institutional_bonus"]["order_block"]
            institutional_score += impact
            inst_resistance = True
            evidence.append({"category": "SMC", "feature": "Order Block", "impact": impact, "explanation": "Price rejecting from bearish OB."})

        regime_mult = SR_CONFIG["regime_modifiers"].get(l1['market_regime'], 1.0)
        t_dir = float(l1.get('trend_direction', 0.0))
        if l1['market_regime'] == "Trending" and t_dir > EPSILON:
            regime_mult = 0.8

        raw_conf = ((base_str * SR_CONFIG["weights"]["base_strength"]) +
                   (vol_conf * SR_CONFIG["weights"]["volume_conf"]) +
                   (institutional_score * SR_CONFIG["weights"]["smc_confluence"])) * regime_mult

        conf = np.clip(raw_conf, 0.0, SR_CONFIG["thresholds"]["max_probability"]) if available else 0.0
        quality = np.clip(conf * (1.0 if dist_pct < 2.0 else 0.8), 0.0, 100.0)

        if available:
            evidence.append({"category": "Structure", "feature": res_type, "impact": base_str, "explanation": f"Nearest resistance at {nearest_res:.2f}."})

        return {
            "resistance_available": available, "resistance_type": res_type, "resistance_confidence": round(conf, 2),
            "resistance_quality": round(quality, 2), "nearest_resistance": round(nearest_res, 2),
            "distance_to_resistance_pct": round(dist_pct, 2), "institutional_resistance": inst_resistance, "evidence": evidence
        }

    def _analyze_breakout(self, l1: Dict[str, Any], current_close: float, res_result: ResistanceResult) -> BreakoutResult:
        evidence: List[EvidenceItem] = []
        vol_conf = l1['volume_confirmation']

        close_cleared_res = res_result['resistance_available'] and current_close > res_result['nearest_resistance'] * (1 + SR_CONFIG["breakout_clearance_pct"])
        bullish_bo = (l1['bos'] == 1.0) or close_cleared_res

        bo_quality, bo_strength, false_risk = 0.0, 0.0, 0.0
        inst_bo, continuation = False, False

        if bullish_bo:
            comp_bonus = l1.get('compression_pct', 0.0) * 0.2
            bo_strength = np.clip((l1['breakout_pressure'] * 0.3) + comp_bonus + (l1['trend_strength'] * 0.3) + (vol_conf * 20.0), 0.0, 100.0)

            if vol_conf > SR_CONFIG["thresholds"]["volume_expansion"]:
                impact = SR_CONFIG["institutional_bonus"]["volume_expansion"]
                bo_quality += impact
                evidence.append({"category": "Volume", "feature": "Volume Expansion", "impact": impact, "explanation": "Breakout supported by volume."})
            else:
                impact = SR_CONFIG["institutional_bonus"]["low_volume_penalty"]
                false_risk += abs(impact)
                evidence.append({"category": "Volume", "feature": "Low Volume", "impact": impact, "explanation": "Lacks volume backing. High risk."})

            if l1['fvg'] == 1.0:
                impact = SR_CONFIG["institutional_bonus"]["fvg"]
                inst_bo, continuation = True, True
                bo_quality += impact
                evidence.append({"category": "SMC", "feature": "Fair Value Gap", "impact": impact, "explanation": "Bullish FVG created."})

            if l1['choch'] == 1.0:
                impact = SR_CONFIG["institutional_bonus"]["choch"]
                inst_bo = True
                bo_quality += impact
                evidence.append({"category": "SMC", "feature": "CHoCH", "impact": impact, "explanation": "Major structural shift (CHoCH)."})

            base_risk = 60.0 if l1['market_regime'] in ["Mean Reversion", "Distribution"] else 20.0
            false_risk = np.clip(base_risk + false_risk - (bo_strength * 0.4) - (l1['pattern_confidence'] * 0.2), 0.0, 100.0)
            final_quality = np.clip(bo_quality + bo_strength - false_risk, 0.0, SR_CONFIG["thresholds"]["max_probability"])
        else:
            final_quality = 0.0

        return {
            "bullish_breakout": bullish_bo, "breakout_type": "Structural BOS" if l1['bos'] == 1.0 else ("Geometrical" if bullish_bo else "None"),
            "breakout_quality": round(final_quality, 2), "breakout_confirmation": bullish_bo and final_quality > SR_CONFIG["thresholds"]["confirmation_min"] and vol_conf > 1.0,
            "breakout_strength": round(bo_strength, 2), "volume_confirmation": round(vol_conf, 2),
            "false_breakout_risk": round(false_risk, 2), "institutional_breakout": inst_bo,
            "expected_continuation": continuation or final_quality > SR_CONFIG["thresholds"]["high_quality"], "evidence": evidence
        }

    def _analyze_breakdown(self, l1: Dict[str, Any], current_close: float, sup_result: SupportResult) -> BreakdownResult:
        evidence: List[EvidenceItem] = []
        vol_conf = l1['volume_confirmation']

        close_broke_sup = sup_result['support_available'] and current_close < sup_result['nearest_support'] * (1 - SR_CONFIG["breakout_clearance_pct"])
        breakdown = (l1['bos'] == -1.0) or close_broke_sup

        bd_quality, bd_strength, false_risk = 0.0, 0.0, 0.0
        inst_selling, continuation = False, False

        if breakdown:
            comp_bonus = l1.get('compression_pct', 0.0) * 0.2
            bd_strength = np.clip((l1['breakout_pressure'] * 0.3) + comp_bonus + (l1['trend_strength'] * 0.3) + (vol_conf * 20.0), 0.0, 100.0)

            if vol_conf > SR_CONFIG["thresholds"]["volume_expansion"]:
                impact = SR_CONFIG["institutional_bonus"]["volume_expansion"]
                bd_quality += impact
                evidence.append({"category": "Volume", "feature": "Volume Expansion", "impact": impact, "explanation": "Breakdown supported by volume."})
            else:
                impact = SR_CONFIG["institutional_bonus"]["low_volume_penalty"]
                false_risk += abs(impact)
                evidence.append({"category": "Volume", "feature": "Low Volume", "impact": impact, "explanation": "Potential fakeout. Low volume."})

            if l1['fvg'] == -1.0:
                impact = SR_CONFIG["institutional_bonus"]["fvg"]
                inst_selling, continuation = True, True
                bd_quality += impact
                evidence.append({"category": "SMC", "feature": "Fair Value Gap", "impact": impact, "explanation": "Bearish FVG indicates aggressive distribution."})

            if l1['choch'] == -1.0:
                impact = SR_CONFIG["institutional_bonus"]["choch"]
                inst_selling = True
                bd_quality += impact
                evidence.append({"category": "SMC", "feature": "CHoCH", "impact": impact, "explanation": "Bearish CHoCH."})

            base_risk = 60.0 if l1['market_regime'] in ["Mean Reversion", "Accumulation"] else 20.0
            false_risk = np.clip(base_risk + false_risk - (bd_strength * 0.4) - (l1['pattern_confidence'] * 0.2), 0.0, 100.0)
            final_quality = np.clip(bd_quality + bd_strength - false_risk, 0.0, SR_CONFIG["thresholds"]["max_probability"])
        else:
            final_quality = 0.0

        return {
            "support_breakdown": breakdown, "breakdown_type": "Structural BOS" if l1['bos'] == -1.0 else ("Geometrical" if breakdown else "None"),
            "breakdown_quality": round(final_quality, 2), "breakdown_confirmation": breakdown and final_quality > SR_CONFIG["thresholds"]["confirmation_min"] and vol_conf > 1.0,
            "breakdown_strength": round(bd_strength, 2), "volume_confirmation": round(vol_conf, 2),
            "false_breakdown_risk": round(false_risk, 2), "institutional_selling": inst_selling,
            "expected_continuation": continuation or final_quality > SR_CONFIG["thresholds"]["high_quality"], "evidence": evidence
        }

    def _analyze_retest(self, l1: Dict[str, Any], current_close: float, sup: SupportResult, res: ResistanceResult) -> RetestResult:
        evidence: List[EvidenceItem] = []
        bullish_retest, bearish_retest = False, False
        successful, failed = False, False
        conf, str_val, qual = 0.0, 0.0, 0.0

        adaptive_tol_pct = (self._get_adaptive_tolerance(l1, current_close) / current_close) * 100.0

        if sup['support_available'] and 0 <= sup['distance_to_support_pct'] <= adaptive_tol_pct:
            if l1['trend_strength'] > SR_CONFIG["thresholds"]["moderate_trend"] and l1['market_phase'] in ['Trending', 'Breakout']:
                bullish_retest = True
                conf = sup['support_confidence']
                str_val = np.clip(conf + (l1['volume_confirmation'] * 10.0), 0.0, SR_CONFIG["thresholds"]["max_probability"])

                if l1['liquidity_sweep'] == 1.0:
                    successful = True
                    qual += SR_CONFIG["institutional_bonus"]["retest_success"]
                    evidence.append({"category": "SMC", "feature": "Liquidity Sweep", "impact": SR_CONFIG["institutional_bonus"]["retest_success"], "explanation": "Retest swept local liquidity and rejected upward."})
                elif current_close >= sup['nearest_support']:
                    successful = True
                    qual += SR_CONFIG["institutional_bonus"]["retest_success"]
                else:
                    failed = True
                    evidence.append({"category": "Price Action", "feature": "Level Breach", "impact": SR_CONFIG["institutional_bonus"]["retest_failure"], "explanation": "Price failed to hold retest level."})

        elif res['resistance_available'] and 0 <= res['distance_to_resistance_pct'] <= adaptive_tol_pct:
            if l1['trend_strength'] < SR_CONFIG["thresholds"]["moderate_trend"] and l1['market_phase'] in ['Trending', 'Breakdown']:
                bearish_retest = True
                conf = res['resistance_confidence']
                str_val = np.clip(conf + (l1['volume_confirmation'] * 10.0), 0.0, SR_CONFIG["thresholds"]["max_probability"])

                if l1['liquidity_sweep'] == -1.0:
                    successful = True
                    qual += SR_CONFIG["institutional_bonus"]["retest_success"]
                    evidence.append({"category": "SMC", "feature": "Liquidity Sweep", "impact": SR_CONFIG["institutional_bonus"]["retest_success"], "explanation": "Bearish retest swept buy-side liquidity."})
                elif current_close <= res['nearest_resistance']:
                    successful = True
                    qual += SR_CONFIG["institutional_bonus"]["retest_success"]
                else:
                    failed = True

        qual = np.clip(qual + (conf * 0.5), 0.0, SR_CONFIG["thresholds"]["max_probability"]) if successful else 0.0

        return {
            "bullish_retest": bullish_retest, "bearish_retest": bearish_retest,
            "retest_successful": successful, "retest_failed": failed,
            "retest_confidence": round(conf, 2), "retest_strength": round(str_val, 2),
            "retest_quality": round(qual, 2), "evidence": evidence
        }

    def _generate_summary(self, l1: Dict[str, Any], sup: SupportResult, res: ResistanceResult, bo: BreakoutResult, bd: BreakdownResult, ret: RetestResult) -> SummaryResult:

        t_dir = float(l1.get('trend_direction', 0.0))
        if t_dir > EPSILON: bias_score = l1['trend_strength']
        elif t_dir < -EPSILON: bias_score = -l1['trend_strength']
        else: bias_score = 0.0

        if bias_score > 60: bias = "Strong Bullish"
        elif bias_score > 20: bias = "Bullish"
        elif bias_score < -60: bias = "Strong Bearish"
        elif bias_score < -20: bias = "Bearish"
        else: bias = "Neutral / Ranging"

        s_stat = f"Holding at {sup['nearest_support']}" if sup['support_holding'] else "Vulnerable" if bd['support_breakdown'] else "Building"
        r_stat = f"Testing {res['nearest_resistance']}" if bo['bullish_breakout'] else "Intact" if res['resistance_available'] else "Clear Sky"

        inst_score = 0
        if sup['institutional_support']: inst_score += 1
        if bo['institutional_breakout']: inst_score += 2
        if bd['institutional_selling']: inst_score -= 2
        if res['institutional_resistance']: inst_score -= 1

        inst_view = "Smart Money Accumulation / Expansion" if inst_score >= 2 else "Smart Money Distribution" if inst_score <= -2 else "Passive / Rotational"

        risk = "Low" if (bo['breakout_confirmation'] and bo['false_breakout_risk'] < 30) or (sup['support_holding'] and sup['distance_to_support_pct'] < 2.0) else "High" if bo['false_breakout_risk'] > 60 or bd['support_breakdown'] else "Moderate"

        if bo['breakout_confirmation'] and bo['expected_continuation']: action = "BUY (Breakout Confirmed)"
        elif ret['bullish_retest'] and ret['retest_successful']: action = "BUY (Retest Validated)"
        elif bd['breakdown_confirmation']: action = "SELL (Structural Failure)"
        elif res['institutional_resistance'] and res['distance_to_resistance_pct'] < 1.0: action = "TAKE PROFIT (Heavy Resistance)"
        else: action = "WAIT (No Clear Edge)"

        bo_prob = min(round(bo['breakout_quality'], 2), SR_CONFIG["thresholds"]["max_probability"]) if bo['bullish_breakout'] else 0.0
        bd_prob = min(round(bd['breakdown_quality'], 2), SR_CONFIG["thresholds"]["max_probability"]) if bd['support_breakdown'] else 0.0

        return {
            "overall_structural_bias": bias, "support_status": s_stat, "resistance_status": r_stat,
            "breakout_probability": bo_prob, "breakdown_probability": bd_prob,
            "retest_status": "Successful" if ret['retest_successful'] else "Failed" if ret['retest_failed'] else "None",
            "institutional_view": inst_view, "risk_level": risk, "action": action
        }

    def analyze(self, df: pd.DataFrame) -> SRAnalysisResult:
        """Main orchestration method for Layer-2 Support & Resistance Engine."""
        try:
            if df.empty or not {'close', 'high', 'low'}.issubset(df.columns):
                logger.error("S&R Analyzer: Required OHLC columns missing or DataFrame is empty.")
                raise ValueError("S&R Analyzer requires basic OHLC columns: ['close', 'high', 'low']")

            l1_data = self._extract_l1_data(df)
            current_close = float(df['close'].iloc[-1])

            logger.debug(f"S&R Analyzer executing on {current_close:.2f} close price.")

            support_res = self._analyze_support(l1_data, current_close)
            resistance_res = self._analyze_resistance(l1_data, current_close)

            breakout_res = self._analyze_breakout(l1_data, current_close, resistance_res)
            if breakout_res['bullish_breakout']: logger.info("S&R Analyzer: Potential Bullish Breakout detected.")

            breakdown_res = self._analyze_breakdown(l1_data, current_close, support_res)
            if breakdown_res['support_breakdown']: logger.info("S&R Analyzer: Potential Support Breakdown detected.")

            retest_res = self._analyze_retest(l1_data, current_close, support_res, resistance_res)

            summary_res = self._generate_summary(l1_data, support_res, resistance_res, breakout_res, breakdown_res, retest_res)

            return {
                "strong_support": support_res,
                "strong_resistance": resistance_res,
                "breakout": breakout_res,
                "breakdown": breakdown_res,
                "retest": retest_res,
                "summary": summary_res
            }

        except Exception as e:
            logger.exception(f"S&R Analyzer encountered a critical error: {e}")
            raise

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "SupportResistanceAnalyzer",
    "SRAnalysisResult",
    "SupportResult",
    "ResistanceResult",
    "BreakoutResult",
    "BreakdownResult",
    "RetestResult",
    "SummaryResult",
    "EvidenceItem"
]
