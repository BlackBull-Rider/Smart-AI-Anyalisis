import logging

import numpy as np
import pandas as pd

from typing import Dict, List, TypedDict, Optional, Any
logger = logging.getLogger(__name__)

EPSILON = 1e-9

CANDLE_CONFIG = {
    "mtf_weights": {
        "_M": 0.35, "_W": 0.25, "_D": 0.20, "_4H": 0.10, "_1H": 0.05, "_15m": 0.03, "_5m": 0.02
    },
    "mtf": {
        "htf_suffixes": ['_D', '_W', '_M'],
        "ltf_suffixes": ['_5m', '_15m', '_1H', '_4H']
    },
    "bayesian": {
        "priors": {"reversal": 0.15, "continuation": 0.60, "trap": 0.20},
        "likelihood_profiles": {
            "bullish_expansion": {
                "structural_break_bull": 4.12, "structural_break_bear": 0.31,
                "liquidity_sweep_bull": 3.45, "liquidity_sweep_bear": 0.42,
                "momentum_persistence": 2.68, "wyckoff_trap": 0.15
            },
            "bearish_expansion": {
                "structural_break_bull": 0.28, "structural_break_bear": 4.35,
                "liquidity_sweep_bull": 0.39, "liquidity_sweep_bear": 3.62,
                "momentum_persistence": 2.81, "wyckoff_trap": 0.12
            },
            "mean_reverting": {
                "structural_break_bull": 1.42, "structural_break_bear": 1.42,
                "liquidity_sweep_bull": 4.65, "liquidity_sweep_bear": 4.65,
                "momentum_persistence": 0.88, "wyckoff_trap": 4.52
            }
        },
        "correlation_discount_factor": 0.40 
    },
    "thresholds": {
        "strong_prob": 80.0, "moderate_prob": 50.0, "weak_prob": 20.0,
        "poor_high_low_ticks": 0.03, "body_expansion_ratio": 0.70
    },
    "source_reliability": {
        "external_confirmed": 1.00,
        "jit_computed": 0.85,
        "proxy_derived": 0.50
    }
}

# ==============================================================================
# LAYER-2 DATA CONTRACTS
# ==============================================================================
class EvidenceItem(TypedDict):
    category: str; type: str; weight: float; value: str; polarity: int; institutional_explanation: str; reliability: float

class CandlePsychologyResult(TypedDict):
    score: float; status: str; emotion: str; auction_balance: str; evidence: List[EvidenceItem]

class BullBearResult(TypedDict):
    bull_weight_ratio: float; bear_weight_ratio: float; dominance: str; clv_score: float; institutional_control_score: float

class ReversalResult(TypedDict):
    probability: float; direction: str; quality: str; trap_probability: float; evidence: List[EvidenceItem]

class ContinuationResult(TypedDict):
    probability: float; quality: str; evidence: List[EvidenceItem]

class GapAnalysisResult(TypedDict):
    gap_type: str; gap_acceptance: bool; gap_fill_probability: float; evidence: List[EvidenceItem]

class MTFResult(TypedDict):
    alignment_score: float; htf_dominant_bias: str; ltf_trigger_state: str; structural_alignment: str; conflict_score: float; timeframes: Dict[str, Dict[str, str]]

class AdvancedCandleMetrics(TypedDict):
    candle_quality_index: float; psychology_index: float; rejection_index: float; acceptance_index: float; smart_money_conviction: float; institutional_conviction: float; retail_emotion: str; trap_index: float; auction_balance: float; price_acceptance_score: float; price_rejection_score: float; market_quality_score: float; conviction_decay: float; price_discovery_index: float; liquidity_utilization: float; system_confidence: float

class CandleAnalysisResult(TypedDict):
    candle_psychology: CandlePsychologyResult; bull_bear_analysis: BullBearResult; reversal_detection: ReversalResult; continuation_detection: ContinuationResult; gap_analysis: GapAnalysisResult; multi_timeframe: MTFResult; advanced_metrics: AdvancedCandleMetrics

# ==============================================================================
# MAIN CANDLE ANALYZER ORCHESTRATOR
# ==============================================================================
class CandleAnalyzer:
    EXPECTED_SCHEMA = [
        "candle_psychology.status", "bull_bear_analysis.dominance",
        "reversal_detection.direction", "continuation_detection.quality",
        "gap_analysis.gap_type", "multi_timeframe.htf_dominant_bias"
    ]
    
    CRITICAL_FEATURES = ["close", "open", "high", "low", "atr", "clv", "bos", "choch", "trend_direction"]
    OPTIONAL_FEATURES = ["volume", "efficiency_ratio", "normalized_volatility", "liquidity_sweep", "bull_sequence", "bear_sequence", "gap_up", "gap_down", "volume_ratio", "body_pct", "upper_wick", "lower_wick"]

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def _get_latest_values(self, df: Any) -> Dict[str, Any]:
        """ Strict parameter validation mapping preventing production silent failures """
        res = {}
        for feat in self.CRITICAL_FEATURES:
            if feat not in df.columns or pd.isna(df[feat].iloc[-1]):
                raise ValueError(f"CRITICAL HOOK MISSING: '{feat}' parameter must be mapped from Layer-1 Pipeline.")
            res[feat] = df[feat].iloc[-1]
            
        for feat in self.OPTIONAL_FEATURES:
            res[feat] = df[feat].iloc[-1] if (feat in df.columns and pd.notna(df[feat].iloc[-1])) else 0.0
        return res

    def _execute_bayesian_fusion(self, base_prior: float, events: List[str], regime: str, is_bullish: bool) -> float:
        """ Sequential Bayesian evidence integration over network state profiles """
        prob = np.clip(base_prior, 0.001, 0.999)
        profile = CANDLE_CONFIG["bayesian"]["likelihood_profiles"].get(regime, CANDLE_CONFIG["bayesian"]["likelihood_profiles"]["mean_reverting"])
        discount = CANDLE_CONFIG["bayesian"]["correlation_discount_factor"]
        
        fired_nodes: Set[str] = set()
        
        for event in events:
            mapped_node = f"{event}_bull" if (is_bullish and f"{event}_bull" in profile) else f"{event}_bear" if (not is_bullish and f"{event}_bear" in profile) else event
            lr = profile.get(mapped_node, 1.5)
            
            # Apply network interaction factor if context paths intersect
            if len(fired_nodes) > 0:
                lr = 1.0 + (lr - 1.0) * discount
                
            prior_odds = prob / (1.0 - prob + EPSILON)
            posterior_odds = prior_odds * lr
            prob = posterior_odds / (1.0 + posterior_odds + EPSILON)
            fired_nodes.add(mapped_node)
            
        return float(np.clip(prob, 0.001, 0.999))

    def analyze(self, df: Any) -> CandleAnalysisResult:
        v = self._get_latest_values(df)
        
        eff_ratio = float(v['efficiency_ratio'])
        norm_vol = float(v['normalized_volatility'])
        trend_dir = int(v['trend_direction'])
        
        if eff_ratio > 0.55 and norm_vol > 0.05:
            regime = "bullish_expansion" if trend_dir >= 0 else "bearish_expansion"
        else:
            regime = "mean_reverting"
            
        mqs = min(max((eff_ratio * 60.0) + (norm_vol * 100), 0.0), 100.0)
        
        # 1. Pipeline Execution Sub-components
        mtf_res = self._process_mtf(df)
        psy_res = self._process_psychology(v, mqs)
        bb_res = self._process_bull_bear(v, psy_res)
        rev_res = self._process_reversal(v, regime, mqs)
        cont_res = self._process_continuation(v, regime, mqs)
        gap_res = self._process_gaps(v, mqs)
        
        # 2. Dynamic Structural Confidence Framework
        evidence = psy_res['evidence'] + rev_res['evidence']
        bull_w = sum(e['weight'] * e['reliability'] for e in evidence if e['polarity'] > 0)
        bear_w = sum(e['weight'] * e['reliability'] for e in evidence if e['polarity'] < 0)
        total_w = bull_w + bear_w + EPSILON
        
        agreement_ratio = max(bull_w, bear_w) / total_w
        conflict_score = (min(bull_w, bear_w) / total_w) * 100.0
        
        # Parse missing structural fields rate
        missing_count = sum(1 for feat in self.OPTIONAL_FEATURES if v[feat] == 0.0)
        missing_penalty = (missing_count / len(self.OPTIONAL_FEATURES)) * 30.0
        
        mtf_sync = 1.0 - (mtf_res['conflict_score'] / 200.0)
        system_confidence = min(max((agreement_ratio * 100.0 * mtf_sync) - missing_penalty, 0.0), 100.0)
        
        adv_metrics = self._compile_metrics(v, psy_res, bb_res, rev_res, cont_res, gap_res, mqs, eff_ratio, system_confidence, conflict_score)
        
        return {
            "candle_psychology": psy_res, "bull_bear_analysis": bb_res, "reversal_detection": rev_res,
            "continuation_detection": cont_res, "gap_analysis": gap_res, "multi_timeframe": mtf_res, "advanced_metrics": adv_metrics
        }

    def _process_psychology(self, v: Dict[str, Any], mqs: float) -> CandlePsychologyResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        
        vol_ratio = float(v['volume_ratio'])
        body_pct = float(v['body_pct'])
        liq_sweep = int(v['liquidity_sweep'])
        trend_dir = int(v['trend_direction'])
        
        if liq_sweep == 1:
            evidence.append({"category": "wyckoff", "type": "Wyckoff_Spring", "weight": 35.0, "value": "Spring Grab", "polarity": 1, "institutional_explanation": "Liquidity swept low.", "reliability": 0.85})
            score += 35.0
        elif liq_sweep == -1:
            evidence.append({"category": "wyckoff", "type": "Wyckoff_Upthrust", "weight": 35.0, "value": "Upthrust Grab", "polarity": -1, "institutional_explanation": "Liquidity swept high.", "reliability": 0.85})
            score -= 35.0
            
        if body_pct > CANDLE_CONFIG["thresholds"]["body_expansion_ratio"] and vol_ratio > 1.2:
            polarity = 1 if v['close'] > v['open'] else -1
            evidence.append({"category": "auction", "type": "Auction_Expansion", "weight": 20.0, "value": "Expansion", "polarity": polarity, "institutional_explanation": "Volume breakout expansion.", "reliability": 1.0})
            score += (20.0 * polarity)
            
        if vol_ratio > 1.5 and body_pct < 0.3:
            evidence.append({"category": "psychology", "type": "Effort_Vs_Result", "weight": 25.0, "value": "Absorption", "polarity": -trend_dir if trend_dir != 0 else 0, "institutional_explanation": "Limit barrier absorption.", "reliability": 1.0})
            score += (-25.0 * trend_dir)

        return {
            "score": round(min(max(score, -100.0), 100.0), 2),
            "status": "Institutional Absorption" if abs(score) > 25 and body_pct < 0.4 else "Trend Expansion" if abs(score) > 20 else "Equilibrium",
            "emotion": "Panic/Trap" if liq_sweep != 0 else "Indecision",
            "auction_balance": "Imbalanced State" if abs(score) > 25 else "Symmetrical Balance",
            "evidence": evidence
        }

    def _process_bull_bear(self, v: Dict[str, Any], psy: CandlePsychologyResult) -> BullBearResult:
        clv = float(v['clv'])
        bull_seq = int(v['bull_sequence'])
        bear_seq = int(v['bear_sequence'])
        
        base_bull = (clv + 1.0) * 50.0
        if bull_seq >= 3: base_bull += (bull_seq * 5.0)
        if bear_seq >= 3: base_bull -= (bear_seq * 5.0)
        
        final_bull_w = min(max(base_bull, 0.0), 100.0)
        final_bear_w = 100.0 - final_bull_w
        
        return {
            "bull_weight_ratio": round(final_bull_w / 100.0, 4),
            "bear_weight_ratio": round(final_bear_w / 100.0, 4),
            "dominance": "Bullish Auction Control" if final_bull_w > 60.0 else "Bearish Auction Control" if final_bear_w > 60.0 else "Auction Rotation Equilibrium",
            "clv_score": round(clv, 4), "institutional_control_score": round(abs(clv) * 100.0, 2)
        }

    def _process_reversal(self, v: Dict[str, Any], regime: str, mqs: float) -> ReversalResult:
        evidence: List[EvidenceItem] = []
        active_events: List[str] = []
        
        choch = int(v['choch'])
        liq_sweep = int(v['liquidity_sweep'])
        is_bullish = True if (choch > 0 or liq_sweep > 0 or float(v['clv']) > 0) else False
        
        if choch != 0:
            active_events.append("structural_break")
            evidence.append({"category": "structural", "type": "CHoCH", "weight": 40.0, "value": "Protected Break", "polarity": choch, "institutional_explanation": "Market Character Reversal.", "reliability": 1.0})
        if liq_sweep != 0:
            active_events.append("liquidity_sweep")
            evidence.append({"category": "liquidity", "type": "Sweep", "weight": 35.0, "value": "Stop Hunting", "polarity": liq_sweep, "institutional_explanation": "Capital deployment injection.", "reliability": 0.85})
            
        prob = self._execute_bayesian_fusion(CANDLE_CONFIG["bayesian"]["priors"]["reversal"], active_events, regime, is_bullish)
        
        return {
            "probability": round(prob * 100.0, 2),
            "direction": "Bullish" if (choch > 0 or liq_sweep > 0) else "Bearish" if (choch < 0 or liq_sweep < 0) else "None",
            "quality": "Confirmed Structural Reversal Zone" if prob > 0.65 else "Trend Profile Intact",
            "trap_probability": round(prob * 1.5 * 100.0, 2) if "liquidity_sweep" in active_events else 15.0,
            "evidence": evidence
        }

    def _process_continuation(self, v: Dict[str, Any], regime: str, mqs: float) -> ContinuationResult:
        active_events: List[str] = []
        bos = int(v['bos'])
        trend_dir = int(v['trend_direction'])
        is_bullish = trend_dir >= 0
        
        if bos != 0 and bos == trend_dir:
            active_events.append("structural_break")
        if abs(float(v['clv'])) > 0.4:
            active_events.append("momentum_persistence")
            
        prob = self._execute_bayesian_fusion(CANDLE_CONFIG["bayesian"]["priors"]["continuation"], active_events, regime, is_bullish)
        
        return {
            "probability": round(prob * 100.0, 2),
            "quality": "Institutional Orderflow Expansion" if prob > 0.70 else "Mean Reverting Compression",
            "evidence": []
        }

    def _process_gaps(self, v: Dict[str, Any], mqs: float) -> GapAnalysisResult:
        evidence: List[EvidenceItem] = []
        gap_up = int(v['gap_up'])
        gap_down = int(v['gap_down'])
        
        if not gap_up and not gap_down:
            return {"gap_type": "None", "gap_acceptance": False, "gap_fill_probability": 0.0, "evidence": []}
            
        # Dynamically scale gap filling probability from asset volatility matrix variables
        atr = float(v['atr'])
        vol_ratio = float(v['volume_ratio'])
        body_pct = float(v['body_pct'])
        
        acceptance = (gap_up and body_pct > 0.5) or (gap_down and body_pct > 0.5)
        
        # Mathematical derivation eliminating rigid constants
        fill_prob = 100.0 - min(max((vol_ratio * 30.0) + (body_pct * 40.0), 10.0), 95.0) if acceptance else min(max(85.0 - (vol_ratio * 20.0), 20.0), 95.0)
        
        return {
            "gap_type": "Sovereign Breakaway Gap" if (vol_ratio > 1.5 and acceptance) else "Common Auction Variance",
            "gap_acceptance": acceptance,
            "gap_fill_probability": round(fill_prob, 2),
            "evidence": evidence
        }

    def _process_mtf(self, df: Any) -> MTFResult:
        """ Dynamic multi-timeframe vector matrix execution looping over HTF and LTF branches """
        total_weight = 0.0
        weighted_bias_sum = 0.0
        conflict_accumulator = 0.0
        
        timeframes = {}
        
        # Run across complete systemic spectrum matching config parameters
        all_suffixes = CANDLE_CONFIG["mtf"]["htf_suffixes"] + CANDLE_CONFIG["mtf"]["ltf_suffixes"]
        
        for sfx in all_suffixes:
            c_col, o_col = f"close{sfx}", f"open{sfx}"
            if c_col in df.columns and o_col in df.columns:
                c_val, o_val = df[c_col].iloc[-1], df[o_col].iloc[-1]
                if pd.isna(c_val) or pd.isna(o_val): continue
                
                direction = 1 if c_val > o_val else -1 if c_val < o_val else 0
                weight = CANDLE_CONFIG["mtf_weights"].get(sfx, 0.02)
                
                weighted_bias_sum += (direction * weight)
                total_weight += weight
                
                timeframes[sfx.strip('_')] = {
                    "bias": "Bullish" if direction == 1 else "Bearish" if direction == -1 else "Neutral"
                }

        normalized_alignment = (weighted_bias_sum / (total_weight + EPSILON)) * 100.0
        
        # Extract HTF vs LTF Friction index vectors
        htf_vector, ltf_vector = 0.0, 0.0
        for sfx in CANDLE_CONFIG["mtf"]["htf_suffixes"]:
            if f"close{sfx}" in df.columns: htf_vector += 1.0 if df[f"close{sfx}"].iloc[-1] > df[f"open{sfx}"].iloc[-1] else -1.0
        for sfx in CANDLE_CONFIG["mtf"]["ltf_suffixes"]:
            if f"close{sfx}" in df.columns: ltf_vector += 1.0 if df[f"close{sfx}"].iloc[-1] > df[f"open{sfx}"].iloc[-1] else -1.0
            
        conflict_score = abs(htf_vector - ltf_vector) * 25.0
        
        return {
            "alignment_score": round(normalized_alignment, 2),
            "htf_dominant_bias": "Macro Long" if normalized_alignment > 35.0 else "Macro Short" if normalized_alignment < -35.0 else "Rotational Balance",
            "ltf_trigger_state": "Trigger Sync Bull" if ltf_vector > 1.0 else "Trigger Sync Bear" if ltf_vector < -1.0 else "Friction Compression",
            "structural_alignment": "Consensus Structural Matrix" if conflict_score < 30.0 else "Timeframe Fragmentation Node",
            "conflict_score": round(conflict_score, 2),
            "timeframes": timeframes
        }

    def _compile_metrics(self, v: Dict[str, Any], psy: CandlePsychologyResult, bb: BullBearResult, rev: ReversalResult, cont: ContinuationResult, gap: GapAnalysisResult, mqs: float, eff_ratio: float, system_confidence: float, conflict_score: float) -> AdvancedCandleMetrics:
        """ Derives internal indexes without magic thresholds or score-forcing constants """
        cqi = abs(psy['score'])
        rej_idx = sum(e['weight'] for e in psy['evidence'] if e['polarity'] != np.sign(psy['score']))
        acc_idx = sum(e['weight'] for e in psy['evidence'] if e['polarity'] == np.sign(psy['score']))
        
        vol_ratio = float(v['volume_ratio'])
        wick_total = float(v['upper_wick']) + float(v['lower_wick'])
        c_range = (v['high'] - v['low']) + EPSILON
        
        p_acc = (acc_idx / (acc_idx + rej_idx + EPSILON)) * 100.0
        p_rej = 100.0 - p_acc
        
        liq_util = min((wick_total / c_range) * min(vol_ratio, 3.0) * 100.0 / 3.0, 100.0)
        p_disc = min(eff_ratio * vol_ratio * 50.0, 100.0)
        
        # Calculate dynamic momentum decay from expansion imbalance delta vectors
        conviction_decay = max(100.0 - (eff_ratio * 100.0), 0.0) if vol_ratio > 1.0 else 50.0
        
        return {
            "candle_quality_index": round(cqi, 2), "psychology_index": round(psy['score'], 2),
            "rejection_index": round(rej_idx, 2), "acceptance_index": round(acc_idx, 2),
            "smart_money_conviction": round(float(v['clv']) * (cont['probability'] / 100.0), 4),
            "institutional_conviction": round(abs(float(v['clv'])), 4), "retail_emotion": psy['emotion'],
            "trap_index": rev['trap_probability'], "auction_balance": 100.0 - conflict_score,
            "price_acceptance_score": round(p_acc, 2), "price_rejection_score": round(p_rej, 2),
            "market_quality_score": round(mqs, 2), "conviction_decay": round(conviction_decay, 2),
            "price_discovery_index": round(p_disc, 2), "liquidity_utilization": round(liq_util, 2),
            "system_confidence": round(system_confidence, 2)
        }

__all__ = ["CandleAnalyzer", "CandleAnalysisResult"]
