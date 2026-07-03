import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS (100% CONFIG-DRIVEN)
# ==============================================================================
EPSILON = 1e-9

SMART_MONEY_CONFIG = {
    "bayesian_priors": {
        "trend_continuation": 0.60,
        "reversal_base": 0.25,
        "liquidity_trap": 0.35,
        "institutional_presence": 0.20
    },
    "likelihood_ratios": {
        "strong_volume": 2.5,
        "weak_volume": 0.4,
        "trend_alignment": 1.8,
        "counter_trend": 0.5,
        "liquidity_sweep": 3.2,
        "fresh_zone": 2.2,
        "mitigated_zone": 0.35,
        "fvg_magnet": 2.4,
        "equal_high_low": 1.6,
        "breaker_conversion": 2.9,
        "swing_failure": 3.6,
        "inducement": 2.0
    },
    "bayesian_correlation_correction": {
        "base_dampening_exponent": 0.85,
        "multi_variable_penalty_step": 0.05,
        "max_dampening_limit": 0.50
    },
    "scoring_weights": {
        "bos_base_quality": 40.0,
        "bos_volume_bonus": 25.0,
        "bos_trend_bonus": 20.0,
        "bos_trend_penalty": -15.0,
        "choch_base_quality": 45.0,
        "choch_sweep_bonus": 30.0,
        "choch_failure_bonus": 20.0,
        "ob_base_strength": 35.0,
        "ob_fresh_bonus": 35.0,
        "ob_mitigated_penalty": -20.0,
        "ob_breaker_bonus": 30.0,
        "fvg_base_quality": 30.0,
        "fvg_unfilled_bonus": 30.0,
        "fvg_partial_bonus": 15.0,
        "fvg_inverse_bonus": 25.0,
        "liquidity_base_efficiency": 50.0,
        "liquidity_sweep_bonus": 40.0,
        "liquidity_pool_penalty": -20.0,
        "structure_alignment_bonus": 25.0
    },
    "evidence_reliabilities": {
        "expansion_bos": 0.88,
        "low_vol_bos": 0.90,
        "pro_trend": 0.85,
        "counter_trend": 0.70,
        "sweep_choch": 0.96,
        "swing_failure": 0.91,
        "fractal_sync": 0.93,
        "coiled_compression": 0.80,
        "fresh_ob": 0.94,
        "mitigated_ob": 0.75,
        "breaker_flip": 0.89,
        "unfilled_fvg": 0.92,
        "partial_fvg": 0.70,
        "inverse_fvg": 0.86,
        "institutional_raid": 0.95,
        "retail_pool": 0.89
    },
    "signal_hierarchy_routing": {
        "priority_order": [
            "external_choch", "external_bos", "internal_choch", "internal_bos",
            "liquidity_sweep", "fresh_ob", "fvg", "volume", "equal_high_low"
        ],
        "weights": {
            "external_choch": 5.0,
            "external_bos": 4.0,
            "internal_choch": 3.0,
            "internal_bos": 2.5,
            "liquidity_sweep": 3.5,
            "fresh_ob": 2.0,
            "fvg": 1.5,
            "volume": 1.5
        },
        "penalties": {
            "equal_high_low_conviction_hit": 15.0
        },
        "conflict_threshold": 50.0,
        "conviction_multiplier": 20.0
    },
    "feature_importance": {
        "bos": 1.0, "choch": 1.0, "order_block": 1.0, "fair_value_gap": 1.0,
        "liquidity_sweep": 1.0, "trend_direction": 0.8, "volume_confirmation": 0.9,
        "breaker_block": 0.7, "mitigation_block": 0.6, "fvg_fill_ratio": 0.8,
        "protected_high": 0.9, "protected_low": 0.9, "swing_failure": 0.9,
        "internal_liquidity": 0.6, "external_liquidity": 0.6, "equal_highs": 0.7
    },
    "lifecycle_factors": {
        "ob_decay_rate": 0.10,
        "ob_max_decay": 20.0,
        "ob_expiry_age": 100.0,
        "fvg_decay_rate": 0.05,
        "fvg_max_decay": 15.0,
        "fvg_nested_multiplier": 1.10,
        "fvg_stack_priority": 2.0,
        "fvg_base_priority": 1.0
    },
    "penalties": {
        "invalid_state_penalty": 10.0,
        "missing_feature_multiplier": 1.5
    },
    "thresholds": {
        "moderate_confirmation": 50.0,
        "volume_breakout_min": 1.25,
        "fvg_partial_fill_max": 0.98,
        "fvg_unfilled_max": 0.01,
        "compression_max": 0.50
    },
    "action_thresholds": {
        "reversal_min": 65.0,
        "continuation_min": 65.0,
        "trap_min": 60.0
    },
    "defaults": {
        "fallback_weight": 0.1,
        "evidence_reliability": 0.50
    }
}

# ==============================================================================
# SCHEMAS & TYPE DEFINITIONS
# ==============================================================================
class EvidenceItem(TypedDict):
    category: str
    feature: str
    weight: float
    polarity: int
    reliability: float
    institutional_explanation: str

class BOSResult(TypedDict):
    bullish_bos: bool
    bearish_bos: bool
    major_bos: bool
    minor_bos: bool
    internal_bos: bool
    external_bos: bool
    bos_quality: float
    bos_confirmation: bool
    bos_strength: float
    continuation_probability: float
    evidence: List[EvidenceItem]

class CHOCHResult(TypedDict):
    bullish_choch: bool
    bearish_choch: bool
    internal_choch: bool
    external_choch: bool
    reversal_probability: float
    structural_shift_quality: float
    evidence: List[EvidenceItem]

class MarketStructureResult(TypedDict):
    hh_hl: bool
    lh_ll: bool
    strong_swing: bool
    weak_swing: bool
    protected_swing: bool
    violation_level: float
    internal_external_alignment_score: float
    fractal_quality: float
    expansion: bool
    compression: bool
    trend_state: str
    internal_structure: str
    external_structure: str
    structure_quality: float
    structure_score: float
    evidence: List[EvidenceItem]

class OrderBlockResult(TypedDict):
    bullish_ob_detected: bool
    bearish_ob_detected: bool
    active_order_block: bool
    fresh: bool
    tapped_once: bool
    tapped_multiple: bool
    consumed: bool
    invalidated: bool
    breaker_converted: bool
    expired: bool
    ob_age_score: float
    reaction_score: float
    decay_score: float
    institutional_strength: float
    reaction_probability: float
    evidence: List[EvidenceItem]

class FairValueGapResult(TypedDict):
    bullish_fvg: bool
    bearish_fvg: bool
    inverse_fvg: bool
    nested_quality: float
    fvg_stack_priority: float
    premium_discount_position: str
    htf_alignment: str
    filled: bool
    unfilled: bool
    partial_fill: bool
    fvg_width: float
    fvg_fill_ratio: float
    fvg_age: float
    decay_score: float
    imbalance_strength: float
    fvg_quality: float
    magnet_probability: float
    evidence: List[EvidenceItem]

class LiquidityResult(TypedDict):
    liquidity_pools_active: bool
    liquidity_heatmap_score: float
    pool_ranking_score: float
    sweep_efficiency: float
    engineered_liquidity: bool
    liquidity_density: float
    liquidity_age: float
    equal_highs_strength: float
    equal_lows_strength: float
    internal_liquidity: bool
    external_liquidity: bool
    resting_liquidity: bool
    liquidity_void: bool
    liquidity_exhaustion: bool
    liquidity_consumption: bool
    buy_side_liquidity: bool
    sell_side_liquidity: bool
    liquidity_sweep: bool
    sweep_strength: float
    stop_hunt: bool
    inducement: bool
    liquidity_efficiency: float
    liquidity_pool_strength: float
    evidence: List[EvidenceItem]

class InstitutionalRoutingResult(TypedDict):
    primary_event: str
    primary_zone: str
    primary_liquidity: str
    primary_trigger: str
    primary_smc_signal: str
    primary_institutional_bias: str
    secondary_confirmation: str
    signal_hierarchy: str
    conflict_score: float
    consensus_score: float
    institutional_conviction: float

class AdvancedSmartMoneyMetrics(TypedDict):
    smart_money_score: float
    institutional_control: str
    institutional_volume_score: float
    market_structure_quality: float
    bos_quality: float
    choch_quality: float
    order_block_strength: float
    fvg_quality: float
    liquidity_quality: float
    liquidity_efficiency: float
    trap_probability: float
    continuation_probability: float
    reversal_probability: float
    system_confidence: float
    market_quality_score: float

class SmartMoneyAnalysisResult(TypedDict):
    bos: BOSResult
    choch: CHOCHResult
    market_structure: MarketStructureResult
    order_block: OrderBlockResult
    fair_value_gap: FairValueGapResult
    liquidity: LiquidityResult
    routing: InstitutionalRoutingResult
    advanced_metrics: AdvancedSmartMoneyMetrics
    summary: Dict[str, Any]

# ==============================================================================
# MAIN ANALYZER ENGINE
# ==============================================================================
class SmartMoneyAnalyzer:
    """
    Sovereign SMC Layer-2 Engine. V114 Gold Final.
    """
    L1_FEATURES = [
        'close', 'volume', 'atr_14', 'swing_high', 'swing_low', 'trend_direction',
        'trend_strength', 'volume_confirmation', 'market_regime', 'breakout_pressure',
        'compression_pct', 'bos', 'major_bos', 'minor_bos', 'internal_bos', 'external_bos',
        'choch', 'internal_choch', 'external_choch', 'displacement_strength', 'protected_high',
        'protected_low', 'swing_failure', 'structure_score', 'violation_level', 'order_block',
        'bullish_ob_detected', 'bearish_ob_detected', 'active_order_block', 'mitigation_block',
        'breaker_block', 'ob_age', 'ob_efficiency', 'reaction_count', 'fair_value_gap',
        'inverse_fvg', 'nested_fvg', 'fvg_stack', 'fvg_width', 'fvg_fill_ratio', 'fvg_age',
        'imbalance_strength', 'htf_alignment', 'liquidity_sweep', 'sweep_strength', 'equal_highs',
        'equal_lows', 'equal_highs_strength', 'equal_lows_strength', 'internal_liquidity',
        'external_liquidity', 'resting_liquidity', 'liquidity_void', 'liquidity_exhaustion',
        'liquidity_consumption', 'liquidity_pool_strength', 'inducement', 'liquidity_age',
        'institutional_volume_score'
    ]

    def __init__(self):
        self.config = SMART_MONEY_CONFIG
        self._missing_features: List[str] = []
        self._state_issues: List[str] = []
        self._base_confidence: float = 100.0
        self._evidence_tracker: List[Tuple[float, float]] = []

    def _validate_l1_integrity(self, l1: Dict[str, Any]) -> float:
        penalty = 0.0
        p_val = self.config['penalties']['invalid_state_penalty']
        has_sub_bos = l1.get('major_bos') or l1.get('minor_bos') or l1.get('internal_bos') or l1.get('external_bos')
        if has_sub_bos and l1.get('bos') == 0.0:
            penalty += p_val
            self._state_issues.append("Sub-BOS triggered without Primary BOS active.")
        has_sub_choch = l1.get('internal_choch') or l1.get('external_choch')
        if has_sub_choch and l1.get('choch') == 0.0:
            penalty += p_val
            self._state_issues.append("Sub-CHOCH triggered without Primary CHOCH active.")
        if l1.get('active_order_block') and not (l1.get('bullish_ob_detected') or l1.get('bearish_ob_detected')):
            penalty += p_val
            self._state_issues.append("Active OB lacks directional polarity mapping.")
        return penalty

    def _extract_l1_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        self._missing_features.clear()
        self._state_issues.clear()
        self._evidence_tracker.clear()
        extracted = {}
        latest = df.iloc[-1]
        penalty = 0.0
        total_weight = sum(self.config['feature_importance'].values())
        fallback_w = self.config['defaults']['fallback_weight']
        
        for feat in self.L1_FEATURES:
            if feat in df.columns:
                val = latest[feat]
                if pd.isna(val):
                    extracted[feat] = "Neutral" if feat in ['market_regime', 'htf_alignment'] else 0.0
                elif isinstance(val, (bool, np.bool_)):
                    extracted[feat] = 1.0 if val else 0.0
                else:
                    extracted[feat] = str(val) if isinstance(val, str) else float(val)
            else:
                extracted[feat] = "Neutral" if feat in ['market_regime', 'htf_alignment'] else 0.0
                self._missing_features.append(feat)
                
            feat_weight = self.config['feature_importance'].get(feat, fallback_w)
            if feat not in df.columns:
                penalty += (feat_weight / total_weight) * 100.0
                
        state_penalty = self._validate_l1_integrity(extracted)
        mult = self.config['penalties']['missing_feature_multiplier']
        self._base_confidence = np.clip(100.0 - (penalty * mult) - state_penalty, 0.0, 100.0)
        return extracted

    def _dampened_bayesian_update(self, prior_prob: float, likelihood_ratios: List[float]) -> float:
        if prior_prob <= 0.0 or prior_prob >= 1.0 or not likelihood_ratios:
            return prior_prob * 100.0
        prior_odds = prior_prob / (1.0 - prior_prob)
        corr_cfg = self.config['bayesian_correlation_correction']
        num_evidence = len(likelihood_ratios)
        dampening_exponent = max(
            corr_cfg['base_dampening_exponent'] - (num_evidence * corr_cfg['multi_variable_penalty_step']),
            corr_cfg['max_dampening_limit']
        )
        product_lr = 1.0
        for lr in likelihood_ratios:
            product_lr *= lr
        corrected_lr = product_lr ** dampening_exponent
        posterior_odds = prior_odds * corrected_lr
        posterior_prob = posterior_odds / (1.0 + posterior_odds)
        return round(posterior_prob * 100.0, 2)

    def _add_evidence(self, ev_list: List[EvidenceItem], category: str, feature: str, weight: float, polarity: int, reliability_key: str, explanation: str):
        rel = self.config['evidence_reliabilities'].get(reliability_key, self.config['defaults']['evidence_reliability'])
        self._evidence_tracker.append((float(weight), float(rel)))
        ev_list.append({
            "category": category,
            "feature": feature,
            "weight": float(weight),
            "polarity": int(polarity),
            "reliability": float(rel),
            "institutional_explanation": explanation
        })

    def _process_bos(self, l1: Dict[str, Any]) -> BOSResult:
        evidence: List[EvidenceItem] = []
        lrs: List[float] = []
        sc = self.config['scoring_weights']
        bos_val = l1.get('bos', 0.0)
        vol_conf = l1.get('volume_confirmation', 1.0)
        trend_dir = l1.get('trend_direction', 0.0)
        disp_str = l1.get('displacement_strength', 0.0)
        bullish_bos = bos_val >= 1.0
        bearish_bos = bos_val <= -1.0
        bos_quality = 0.0
        bos_strength = 0.0
        continuation_prob = 0.0
        
        if bullish_bos or bearish_bos:
            polarity = 1 if bullish_bos else -1
            bos_quality += sc['bos_base_quality']
            if vol_conf > self.config['thresholds']['volume_breakout_min']:
                lrs.append(self.config['likelihood_ratios']['strong_volume'])
                bos_quality += sc['bos_volume_bonus']
                self._add_evidence(evidence, "Volume", "Expansion BOS", sc['bos_volume_bonus'], polarity, "expansion_bos", "BOS validated by genuine volume expansion matrix.")
            else:
                lrs.append(self.config['likelihood_ratios']['weak_volume'])
                self._add_evidence(evidence, "Volume", "Low Volume BOS", sc['bos_volume_bonus'], -polarity, "low_vol_bos", "Imbalance failed volume validation. Highly prospective fakeout structure.")
                
            if (bullish_bos and trend_dir > 0) or (bearish_bos and trend_dir < 0):
                lrs.append(self.config['likelihood_ratios']['trend_alignment'])
                bos_quality += sc['bos_trend_bonus']
                self._add_evidence(evidence, "Structure", "Pro-Trend BOS", sc['bos_trend_bonus'], polarity, "pro_trend", "Structural breakout aligns smoothly with order-flow direction.")
            else:
                lrs.append(self.config['likelihood_ratios']['counter_trend'])
                bos_quality += sc['bos_trend_penalty']
                self._add_evidence(evidence, "Structure", "Counter-Trend BOS", abs(sc['bos_trend_penalty']), -polarity, "counter_trend", "Counter-trend displacement. Susceptible to immediate premium pullbacks.")
                
            continuation_prob = self._dampened_bayesian_update(self.config['bayesian_priors']['trend_continuation'], lrs)
            bos_quality = np.clip(bos_quality + disp_str, 0.0, 100.0)
            bos_strength = np.clip(disp_str * 10.0 + (vol_conf * 15.0), 0.0, 100.0)
            
        return {
            "bullish_bos": bullish_bos, "bearish_bos": bearish_bos,
            "major_bos": l1.get('major_bos', 0.0) != 0.0, "minor_bos": l1.get('minor_bos', 0.0) != 0.0,
            "internal_bos": l1.get('internal_bos', 0.0) != 0.0, "external_bos": l1.get('external_bos', 0.0) != 0.0,
            "bos_quality": round(bos_quality, 2), "bos_confirmation": bos_quality >= self.config['thresholds']['moderate_confirmation'],
            "bos_strength": round(bos_strength, 2), "continuation_probability": continuation_prob if (bullish_bos or bearish_bos) else 0.0,
            "evidence": evidence
        }

    def _process_choch(self, l1: Dict[str, Any]) -> CHOCHResult:
        evidence: List[EvidenceItem] = []
        lrs: List[float] = []
        sc = self.config['scoring_weights']
        choch_val = l1.get('choch', 0.0)
        liq_sweep = l1.get('liquidity_sweep', 0.0)
        sf = l1.get('swing_failure', 0.0)
        bullish_choch = choch_val >= 1.0
        bearish_choch = choch_val <= -1.0
        reversal_prob = 0.0
        shift_quality = 0.0
        
        if bullish_choch or bearish_choch:
            polarity = 1 if bullish_choch else -1
            shift_quality = sc['choch_base_quality']
            if (bullish_choch and liq_sweep <= -1.0) or (bearish_choch and liq_sweep >= 1.0):
                lrs.append(self.config['likelihood_ratios']['liquidity_sweep'])
                shift_quality += sc['choch_sweep_bonus']
                self._add_evidence(evidence, "Liquidity", "Sweep Induced CHOCH", sc['choch_sweep_bonus'], polarity, "sweep_choch", "Liquidity swept cleanly before character shift occurred.")
            if sf != 0.0:
                lrs.append(self.config['likelihood_ratios']['swing_failure'])
                shift_quality += sc['choch_failure_bonus']
                self._add_evidence(evidence, "Structure", "Swing Failure Reversal", sc['choch_failure_bonus'], polarity, "swing_failure", "Failed swing confirms macro distribution matrix.")
            reversal_prob = self._dampened_bayesian_update(self.config['bayesian_priors']['reversal_base'], lrs)
            shift_quality = np.clip(shift_quality, 0.0, 100.0)
            
        return {
            "bullish_choch": bullish_choch, "bearish_choch": bearish_choch,
            "internal_choch": l1.get('internal_choch', 0.0) != 0.0, "external_choch": l1.get('external_choch', 0.0) != 0.0,
            "reversal_probability": reversal_prob if (bullish_choch or bearish_choch) else 0.0,
            "structural_shift_quality": round(shift_quality, 2), "evidence": evidence
        }

    def _process_market_structure(self, l1: Dict[str, Any], bos_res: BOSResult, choch_res: CHOCHResult) -> MarketStructureResult:
        evidence: List[EvidenceItem] = []
        sc = self.config['scoring_weights']
        trend_dir = l1.get('trend_direction', 0.0)
        compression = l1.get('compression_pct', 0.0)
        hh_hl = trend_dir > 0
        lh_ll = trend_dir < 0
        is_expansion = l1.get('market_regime') == "Trending"
        is_compression = compression < self.config['thresholds']['compression_max'] and not is_expansion
        t_state = "Bullish Expansion" if (hh_hl and is_expansion) else "Bearish Expansion" if (lh_ll and is_expansion) else "Consolidation"
        int_struct = "Bullish" if choch_res['bullish_choch'] or bos_res['bullish_bos'] else "Bearish" if choch_res['bearish_choch'] or bos_res['bearish_bos'] else "Neutral"
        ext_struct = "Bullish" if hh_hl else "Bearish" if lh_ll else "Neutral"
        align_score = sc['structure_alignment_bonus'] if int_struct == ext_struct and int_struct != "Neutral" else 0.0
        quality = np.clip(l1.get('trend_strength', 0.0) + align_score, 0.0, 100.0)
        
        if align_score > 0.0:
            self._add_evidence(evidence, "Alignment", "Fractal Order Sync", align_score, 1 if hh_hl else -1, "fractal_sync", "Macro matrix and local delivery frames synchronized.")
        if is_compression:
            self._add_evidence(evidence, "Volatility", "Structural Compression", 15.0, 0, "coiled_compression", "Price coiled. Imminent volatility expansion likely.")
            
        return {
            "hh_hl": hh_hl, "lh_ll": lh_ll,
            "strong_swing": l1.get('protected_high', 0.0) != 0.0 or l1.get('protected_low', 0.0) != 0.0,
            "weak_swing": l1.get('swing_failure', 0.0) != 0.0,
            "protected_swing": l1.get('protected_high', 0.0) != 0.0 or l1.get('protected_low', 0.0) != 0.0,
            "violation_level": float(l1.get('violation_level', 0.0)), "internal_external_alignment_score": float(align_score),
            "fractal_quality": round(quality, 2), "expansion": is_expansion, "compression": is_compression,
            "trend_state": t_state, "internal_structure": int_struct, "external_structure": ext_struct,
            "structure_quality": round(quality, 2), "structure_score": round(l1.get('structure_score', 0.0), 2),
            "evidence": evidence
        }

    def _process_order_blocks(self, l1: Dict[str, Any]) -> OrderBlockResult:
        evidence: List[EvidenceItem] = []
        lrs: List[float] = []
        sc = self.config['scoring_weights']
        lc = self.config['lifecycle_factors']
        bull_ob = l1.get('bullish_ob_detected', 0.0) != 0.0
        bear_ob = l1.get('bearish_ob_detected', 0.0) != 0.0
        active_ob = l1.get('active_order_block', 0.0) != 0.0
        breaker = l1.get('breaker_block', 0.0) != 0.0
        react_cnt = l1.get('reaction_count', 0.0)
        ob_age = l1.get('ob_age', 0.0)
        is_fresh = active_ob and react_cnt == 0
        tapped_once = active_ob and react_cnt == 1
        tapped_mult = active_ob and react_cnt > 1
        consumed = not active_ob and react_cnt > 0
        ob_strength = sc['ob_base_strength']
        
        if bull_ob or bear_ob:
            polarity = 1 if bull_ob else -1
            if is_fresh:
                lrs.append(self.config['likelihood_ratios']['fresh_zone'])
                ob_strength += sc['ob_fresh_bonus']
                self._add_evidence(evidence, "SMC", "Fresh Institutional Block", sc['ob_fresh_bonus'], polarity, "fresh_ob", "Unmitigated block containing significant residual orders.")
            elif tapped_once or tapped_mult:
                lrs.append(self.config['likelihood_ratios']['mitigated_zone'])
                ob_strength += sc['ob_mitigated_penalty']
                self._add_evidence(evidence, "SMC", "Mitigated Block", abs(sc['ob_mitigated_penalty']), -polarity, "mitigated_ob", "Zone pre-tested. Order density severely depleted.")
            if breaker:
                lrs.append(self.config['likelihood_ratios']['breaker_conversion'])
                ob_strength += sc['ob_breaker_bonus']
                self._add_evidence(evidence, "SMC", "Breaker Pivot Conversion", sc['ob_breaker_bonus'], -polarity, "breaker_flip", "Breached OB validated as an institutional mitigation flip zone.")
            react_prob = self._dampened_bayesian_update(self.config['bayesian_priors']['institutional_presence'], lrs)
        else:
            react_prob = 0.0
            
        decay = np.clip(ob_age * lc['ob_decay_rate'], 0.0, lc['ob_max_decay'])
        return {
            "bullish_ob_detected": bull_ob, "bearish_ob_detected": bear_ob, "active_order_block": active_ob,
            "fresh": is_fresh, "tapped_once": tapped_once, "tapped_multiple": tapped_mult, "consumed": consumed,
            "invalidated": not active_ob and not breaker, "breaker_converted": breaker, "expired": ob_age > lc['ob_expiry_age'],
            "ob_age_score": float(ob_age), "reaction_score": float(react_cnt), "decay_score": float(decay),
            "institutional_strength": round(np.clip(ob_strength - decay, 0.0, 100.0), 2), "reaction_probability": round(react_prob, 2),
            "evidence": evidence
        }

    def _process_fair_value_gap(self, l1: Dict[str, Any]) -> FairValueGapResult:
        evidence: List[EvidenceItem] = []
        lrs: List[float] = []
        sc = self.config['scoring_weights']
        lc = self.config['lifecycle_factors']
        th = self.config['thresholds']
        fvg_val = l1.get('fair_value_gap', 0.0)
        inv_fvg = l1.get('inverse_fvg', 0.0) != 0.0
        fill_ratio = l1.get('fvg_fill_ratio', 0.0)
        fvg_age = l1.get('fvg_age', 0.0)
        bullish_fvg = fvg_val >= 1.0
        bearish_fvg = fvg_val <= -1.0
        is_filled = fill_ratio >= th['fvg_partial_fill_max']
        is_unfilled = fill_ratio <= th['fvg_unfilled_max']
        partial_fill = not is_filled and not is_unfilled
        fvg_q = sc['fvg_base_quality']
        
        if bullish_fvg or bearish_fvg or inv_fvg:
            polarity = 1 if bullish_fvg else -1
            if is_unfilled:
                lrs.append(self.config['likelihood_ratios']['fvg_magnet'])
                fvg_q += sc['fvg_unfilled_bonus']
                self._add_evidence(evidence, "Imbalance", "Pristine FVG Imbalance", sc['fvg_unfilled_bonus'], polarity, "unfilled_fvg", "Pure algorithmic inefficiency creates major magnetic pull.")
            elif partial_fill:
                fvg_q += sc['fvg_partial_bonus']
                self._add_evidence(evidence, "Imbalance", "Partial Inefficiency Fill", sc['fvg_partial_bonus'], polarity, "partial_fvg", "Imbalance partially mitigated by re-delivery.")
            if inv_fvg:
                lrs.append(self.config['likelihood_ratios']['breaker_conversion'])
                fvg_q += sc['fvg_inverse_bonus']
                self._add_evidence(evidence, "Imbalance", "Inverse FVG Frame", sc['fvg_inverse_bonus'], -polarity, "inverse_fvg", "Imbalance breached and flipped to structural validation zone.")
            mag_prob = self._dampened_bayesian_update(self.config['bayesian_priors']['institutional_presence'], lrs)
        else:
            mag_prob = 0.0
            
        decay = np.clip(fvg_age * lc['fvg_decay_rate'], 0.0, lc['fvg_max_decay'])
        htf_str = str(l1.get('htf_alignment', 'None'))
        return {
            "bullish_fvg": bullish_fvg, "bearish_fvg": bearish_fvg, "inverse_fvg": inv_fvg,
            "nested_quality": float(fvg_q * lc['fvg_nested_multiplier']) if l1.get('nested_fvg') else float(fvg_q),
            "fvg_stack_priority": lc['fvg_stack_priority'] if l1.get('fvg_stack') else lc['fvg_base_priority'],
            "premium_discount_position": "Discount" if bullish_fvg else "Premium" if bearish_fvg else "Equilibrium",
            "htf_alignment": htf_str, "filled": is_filled, "unfilled": is_unfilled, "partial_fill": partial_fill,
            "fvg_width": round(l1.get('fvg_width', 0.0), 4), "fvg_fill_ratio": round(fill_ratio, 2),
            "fvg_age": float(fvg_age), "decay_score": float(decay), "imbalance_strength": round(l1.get('imbalance_strength', 0.0), 2),
            "fvg_quality": round(np.clip(fvg_q - decay, 0.0, 100.0), 2), "magnet_probability": mag_prob if not is_filled else 0.0,
            "evidence": evidence
        }

    def _process_liquidity(self, l1: Dict[str, Any]) -> LiquidityResult:
        evidence: List[EvidenceItem] = []
        sc = self.config['scoring_weights']
        sweep_val = l1.get('liquidity_sweep', 0.0)
        eq_h = l1.get('equal_highs', 0.0) >= 1.0
        eq_l = l1.get('equal_lows', 0.0) >= 1.0
        eq_h_str = l1.get('equal_highs_strength', 0.0)
        eq_l_str = l1.get('equal_lows_strength', 0.0)
        liq_sweep = sweep_val != 0.0
        liq_eff = sc['liquidity_base_efficiency']
        
        if liq_sweep:
            liq_eff += sc['liquidity_sweep_bonus']
            polarity = 1 if sweep_val >= 1.0 else -1
            self._add_evidence(evidence, "Liquidity", "Institutional Raid", sc['liquidity_sweep_bonus'], polarity, "institutional_raid", "Stops engineered and completely purged.")
        if eq_h or eq_l:
            liq_eff += sc['liquidity_pool_penalty']
            self._add_evidence(evidence, "Liquidity", "Retail Engine Pool", abs(sc['liquidity_pool_penalty']), -1 if eq_h else 1, "retail_pool", "Clean retail double levels offer premium fuel targets.")
            
        pool_str = np.clip(eq_h_str + eq_l_str + l1.get('liquidity_pool_strength', 0.0), 0.0, 100.0)
        return {
            "liquidity_pools_active": eq_h or eq_l, "liquidity_heatmap_score": round(pool_str * 1.1, 2),
            "pool_ranking_score": round(pool_str * 0.9, 2), "sweep_efficiency": round(l1.get('sweep_strength', 50.0), 2),
            "engineered_liquidity": eq_h or eq_l, "liquidity_density": round(pool_str, 2),
            "liquidity_age": float(l1.get('liquidity_age', 0.0)), "equal_highs_strength": round(eq_h_str, 2),
            "equal_lows_strength": round(eq_l_str, 2), "internal_liquidity": l1.get('internal_liquidity', 0.0) != 0.0,
            "external_liquidity": l1.get('external_liquidity', 0.0) != 0.0, "resting_liquidity": l1.get('resting_liquidity', 0.0) != 0.0,
            "liquidity_void": l1.get('liquidity_void', 0.0) != 0.0, "liquidity_exhaustion": l1.get('liquidity_exhaustion', 0.0) != 0.0,
            "liquidity_consumption": l1.get('liquidity_consumption', 0.0) != 0.0, "buy_side_liquidity": sweep_val <= -1.0 or eq_h,
            "sell_side_liquidity": sweep_val >= 1.0 or eq_l, "liquidity_sweep": liq_sweep,
            "sweep_strength": round(l1.get('sweep_strength', 0.0), 2), "stop_hunt": liq_sweep,
            "inducement": l1.get('inducement', 0.0) != 0.0, "liquidity_efficiency": round(np.clip(liq_eff, 0.0, 100.0), 2),
            "liquidity_pool_strength": round(pool_str, 2), "evidence": evidence
        }

    def _route_and_arbitrate(self, bos: BOSResult, choch: CHOCHResult, ob: OrderBlockResult, fvg: FairValueGapResult, liq: LiquidityResult, l1: Dict[str, Any]) -> InstitutionalRoutingResult:
        r_cfg = self.config['signal_hierarchy_routing']
        weights = r_cfg['weights']
        bull_score = 0.0
        bear_score = 0.0
        triggers = []
        
        if choch['external_choch']:
            triggers.append("external_choch")
            bull_score += weights['external_choch'] if choch['bullish_choch'] else 0.0
            bear_score += weights['external_choch'] if choch['bearish_choch'] else 0.0
        if bos['external_bos']:
            triggers.append("external_bos")
            bull_score += weights['external_bos'] if bos['bullish_bos'] else 0.0
            bear_score += weights['external_bos'] if bos['bearish_bos'] else 0.0
        if choch['internal_choch']:
            triggers.append("internal_choch")
            bull_score += weights['internal_choch'] if choch['bullish_choch'] else 0.0
            bear_score += weights['internal_choch'] if choch['bearish_choch'] else 0.0
        if bos['internal_bos']:
            triggers.append("internal_bos")
            bull_score += weights['internal_bos'] if bos['bullish_bos'] else 0.0
            bear_score += weights['internal_bos'] if bos['bearish_bos'] else 0.0
        if liq['liquidity_sweep']:
            triggers.append("liquidity_sweep")
            bull_score += weights['liquidity_sweep'] if liq['sell_side_liquidity'] else 0.0
            bear_score += weights['liquidity_sweep'] if liq['buy_side_liquidity'] else 0.0
        if ob['fresh']:
            triggers.append("fresh_ob")
            bull_score += weights['fresh_ob'] if ob['bullish_ob_detected'] else 0.0
            bear_score += weights['fresh_ob'] if ob['bearish_ob_detected'] else 0.0
        if fvg['unfilled'] or fvg['partial_fill']:
            triggers.append("fvg")
            bull_score += weights['fvg'] if fvg['bullish_fvg'] else 0.0
            bear_score += weights['fvg'] if fvg['bearish_fvg'] else 0.0
        if l1.get('volume_confirmation', 1.0) > self.config['thresholds']['volume_breakout_min']:
            triggers.append("volume")
            if l1.get('trend_direction', 0.0) > 0:
                bull_score += weights['volume']
            elif l1.get('trend_direction', 0.0) < 0:
                bear_score += weights['volume']
                
        pool_conviction_hit = 0.0
        if liq['liquidity_pools_active']:
            triggers.append("equal_high_low")
            pool_conviction_hit = r_cfg['penalties']['equal_high_low_conviction_hit']
            
        primary_trigger = "None"
        for priority in r_cfg['priority_order']:
            if priority in triggers:
                primary_trigger = priority.upper().replace('_', ' ')
                break
                
        conflict = 0.0
        if bull_score > 0.0 and bear_score > 0.0:
            conflict = (min(bull_score, bear_score) / max(bull_score, bear_score)) * 100.0
            
        bias = "Neutral"
        if conflict < r_cfg['conflict_threshold']:
            if bull_score > bear_score:
                bias = "Bullish"
            elif bear_score > bull_score:
                bias = "Bearish"
                
        consensus = 100.0 - conflict
        conviction = max(bull_score, bear_score) * r_cfg['conviction_multiplier'] * (consensus / 100.0)
        conviction = np.clip(conviction - pool_conviction_hit, 0.0, 100.0)
        
        return {
            "primary_event": "Structural Reversal" if "CHOCH" in primary_trigger else "Structural Continuation" if "BOS" in primary_trigger else "Liquidity Raid",
            "primary_zone": "Order Block Matrix" if ob['active_order_block'] else "Fair Value Inefficiency" if fvg['unfilled'] else "Equilibrium",
            "primary_liquidity": "Engineered Retail Pools" if liq['liquidity_pools_active'] else "Sovereign Target Swept",
            "primary_trigger": primary_trigger, "primary_smc_signal": primary_trigger if primary_trigger != "None" else "No Core Trigger",
            "primary_institutional_bias": bias, "secondary_confirmation": "Order Flow Confirmed" if consensus > 70.0 and bias != "Neutral" else "Mixed Flow",
            "signal_hierarchy": f"Level-2 Priority: {primary_trigger}", "conflict_score": round(conflict, 2),
            "consensus_score": round(consensus, 2), "institutional_conviction": round(conviction, 2)
        }

    def _compile_advanced_metrics(self, bos: BOSResult, choch: CHOCHResult, ms: MarketStructureResult, ob: OrderBlockResult, fvg: FairValueGapResult, liq: LiquidityResult, route: InstitutionalRoutingResult, l1: Dict[str, Any]) -> AdvancedSmartMoneyMetrics:
        w = self.config['scoring_weights']
        sms = ((bos['bos_quality'] / 100.0) * w['bos_base_quality']) + \
              ((choch['structural_shift_quality'] / 100.0) * w['choch_base_quality']) + \
              ((ob['institutional_strength'] / 100.0) * w['ob_base_strength']) + \
              ((fvg['fvg_quality'] / 100.0) * w['fvg_base_quality']) + \
              ((liq['liquidity_efficiency'] / 100.0) * w['liquidity_base_efficiency'])
              
        trap_prob = 0.0
        lrs = []
        if liq['liquidity_pools_active']:
            lrs.append(self.config['likelihood_ratios']['equal_high_low'])
        if liq['inducement']:
            lrs.append(self.config['likelihood_ratios']['inducement'])
        if lrs:
            trap_prob = self._dampened_bayesian_update(self.config['bayesian_priors']['liquidity_trap'], lrs)
            
        if self._evidence_tracker:
            total_w = sum(e[0] for e in self._evidence_tracker)
            avg_rel = sum(e[0] * e[1] for e in self._evidence_tracker) / total_w if total_w > 0 else self.config['defaults']['evidence_reliability']
        else:
            avg_rel = self.config['defaults']['evidence_reliability']
            
        final_system_confidence = np.clip(self._base_confidence * avg_rel, 0.0, 100.0)
        mq_score = np.clip((sms * 0.3) + (ms['structure_quality'] * 0.3) + (route['consensus_score'] * 0.2) + (final_system_confidence * 0.2), 0.0, 100.0)
        
        return {
            "smart_money_score": round(np.clip(sms, 0.0, 100.0), 2),
            "institutional_control": route['primary_institutional_bias'],
            "institutional_volume_score": round(l1.get('institutional_volume_score', 0.0), 2),
            "market_structure_quality": ms['structure_quality'], "bos_quality": bos['bos_quality'],
            "choch_quality": choch['structural_shift_quality'], "order_block_strength": ob['institutional_strength'],
            "fvg_quality": fvg['fvg_quality'], "liquidity_quality": liq['liquidity_pool_strength'],
            "liquidity_efficiency": liq['liquidity_efficiency'], "trap_probability": trap_prob,
            "continuation_probability": bos['continuation_probability'], "reversal_probability": choch['reversal_probability'],
            "system_confidence": round(final_system_confidence, 2), "market_quality_score": round(mq_score, 2)
        }

    def analyze(self, df: pd.DataFrame) -> SmartMoneyAnalysisResult:
        try:
            if df.empty or not {'close'}.issubset(df.columns):
                logger.error("SmartMoneyAnalyzer: Required core OHLC framework is missing.")
                raise ValueError("SmartMoneyAnalyzer critical init mismatch.")
                
            l1_data = self._extract_l1_data(df)
            bos_res = self._process_bos(l1_data)
            choch_res = self._process_choch(l1_data)
            ms_res = self._process_market_structure(l1_data, bos_res, choch_res)
            ob_res = self._process_order_blocks(l1_data)
            fvg_res = self._process_fair_value_gap(l1_data)
            liq_res = self._process_liquidity(l1_data)
            route_res = self._route_and_arbitrate(bos_res, choch_res, ob_res, fvg_res, liq_res, l1_data)
            adv_metrics = self._compile_advanced_metrics(bos_res, choch_res, ms_res, ob_res, fvg_res, liq_res, route_res, l1_data)
            
            act_th = self.config['action_thresholds']
            action = "WAIT (No Clear SMC Edge)"
            if adv_metrics['reversal_probability'] >= act_th['reversal_min'] and route_res['primary_institutional_bias'] != "Neutral":
                action = f"PREPARE REVERSAL ({route_res['primary_institutional_bias'].upper()})"
            elif adv_metrics['continuation_probability'] >= act_th['continuation_min']:
                action = f"TREND CONTINUATION ({route_res['primary_institutional_bias'].upper()})"
            elif adv_metrics['trap_probability'] >= act_th['trap_min']:
                action = "STAY OUT (High Pool Trap Risk)"
                
            all_evidence = bos_res['evidence'] + choch_res['evidence'] + ms_res['evidence'] + ob_res['evidence'] + fvg_res['evidence'] + liq_res['evidence']
            top_evidence = sorted(all_evidence, key=lambda x: x['weight'] * x['reliability'], reverse=True)[:3]
            top_evidence_titles = [e['feature'] for e in top_evidence]
            
            conflict_val = route_res['conflict_score']
            conflict_level = "Low" if conflict_val < 25 else "Moderate" if conflict_val < 50 else "High" if conflict_val < 75 else "Severe"
            
            summary = {
                "action": action,
                "primary_signal": route_res['primary_smc_signal'],
                "dominant_institutional_event": route_res['primary_event'],
                "dominant_zone": route_res['primary_zone'],
                "dominant_bias": route_res['primary_institutional_bias'],
                "signal_conflict_level": conflict_level,
                "smart_money_score": adv_metrics['smart_money_score'],
                "confidence_level": adv_metrics['system_confidence'],
                "top_3_evidence": top_evidence_titles
            }
            
            return {
                "bos": bos_res, "choch": choch_res, "market_structure": ms_res,
                "order_block": ob_res, "fair_value_gap": fvg_res, "liquidity": liq_res,
                "routing": route_res, "advanced_metrics": adv_metrics, "summary": summary
            }
        except Exception as e:
            logger.exception(f"SmartMoneyAnalyzer Engine Exception: {e}")
            raise

__all__ = [
    "SmartMoneyAnalyzer",
    "SmartMoneyAnalysisResult",
    "BOSResult",
    "CHOCHResult",
    "MarketStructureResult",
    "OrderBlockResult",
    "FairValueGapResult",
    "LiquidityResult",
    "InstitutionalRoutingResult",
    "AdvancedSmartMoneyMetrics",
    "EvidenceItem"
]
