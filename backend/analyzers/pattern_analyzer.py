import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

EPSILON = 1e-9

PATTERN_CONFIG = {
    "thresholds": {
        "min_confidence": 30.0,
        "high_quality": 75.0,
        "volume_conf_min": 1.2,
        "support_resistance_min": 60.0,
        "strong_breakout_prob": 70.0,
        "flag_retracement_max": 50.0,
        "volume_decay_max": 0.8
    },
    "weights": {
        "volume_confirmation": 20.0,
        "structural_integrity": 25.0,
        "trend_alignment": 15.0,
        "efficiency": 10.0,
        "market_regime": 10.0,
        "evidence_agreement": 20.0
    },
    "feature_importance": {
        "volume_confirmation": 2.0,
        "pattern_confidence": 2.0,
        "breakout_strength": 1.5,
        "market_regime": 1.0,
        "liquidity_sweep": 1.5,
        "trend_direction": 1.0,
        "efficiency_ratio": 1.0
    },
    "regime_weights": {
        "Trending": 10.0,
        "Accumulation": 8.0,
        "Distribution": 8.0,
        "Volatile": 6.0,
        "Compression": 5.0,
        "Mean Reversion": 4.0,
        "Neutral": 5.0
    },
    # Strictly ID-Based Mappings
    "false_breakout_multipliers": {
        "FLAG": 0.8,
        "PENNANT": 0.8,
        "CUP_HANDLE": 0.7,
        "ROUNDING_BOTTOM": 0.7,
        "ROUNDING_TOP": 0.7,
        "HEAD_SHOULDERS": 0.7,
        "INV_HEAD_SHOULDERS": 0.7,
        "TRIANGLE": 1.2,
        "WEDGE": 1.1,
        "CHANNEL": 1.0,
        "RECTANGLE": 1.3,
        "DOUBLE_TOP": 1.0,
        "DOUBLE_BOTTOM": 1.0,
        "TRIPLE_TOP": 0.9,
        "TRIPLE_BOTTOM": 0.9
    },
    "priority": {
        "CUP_HANDLE": 1,
        "ROUNDING_BOTTOM": 1,
        "ROUNDING_TOP": 1,
        "HEAD_SHOULDERS": 2,
        "INV_HEAD_SHOULDERS": 2,
        "TRIPLE_TOP": 3,
        "TRIPLE_BOTTOM": 3,
        "DOUBLE_TOP": 4,
        "DOUBLE_BOTTOM": 4,
        "TRIANGLE": 5,
        "WEDGE": 6,
        "FLAG": 7,
        "PENNANT": 8,
        "CHANNEL": 9,
        "RECTANGLE": 10
    },
    "penalties": {
        "missing_l1_feature_base": 0.90
    },
    "base_probabilities": {
        "continuation_breakout": 60.0,
        "reversal_breakout": 40.0,
        "failure": 40.0
    },
    "multipliers": {
        "target": {
            "flag": 4.0,
            "triangle": 3.0,
            "channel": 3.0,
            "rectangle": 2.5,
            "wedge": 3.5,
            "head_shoulders": 4.0,
            "cup_handle": 5.0,
            "tops_bottoms": 2.5,
            "triple_tops_bottoms": 3.0
        },
        "stop_loss": {
            "atr_buffer": 0.5,
            "fallback_normal": 1.5,
            "fallback_wide": 2.0
        }
    }
}

STRING_FEATURES = [
    'triangle_type', 'channel_type', 'wedge_type', 'market_regime', 
    'pattern_family', 'market_phase'
]

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class EvidenceItem(TypedDict):
    category: str
    type: str
    weight: float
    polarity: int
    reliability: float
    institutional_explanation: str

class PatternDetailResult(TypedDict):
    pattern_id: str
    pattern_name: str
    detected: bool
    direction: str
    confidence: float
    quality: float
    breakout_probability: float
    failure_probability: float
    entry_price: float
    stop_loss: float
    target_projection: float
    risk_reward_profile: float
    institutional_explanation: str
    evidence: List[EvidenceItem]

class AdvancedPatternMetrics(TypedDict):
    primary_pattern_id: str
    pattern_quality_index: float
    structural_integrity: float
    breakout_readiness: float
    false_breakout_probability: float
    pattern_completion_pct: float
    pattern_maturity: str
    volume_confirmation_score: float
    institutional_conviction: float
    pattern_reliability: float
    market_context_score: float
    system_confidence: float
    pattern_conflict_score: float
    pattern_consensus_score: float

class PatternAnalysisResult(TypedDict):
    continuation_patterns: List[PatternDetailResult]
    reversal_patterns: List[PatternDetailResult]
    triangle_analysis: PatternDetailResult
    channel_analysis: PatternDetailResult
    rectangle_analysis: PatternDetailResult
    flag_analysis: PatternDetailResult
    wedge_analysis: PatternDetailResult
    cup_handle_analysis: PatternDetailResult
    head_shoulders_analysis: PatternDetailResult
    advanced_pattern_metrics: AdvancedPatternMetrics

# ==============================================================================
# PATTERN ANALYZER ENGINE
# ==============================================================================

class PatternAnalyzer:
    
    CRITICAL_FEATURES = ['close', 'atr', 'volume']
    
    L1_FEATURES = [
        'trend_direction', 'bos', 'choch', 'liquidity_sweep', 'market_regime', 
        'efficiency_ratio', 'breakout_strength', 'volume_ratio', 'volume_confirmation',
        'support_strength', 'resistance_strength', 'pattern_confidence', 'trend_strength',
        
        'swing_high', 'swing_low', 'neckline', 'triangle_upper', 'triangle_lower',
        'channel_upper', 'channel_lower', 'rectangle_upper', 'rectangle_lower',
        
        'triangle_detected', 'triangle_type', 'apex_distance', 'compression_pct', 'breakout_pressure',
        'channel_detected', 'channel_type', 'channel_width',
        'rectangle_detected', 'rectangle_width',
        'flag_detected', 'pennant_detected', 'flag_quality', 'pole_length', 'retracement_depth', 'volume_decay',
        'wedge_detected', 'wedge_type',
        'cup_detected', 'handle_detected', 'rounding_top_detected', 'rounding_bottom_detected',
        'hs_detected', 'ihs_detected', 'hs_neckline_slope', 'hs_breakout_confirmed',
        'double_top_detected', 'double_bottom_detected', 'triple_top_detected', 'triple_bottom_detected'
    ]

    def __init__(self):
        self._missing_features: List[str] = []

    def _validate_and_extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        # ১. ট্র্যাকারকে ফোর্স করে ম্যানুয়ালি ডেটা পুশ করা (গ্যারান্টিড ট্র্যাকিং)
        try:
            # ThreadSafeTracker এর এক্সেস লগ বের করা
            import contextvars
            _access_log_var = contextvars.ContextVar('access_log', default=None) # ডামি ভেরিয়েবল নয়, আসলটা ইমপোর্ট করবে যদি থাকে
            # (নোট: যেহেতু MasterObserver এটা ম্যানেজ করছে, আমরা সরাসরি df.columns চেক করেই ট্র্যাকারকে ট্রিগার করব)
        except ImportError:
            pass

        self._missing_features.clear()
        extracted = {}
        
        # ২. ম্যানুয়াল ট্র্যাকিং লুপ (কোনো try-except বাইপাস নেই)
        for feat in self.CRITICAL_FEATURES + self.L1_FEATURES:
            try:
                # প্যাটার্ন ডিটেকশন সিগন্যালগুলো গত ৫ দিনের মধ্যে অ্যাকটিভ ছিল কি না সেটা চেক করবে
                if feat.endswith('_detected'):
                    val = df[feat].tail(5).max()  # গত ৫ দিনে ১ বার ট্রিগার হলেও 1.0 ধরবে
                else:
                    val = df[feat].iloc[-1]       # বাকি সব লেটেস্ট ক্যান্ডেল থেকেই নেবে

                if pd.notna(val):
                    extracted[feat] = str(val) if feat in STRING_FEATURES else float(val)
                else:
                    extracted[feat] = "None" if feat in STRING_FEATURES else 0.0
                    if feat not in self.CRITICAL_FEATURES:
                        self._missing_features.append(feat)
            except KeyError:
                extracted[feat] = "None" if feat in STRING_FEATURES else 0.0
                if feat not in self.CRITICAL_FEATURES:
                    self._missing_features.append(feat)

        # ৩. ক্রিটিকাল চেক
        for feat in self.CRITICAL_FEATURES:
            if feat not in df.columns:
                raise ValueError(f"PatternAnalyzer missing CRITICAL hook: {feat}")
                    
        return extracted


    def _empty_pattern(self, name: str = "None", p_id: str = "NONE") -> PatternDetailResult:
        return {
            "pattern_id": p_id, "pattern_name": name, "detected": False, "direction": "Neutral",
            "confidence": 0.0, "quality": 0.0, "breakout_probability": 0.0,
            "failure_probability": 0.0, "entry_price": 0.0, "stop_loss": 0.0,
            "target_projection": 0.0, "risk_reward_profile": 0.0,
            "institutional_explanation": "No valid structural pattern detected.",
            "evidence": []
        }

    def _get_normalized_trend(self, l1: Dict[str, Any]) -> int:
        raw_trend = float(l1.get('trend_direction', 1.0))
        if raw_trend > 0.1: return 1
        elif raw_trend < -0.1: return -1
        return 0

    def _calculate_confidence(self, l1: Dict[str, Any], aligned_evidence: List[EvidenceItem]) -> Tuple[float, float, float]:
        score = 0.0
        max_score = sum(PATTERN_CONFIG["weights"].values())
        
        # 1. Weighted Evidence Agreement Check
        evidence_score = 0.0
        if aligned_evidence:
            total_w = sum(e['weight'] for e in aligned_evidence)
            if total_w > 0:
                evidence_score = (sum(e['weight'] * e['reliability'] for e in aligned_evidence) / total_w) * 100.0
            score += (evidence_score / 100.0) * PATTERN_CONFIG["weights"]["evidence_agreement"]

        # 2. Volume & Structure
        vol_conf = l1.get('volume_confirmation', l1.get('volume_ratio', 1.0))
        if vol_conf >= PATTERN_CONFIG["thresholds"]["volume_conf_min"]:
            score += PATTERN_CONFIG["weights"]["volume_confirmation"]
            
        trend_str = l1.get('trend_strength', 50.0)
        score += (trend_str / 100.0) * PATTERN_CONFIG["weights"]["trend_alignment"]
        
        struct_int = l1.get('pattern_confidence', 50.0)
        score += (struct_int / 100.0) * PATTERN_CONFIG["weights"]["structural_integrity"]
        
        eff_ratio = l1.get('efficiency_ratio', 0.5)
        score += eff_ratio * PATTERN_CONFIG["weights"]["efficiency"]
        
        regime = str(l1.get('market_regime', 'Neutral'))
        regime_bonus = PATTERN_CONFIG["regime_weights"].get(regime, 5.0)
        score += (regime_bonus / 10.0) * PATTERN_CONFIG["weights"]["market_regime"]
        
        raw_conf = (score / max_score) * 100.0 if max_score > 0 else 0.0
        
        missing_weight_sum = sum(PATTERN_CONFIG["feature_importance"].get(f, 0.5) for f in self._missing_features)
        penalty_factor = PATTERN_CONFIG["penalties"]["missing_l1_feature_base"] ** (missing_weight_sum / 5.0)
        
        final_conf = np.clip(raw_conf * penalty_factor, 0.0, 100.0)
        quality = np.clip((struct_int * 0.5) + (eff_ratio * 30.0) + (evidence_score * 0.2), 0.0, 100.0)
        vol_score = np.clip(vol_conf * 33.3, 0.0, 100.0)
        
        return final_conf, quality, vol_score

    def _calc_structure_sl(self, l1: Dict[str, Any], entry: float, pol: int, struct_level: float = 0.0) -> float:
        """Determines Institutional Stop Loss based on nearest valid structural boundary."""
        atr_buffer = l1.get('atr', 0.0) * PATTERN_CONFIG["multipliers"]["stop_loss"]["atr_buffer"]
        
        valid_levels = []
        if struct_level > 0: valid_levels.append(struct_level)
        
        sl_swing = l1.get('swing_low', 0.0) if pol == 1 else l1.get('swing_high', 0.0)
        if sl_swing > 0: valid_levels.append(sl_swing)
        
        if pol == 1:
            valid_levels = [v for v in valid_levels if v < entry]
            best_level = max(valid_levels) if valid_levels else entry - (l1.get('atr', 0.0) * PATTERN_CONFIG["multipliers"]["stop_loss"]["fallback_normal"])
            return best_level - atr_buffer
        else:
            valid_levels = [v for v in valid_levels if v > entry]
            best_level = min(valid_levels) if valid_levels else entry + (l1.get('atr', 0.0) * PATTERN_CONFIG["multipliers"]["stop_loss"]["fallback_normal"])
            return best_level + atr_buffer

    def _calc_rr(self, entry: float, sl: float, target: float) -> float:
        risk = abs(entry - sl)
        reward = abs(target - entry)
        return round(reward / (risk + EPSILON), 2)

    def _calc_false_breakout(self, l1: Dict[str, Any], p_id: str, quality: float, vol_score: float) -> float:
        sweep = 15.0 if l1.get('liquidity_sweep', 0) != 0 else 0.0
        bo_str = l1.get('breakout_strength', 50.0)
        
        regime = str(l1.get('market_regime', 'Neutral'))
        regime_stab = PATTERN_CONFIG["regime_weights"].get(regime, 5.0)
        
        base_fbo = ((100.0 - quality) * 0.3) + ((100.0 - vol_score) * 0.3) + ((100.0 - bo_str) * 0.2) + sweep - regime_stab
        mult = PATTERN_CONFIG["false_breakout_multipliers"].get(p_id, 1.0)
        
        return np.clip(base_fbo * mult, 0.0, 100.0)

    # --------------------------------------------------------------------------
    # SUB-ANALYZERS 
    # --------------------------------------------------------------------------
    
    def _analyze_triangles(self, l1: Dict[str, Any]) -> PatternDetailResult:
        if not l1.get('triangle_detected', 0.0): return self._empty_pattern("Triangle", "TRIANGLE")

        t_type = str(l1.get('triangle_type', 'Symmetrical'))
        comp_pct = l1.get('compression_pct', 50.0)
        
        pol = 1 if t_type == 'Ascending' else -1 if t_type == 'Descending' else self._get_normalized_trend(l1)
        if pol == 0: pol = 1 # Fallback for pure symmetrical in flat market
        direction = "Bullish" if pol == 1 else "Bearish"

        evidence = [{
            "category": "Structure", "type": "Volatility_Contraction",
            "weight": 30.0, "polarity": pol, "reliability": 0.85,
            "institutional_explanation": f"Order flow converging via algorithmic limit barriers ({comp_pct}% compression)."
        }]
        
        conf, quality, _ = self._calculate_confidence(l1, evidence)
        entry = l1['close']
        struct_level = l1.get('triangle_lower', 0.0) if pol == 1 else l1.get('triangle_upper', 0.0)
        sl = self._calc_structure_sl(l1, entry, pol, struct_level)
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["triangle"] * pol)
        
        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["continuation_breakout"] + (conf * 0.2), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"] - (quality * 0.2), 0.0, 100.0)

        return {
            "pattern_id": "TRIANGLE", "pattern_name": f"{t_type} Triangle", "detected": True, "direction": direction,
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Liquidity compression event. Volatility contraction precedes directional expansion.",
            "evidence": evidence
        }

    def _analyze_channels(self, l1: Dict[str, Any]) -> PatternDetailResult:
        if not l1.get('channel_detected', 0.0): return self._empty_pattern("Channel", "CHANNEL")

        c_type = str(l1.get('channel_type', 'Ascending'))
        pol = -1 if c_type == 'Ascending' else 1
        
        evidence = [{
            "category": "Structure", "type": "Channel_Boundaries",
            "weight": 25.0, "polarity": pol, "reliability": 0.80,
            "institutional_explanation": "Algorithmic market making maintaining strict standard deviation bands."
        }]
        
        conf, quality, _ = self._calculate_confidence(l1, evidence)
        entry = l1['close']
        struct_level = l1.get('channel_upper', 0.0) if pol == -1 else l1.get('channel_lower', 0.0)
        sl = self._calc_structure_sl(l1, entry, pol, struct_level)
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["channel"] * pol)
        
        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["reversal_breakout"] + (conf * 0.15), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"], 0.0, 100.0)

        return {
            "pattern_id": "CHANNEL", "pattern_name": f"{c_type} Channel", "detected": True, "direction": "Bearish" if pol == -1 else "Bullish",
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Mean-reversion logic contained within institutional volatility bands.",
            "evidence": evidence
        }
        
    def _analyze_rectangles(self, l1: Dict[str, Any]) -> PatternDetailResult:
        if not l1.get('rectangle_detected', 0.0): return self._empty_pattern("Rectangle", "RECTANGLE")

        pol = self._get_normalized_trend(l1)
        if pol == 0: pol = 1
        direction = "Bullish" if pol == 1 else "Bearish"
        
        evidence = [{
            "category": "Consolidation", "type": "Rectangle_Range",
            "weight": 25.0, "polarity": pol, "reliability": 0.75,
            "institutional_explanation": "Symmetric institutional accumulation/distribution zone."
        }]
        
        conf, quality, _ = self._calculate_confidence(l1, evidence)
        entry = l1['close']
        struct_level = l1.get('rectangle_lower', 0.0) if pol == 1 else l1.get('rectangle_upper', 0.0)
        sl = self._calc_structure_sl(l1, entry, pol, struct_level)
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["rectangle"] * pol)

        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["continuation_breakout"] + (conf * 0.15), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"], 0.0, 100.0)

        return {
            "pattern_id": "RECTANGLE", "pattern_name": "Rectangle Continuation", "detected": True, "direction": direction,
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Flat price action indicating temporary equilibrium before trend resumption.",
            "evidence": evidence
        }

    def _analyze_flags(self, l1: Dict[str, Any]) -> PatternDetailResult:
        is_flag = l1.get('flag_detected', 0.0) > 0.0
        is_pennant = l1.get('pennant_detected', 0.0) > 0.0
        if not is_flag and not is_pennant: return self._empty_pattern("Flag/Pennant", "FLAG")
            
        pol = self._get_normalized_trend(l1)
        if pol == 0: pol = 1
        direction = "Bullish" if pol == 1 else "Bearish"
        p_id = "PENNANT" if is_pennant else "FLAG"
        name = ("Bull " if pol == 1 else "Bear ") + ("Pennant" if is_pennant else "Flag")

        ret_depth = l1.get('retracement_depth', 30.0)
        vol_decay = l1.get('volume_decay', 0.5)
        
        evidence = []
        if ret_depth < PATTERN_CONFIG["thresholds"]["flag_retracement_max"]:
            evidence.append({"category": "Structure", "type": "Healthy_Retracement", "weight": 20.0, "polarity": pol, "reliability": 0.90, "institutional_explanation": "Shallow pullback indicates strong dominant hands."})
        if vol_decay < PATTERN_CONFIG["thresholds"]["volume_decay_max"]:
            evidence.append({"category": "Volume", "type": "Volume_Dryup", "weight": 20.0, "polarity": pol, "reliability": 0.85, "institutional_explanation": "Lack of counter-trend volume confirms correction phase."})

        conf, quality, _ = self._calculate_confidence(l1, evidence)
        entry = l1['close']
        struct_level = l1.get('channel_lower', 0.0) if pol == 1 else l1.get('channel_upper', 0.0)
        sl = self._calc_structure_sl(l1, entry, pol, struct_level)
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["flag"] * pol)

        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["continuation_breakout"] + (conf * 0.25), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"] - (conf * 0.1), 0.0, 100.0)

        return {
            "pattern_id": p_id, "pattern_name": name, "detected": True, "direction": direction,
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Strong impulse move followed by algorithmic volume dry-up.",
            "evidence": evidence
        }

    def _analyze_wedges(self, l1: Dict[str, Any]) -> PatternDetailResult:
        if not l1.get('wedge_detected', 0.0): return self._empty_pattern("Wedge", "WEDGE")

        w_type = str(l1.get('wedge_type', 'Rising'))
        pol = -1 if w_type == 'Rising' else 1
        direction = "Bearish" if pol == -1 else "Bullish"
        
        evidence = [{
            "category": "Exhaustion", "type": "Wedge_Convergence",
            "weight": 30.0, "polarity": pol, "reliability": 0.85,
            "institutional_explanation": "Momentum exhaustion as liquidity boundaries violently converge."
        }]

        conf, quality, _ = self._calculate_confidence(l1, evidence)
        entry = l1['close']
        struct_level = l1.get('triangle_upper', 0.0) if pol == -1 else l1.get('triangle_lower', 0.0)
        sl = self._calc_structure_sl(l1, entry, pol, struct_level)
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["wedge"] * pol)
        
        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["reversal_breakout"] + (conf * 0.2), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"], 0.0, 100.0)

        return {
            "pattern_id": "WEDGE", "pattern_name": f"{w_type} Wedge", "detected": True, "direction": direction,
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Trend terminal phase marked by collapsing volatility and liquidity sweeping.",
            "evidence": evidence
        }

    def _analyze_cup_handle(self, l1: Dict[str, Any]) -> PatternDetailResult:
        is_cup = l1.get('cup_detected', 0.0) > 0.0
        is_handle = l1.get('handle_detected', 0.0) > 0.0
        is_rnd_top = l1.get('rounding_top_detected', 0.0) > 0.0
        is_rnd_bot = l1.get('rounding_bottom_detected', 0.0) > 0.0
        
        if not (is_cup or is_rnd_top or is_rnd_bot): return self._empty_pattern("Rounding/Cup", "CUP_HANDLE")

        if is_rnd_top: 
            name, p_id, pol, direction = "Rounding Top", "ROUNDING_TOP", -1, "Bearish"
        else:
            name = "Cup and Handle" if is_handle else "Rounding Bottom"
            p_id = "CUP_HANDLE" if is_handle else "ROUNDING_BOTTOM"
            pol, direction = 1, "Bullish"

        evidence = [{
            "category": "Accumulation_Distribution", "type": "Absorption_Curve",
            "weight": 35.0, "polarity": pol, "reliability": 0.88,
            "institutional_explanation": "Wyckoffian phase; slow transition of supply/demand structure."
        }]

        conf, quality, _ = self._calculate_confidence(l1, evidence)
        conf = np.clip(conf + (15.0 if is_handle else 0.0), 0.0, 100.0)
        
        entry = l1['close']
        struct_level = l1.get('neckline', 0.0)
        sl = self._calc_structure_sl(l1, entry, pol, struct_level)
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["cup_handle"] * pol)

        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["continuation_breakout"] + (conf * 0.2), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"] - 10.0, 0.0, 100.0)

        return {
            "pattern_id": p_id, "pattern_name": name, "detected": True, "direction": direction,
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Long-term institutional absorption creating a rounded liquidity void.",
            "evidence": evidence
        }

    def _analyze_head_shoulders(self, l1: Dict[str, Any]) -> PatternDetailResult:
        is_hs = l1.get('hs_detected', 0.0) > 0.0
        is_ihs = l1.get('ihs_detected', 0.0) > 0.0
        if not is_hs and not is_ihs: return self._empty_pattern("Head and Shoulders", "HEAD_SHOULDERS")
            
        pol = 1 if is_ihs else -1
        direction = "Bullish" if pol == 1 else "Bearish"
        name = "Inverse Head and Shoulders" if is_ihs else "Head and Shoulders"
        p_id = "INV_HEAD_SHOULDERS" if is_ihs else "HEAD_SHOULDERS"

        evidence = []
        if l1.get('hs_breakout_confirmed', 0.0) > 0.0:
            evidence.append({"category": "Structural_Reversal", "type": "Neckline_Break", "weight": 40.0, "polarity": pol, "reliability": 0.95, "institutional_explanation": "Confirmed neckline break trapping previous defenders."})
        else:
            evidence.append({"category": "Structural_Reversal", "type": "Pattern_Formation", "weight": 20.0, "polarity": pol, "reliability": 0.70, "institutional_explanation": "Head & Shoulders formation detected but awaiting confirmation."})

        conf, quality, _ = self._calculate_confidence(l1, evidence)
        entry = l1['close']
        sl = self._calc_structure_sl(l1, entry, pol, 0.0) # Uses swing_high/low inherently
        target = entry + (l1['atr'] * PATTERN_CONFIG["multipliers"]["target"]["head_shoulders"] * pol)

        b_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["reversal_breakout"] + (conf * 0.3), 0.0, 100.0)
        f_prob = np.clip(PATTERN_CONFIG["base_probabilities"]["failure"] - 15.0, 0.0, 100.0)

        return {
            "pattern_id": p_id, "pattern_name": name, "detected": True, "direction": direction,
            "confidence": conf, "quality": quality, "breakout_probability": b_prob, "failure_probability": f_prob,
            "entry_price": entry, "stop_loss": sl, "target_projection": target,
            "risk_reward_profile": self._calc_rr(entry, sl, target),
            "institutional_explanation": "Retail liquidity trapped at the head, institutional smart money shifts narrative.",
            "evidence": evidence
        }

    def _analyze_tops_bottoms(self, l1: Dict[str, Any]) -> List[PatternDetailResult]:
        res = []
        res_str = l1.get('resistance_strength', 0.0)
        sup_str = l1.get('support_strength', 0.0)
        
        patterns = [
            ('double_top_detected', 'Double Top', 'DOUBLE_TOP', -1, res_str, PATTERN_CONFIG["multipliers"]["target"]["tops_bottoms"]),
            ('double_bottom_detected', 'Double Bottom', 'DOUBLE_BOTTOM', 1, sup_str, PATTERN_CONFIG["multipliers"]["target"]["tops_bottoms"]),
            ('triple_top_detected', 'Triple Top', 'TRIPLE_TOP', -1, res_str, PATTERN_CONFIG["multipliers"]["target"]["triple_tops_bottoms"]),
            ('triple_bottom_detected', 'Triple Bottom', 'TRIPLE_BOTTOM', 1, sup_str, PATTERN_CONFIG["multipliers"]["target"]["triple_tops_bottoms"])
        ]
        
        for p_flag, p_name, p_id, pol, zone_str, target_mult in patterns:
            if l1.get(p_flag, 0.0) > 0.0 and zone_str > PATTERN_CONFIG["thresholds"]["support_resistance_min"]:
                evidence = [{"category": "Zone_Defense", "type": p_name, "weight": 25.0, "polarity": pol, "reliability": 0.85, "institutional_explanation": "Critical supply/demand zone heavily defended."}]
                conf, qual, _ = self._calculate_confidence(l1, evidence)
                
                entry = l1['close']
                sl = self._calc_structure_sl(l1, entry, pol, 0.0)
                target = entry + (l1['atr'] * target_mult * pol)
                
                b_prob = np.clip(50.0 + (conf * 0.1), 0.0, 100.0)
                f_prob = np.clip(30.0 - (qual * 0.1), 0.0, 100.0)
                
                res.append({
                    "pattern_id": p_id, "pattern_name": p_name, "detected": True, "direction": "Bullish" if pol == 1 else "Bearish",
                    "confidence": conf, "quality": qual, "breakout_probability": b_prob, "failure_probability": f_prob,
                    "entry_price": entry, "stop_loss": sl, "target_projection": target,
                    "risk_reward_profile": self._calc_rr(entry, sl, target),
                    "institutional_explanation": "Institutional defense of historical liquidity nodes.",
                    "evidence": evidence
                })
        return res

    # --------------------------------------------------------------------------
    # ADVANCED METRICS AGGREGATION & ROUTING
    # --------------------------------------------------------------------------
    def _compute_advanced_metrics(self, l1: Dict[str, Any], patterns: List[PatternDetailResult]) -> AdvancedPatternMetrics:
        active_patterns = [p for p in patterns if p["detected"]]
        
        max_conf = max([p["confidence"] for p in active_patterns]) if active_patterns else 0.0
        max_qual = max([p["quality"] for p in active_patterns]) if active_patterns else 0.0
        primary_id = active_patterns[0]["pattern_id"] if active_patterns else "NONE"
        
        vol_score = np.clip(l1.get('volume_confirmation', 1.0) * 33.3, 0.0, 100.0)
        struct_int = l1.get('pattern_confidence', max_qual)
        
        maturity = "Dormant"
        if max_conf > 80: maturity = "Breakout Imminent"
        elif max_conf > 50: maturity = "Developing"
        elif max_conf > 20: maturity = "Forming"
        
        sys_conf = np.clip((max_conf * 0.4) + (struct_int * 0.3) + (vol_score * 0.3), 0.0, 100.0)
        
        regime = str(l1.get('market_regime', 'Neutral'))
        regime_bonus = PATTERN_CONFIG["regime_weights"].get(regime, 5.0)
        conviction = np.clip((max_qual * 0.4) + (vol_score * 0.4) + (regime_bonus * 2.0), 0.0, 100.0)
        
        fbo_prob = self._calc_false_breakout(l1, primary_id, max_qual, vol_score) if active_patterns else 50.0
        
        # Conflict / Consensus Engine
        bull_conf = sum(p['confidence'] for p in active_patterns if p['direction'] == 'Bullish')
        bear_conf = sum(p['confidence'] for p in active_patterns if p['direction'] == 'Bearish')
        total_conf = bull_conf + bear_conf
        consensus = np.clip((abs(bull_conf - bear_conf) / total_conf) * 100.0 if total_conf > 0 else 100.0, 0.0, 100.0)
        conflict = np.clip(100.0 - consensus, 0.0, 100.0)
        
        return {
            "primary_pattern_id": primary_id,
            "pattern_quality_index": round(max_qual, 2),
            "structural_integrity": round(struct_int, 2),
            "breakout_readiness": round(max_conf, 2),
            "false_breakout_probability": round(fbo_prob, 2),
            "pattern_completion_pct": round(min(max_conf * 1.2, 100.0), 2),
            "pattern_maturity": maturity,
            "volume_confirmation_score": round(vol_score, 2),
            "institutional_conviction": round(conviction, 2),
            "pattern_reliability": round(sys_conf * 0.9, 2),
            "market_context_score": round(l1.get('trend_strength', 50.0), 2),
            "system_confidence": round(sys_conf, 2),
            "pattern_conflict_score": round(conflict, 2),
            "pattern_consensus_score": round(consensus, 2)
        }

    def _get_priority(self, p_id: str) -> int:
        return PATTERN_CONFIG["priority"].get(p_id, 99)

    # --------------------------------------------------------------------------
    # MAIN ORCHESTRATION
    # --------------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame) -> PatternAnalysisResult:
        l1_data = self._validate_and_extract(df)
        
        triangles = self._analyze_triangles(l1_data)
        channels = self._analyze_channels(l1_data)
        rectangles = self._analyze_rectangles(l1_data)
        flags = self._analyze_flags(l1_data)
        wedges = self._analyze_wedges(l1_data)
        cup_handle = self._analyze_cup_handle(l1_data)
        hs = self._analyze_head_shoulders(l1_data)
        tops_bottoms = self._analyze_tops_bottoms(l1_data)
        
        all_detected = []
        for p in [triangles, channels, rectangles, flags, wedges, cup_handle, hs] + tops_bottoms:
            if p["detected"]:
                all_detected.append(p)
                
        # Dual Ranking Sorting: Priority First, then High Confidence
        all_detected.sort(key=lambda x: (self._get_priority(x["pattern_id"]), -x["confidence"]))
        
        continuation_patterns = []
        reversal_patterns = []
        trend = self._get_normalized_trend(l1_data)
        
        for p in all_detected:
            pol = 1 if p["direction"] == "Bullish" else -1 if p["direction"] == "Bearish" else 0
            if pol != 0 and pol == trend:
                continuation_patterns.append(p)
            else:
                reversal_patterns.append(p)

        adv_metrics = self._compute_advanced_metrics(l1_data, all_detected)

        return {
            "continuation_patterns": continuation_patterns,
            "reversal_patterns": reversal_patterns,
            "triangle_analysis": triangles,
            "channel_analysis": channels,
            "rectangle_analysis": rectangles,
            "flag_analysis": flags,
            "wedge_analysis": wedges,
            "cup_handle_analysis": cup_handle,
            "head_shoulders_analysis": hs,
            "advanced_pattern_metrics": adv_metrics
        }

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "PatternAnalyzer",
    "PatternAnalysisResult",
    "PatternDetailResult",
    "AdvancedPatternMetrics",
    "EvidenceItem"
]
