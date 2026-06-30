import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional, Tuple

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

EPSILON = 1e-9

# External configs can override this in production via a config.yaml loader
VOLUME_CONFIG = {
    "lookbacks": {
        "micro": 3,
        "short": 10,
        "medium": 20,
        "long": 50,
        "macro": 100,
        "institutional": 252
    },
    "thresholds": {
        "climax_vol_mult": 3.0,
        "explosive_vol_mult": 2.0,
        "expanding_vol_mult": 1.2,
        "high_delivery_pct": 60.0,
        "extreme_zscore": 2.5,
        "divergence_lookback": 20,
        "dark_pool_displacement_limit": 0.15 # Max ATR fraction for dark pool proxy
    },
    "mtf_weights": {
        "_M": 0.30, "_W": 0.25, "_D": 0.20, "_4H": 0.10, "_1H": 0.05, "_15m": 0.05, "_5m": 0.05
    },
    "oscillator_weights": {
        "obv": 0.20, "cmf": 0.15, "adl": 0.15, "vpt": 0.10, 
        "mfi": 0.10, "money_flow": 0.10, "force_index": 0.10, "accdist": 0.10
    }
}

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class EvidenceItem(TypedDict):
    type: str
    weight: float
    value: str
    polarity: int

class VolumeConfirmationResult(TypedDict):
    score: float
    confidence: float
    status: str
    evidence: List[EvidenceItem]

class VolumeExplosionResult(TypedDict):
    score: float
    confidence: float
    status: str
    evidence: List[EvidenceItem]

class DeliveryAnalysisResult(TypedDict):
    score: float
    confidence: float
    status: str
    evidence: List[EvidenceItem]

class SmartVolumeResult(TypedDict):
    institutional_probability: float
    retail_probability: float
    confidence: float
    dominance: str
    evidence: List[EvidenceItem]

class VolumeDivergenceResult(TypedDict):
    divergence_type: str
    strength: float
    confidence: float
    evidence: List[EvidenceItem]

class MTFVolumeResult(TypedDict):
    alignment_score: float
    dominant_trend: str
    confidence: float
    timeframes: Dict[str, str]

class AdvancedVolumeMetrics(TypedDict):
    volume_quality: float
    volume_efficiency: float
    participation_regime: str
    smart_participation_index: float
    retail_participation_index: float
    volume_stability: float
    volume_persistence: float
    volume_dry_up: bool
    volume_regime_shift: bool
    liquidity_absorption: float
    volume_compression_score: float
    volume_expansion_probability: float
    block_trade_proxy: bool
    dark_pool_proxy: bool
    volume_fractal_index: float

class VolumeAnalysisResult(TypedDict):
    volume_confirmation: VolumeConfirmationResult
    volume_explosion: VolumeExplosionResult
    delivery_analysis: DeliveryAnalysisResult
    smart_volume: SmartVolumeResult
    volume_divergence: VolumeDivergenceResult
    multi_timeframe: MTFVolumeResult
    advanced_metrics: AdvancedVolumeMetrics

# ==============================================================================
# NUMBA JIT ACCELERATED ENGINES
# ==============================================================================

@jit(nopython=True, cache=True)
def _find_pivots_jit(arr: np.ndarray, window: int) -> Tuple[List[int], List[int]]:
    """Identifies indices of Pivot Highs and Pivot Lows."""
    highs = []
    lows = []
    n = len(arr)
    if n < window * 2 + 1:
        return highs, lows
        
    for i in range(window, n - window):
        is_high = True
        is_low = True
        for j in range(i - window, i + window + 1):
            if i == j: continue
            if arr[i] <= arr[j]: is_high = False
            if arr[i] >= arr[j]: is_low = False
        
        if is_high: highs.append(i)
        if is_low: lows.append(i)
        
    return highs, lows

@jit(nopython=True, cache=True)
def _pivot_divergence_engine_jit(price: np.ndarray, osc: np.ndarray, lookback: int) -> Tuple[float, int]:
    """
    Pivot-based Divergence Scanner. 
    1: RegBull, -1: RegBear, 2: HidBull, -2: HidBear
    """
    n = len(price)
    if n < lookback: return 0.0, 0
    
    p_window = price[-lookback:]
    o_window = osc[-lookback:]
    
    p_highs, p_lows = _find_pivots_jit(p_window, 2)
    o_highs, o_lows = _find_pivots_jit(o_window, 2)
    
    # Needs at least 2 pivots to compare
    if len(p_highs) >= 2 and len(o_highs) >= 2:
        last_p_high, prev_p_high = p_window[p_highs[-1]], p_window[p_highs[-2]]
        last_o_high, prev_o_high = o_window[o_highs[-1]], o_window[o_highs[-2]]
        
        # Regular Bearish
        if last_p_high > prev_p_high and last_o_high < prev_o_high:
            str_val = min(((prev_o_high - last_o_high) / (np.abs(prev_o_high) + EPSILON)) * 200.0, 100.0)
            return str_val, -1
            
        # Hidden Bearish
        if last_p_high < prev_p_high and last_o_high > prev_o_high:
            str_val = min(((last_o_high - prev_o_high) / (np.abs(prev_o_high) + EPSILON)) * 150.0, 100.0)
            return str_val, -2

    if len(p_lows) >= 2 and len(o_lows) >= 2:
        last_p_low, prev_p_low = p_window[p_lows[-1]], p_window[p_lows[-2]]
        last_o_low, prev_o_low = o_window[o_lows[-1]], o_window[o_lows[-2]]
        
        # Regular Bullish
        if last_p_low < prev_p_low and last_o_low > prev_o_low:
            str_val = min(((last_o_low - prev_o_low) / (np.abs(prev_o_low) + EPSILON)) * 200.0, 100.0)
            return str_val, 1
            
        # Hidden Bullish
        if last_p_low > prev_p_low and last_o_low < prev_o_low:
            str_val = min(((prev_o_low - last_o_low) / (np.abs(prev_o_low) + EPSILON)) * 150.0, 100.0)
            return str_val, 2

    return 0.0, 0

# ==============================================================================
# VOLUME ANALYZER ENGINE (PURE LAYER-2)
# ==============================================================================

class VolumeAnalyzer:
    """
    Institutional Volume Analyzer (Layer-2 V2).
    Implements Dynamic Feature Resolution, Pivot-Based JIT Divergence, 
    and Ensemble Probability Modeling.
    """

    def __init__(self):
        self.req_cols = ['open', 'high', 'low', 'close', 'volume']
        self._feature_cache: Dict[str, Optional[str]] = {}
        self.mtf_suffixes = ['_5m', '_15m', '_1H', '_4H', '_D', '_W', '_M']

    def _resolve_feature(self, df: pd.DataFrame, base_name: str) -> Optional[str]:
        """Robust Regex/Alias based O(1) feature resolver."""
        cache_key = base_name.lower()
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
            
        # Match exact, or separated by underscore/dot (e.g., momentum.obv, obv_14)
        pattern = re.compile(rf'(^|\.){re.escape(base_name)}(_|$)')
        for col in df.columns:
            if pattern.search(col.lower()) or col.lower() == cache_key:
                self._feature_cache[cache_key] = col
                return col
                
        self._feature_cache[cache_key] = None
        return None

    def _get_val(self, df: pd.DataFrame, base_name: str, default: float = 0.0) -> float:
        col = self._resolve_feature(df, base_name)
        if col and pd.notna(df[col].iloc[-1]): 
            return float(df[col].iloc[-1])
        return default

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.req_cols if col not in df.columns]
        if missing:
            logger.error(f"VolumeAnalyzer missing core OHLCV: {missing}")
            raise ValueError(f"VolumeAnalyzer requires basic columns: {missing}")

        working_df = df.copy()
        num_cols = working_df.select_dtypes(include=[np.number]).columns
        if np.isinf(working_df[num_cols]).any().any():
            working_df[num_cols] = working_df[num_cols].replace([np.inf, -np.inf], np.nan)

        working_df.ffill(inplace=True)
        working_df.bfill(inplace=True)
        return working_df

    def _get_internal_trend_proxy(self, df: pd.DataFrame) -> int:
        """Internal robust trend proxy if L3 trend is unavailable."""
        ema_20 = self._get_val(df, 'ema_20')
        vwap = self._get_val(df, 'vwap')
        close = df['close'].iloc[-1]
        
        bull_score = sum([close > ema_20, close > vwap, df['close'].iloc[-1] > df['close'].iloc[-10]])
        if bull_score >= 2: return 1
        elif bull_score == 0: return -1
        return 0

    def analyze(self, df: pd.DataFrame) -> VolumeAnalysisResult:
        safe_df = self._validate_data(df)
        eval_len = VOLUME_CONFIG["lookbacks"]["institutional"]
        working_df = safe_df.tail(min(len(safe_df), eval_len))

        trend_dir = self._get_internal_trend_proxy(working_df)

        conf_res = self._analyze_confirmation(working_df, trend_dir)
        expl_res = self._analyze_explosion(working_df)
        del_res = self._analyze_delivery(working_df)
        div_res = self._analyze_divergence(working_df)
        mtf_res = self._analyze_mtf(working_df)
        
        adv_metrics = self._analyze_advanced_metrics(working_df, conf_res, expl_res, del_res, div_res)
        smart_res = self._analyze_smart_volume(working_df, conf_res, expl_res, del_res, div_res, mtf_res, adv_metrics)

        return {
            "volume_confirmation": conf_res,
            "volume_explosion": expl_res,
            "delivery_analysis": del_res,
            "smart_volume": smart_res,
            "volume_divergence": div_res,
            "multi_timeframe": mtf_res,
            "advanced_metrics": adv_metrics
        }

    # --------------------------------------------------------------------------
    # 1. VOLUME CONFIRMATION
    # --------------------------------------------------------------------------
    def _analyze_confirmation(self, df: pd.DataFrame, trend_dir: int) -> VolumeConfirmationResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        max_possible_weight = 0.0
        active_weight = 0.0

        features = [
            ('obv', 15.0), ('cmf', 15.0), ('vpt', 10.0), ('mfi', 10.0),
            ('force_index', 10.0), ('adl', 15.0), ('money_flow', 10.0),
            ('nvi', 5.0), ('pvi', 5.0), ('accdist', 5.0)
        ]

        for feat, weight in features:
            max_possible_weight += weight
            col = self._resolve_feature(df, feat)
            if col and len(df) > 1:
                active_weight += weight
                val, prev_val = df[col].iloc[-1], df[col].iloc[-2]
                
                if feat in ['cmf', 'money_flow']:
                    aligned = 1 if (val > 0 and trend_dir == 1) else -1 if (val < 0 and trend_dir == -1) else 0
                elif feat == 'mfi':
                    aligned = 1 if (val > 50 and trend_dir == 1) else -1 if (val < 50 and trend_dir == -1) else 0
                else:
                    slope = val - prev_val
                    aligned = 1 if (slope > 0 and trend_dir == 1) else -1 if (slope < 0 and trend_dir == -1) else 0
                
                if aligned == 1:
                    score += weight
                    evidence.append({"type": feat.upper(), "weight": weight, "value": f"Aligned with Trend", "polarity": 1})
                elif aligned == -1:
                    score -= weight
                    evidence.append({"type": feat.upper(), "weight": weight, "value": f"Against Trend", "polarity": -1})
                else:
                    evidence.append({"type": feat.upper(), "weight": weight, "value": f"Neutral/Divergent", "polarity": 0})

        # VWAP & Relative Volume Validation
        vwap_col = self._resolve_feature(df, 'vwap')
        rvol_col = self._resolve_feature(df, 'relative_volume')
        if vwap_col and rvol_col and pd.notna(df[rvol_col].iloc[-1]):
            w = 15.0
            max_possible_weight += w
            active_weight += w
            
            close, vwap, rvol = df['close'].iloc[-1], df[vwap_col].iloc[-1], df[rvol_col].iloc[-1]
            if rvol > 1.2:
                if close > vwap and trend_dir == 1:
                    score += w
                    evidence.append({"type": "VWAP_RVOL", "weight": w, "value": "High Vol hold above VWAP", "polarity": 1})
                elif close < vwap and trend_dir == -1:
                    score -= w
                    evidence.append({"type": "VWAP_RVOL", "weight": w, "value": "High Vol rejection below VWAP", "polarity": -1})

        norm_score = np.clip((score / active_weight) * 100.0 if active_weight > 0 else 0.0, -100.0, 100.0)
        status = "Strong Bullish" if norm_score >= 50 else "Bullish" if norm_score >= 15 else "Neutral" if norm_score > -15 else "Bearish" if norm_score > -50 else "Strong Bearish"

        data_quality = active_weight / max_possible_weight if max_possible_weight > 0 else 0.0
        target_pol = 1 if norm_score > 0 else -1 if norm_score < 0 else 0
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == target_pol)
        agreement = aligned_w / active_weight if active_weight > 0 else 0.0
        confidence = np.clip((data_quality * 0.4 + agreement * 0.6) * 100.0, 0.0, 100.0)

        return {"score": round(norm_score, 2), "confidence": round(confidence, 2), "status": status, "evidence": evidence}

    # --------------------------------------------------------------------------
    # 2. VOLUME EXPLOSION
    # --------------------------------------------------------------------------
    def _analyze_explosion(self, df: pd.DataFrame) -> VolumeExplosionResult:
        evidence: List[EvidenceItem] = []
        score, max_conf, conf_pts = 0.0, 0.0, 0.0
        
        vol_z = self._get_val(df, 'volume_zscore', default=0.0)
        vol_pct = self._get_val(df, 'volume_percentile', default=50.0)
        rvol = self._get_val(df, 'relative_volume', default=1.0)
        
        # 1. Percentile & Regime
        max_conf += 40.0
        if self._resolve_feature(df, 'volume_percentile'):
            conf_pts += 40.0
            if vol_pct > 95.0 or vol_z > 3.0:
                score += 40.0
                evidence.append({"type": "Percentile", "weight": 40.0, "value": f"Extreme Z-Score ({vol_z:.1f})", "polarity": 1})
            elif vol_pct > 80.0:
                score += 25.0
                evidence.append({"type": "Percentile", "weight": 40.0, "value": f"High Volume Regime", "polarity": 1})
            elif vol_pct < 20.0:
                evidence.append({"type": "Percentile", "weight": 40.0, "value": f"Volume Dry-up", "polarity": -1})
                
        # 2. Expansion Velocity
        max_conf += 30.0
        if len(df) > 5:
            conf_pts += 30.0
            vol_roc = df['volume'].pct_change().tail(3)
            if (vol_roc > 0).all():
                score += 30.0
                evidence.append({"type": "Velocity", "weight": 30.0, "value": "Multi-bar Acceleration", "polarity": 1})

        # 3. Exhaustion (Climax)
        max_conf += 30.0
        atr_col = self._resolve_feature(df, 'atr')
        if atr_col and pd.notna(df[atr_col].iloc[-1]):
            conf_pts += 30.0
            body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            spread = df['high'].iloc[-1] - df['low'].iloc[-1] + EPSILON
            if rvol > 2.5 and (body / spread) < 0.3:
                score += 30.0 
                evidence.append({"type": "Climax", "weight": 30.0, "value": "Climax/Exhaustion Risk", "polarity": -1})

        norm_score = np.clip(score, 0.0, 100.0)
        has_climax = any(e['type'] == 'Climax' for e in evidence)
        status = "Climax (Exhaustion)" if has_climax else "Explosive" if norm_score >= 80 else "Expanding" if norm_score >= 50 else "Building" if norm_score >= 20 else "Dormant"
        confidence = (conf_pts / max_conf) * 100.0 if max_conf > 0 else 0.0

        return {"score": round(norm_score, 2), "confidence": round(confidence, 2), "status": status, "evidence": evidence}

    # --------------------------------------------------------------------------
    # 3. DELIVERY ANALYSIS
    # --------------------------------------------------------------------------
    def _analyze_delivery(self, df: pd.DataFrame) -> DeliveryAnalysisResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        
        del_pct_col = self._resolve_feature(df, 'delivery_percent')
        del_qty_col = self._resolve_feature(df, 'delivery_quantity')
        
        if not del_pct_col or pd.isna(df[del_pct_col].iloc[-1]):
            return {"score": 0.0, "confidence": 0.0, "status": "Unavailable", "evidence": []}

        del_pct = df[del_pct_col].iloc[-1]
        close_chg = df['close'].diff().iloc[-1]
        
        # Delivery % Quality & Rolling Rank
        if del_pct > VOLUME_CONFIG["thresholds"]["high_delivery_pct"]:
            score += 30.0
            evidence.append({"type": "Del_Pct", "weight": 30.0, "value": f"High Delivery ({del_pct:.1f}%)", "polarity": 1})
            
        if len(df) > 20:
            del_rank = df[del_pct_col].tail(20).rank(pct=True).iloc[-1] * 100.0
            if del_rank > 80.0:
                score += 20.0
                evidence.append({"type": "Del_Rank", "weight": 20.0, "value": f"Top 20d Rank", "polarity": 1})
            
            del_z = (del_pct - df[del_pct_col].tail(20).mean()) / (df[del_pct_col].tail(20).std() + EPSILON)
            if del_z > 2.0:
                score += 20.0
                evidence.append({"type": "Del_ZScore", "weight": 20.0, "value": "Statistical Anomaly", "polarity": 1})

        # Quantity Alignment
        if del_qty_col and pd.notna(df[del_qty_col].iloc[-1]):
            del_qty = df[del_qty_col].iloc[-1]
            qty_sma = df[del_qty_col].tail(10).mean() + EPSILON
            if del_qty > qty_sma * 1.5:
                if close_chg > 0:
                    score += 30.0
                    evidence.append({"type": "Accumulation", "weight": 30.0, "value": "Heavy Delivery (Up Day)", "polarity": 1})
                elif close_chg < 0:
                    score -= 30.0
                    evidence.append({"type": "Distribution", "weight": 30.0, "value": "Heavy Delivery (Down Day)", "polarity": -1})

        norm_score = np.clip((score / 100.0) * 100.0, -100.0, 100.0)
        status = "Strong Accumulation" if norm_score >= 60 else "Mild Accumulation" if norm_score >= 20 else "Neutral" if norm_score > -20 else "Mild Distribution" if norm_score > -60 else "Strong Distribution"
        
        conflicting = len([e for e in evidence if e['polarity'] == -1]) > 0 and len([e for e in evidence if e['polarity'] == 1]) > 0
        confidence = 100.0 if del_qty_col else 60.0
        if conflicting: confidence -= 30.0

        return {"score": round(norm_score, 2), "confidence": round(confidence, 2), "status": status, "evidence": evidence}

    # --------------------------------------------------------------------------
    # 4. ENSEMBLE PROBABILITY MODEL (SMART VOLUME)
    # --------------------------------------------------------------------------
    def _analyze_smart_volume(self, df: pd.DataFrame, conf_res: VolumeConfirmationResult, expl_res: VolumeExplosionResult, del_res: DeliveryAnalysisResult, div_res: VolumeDivergenceResult, mtf_res: MTFVolumeResult, adv: AdvancedVolumeMetrics) -> SmartVolumeResult:
        evidence: List[EvidenceItem] = []
        
        # Base Probabilities (Bayesian updating approach)
        p_inst = 0.5 
        p_ret = 0.5
        
        def update_prob(prob: float, weight: float, event_true: bool) -> float:
            """Simple ensemble weight shift."""
            if event_true: return prob + weight * (1.0 - prob)
            return prob - weight * prob

        # 1. SMC Structure Footprints
        smc_feats = ['ob_active', 'fvg_active', 'liq_sweep', 'bos', 'choch', 'breaker', 'mitigation']
        struct_active = False
        for feat in smc_feats:
            col = self._resolve_feature(df, feat)
            if col and pd.notna(df[col].iloc[-1]) and df[col].iloc[-1] != 0:
                struct_active = True
                p_inst = update_prob(p_inst, 0.20, True)
                evidence.append({"type": "SMC", "weight": 20.0, "value": f"Structure {feat.upper()} Active", "polarity": 1})

        # 2. Advanced Institutional Behaviors
        if adv['dark_pool_proxy']:
            p_inst = update_prob(p_inst, 0.25, True)
            evidence.append({"type": "Dark_Pool", "weight": 25.0, "value": "Dark Pool / Hidden Absorption Proxy", "polarity": 1})
            
        if adv['block_trade_proxy']:
            p_inst = update_prob(p_inst, 0.20, True)
            evidence.append({"type": "Block_Trade", "weight": 20.0, "value": "Institutional Block Trade Profile", "polarity": 1})

        # 3. Delivery Confirmation
        if del_res['score'] > 40:
            p_inst = update_prob(p_inst, 0.15, True)
            evidence.append({"type": "Delivery", "weight": 15.0, "value": "Validates Smart Accumulation", "polarity": 1})

        # 4. Retail FOMO / Traps
        rsi = self._get_val(df, 'rsi', 50.0)
        is_retail_trap = not struct_active and expl_res['score'] > 50 and (rsi > 70 or rsi < 30)
        if is_retail_trap:
            p_ret = update_prob(p_ret, 0.35, True)
            p_inst = update_prob(p_inst, 0.20, False)
            evidence.append({"type": "Retail_Trap", "weight": 35.0, "value": "Explosive Vol at Extreme w/o Structure", "polarity": -1})

        # 5. Divergence Penalty
        if div_res['divergence_type'] != "None" and div_res['strength'] > 50:
            p_ret = update_prob(p_ret, 0.15, True)
            evidence.append({"type": "Divergence", "weight": 15.0, "value": "Volume Divergence flags weakness", "polarity": -1})

        inst_prob_final = np.clip(p_inst * 100.0, 0.0, 100.0)
        ret_prob_final = np.clip(p_ret * 100.0, 0.0, 100.0)

        dom = "Institutional" if inst_prob_final > ret_prob_final + 20 else "Retail" if ret_prob_final > inst_prob_final + 20 else "Mixed"
        conf = np.clip(((inst_prob_final + ret_prob_final) / 200.0) * 100.0 + 30.0, 20.0, 95.0)

        return {
            "institutional_probability": round(inst_prob_final, 2),
            "retail_probability": round(ret_prob_final, 2),
            "confidence": round(conf, 2),
            "dominance": dom,
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 5. MULTI-OSCILLATOR DIVERGENCE (PIVOT BASED JIT)
    # --------------------------------------------------------------------------
    def _analyze_divergence(self, df: pd.DataFrame) -> VolumeDivergenceResult:
        evidence: List[EvidenceItem] = []
        votes = {1: 0.0, -1: 0.0, 2: 0.0, -2: 0.0} 
        total_weight = 0.0
        
        lookback = VOLUME_CONFIG["thresholds"]["divergence_lookback"]
        price_arr = df['close'].to_numpy(dtype=np.float64)
        
        for osc, weight in VOLUME_CONFIG["oscillator_weights"].items():
            col = self._resolve_feature(df, osc)
            if col and pd.notna(df[col].iloc[-1]) and len(df) >= lookback:
                total_weight += weight
                osc_arr = df[col].to_numpy(dtype=np.float64)
                
                str_val, div_type = _pivot_divergence_engine_jit(price_arr, osc_arr, lookback)
                if div_type != 0:
                    votes[div_type] += (weight * str_val)
                    evidence.append({"type": osc.upper(), "weight": weight * 100.0, "value": "Pivot Divergence Flagged", "polarity": np.sign(div_type)})

        if total_weight == 0 or max(votes.values()) == 0:
            return {"divergence_type": "None", "strength": 0.0, "confidence": 100.0 if total_weight > 0.5 else 50.0, "evidence": evidence}
            
        dominant_type = max(votes, key=votes.get)
        strength = np.clip((votes[dominant_type] / total_weight), 0.0, 100.0)
        conf = np.clip((sum(v for k,v in votes.items() if np.sign(k) == np.sign(dominant_type)) / total_weight) * 100.0, 0.0, 100.0)
        
        d_map = {1: "Regular Bullish", -1: "Regular Bearish", 2: "Hidden Bullish", -2: "Hidden Bearish"}
        
        return {
            "divergence_type": d_map[dominant_type],
            "strength": round(strength, 2),
            "confidence": round(conf, 2),
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 6. EXPANDED MULTI-TIMEFRAME ALIGNMENT
    # --------------------------------------------------------------------------
    def _analyze_mtf(self, df: pd.DataFrame) -> MTFVolumeResult:
        timeframes = {}
        score, act_w = 0.0, 0.0
        
        for sfx in self.mtf_suffixes:
            w = VOLUME_CONFIG["mtf_weights"].get(sfx, 0.0)
            tf_score = 0.0
            tf_count = 0
            
            # Check multiple volume indicators in MTF
            for base in ['volume', 'obv', 'cmf', 'relative_volume', 'vwap']:
                col = self._resolve_feature(df, f"{base}{sfx}")
                if col and pd.notna(df[col].iloc[-1]):
                    tf_count += 1
                    val = df[col].iloc[-1]
                    if base == 'volume':
                        ma_col = self._resolve_feature(df, f"volume_ma_20{sfx}")
                        if ma_col and val > df[ma_col].iloc[-1]: tf_score += 1
                        else: tf_score -= 1
                    elif base in ['obv', 'cmf']:
                        if val > df[col].iloc[-2]: tf_score += 1
                        else: tf_score -= 1
                    elif base == 'vwap':
                        if df[f'close{sfx}'].iloc[-1] > val: tf_score += 1
                        else: tf_score -= 1
            
            if tf_count > 0:
                act_w += w
                norm_tf = tf_score / tf_count
                if norm_tf > 0:
                    timeframes[sfx.strip('_')] = "Bullish / Expansion"
                    score += w
                else:
                    timeframes[sfx.strip('_')] = "Bearish / Compression"
                    score -= w

        norm = (score / act_w * 100.0) if act_w > 0 else 0.0
        trend = "Macro Accumulation" if norm > 50 else "Macro Distribution" if norm < -50 else "Mixed Context"
        
        return {
            "alignment_score": round(norm, 2),
            "dominant_trend": trend,
            "confidence": round(act_w * 100.0, 2),
            "timeframes": timeframes
        }

    # --------------------------------------------------------------------------
    # 7. ADVANCED VOLUME METRICS & INSTITUTIONAL PROXIES
    # --------------------------------------------------------------------------
    def _analyze_advanced_metrics(self, df: pd.DataFrame, conf: VolumeConfirmationResult, expl: VolumeExplosionResult, deliv: DeliveryAnalysisResult, div: VolumeDivergenceResult) -> AdvancedVolumeMetrics:
        latest = df.iloc[-1]
        
        # Stability & Efficiency
        vol_arr = df['volume'].tail(20)
        vol_std, vol_mean = vol_arr.std() + EPSILON, vol_arr.mean() + EPSILON
        vol_stability = np.clip((1.0 - (vol_std / vol_mean)) * 100.0, 0.0, 100.0)
        
        price_change = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        vol_efficiency = np.clip((price_change / (latest['volume'] + EPSILON)) * 1e6, 0.0, 100.0)
        
        # Dry-up & Shifts
        vol_pct = self._get_val(df, 'volume_percentile', 50.0)
        dry_up = bool(vol_pct < 15.0)
        regime_shift = bool(expl['status'] in ["Explosive", "Climax"] and vol_arr.iloc[-2] < vol_mean)
        
        # Liquidity Absorption
        atr = self._get_val(df, 'atr', EPSILON)
        abs_prob = 80.0 if (expl['score'] > 50 and price_change < atr * 0.5) else 0.0
        
        # Compression vs Expansion Prob
        comp_score = 100.0 - vol_pct if dry_up else 0.0
        exp_prob = np.clip(expl['score'], 0.0, 100.0)

        # Regimes & Indices
        spi = np.clip((deliv['score'] * 0.6) + (conf['score'] * 0.4), 0.0, 100.0)
        rpi = np.clip(expl['score'] - spi, 0.0, 100.0)
        p_regime = "Smart Accumulation" if spi > 60 else "Retail Speculation" if rpi > 60 else "Equilibrium"

        # VFI (Volume Fractal Index) Proxy
        vfi = 0.0
        if len(df) > 20:
            tr = df['high'] - df['low']
            vfi_series = ((df['close'] - df['close'].shift(1)) / (tr + EPSILON)) * df['volume']
            vfi = vfi_series.tail(20).sum() / (vol_mean * 20.0 + EPSILON) * 100.0

        # Institutional Proxies
        rvol = self._get_val(df, 'relative_volume', 1.0)
        spread = df['high'].iloc[-1] - df['low'].iloc[-1] + EPSILON
        
        # Dark Pool Proxy: Massive Volume, negligible price displacement
        dp_proxy = bool(rvol > 2.0 and spread < atr * VOLUME_CONFIG["thresholds"]["dark_pool_displacement_limit"])
        
        # Block Trade Proxy: Single bar massive spike on low overall volatility
        block_proxy = bool(rvol > 3.0 and vol_pct > 90.0 and spread < atr)

        return {
            "volume_quality": round(np.clip((conf['score'] + 100.0) / 2.0, 0.0, 100.0), 2),
            "volume_efficiency": round(vol_efficiency, 2),
            "participation_regime": p_regime,
            "smart_participation_index": round(spi, 2),
            "retail_participation_index": round(rpi, 2),
            "volume_stability": round(vol_stability, 2),
            "volume_persistence": round(expl['score'], 2),
            "volume_dry_up": dry_up,
            "volume_regime_shift": regime_shift,
            "liquidity_absorption": round(abs_prob, 2),
            "volume_compression_score": round(comp_score, 2),
            "volume_expansion_probability": round(exp_prob, 2),
            "block_trade_proxy": block_proxy,
            "dark_pool_proxy": dp_proxy,
            "volume_fractal_index": round(vfi, 2)
        }

# ==============================================================================
# EXPORTS
# ==============================================================================
__all__ = [
    "VolumeAnalyzer",
    "VolumeAnalysisResult",
    "VolumeConfirmationResult",
    "VolumeExplosionResult",
    "DeliveryAnalysisResult",
    "SmartVolumeResult",
    "VolumeDivergenceResult",
    "MTFVolumeResult",
    "AdvancedVolumeMetrics",
    "EvidenceItem"
]
