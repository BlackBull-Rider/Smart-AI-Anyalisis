import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Tuple

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

MOMENTUM_CONFIG = {
    "lookbacks": {
        "micro": 3,
        "short": 5,
        "medium": 14,
        "long": 30,
        "divergence": 20
    },
    "thresholds": {
        "rsi_ob": 70.0,
        "rsi_os": 30.0,
        "adx_trend": 25.0,
        "vol_expansion": 1.5,
        "climax_vol_mult": 3.0,
        "compression_limit": 0.5,
        "atr_expansion": 1.2
    },
    "weights": {
        "macd": 20.0,
        "rsi": 20.0,
        "roc": 15.0,
        "momentum": 15.0,
        "adx": 15.0,
        "slope": 15.0
    },
    "mtf_weights": {
        "_M": 0.30,
        "_W": 0.25,
        "_D": 0.20,
        "_4H": 0.10,
        "_1H": 0.05,
        "_15m": 0.05,
        "_5m": 0.05
    }
}

# ==============================================================================
# TYPE DEFINITIONS FOR OUTPUT STRUCTURE
# ==============================================================================

class EvidenceItem(TypedDict):
    type: str
    weight: float
    value: str
    polarity: int

class MomentumStrengthResult(TypedDict):
    status: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class MomentumAccelerationResult(TypedDict):
    status: str
    score: float
    confidence: float
    acceleration_rate: float
    evidence: List[EvidenceItem]

class MomentumExhaustionResult(TypedDict):
    status: str
    score: float
    confidence: float
    index: float
    evidence: List[EvidenceItem]

class MomentumShiftResult(TypedDict):
    status: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class MomentumIgnitionResult(TypedDict):
    stage: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class MomentumCompressionResult(TypedDict):
    status: str
    score: float
    evidence: List[EvidenceItem]

class MomentumCycleResult(TypedDict):
    stage: str
    confidence: float

class RSIAnalysisResult(TypedDict):
    status: str
    score: float
    confidence: float
    regime: str
    evidence: List[EvidenceItem]

class MACDAnalysisResult(TypedDict):
    status: str
    score: float
    confidence: float
    phase: str
    evidence: List[EvidenceItem]

class DivergenceResult(TypedDict):
    status: str
    strength: float
    confidence: float
    divergence_type: str
    multi_oscillator_score: float
    evidence: List[EvidenceItem]

class InstitutionalMomentumResult(TypedDict):
    institutional_score: float
    retail_score: float
    dominance: str
    evidence: List[EvidenceItem]

class MTFMomentumResult(TypedDict):
    alignment: str
    score: float
    timeframes: Dict[str, str]

class SwingReadinessResult(TypedDict):
    state: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class AdvancedMomentumMetrics(TypedDict):
    momentum_persistence: float
    momentum_efficiency: float
    momentum_stability: float
    momentum_consistency: float
    impulse_strength: float
    pullback_strength: float
    reversal_probability: float
    continuation_probability: float

class MomentumAnalysisResult(TypedDict):
    momentum_strength: MomentumStrengthResult
    momentum_acceleration: MomentumAccelerationResult
    momentum_exhaustion: MomentumExhaustionResult
    momentum_shift: MomentumShiftResult
    momentum_ignition: MomentumIgnitionResult
    momentum_compression: MomentumCompressionResult
    momentum_cycle: MomentumCycleResult
    institutional_momentum: InstitutionalMomentumResult
    multi_timeframe: MTFMomentumResult
    rsi_analysis: RSIAnalysisResult
    macd_analysis: MACDAnalysisResult
    divergence: DivergenceResult
    swing_readiness: SwingReadinessResult
    advanced_metrics: AdvancedMomentumMetrics

# ==============================================================================
# NUMBA JIT ACCELERATED ENGINES
# ==============================================================================

@jit(nopython=True, cache=True)
def _get_nanmin_nanmax(arr: np.ndarray) -> Tuple[float, float]:
    min_val = np.inf
    max_val = -np.inf
    for x in arr:
        if not np.isnan(x):
            if x < min_val: min_val = x
            if x > max_val: max_val = x
    if np.isinf(min_val):
        return np.nan, np.nan
    return min_val, max_val

@jit(nopython=True, cache=True)
def _divergence_engine_jit(price: np.ndarray, osc: np.ndarray, lookback: int, is_rsi: bool) -> Tuple[float, int]:
    """
    O(N) Divergence Scanner. 
    Returns: (strength [0-100], div_type [1: RegBull, -1: RegBear, 2: HidBull, -2: HidBear, 0: None])
    """
    n = len(price)
    if n < lookback: return 0.0, 0
    half_lookback = lookback // 2
    
    p_past, p_rec = price[-lookback : -half_lookback], price[-half_lookback :]
    o_past, o_rec = osc[-lookback : -half_lookback], osc[-half_lookback :]
    
    max_p_past, max_p_rec = _get_nanmin_nanmax(p_past)[1], _get_nanmin_nanmax(p_rec)[1]
    min_p_past, min_p_rec = _get_nanmin_nanmax(p_past)[0], _get_nanmin_nanmax(p_rec)[0]
    
    max_o_past, max_o_rec = _get_nanmin_nanmax(o_past)[1], _get_nanmin_nanmax(o_rec)[1]
    min_o_past, min_o_rec = _get_nanmin_nanmax(o_past)[0], _get_nanmin_nanmax(o_rec)[0]
    
    if np.isnan(max_p_past) or np.isnan(max_o_past): return 0.0, 0
        
    ob_thresh = 65.0 if is_rsi else 0.0
    os_thresh = 35.0 if is_rsi else 0.0

    if max_p_rec > max_p_past and max_o_rec < max_o_past and max_o_past > ob_thresh:
        return min(((max_o_past - max_o_rec) / (np.abs(max_o_past) + EPSILON)) * 200.0, 100.0), -1
    if min_p_rec < min_p_past and min_o_rec > min_o_past and min_o_past < os_thresh:
        return min(((min_o_rec - min_o_past) / (np.abs(min_o_past) + EPSILON)) * 200.0, 100.0), 1
    if max_p_rec < max_p_past and max_o_rec > max_o_past and max_o_rec > ob_thresh:
        return min(((max_o_rec - max_o_past) / (np.abs(max_o_past) + EPSILON)) * 150.0, 100.0), -2
    if min_p_rec > min_p_past and min_o_rec < min_o_past and min_o_rec < os_thresh:
        return min(((min_o_past - min_o_rec) / (np.abs(min_o_past) + EPSILON)) * 150.0, 100.0), 2

    return 0.0, 0

# ==============================================================================
# MOMENTUM ANALYZER ENGINE
# ==============================================================================

class MomentumAnalyzer:
    """
    Institutional Momentum Analyzer evaluating strength, shifts, divergences, 
    and multi-factor swing readiness using pure functional arrays.
    """

    def __init__(self):
        self.req_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'macd_line', 'macd_signal', 'macd_histogram',
            'rsi', 'adx', 'linreg_slope', 'linreg_r2', 'roc', 'momentum', 'atr_14'
        ]
        self.opt_cols = [
            'bos', 'choch', 'ob_active', 'fvg_active', 'liq_sweep'
        ]
        self.mtf_suffixes = ['_5m', '_15m', '_1H', '_4H', '_D', '_W', '_M']

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.req_cols if col not in df.columns]
        if missing:
            logger.error(f"MomentumAnalyzer missing critical columns: {missing}")
            raise ValueError(f"MomentumAnalyzer requires missing columns: {missing}")
            
        min_len = MOMENTUM_CONFIG["lookbacks"]["long"] + 1
        if len(df) < min_len:
            raise ValueError(f"MomentumAnalyzer requires at least {min_len} bars.")

        working_df = df.copy()
        num_cols = working_df.select_dtypes(include=[np.number]).columns
        if np.isinf(working_df[num_cols]).any().any():
            working_df[num_cols] = working_df[num_cols].replace([np.inf, -np.inf], np.nan)

        if working_df[self.req_cols].isna().any().any():
            working_df[self.req_cols] = working_df[self.req_cols].ffill().bfill()
            
        if working_df[self.req_cols].isna().any().any():
            raise ValueError("Data contains unresolvable NaN columns.")
            
        return working_df

    def analyze(self, df: pd.DataFrame) -> MomentumAnalysisResult:
        safe_df = self._validate_data(df)
        working_df = safe_df.tail(MOMENTUM_CONFIG["lookbacks"]["long"])

        macd_res = self._analyze_macd(working_df)
        rsi_res = self._analyze_rsi(working_df)
        
        strength_res = self._analyze_strength(working_df, macd_res, rsi_res)
        accel_res = self._analyze_acceleration(working_df)
        exhaust_res = self._analyze_exhaustion(working_df)
        
        shift_res = self._analyze_shift(working_df)
        comp_res = self._analyze_compression(working_df)
        inst_res = self._analyze_institutional(working_df)
        ignit_res = self._analyze_ignition(shift_res, comp_res, inst_res, working_df)
        
        div_res = self._analyze_divergence(working_df)
        mtf_res = self._analyze_mtf(working_df)
        
        cycle_res = self._analyze_cycle(comp_res, ignit_res, strength_res, accel_res, exhaust_res, shift_res)
        adv_metrics = self._analyze_advanced_metrics(strength_res, accel_res, exhaust_res, comp_res, div_res, inst_res, mtf_res, working_df)
        readiness_res = self._analyze_swing_readiness(ignit_res, shift_res, inst_res, mtf_res, exhaust_res, strength_res, div_res, adv_metrics, working_df)
        
        return {
            "momentum_strength": strength_res,
            "momentum_acceleration": accel_res,
            "momentum_exhaustion": exhaust_res,
            "momentum_shift": shift_res,
            "momentum_ignition": ignit_res,
            "momentum_compression": comp_res,
            "momentum_cycle": cycle_res,
            "institutional_momentum": inst_res,
            "multi_timeframe": mtf_res,
            "rsi_analysis": rsi_res,
            "macd_analysis": macd_res,
            "divergence": div_res,
            "swing_readiness": readiness_res,
            "advanced_metrics": adv_metrics
        }

    # --------------------------------------------------------------------------
    # 1. MOMENTUM STRENGTH
    # --------------------------------------------------------------------------
    def _analyze_strength(self, df: pd.DataFrame, macd: MACDAnalysisResult, rsi: RSIAnalysisResult) -> MomentumStrengthResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        max_score = sum(MOMENTUM_CONFIG["weights"].values())
        latest = df.iloc[-1]
        
        w_macd = MOMENTUM_CONFIG["weights"]["macd"]
        if macd['score'] > 0:
            score += w_macd; evidence.append({"type": "MACD", "weight": w_macd, "value": "Bullish MACD Force", "polarity": 1})
        elif macd['score'] < 0:
            score -= w_macd; evidence.append({"type": "MACD", "weight": w_macd, "value": "Bearish MACD Force", "polarity": -1})
        
        w_rsi = MOMENTUM_CONFIG["weights"]["rsi"]
        if rsi['score'] > 0:
            score += w_rsi; evidence.append({"type": "RSI", "weight": w_rsi, "value": "Bullish RSI", "polarity": 1})
        elif rsi['score'] < 0:
            score -= w_rsi; evidence.append({"type": "RSI", "weight": w_rsi, "value": "Bearish RSI", "polarity": -1})
            
        w_roc = MOMENTUM_CONFIG["weights"]["roc"]
        if latest['roc'] > 0:
            score += w_roc; evidence.append({"type": "ROC", "weight": w_roc, "value": "Positive ROC", "polarity": 1})
        elif latest['roc'] < 0:
            score -= w_roc; evidence.append({"type": "ROC", "weight": w_roc, "value": "Negative ROC", "polarity": -1})

        w_mom = MOMENTUM_CONFIG["weights"]["momentum"]
        if latest['momentum'] > 0:
            score += w_mom; evidence.append({"type": "Momentum", "weight": w_mom, "value": "Positive Momentum", "polarity": 1})
        elif latest['momentum'] < 0:
            score -= w_mom; evidence.append({"type": "Momentum", "weight": w_mom, "value": "Negative Momentum", "polarity": -1})

        w_adx = MOMENTUM_CONFIG["weights"]["adx"]
        if latest['adx'] > MOMENTUM_CONFIG["thresholds"]["adx_trend"]:
            aligned = 1 if score > 0 else -1
            score += w_adx if aligned == 1 else -w_adx
            evidence.append({"type": "ADX", "weight": w_adx, "value": "ADX Supporting Trend", "polarity": aligned})

        norm_score = np.clip((score / max_score) * 100.0, -100.0, 100.0)
        
        if norm_score >= 60: status = "Strong Bullish"
        elif norm_score >= 20: status = "Bullish"
        elif norm_score > -20: status = "Neutral"
        elif norm_score > -60: status = "Bearish"
        else: status = "Strong Bearish"

        pol = 1 if norm_score > 0 else -1 if norm_score < 0 else 0
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == pol)
        total_w = sum(e['weight'] for e in evidence)
        conf = np.clip((aligned_w / total_w) * 100.0 if total_w > 0 else 0.0, 0.0, 100.0)

        return {"status": status, "score": round(norm_score, 2), "confidence": round(conf, 2), "evidence": evidence}

    # --------------------------------------------------------------------------
    # 2. ACCELERATION & EXHAUSTION
    # --------------------------------------------------------------------------
    def _analyze_acceleration(self, df: pd.DataFrame) -> MomentumAccelerationResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        accel_rate = 0.0
        recent = df.tail(MOMENTUM_CONFIG["lookbacks"]["short"])
        
        hist_slope = recent['macd_histogram'].diff().mean()
        if hist_slope > 0 and df['macd_histogram'].iloc[-1] > 0:
            score += 30.0; accel_rate += 1.0
            evidence.append({"type": "Hist_Expand", "weight": 30.0, "value": "Bullish Histogram Expansion", "polarity": 1})
        elif hist_slope < 0 and df['macd_histogram'].iloc[-1] < 0:
            score += 30.0; accel_rate += 1.0
            evidence.append({"type": "Hist_Expand", "weight": 30.0, "value": "Bearish Histogram Expansion", "polarity": -1})

        rsi_slope = recent['rsi'].diff().mean()
        if abs(rsi_slope) > 1.5:
            score += 30.0; accel_rate += abs(rsi_slope) * 0.1
            evidence.append({"type": "RSI_Velocity", "weight": 30.0, "value": "High RSI Velocity", "polarity": 1 if rsi_slope > 0 else -1})

        score = np.clip(score, 0.0, 100.0)
        status = "Rapid Acceleration" if score >= 60 else "Accelerating" if score >= 30 else "Constant"
        
        target_pol = 1 if (hist_slope > 0 or rsi_slope > 0) else -1
        total_w = sum(e['weight'] for e in evidence)
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == target_pol)
        conf = np.clip((aligned_w / total_w) * 100.0 if total_w > 0 else 0.0, 0.0, 100.0)

        return {"status": status, "score": round(score, 2), "confidence": round(conf, 2), "acceleration_rate": round(accel_rate, 2), "evidence": evidence}

    def _analyze_exhaustion(self, df: pd.DataFrame) -> MomentumExhaustionResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        latest = df.iloc[-1]
        
        vol_sma = df['volume'].tail(20).mean()
        if latest['volume'] > vol_sma * MOMENTUM_CONFIG["thresholds"]["climax_vol_mult"]:
            body_pct = abs(latest['close'] - latest['open']) / (latest['high'] - latest['low'] + EPSILON)
            if body_pct < 0.4:
                score += 40.0
                evidence.append({"type": "Volume_Climax", "weight": 40.0, "value": "Blowoff / Climax Volume w/ Rejection", "polarity": -1})
            
        if latest['rsi'] > 80 or latest['rsi'] < 20:
            score += 30.0
            evidence.append({"type": "RSI_Extreme", "weight": 30.0, "value": "RSI at Extreme Exhaustion", "polarity": -1})

        roc_std = df['roc'].tail(10).std()
        roc_std_past = df['roc'].iloc[-20:-10].std()
        if roc_std < roc_std_past * 0.5:
            score += 30.0
            evidence.append({"type": "Volatility_Collapse", "weight": 30.0, "value": "Momentum Volatility Collapsing post-peak", "polarity": -1})

        score = np.clip(score, 0.0, 100.0)
        status = "Climax/Blowoff" if score > 70 else "Exhausting" if score > 35 else "Normal"
        
        total_w = sum(e['weight'] for e in evidence)
        active_w = sum(e['weight'] for e in evidence if e['polarity'] != 0)
        conf = np.clip((active_w / total_w) * 100.0 if total_w > 0 else 0.0, 0.0, 100.0)
        
        return {"status": status, "score": round(score, 2), "confidence": round(conf, 2), "index": round(score, 2), "evidence": evidence}

    # --------------------------------------------------------------------------
    # 3. EARLY SWING: SHIFT, COMPRESSION & IGNITION (ADVANCED)
    # --------------------------------------------------------------------------
    def _analyze_shift(self, df: pd.DataFrame) -> MomentumShiftResult:
        evidence: List[EvidenceItem] = []
        recent = df.tail(3)
        score = 0.0
        
        if recent['macd_histogram'].iloc[-2] < 0 and recent['macd_histogram'].iloc[-1] > 0:
            score += 25.0; evidence.append({"type": "MACD_Hist", "weight": 25.0, "value": "MACD Hist Negative -> Positive", "polarity": 1})
        elif recent['macd_histogram'].iloc[-2] > 0 and recent['macd_histogram'].iloc[-1] < 0:
            score -= 25.0; evidence.append({"type": "MACD_Hist", "weight": 25.0, "value": "MACD Hist Positive -> Negative", "polarity": -1})

        if recent['macd_line'].iloc[-2] < recent['macd_signal'].iloc[-2] and recent['macd_line'].iloc[-1] > recent['macd_signal'].iloc[-1]:
            score += 20.0; evidence.append({"type": "MACD_Line", "weight": 20.0, "value": "MACD Line Bullish Cross", "polarity": 1})
        elif recent['macd_line'].iloc[-2] > recent['macd_signal'].iloc[-2] and recent['macd_line'].iloc[-1] < recent['macd_signal'].iloc[-1]:
            score -= 20.0; evidence.append({"type": "MACD_Line", "weight": 20.0, "value": "MACD Line Bearish Cross", "polarity": -1})

        if recent['rsi'].iloc[-2] < 50 and recent['rsi'].iloc[-1] >= 50:
            score += 20.0; evidence.append({"type": "RSI_Cross", "weight": 20.0, "value": "RSI Crosses > 50", "polarity": 1})
        elif recent['rsi'].iloc[-2] > 50 and recent['rsi'].iloc[-1] <= 50:
            score -= 20.0; evidence.append({"type": "RSI_Cross", "weight": 20.0, "value": "RSI Crosses < 50", "polarity": -1})

        if recent['momentum'].iloc[-2] < 0 and recent['momentum'].iloc[-1] > 0:
            score += 15.0; evidence.append({"type": "Mom_Cross", "weight": 15.0, "value": "Momentum Zero Cross Up", "polarity": 1})
        elif recent['momentum'].iloc[-2] > 0 and recent['momentum'].iloc[-1] < 0:
            score -= 15.0; evidence.append({"type": "Mom_Cross", "weight": 15.0, "value": "Momentum Zero Cross Down", "polarity": -1})

        status = "Positive Shift" if score >= 30 else "Negative Shift" if score <= -30 else "No Major Shift"
        
        target_pol = 1 if score > 0 else -1 if score < 0 else 0
        total_w = sum(e['weight'] for e in evidence)
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == target_pol)
        conf = np.clip((aligned_w / total_w) * 100.0 if total_w > 0 else 0.0, 0.0, 100.0)

        return {"status": status, "score": round(score, 2), "confidence": round(conf, 2), "evidence": evidence}

    def _analyze_compression(self, df: pd.DataFrame) -> MomentumCompressionResult:
        evidence: List[EvidenceItem] = []
        recent = df.tail(20)
        score = 0.0
        
        roc_std = recent['roc'].std()
        if roc_std < MOMENTUM_CONFIG["thresholds"]["compression_limit"]:
            score += 35.0
            evidence.append({"type": "ROC_Squeeze", "weight": 35.0, "value": "Momentum heavily compressed", "polarity": 0})
            
        hh, ll = recent['high'].max(), recent['low'].min()
        atr = recent['atr_14'].iloc[-1]
        donch_width = (hh - ll) / (atr + EPSILON)
        
        if donch_width < 4.0: 
            score += 35.0
            evidence.append({"type": "Price_Squeeze", "weight": 35.0, "value": "Price Volatility Squeeze (Donchian/BB Proxy)", "polarity": 0})
            
        atr_sma = recent['atr_14'].mean()
        if atr < atr_sma * 0.8:
            score += 30.0
            evidence.append({"type": "ATR_Squeeze", "weight": 30.0, "value": "ATR Compression Phase", "polarity": 0})
            
        status = "High Compression" if score >= 70 else "Building" if score >= 35 else "Expanded"
        return {"status": status, "score": round(score, 2), "evidence": evidence}

    def _analyze_ignition(self, shift: MomentumShiftResult, comp: MomentumCompressionResult, inst: InstitutionalMomentumResult, df: pd.DataFrame) -> MomentumIgnitionResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        latest = df.iloc[-1]
        
        if abs(shift['score']) >= 30:
            score += 25.0
            evidence.append({"type": "Shift", "weight": 25.0, "value": "Directional Momentum Shift detected", "polarity": np.sign(shift['score'])})
            
        if comp['score'] >= 35 and abs(shift['score']) > 0:
            score += 25.0
            evidence.append({"type": "Compression_Break", "weight": 25.0, "value": "Breaking out of Compression", "polarity": np.sign(shift['score'])})

        # True Institutional Volume Alignment (Ensures volume spike is paired with structural intent)
        vol_sma = df['volume'].tail(20).mean()
        vol_spike = latest['volume'] > vol_sma * MOMENTUM_CONFIG["thresholds"]["vol_expansion"]
        if vol_spike and inst['institutional_score'] > 30:
            score += 25.0
            evidence.append({"type": "Inst_Volume", "weight": 25.0, "value": "SMC Aligned Volume Expansion", "polarity": np.sign(shift['score'])})
            
        atr_sma = df['atr_14'].tail(20).mean()
        if latest['atr_14'] > atr_sma * MOMENTUM_CONFIG["thresholds"]["atr_expansion"]:
            score += 25.0
            evidence.append({"type": "Volatility", "weight": 25.0, "value": "Volatility Expanding with Direction", "polarity": np.sign(shift['score'])})

        score = np.clip(score, 0.0, 100.0)
        stage = "Explosive" if score >= 85 else "Igniting" if score >= 60 else "Building" if score >= 30 else "Dormant"
        
        total_w = sum(e['weight'] for e in evidence)
        target_pol = np.sign(shift['score']) if shift['score'] != 0 else 0
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == target_pol)
        conf = np.clip((aligned_w / total_w) * 100.0 if total_w > 0 else 0.0, 0.0, 100.0)
        
        return {"stage": stage, "score": round(score, 2), "confidence": round(conf, 2), "evidence": evidence}

    # --------------------------------------------------------------------------
    # 4. MULTI-TIMEFRAME (MTF) MOMENTUM ALIGNMENT (Multi-Indicator)
    # --------------------------------------------------------------------------
    def _analyze_mtf(self, df: pd.DataFrame) -> MTFMomentumResult:
        latest = df.iloc[-1]
        tfs = {}
        score = 0.0
        act_w = 0.0
        weights = MOMENTUM_CONFIG["mtf_weights"]
        
        # Require strict local alignment across ROC, RSI, MACD Histogram, Momentum, and ADX context
        local_bull = (latest['roc'] > 0 and latest['rsi'] > 50 and latest['macd_histogram'] > 0 and latest['momentum'] > 0)
        local_bear = (latest['roc'] < 0 and latest['rsi'] < 50 and latest['macd_histogram'] < 0 and latest['momentum'] < 0)
        
        tfs['Local'] = "Bullish" if local_bull else "Bearish" if local_bear else "Mixed"
        act_w += 0.15 
        if local_bull: score += 0.15
        elif local_bear: score -= 0.15
        
        for sfx in self.mtf_suffixes:
            roc_col, rsi_col = f'roc{sfx}', f'rsi{sfx}'
            macd_col, mom_col, adx_col = f'macd_histogram{sfx}', f'momentum{sfx}', f'adx{sfx}'
            
            # Check availability of all required HTF indicators
            if all(c in df.columns and pd.notna(latest[c]) for c in [roc_col, rsi_col, macd_col, mom_col, adx_col]):
                w = weights.get(sfx, 0.0)
                act_w += w
                
                # Strict multi-indicator HTF alignment
                b_align = latest[roc_col] > 0 and latest[rsi_col] > 50 and latest[macd_col] > 0 and latest[mom_col] > 0
                br_align = latest[roc_col] < 0 and latest[rsi_col] < 50 and latest[macd_col] < 0 and latest[mom_col] < 0
                
                if b_align:
                    tfs[sfx.strip('_')] = "Bullish"
                    score += w
                elif br_align:
                    tfs[sfx.strip('_')] = "Bearish"
                    score -= w
                else:
                    tfs[sfx.strip('_')] = "Mixed"
                    
        norm = (score / act_w * 100.0) if act_w > 0 else 0.0
        align = "Strong Bullish Alignment" if norm > 70 else "Strong Bearish Alignment" if norm < -70 else "Mixed / Choppy"
        return {"alignment": align, "score": round(norm, 2), "timeframes": tfs}

    # --------------------------------------------------------------------------
    # 5. CYCLE DETECTION (8-STAGE STABLE)
    # --------------------------------------------------------------------------
    def _analyze_cycle(self, comp: MomentumCompressionResult, ignit: MomentumIgnitionResult, str_res: MomentumStrengthResult, acc_res: MomentumAccelerationResult, exh_res: MomentumExhaustionResult, shift: MomentumShiftResult) -> MomentumCycleResult:
        # Ensures that stages don't flicker by demanding heavier evidence for stage transitions
        if comp['score'] >= 70: stage = "Accumulation"
        elif ignit['score'] >= 60 and shift['score'] != 0: stage = "Early Expansion"
        elif ignit['score'] >= 30 and abs(str_res['score']) > 20: stage = "Expansion"
        elif acc_res['score'] >= 60: stage = "Acceleration"
        elif abs(str_res['score']) >= 60: stage = "Markup/Markdown"
        elif exh_res['score'] >= 60: stage = "Exhaustion"
        elif abs(shift['score']) > 40 and abs(str_res['score']) > 0 and np.sign(shift['score']) != np.sign(str_res['score']): 
            stage = "Distribution"
        else: stage = "Reset"
        
        # Proxied confidence based on clarity of stage
        conf = 100.0 if stage in ["Accumulation", "Acceleration", "Exhaustion"] else 80.0
        return {"stage": stage, "confidence": round(conf, 2)}

    # --------------------------------------------------------------------------
    # 6. INSTITUTIONAL VS RETAIL MOMENTUM (VOLUME ALIGNED)
    # --------------------------------------------------------------------------
    def _analyze_institutional(self, df: pd.DataFrame) -> InstitutionalMomentumResult:
        evidence: List[EvidenceItem] = []
        inst_score = 0.0
        retail_score = 0.0
        latest = df.iloc[-1]
        
        # Ensure optional SMC columns exist safely
        smc_aligned = False
        has_bos = 'bos' in df.columns and pd.notna(latest['bos']) and latest['bos'] != 0
        has_ob = 'ob_active' in df.columns and pd.notna(latest['ob_active']) and latest['ob_active']
        has_fvg = 'fvg_active' in df.columns and pd.notna(latest['fvg_active']) and latest['fvg_active']
        has_liq = 'liq_sweep' in df.columns and pd.notna(latest['liq_sweep']) and latest['liq_sweep'] != 0

        if has_ob: inst_score += 30.0; smc_aligned = True
        if has_fvg: inst_score += 20.0; smc_aligned = True
        if has_liq: inst_score += 20.0; smc_aligned = True
        if has_bos: inst_score += 30.0; smc_aligned = True
        
        # Volume Alignment Penalty/Bonus
        vol_sma = df['volume'].tail(20).mean()
        vol_ratio = latest['volume'] / (vol_sma + EPSILON)
        
        if smc_aligned:
            if vol_ratio > 1.2:
                inst_score += 20.0 * min(vol_ratio, 2.0) # Bonus for volume confirmation
                evidence.append({"type": "SMC_Vol", "weight": inst_score, "value": "SMC Supported by Volume Expansion", "polarity": 1})
            elif vol_ratio < 0.8:
                inst_score *= 0.5 # Severe penalty for structure breaks lacking volume (Trap Risk)
                evidence.append({"type": "SMC_Trap", "weight": inst_score, "value": "SMC Structure without Volume (Trap Risk)", "polarity": -1})
            else:
                evidence.append({"type": "SMC", "weight": inst_score, "value": "Smart Money Footprints Active", "polarity": 1})
            
        # Retail checks (Pure Oscillator extreme chasing)
        if latest['rsi'] > 75 or latest['rsi'] < 25:
            retail_score += 50.0
            evidence.append({"type": "Retail", "weight": 50.0, "value": "Retail Oscillator Extremes", "polarity": -1})
            
        inst_score = np.clip(inst_score, 0.0, 100.0)
        retail_score = np.clip(retail_score, 0.0, 100.0)
        
        dom = "Institutional Dominance" if inst_score > retail_score + 20 else "Retail Dominance" if retail_score > inst_score + 20 else "Mixed Context"
        
        return {"institutional_score": round(inst_score, 2), "retail_score": round(retail_score, 2), "dominance": dom, "evidence": evidence}

    # --------------------------------------------------------------------------
    # 7. SWING READINESS (12-FACTOR ENGINE)
    # --------------------------------------------------------------------------
    def _analyze_swing_readiness(self, ignit: MomentumIgnitionResult, shift: MomentumShiftResult, inst: InstitutionalMomentumResult, mtf: MTFMomentumResult, exh: MomentumExhaustionResult, s: MomentumStrengthResult, div: DivergenceResult, adv: AdvancedMomentumMetrics, df: pd.DataFrame) -> SwingReadinessResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        
        target_pol = np.sign(shift['score']) if shift['score'] != 0 else np.sign(s['score'])
        
        # 1-2. Ignition & Shift (Early Entry Multipliers)
        score += (ignit['score'] / 100.0) * 15.0
        score += (abs(shift['score']) / 100.0) * 10.0
        
        # 3-4. Institutional & Volume Quality
        score += (inst['institutional_score'] / 100.0) * 15.0
        score += (adv['volume_confirmation'] / 100.0) * 10.0
        
        # 5. MTF Alignment
        if target_pol != 0 and np.sign(mtf['score']) == target_pol:
            score += (abs(mtf['score']) / 100.0) * 10.0
            
        # 6-7. Trend Strength & Momentum Quality Validation
        score += (abs(s['score']) / 100.0) * 10.0
        score += (adv['momentum_quality'] / 100.0) * 10.0
        
        # 8-9. ADX Validation & Persistence
        latest_adx = df['adx'].iloc[-1]
        score += min((latest_adx / 50.0) * 10.0, 10.0)
        score += (adv['momentum_persistence'] / 100.0) * 10.0

        # 10. Volatility / ATR Expansion Proxy
        atr_sma = df['atr_14'].tail(20).mean()
        if df['atr_14'].iloc[-1] > atr_sma * 1.2:
            score += 5.0

        # 11-12. Penalties (Exhaustion & Divergence)
        if exh['score'] > 40: score -= (exh['score'] * 0.4)
        if div['status'] == "Active Divergence" and np.sign(div['strength']) != target_pol:
            score -= 30.0 # Heavy penalty if diverging against target trade
            
        score = np.clip(score, 0.0, 100.0)
        
        if score >= 85: state = "Ready (Optimal Entry)"
        elif score >= 65: state = "Early (Building)"
        elif score >= 45: state = "Running"
        elif score >= 25: state = "Late (Extended)"
        elif exh['score'] > 60: state = "Exhausted"
        else: state = "Very Early / Dormant"
        
        evidence.append({"type": "SwingReadiness", "weight": 100.0, "value": f"Composite 12-Factor Readiness: {state}", "polarity": target_pol})
        
        # Proper confidence calculation based on readiness component consistency
        conf = adv['momentum_consistency']

        return {"state": state, "score": round(score, 2), "confidence": round(conf, 2), "evidence": evidence}

    # --------------------------------------------------------------------------
    # ADVANCED METRICS, RSI, MACD & MULTI-OSC DIVERGENCE
    # --------------------------------------------------------------------------
    def _analyze_advanced_metrics(self, s: MomentumStrengthResult, a: MomentumAccelerationResult, exh: MomentumExhaustionResult, comp: MomentumCompressionResult, div: DivergenceResult, inst: InstitutionalMomentumResult, mtf: MTFMomentumResult, df: pd.DataFrame) -> AdvancedMomentumMetrics:
        latest = df.iloc[-1]
        
        # Momentum Persistence
        roc_arr = df['roc'].tail(20).to_numpy()
        pos_roc = np.sum(roc_arr > 0)
        persistence = (pos_roc / 20.0 * 100.0) if s['score'] > 0 else ((20 - pos_roc) / 20.0 * 100.0)
        
        vol_sma = df['volume'].tail(20).mean()
        vol_conf = np.clip((latest['volume'] / (vol_sma + EPSILON)) * 50.0, 0.0, 100.0)
        
        # Check strict agreement across oscillators (RSI, MACD, ROC)
        osc_agree = 100.0 if (latest['rsi'] > 50 and latest['macd_line'] > 0 and latest['roc'] > 0) else (100.0 if latest['rsi'] < 50 and latest['macd_line'] < 0 and latest['roc'] < 0 else 0.0)

        # Base momentum continuation probability (to be combined in Layer 3 Engine with Trend/Vol layers)
        rel = (abs(s['score']) * 0.3) + (persistence * 0.3) + (inst['institutional_score'] * 0.4)
        if div['status'] == "Active Divergence": rel *= 0.5

        return {
            "momentum_persistence": round(persistence, 2),
            "momentum_efficiency": round(np.clip(latest['linreg_r2'] * 100.0, 0.0, 100.0), 2),
            "momentum_stability": round(np.clip(100.0 - comp['score'], 0.0, 100.0), 2),
            "momentum_consistency": round(osc_agree, 2),
            "impulse_strength": round(a['score'], 2),
            "pullback_strength": round(100.0 - a['score'], 2), # Simplified inverse logic
            "reversal_probability": round(exh['score'], 2),
            "continuation_probability": round(rel, 2)
        }

    def _analyze_rsi(self, df: pd.DataFrame) -> RSIAnalysisResult:
        rsi = df['rsi'].iloc[-1]
        sc = np.clip((rsi - 50) * 2.0, -100.0, 100.0)
        return {"status": "Active", "score": round(sc, 2), "confidence": 100.0, "regime": "Bullish" if rsi > 50 else "Bearish", "evidence": []}

    def _analyze_macd(self, df: pd.DataFrame) -> MACDAnalysisResult:
        sc = np.clip(np.sign(df['macd_line'].iloc[-1]) * 50 + np.sign(df['macd_histogram'].iloc[-1]) * 50, -100.0, 100.0)
        return {"status": "Active", "score": round(sc, 2), "confidence": 100.0, "phase": "Expanding" if abs(sc) == 100 else "Transitional", "evidence": []}

    def _analyze_divergence(self, df: pd.DataFrame) -> DivergenceResult:
        """True Multi-Oscillator Divergence Scanning (RSI, MACD, ROC, Momentum)."""
        evidence: List[EvidenceItem] = []
        p = df['close'].to_numpy(dtype=np.float64)
        lb = MOMENTUM_CONFIG["lookbacks"]["divergence"]
        
        rsi_str, rsi_t = _divergence_engine_jit(p, df['rsi'].to_numpy(dtype=np.float64), lb, True)
        macd_str, macd_t = _divergence_engine_jit(p, df['macd_line'].to_numpy(dtype=np.float64), lb, False)
        roc_str, roc_t = _divergence_engine_jit(p, df['roc'].to_numpy(dtype=np.float64), lb, False)
        mom_str, mom_t = _divergence_engine_jit(p, df['momentum'].to_numpy(dtype=np.float64), lb, False)
        
        # Identify dominant divergence type across oscillators
        types = [t for t in [rsi_t, macd_t, roc_t, mom_t] if t != 0]
        dominant_type = max(set(types), key=types.count) if types else 0
        
        max_str = max(rsi_str, macd_str, roc_str, mom_str)
        multi_osc_score = np.clip((len(types) / 4.0) * 100.0, 0.0, 100.0)
        
        div_type_str = "None"
        if dominant_type == -1: div_type_str = "Regular Bearish"
        elif dominant_type == 1: div_type_str = "Regular Bullish"
        elif dominant_type == -2: div_type_str = "Hidden Bearish"
        elif dominant_type == 2: div_type_str = "Hidden Bullish"
        
        if len(types) > 0:
            evidence.append({"type": "Multi_Oscillator", "weight": multi_osc_score, "value": f"{len(types)}/4 Oscillators agree on {div_type_str}", "polarity": np.sign(dominant_type)})
            
        status = "Active Divergence" if len(types) > 0 else "None"
        
        return {
            "status": status, 
            "strength": round(max_str, 2), 
            "confidence": round(multi_osc_score, 2), 
            "divergence_type": div_type_str, 
            "multi_oscillator_score": round(multi_osc_score, 2),
            "evidence": evidence
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "MomentumAnalyzer",
    "MomentumAnalysisResult",
    "MomentumStrengthResult",
    "MomentumAccelerationResult",
    "MomentumExhaustionResult",
    "MomentumShiftResult",
    "MomentumIgnitionResult",
    "MomentumCompressionResult",
    "MomentumCycleResult",
    "InstitutionalMomentumResult",
    "MTFMomentumResult",
    "SwingReadinessResult",
    "RSIAnalysisResult",
    "MACDAnalysisResult",
    "DivergenceResult",
    "AdvancedMomentumMetrics",
    "EvidenceItem"
]
