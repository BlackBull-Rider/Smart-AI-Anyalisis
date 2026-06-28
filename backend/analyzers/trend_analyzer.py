import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS (ZERO MAGIC NUMBERS)
# ==============================================================================

EPSILON = 1e-9

TREND_CONFIG = {
    "lookbacks": {
        "micro": 5,
        "short": 10,
        "medium": 20,
        "long": 50
    },
    "thresholds": {
        "adx_strong": 25.0,
        "efficiency_ratio_good": 0.6,
        "efficiency_ratio_excellent": 0.8,
        "noise_high": 1.5,
        "noise_low": 0.8,
        "climax_volume_mult": 2.5,
        "ema_extension_pct": 0.05,
        "atr_expansion_mult": 1.5
    },
    "weights": {
        "ema_alignment": 20.0,
        "ema_slope": 10.0,
        "structure_bos": 20.0,
        "vwap_relation": 15.0,
        "macd_momentum": 15.0,
        "adx_dmi": 10.0,
        "supertrend": 10.0
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
    polarity: int  # 1 for Bullish, -1 for Bearish, 0 for Neutral

class DirectionResult(TypedDict):
    status: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class StrengthResult(TypedDict):
    status: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]
    components: Dict[str, float]

class QualityResult(TypedDict):
    status: str
    score: float
    confidence: float
    efficiency_ratio: float
    noise_level: str
    evidence: List[EvidenceItem]

class ContinuationResult(TypedDict):
    probability: float
    confidence: float
    category: str
    evidence: List[EvidenceItem]

class ExhaustionResult(TypedDict):
    level: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class RegimeResult(TypedDict):
    regime: str
    phase: str
    confidence: float

class MTFResult(TypedDict):
    alignment: str
    score: float
    timeframes: Dict[str, str]
    context: str

class AdvancedMetrics(TypedDict):
    trend_persistence: float
    trend_efficiency_score: float
    pullback_health: float
    institutional_participation: float
    momentum_quality: float
    smart_money_confirmation: float
    trend_reliability: float

class TrendAnalysisResult(TypedDict):
    direction: DirectionResult
    strength: StrengthResult
    quality: QualityResult
    continuation: ContinuationResult
    exhaustion: ExhaustionResult
    regime: RegimeResult
    multi_timeframe: MTFResult
    advanced_metrics: AdvancedMetrics

# ==============================================================================
# TREND ANALYZER ENGINE
# ==============================================================================

class TrendAnalyzer:
    """
    Institutional Trend Analyzer computing directional arrays, regime models, 
    and multi-factor confidences based purely on Layer 1 data.
    """

    def __init__(self):
        # Strictly Required Columns
        self.required_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'ema_20', 'ema_50', 'vwap', 'supertrend', 
            'adx', 'macd_line', 'macd_signal', 'rsi', 'atr_14', 
            'linreg_slope', 'linreg_r2'
        ]
        # Optional SMC / MTF Columns
        self.optional_cols = [
            'ema_200', 'bos', 'choch', 'liq_sweep', 'fvg_active', 'ob_active'
        ]
        self.mtf_suffixes = ['_5m', '_15m', '_1H', '_4H', '_D', '_W', '_M']

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates columns, handles NaNs, Infs, and ensures minimum length."""
        missing = [col for col in self.required_cols if col not in df.columns]
        if missing:
            logger.error(f"TrendAnalyzer missing critical columns: {missing}")
            raise ValueError(f"TrendAnalyzer requires missing columns: {missing}")
            
        # Minimum data requirement check
        min_len = TREND_CONFIG["lookbacks"]["micro"] + 1
        if len(df) < min_len:
            raise ValueError(f"TrendAnalyzer requires at least {min_len} bars, got {len(df)}.")

        # Create a safe copy
        working_df = df.copy()

        # Handle Inf values safely
        num_cols = working_df.select_dtypes(include=[np.number]).columns
        if np.isinf(working_df[num_cols]).any().any():
            working_df[num_cols] = working_df[num_cols].replace([np.inf, -np.inf], np.nan)

        # Forward fill and backward fill NaNs for robustness
        if working_df[self.required_cols].isna().any().any():
            working_df[self.required_cols] = working_df[self.required_cols].ffill().bfill()
            
        # Final Hard Stop if fully NaN columns exist
        if working_df[self.required_cols].isna().any().any():
            raise ValueError("Data contains all-NaN columns which cannot be mathematically resolved.")
            
        return working_df

    def analyze(self, df: pd.DataFrame) -> TrendAnalysisResult:
        safe_df = self._validate_data(df)

        # Contextual Window
        context_len = TREND_CONFIG["lookbacks"]["long"]
        working_df = safe_df.tail(context_len)

        # Sequential Analysis Core
        direction = self._analyze_direction(working_df)
        strength = self._analyze_strength(working_df)
        quality = self._analyze_quality(working_df)
        exhaustion = self._analyze_exhaustion(working_df)
        advanced = self._analyze_advanced_metrics(direction, strength, quality, working_df)
        regime = self._analyze_regime_and_phase(direction, strength, advanced, working_df)
        continuation = self._analyze_continuation(direction, strength, quality, exhaustion, advanced, working_df)
        mtf = self._analyze_mtf(working_df)

        return {
            "direction": direction,
            "strength": strength,
            "quality": quality,
            "continuation": continuation,
            "exhaustion": exhaustion,
            "regime": regime,
            "multi_timeframe": mtf,
            "advanced_metrics": advanced
        }

    # --------------------------------------------------------------------------
    # 1. TREND DIRECTION ANALYSIS
    # --------------------------------------------------------------------------
    def _analyze_direction(self, df: pd.DataFrame) -> DirectionResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        max_score = sum(TREND_CONFIG["weights"].values())
        
        recent_5 = df.tail(TREND_CONFIG["lookbacks"]["micro"])
        latest = df.iloc[-1]
        
        # 1. EMA Alignment
        w_ema = TREND_CONFIG["weights"]["ema_alignment"]
        has_200 = 'ema_200' in df.columns and pd.notna(latest['ema_200'])
        if latest['ema_20'] > latest['ema_50'] and (not has_200 or latest['ema_50'] > latest['ema_200']):
            score += w_ema
            evidence.append({"type": "EMA_Align", "weight": w_ema, "value": "Bullish Alignment", "polarity": 1})
        elif latest['ema_20'] < latest['ema_50'] and (not has_200 or latest['ema_50'] < latest['ema_200']):
            score -= w_ema
            evidence.append({"type": "EMA_Align", "weight": w_ema, "value": "Bearish Alignment", "polarity": -1})
        else:
            evidence.append({"type": "EMA_Align", "weight": w_ema, "value": "Mixed Alignment", "polarity": 0})

        # 2. EMA Slope
        w_slope = TREND_CONFIG["weights"]["ema_slope"]
        eval_len = min(len(df), TREND_CONFIG["lookbacks"]["micro"])
        if eval_len > 1:
            ema20_slope = df['ema_20'].tail(eval_len).diff().mean()
            if ema20_slope > 0:
                score += w_slope
                evidence.append({"type": "EMA_Slope", "weight": w_slope, "value": "Positive Short-term Slope", "polarity": 1})
            elif ema20_slope < 0:
                score -= w_slope
                evidence.append({"type": "EMA_Slope", "weight": w_slope, "value": "Negative Short-term Slope", "polarity": -1})

        # 3. SMC Structure
        w_struct = TREND_CONFIG["weights"]["structure_bos"]
        struct_len = min(len(df), TREND_CONFIG["lookbacks"]["medium"])
        
        if 'bos' in df.columns and 'choch' in df.columns and struct_len > 1:
            recent_bos = df['bos'].tail(struct_len).sum()
            recent_choch = df['choch'].tail(struct_len).sum()
            struct_val = recent_bos + (recent_choch * 1.5)
            
            if struct_val >= 1:
                score += w_struct
                evidence.append({"type": "Structure", "weight": w_struct, "value": "Bullish Structure (HH/HL)", "polarity": 1})
            elif struct_val <= -1:
                score -= w_struct
                evidence.append({"type": "Structure", "weight": w_struct, "value": "Bearish Structure (LH/LL)", "polarity": -1})
            else:
                evidence.append({"type": "Structure", "weight": w_struct, "value": "Structure Consolidated", "polarity": 0})
        else:
            if eval_len > 1:
                price_slope = df['close'].tail(eval_len).diff().mean()
                if price_slope > 0:
                    score += w_struct
                    evidence.append({"type": "Price_Action", "weight": w_struct, "value": "Bullish Momentum", "polarity": 1})
                else:
                    score -= w_struct
                    evidence.append({"type": "Price_Action", "weight": w_struct, "value": "Bearish Momentum", "polarity": -1})

        # 4. VWAP Anchor
        w_vwap = TREND_CONFIG["weights"]["vwap_relation"]
        if latest['close'] > latest['vwap']:
            score += w_vwap
            evidence.append({"type": "VWAP", "weight": w_vwap, "value": "Price above VWAP", "polarity": 1})
        else:
            score -= w_vwap
            evidence.append({"type": "VWAP", "weight": w_vwap, "value": "Price below VWAP", "polarity": -1})

        # 5. MACD Momentum
        w_macd = TREND_CONFIG["weights"]["macd_momentum"]
        if eval_len > 1:
            macd_hist = df['macd_line'].tail(eval_len) - df['macd_signal'].tail(eval_len)
            macd_slope = macd_hist.diff().mean()
            if latest['macd_line'] > latest['macd_signal'] and macd_slope > 0:
                score += w_macd
                evidence.append({"type": "MACD", "weight": w_macd, "value": "Bullish Expanding Momentum", "polarity": 1})
            elif latest['macd_line'] < latest['macd_signal'] and macd_slope < 0:
                score -= w_macd
                evidence.append({"type": "MACD", "weight": w_macd, "value": "Bearish Expanding Momentum", "polarity": -1})
            else:
                evidence.append({"type": "MACD", "weight": w_macd, "value": "Decaying Momentum", "polarity": 0})
            
        # 6. SuperTrend
        w_st = TREND_CONFIG["weights"]["supertrend"]
        if latest['supertrend'] > 0:
            score += w_st
            evidence.append({"type": "SuperTrend", "weight": w_st, "value": "Bullish Trailing Support", "polarity": 1})
        else:
            score -= w_st
            evidence.append({"type": "SuperTrend", "weight": w_st, "value": "Bearish Trailing Resistance", "polarity": -1})

        # Score Normalization
        norm_score = max(min((score / max_score) * 100.0, 100.0), -100.0) if max_score > 0 else 0.0
        
        if norm_score >= 60: status = "Strong Bullish"
        elif norm_score >= 20: status = "Bullish"
        elif norm_score > -20: status = "Sideways"
        elif norm_score > -60: status = "Bearish"
        else: status = "Strong Bearish"
        
        # Exact Mathematical Weighted Confidence (Including Neutral Partial Credit)
        target_polarity = 1 if norm_score > 0 else -1 if norm_score < 0 else 0
        total_weight = sum(e['weight'] for e in evidence)
        aligned_weight = sum(e['weight'] for e in evidence if e['polarity'] == target_polarity)
        neutral_weight = sum(e['weight'] for e in evidence if e['polarity'] == 0)
        
        # Neutral items offer minor confidence in choppy markets, but don't penalize as heavily as conflicts
        eff_aligned = aligned_weight + (neutral_weight * 0.5) if target_polarity == 0 else aligned_weight
        confidence = (eff_aligned / total_weight) * 100.0 if total_weight > 0 else 0.0

        return {
            "status": status,
            "score": round(norm_score, 2),
            "confidence": round(confidence, 2),
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 2. TREND STRENGTH ANALYSIS
    # --------------------------------------------------------------------------
    def _analyze_strength(self, df: pd.DataFrame) -> StrengthResult:
        evidence: List[EvidenceItem] = []
        components = {}
        latest = df.iloc[-1]
        eval_len = min(len(df), 10)
        score = 0.0
        
        # 1. ADX Persistence
        adx = latest['adx']
        components['adx'] = adx
        
        if eval_len > 1:
            adx_slope = df['adx'].tail(eval_len).diff().mean()
            components['adx_slope'] = float(np.nan_to_num(adx_slope))
        else:
            adx_slope = 0.0

        if adx >= TREND_CONFIG["thresholds"]["adx_strong"]:
            base_str = min(adx * 1.5, 60.0)
            if adx_slope > 0: base_str += 10.0
            score += base_str
            evidence.append({"type": "ADX", "weight": 70.0, "value": f"Strong directional ADX ({adx:.1f})", "polarity": 1})
        else:
            score += min(adx, 30.0)
            evidence.append({"type": "ADX", "weight": 70.0, "value": f"Weak ADX ({adx:.1f})", "polarity": -1})

        # 2. EMA Spread Expansion
        if eval_len > 1:
            ema_diff = (df['ema_20'].tail(eval_len) - df['ema_50'].tail(eval_len)).abs()
            ema_slope = ema_diff.diff().mean()
            components['ema_spread_slope'] = float(np.nan_to_num(ema_slope))
            if ema_slope > 0:
                score += 15.0
                evidence.append({"type": "EMA_Spread", "weight": 15.0, "value": "Moving Averages Expanding", "polarity": 1})
            else:
                evidence.append({"type": "EMA_Spread", "weight": 15.0, "value": "Moving Averages Contracting", "polarity": -1})

        # 3. Regression Fit (R2)
        r2 = np.clip(latest['linreg_r2'], 0.0, 1.0)
        components['r2_fit'] = r2
        if r2 > 0.6:
            score += 15.0
            evidence.append({"type": "Regression", "weight": 15.0, "value": f"High linear fit (R2 {r2:.2f})", "polarity": 1})
        else:
            evidence.append({"type": "Regression", "weight": 15.0, "value": "Messy linear fit", "polarity": -1})

        score = max(min(score, 100.0), 0.0)
        
        if score >= 80: status = "Very Strong"
        elif score >= 60: status = "Strong"
        elif score >= 40: status = "Moderate"
        elif score >= 20: status = "Weak"
        else: status = "Very Weak"

        total_w = sum(e['weight'] for e in evidence)
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == 1)
        confidence = (aligned_w / total_w) * 100.0 if total_w > 0 else 0.0

        return {
            "status": status,
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "components": components
        }

    # --------------------------------------------------------------------------
    # 3. TREND QUALITY ANALYSIS
    # --------------------------------------------------------------------------
    def _analyze_quality(self, df: pd.DataFrame) -> QualityResult:
        evidence: List[EvidenceItem] = []
        period = min(len(df) - 1, TREND_CONFIG["lookbacks"]["medium"])
        
        if period < 1:
            return {"status": "Unknown", "score": 0.0, "confidence": 0.0, "efficiency_ratio": 0.0, "noise_level": "Unknown", "evidence": []}

        # 1. Kaufman Efficiency Ratio (ER)
        net_change = abs(df['close'].iloc[-1] - df['close'].iloc[-period - 1])
        sum_abs_change = df['close'].diff().abs().tail(period).sum()
        er = net_change / (sum_abs_change + EPSILON)
        
        if er >= TREND_CONFIG["thresholds"]["efficiency_ratio_excellent"]:
            evidence.append({"type": "Efficiency", "weight": 40.0, "value": "Excellent efficiency", "polarity": 1})
        elif er >= TREND_CONFIG["thresholds"]["efficiency_ratio_good"]:
            evidence.append({"type": "Efficiency", "weight": 40.0, "value": "Good efficiency", "polarity": 1})
        else:
            evidence.append({"type": "Efficiency", "weight": 40.0, "value": "Choppy action", "polarity": -1})

        # 2. Wick Noise Ratio
        eval_len = min(len(df), 10)
        recent_bars = df.tail(eval_len)
        bodies = (recent_bars['close'] - recent_bars['open']).abs()
        wicks = (recent_bars['high'] - recent_bars['low']) - bodies
        noise_ratio = wicks.mean() / (bodies.mean() + EPSILON)
        
        if noise_ratio > TREND_CONFIG["thresholds"]["noise_high"]:
            noise_level = "High"
            evidence.append({"type": "Noise", "weight": 30.0, "value": "High wick interference", "polarity": -1})
        elif noise_ratio < TREND_CONFIG["thresholds"]["noise_low"]:
            noise_level = "Low"
            evidence.append({"type": "Noise", "weight": 30.0, "value": "Clean bodies", "polarity": 1})
        else:
            noise_level = "Moderate"
            evidence.append({"type": "Noise", "weight": 30.0, "value": "Average noise", "polarity": 0})

        # 3. Trend Smoothness (R2 Consistency)
        r2_mean = np.clip(recent_bars['linreg_r2'].mean(), 0.0, 1.0)
        if r2_mean > 0.7:
            evidence.append({"type": "Smoothness", "weight": 30.0, "value": "Consistently smooth", "polarity": 1})
        else:
            evidence.append({"type": "Smoothness", "weight": 30.0, "value": "Erratic path", "polarity": -1})

        # Score Aggregation
        q_score = (er * 50.0) + (max(0.0, (2.0 - noise_ratio) * 10.0)) + (r2_mean * 30.0)
        q_score = max(min(q_score, 100.0), 0.0)
        
        if q_score >= 75: status = "Excellent"
        elif q_score >= 50: status = "Good"
        elif q_score >= 30: status = "Average"
        else: status = "Poor"

        total_w = sum(e['weight'] for e in evidence)
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == 1)
        conf = (aligned_w / total_w) * 100.0 if total_w > 0 else 0.0
        
        return {
            "status": status,
            "score": round(q_score, 2),
            "confidence": round(conf, 2),
            "efficiency_ratio": round(er, 2),
            "noise_level": noise_level,
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 4. ADVANCED INSTITUTIONAL METRICS
    # --------------------------------------------------------------------------
    def _analyze_advanced_metrics(self, dir_res: DirectionResult, str_res: StrengthResult, qual_res: QualityResult, df: pd.DataFrame) -> AdvancedMetrics:
        trend_sign = 1 if dir_res['score'] > 0 else -1
        eval_len = min(len(df), 20)
        recent_bars = df.tail(eval_len)
        latest = df.iloc[-1]
        
        # 1. Trend Persistence
        if eval_len > 1:
            ema_slopes = recent_bars['ema_20'].diff().dropna()
            if len(ema_slopes) > 0:
                persistence = (ema_slopes > 0).mean() * 100.0 if trend_sign == 1 else (ema_slopes < 0).mean() * 100.0
            else:
                persistence = 50.0
        else:
            persistence = 50.0

        # 2. Pullback Health (ATR Adjusted)
        if eval_len > 1:
            atr = latest['atr_14']
            if trend_sign == 1:
                hh = recent_bars['high'].max()
                drawdown = hh - latest['close']
            else:
                ll = recent_bars['low'].min()
                drawdown = latest['close'] - ll
                
            pb_ratio = drawdown / (atr + EPSILON)
            pullback_health = np.clip(100.0 - (pb_ratio * 20.0), 0.0, 100.0)
        else:
            pullback_health = 50.0

        # 3. Institutional Participation (Strict Validation)
        inst_part = 0.0
        struct_aligned = False
        if 'ob_active' in df.columns and latest['ob_active']: 
            inst_part += 30.0
            struct_aligned = True
        if 'fvg_active' in df.columns and latest['fvg_active']: 
            inst_part += 20.0
            struct_aligned = True
        if 'liq_sweep' in df.columns and latest['liq_sweep'] != 0: 
            inst_part += 20.0
            struct_aligned = True
            
        # Volume expansion valid ONLY if structurally aligned
        if eval_len > 1:
            vol_sma = recent_bars['volume'].mean()
            if latest['volume'] > vol_sma * 1.5 and struct_aligned:
                inst_part += 30.0
                
        inst_part = np.clip(inst_part, 0.0, 100.0)
        
        # 4. Momentum Quality
        mq_score = np.clip((str_res['score'] / 100.0) * qual_res['efficiency_ratio'] * 100.0, 0.0, 100.0)

        # 5. Smart Money Confirmation (Directional alignment checks)
        bos_aligned = 100.0 if ('bos' in df.columns and latest['bos'] == trend_sign) else 0.0
        smc_score = np.clip((inst_part * 0.7) + (bos_aligned * 0.3), 0.0, 100.0)

        # 6. Trend Reliability
        te_score = qual_res['score']
        reliability = np.clip((persistence * 0.4) + (pullback_health * 0.3) + (mq_score * 0.3), 0.0, 100.0)

        return {
            "trend_persistence": round(persistence, 2),
            "trend_efficiency_score": round(te_score, 2),
            "pullback_health": round(pullback_health, 2),
            "institutional_participation": round(inst_part, 2),
            "momentum_quality": round(mq_score, 2),
            "smart_money_confirmation": round(smc_score, 2),
            "trend_reliability": round(reliability, 2)
        }

    # --------------------------------------------------------------------------
    # 5. EXHAUSTION ANALYSIS
    # --------------------------------------------------------------------------
    def _analyze_exhaustion(self, df: pd.DataFrame) -> ExhaustionResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        latest = df.iloc[-1]
        
        if len(df) < 20:
            return {"level": "Unknown", "score": 0.0, "confidence": 0.0, "evidence": []}

        # 1. RSI / MACD Divergence
        recent_rsi = df['rsi'].tail(5).max()
        past_rsi = df['rsi'].iloc[-20:-5].max()
        recent_macd = df['macd_line'].tail(5).max()
        past_macd = df['macd_line'].iloc[-20:-5].max()
        price_high = df['high'].tail(5).max()
        past_price_high = df['high'].iloc[-20:-5].max()
        
        if price_high > past_price_high:
            if recent_rsi < past_rsi and recent_rsi > 60:
                score += 20.0
                evidence.append({"type": "Divergence", "weight": 20.0, "value": "Bearish RSI Divergence", "polarity": 1})
            if recent_macd < past_macd and recent_macd > 0:
                score += 20.0
                evidence.append({"type": "Divergence", "weight": 20.0, "value": "Bearish MACD Divergence", "polarity": 1})

        # 2. ADX Rolling Decline
        adx_slice = df['adx'].tail(5)
        if len(adx_slice) >= 3 and adx_slice.iloc[-1] < adx_slice.iloc[-3] and adx_slice.iloc[-1] > 30:
            score += 20.0
            evidence.append({"type": "ADX", "weight": 20.0, "value": "ADX Decline (Momentum Loss)", "polarity": 1})

        # 3. Volume Exhaustion / Climax
        vol_sma = df['volume'].tail(20).mean()
        if latest['volume'] > (vol_sma * TREND_CONFIG["thresholds"]["climax_volume_mult"]):
            body_pct = abs(latest['close'] - latest['open']) / (latest['high'] - latest['low'] + EPSILON)
            if body_pct < 0.4:
                score += 20.0
                evidence.append({"type": "Volume", "weight": 20.0, "value": "Climax Volume w/ Rejection", "polarity": 1})

        # 4. Over-extension from Mean
        ext_pct = TREND_CONFIG["thresholds"]["ema_extension_pct"]
        if abs(latest['close'] - latest['ema_20']) / (latest['ema_20'] + EPSILON) > ext_pct:
            score += 20.0
            evidence.append({"type": "Extension", "weight": 20.0, "value": "Hyper-extended from EMA20", "polarity": 1})

        score = np.clip(score, 0.0, 100.0)
        if score >= 75: level = "Critical"
        elif score >= 50: level = "High"
        elif score >= 25: level = "Moderate"
        else: level = "Low"

        # If we had any actual evidence logic trigger, scale confidence accordingly
        confidence = min((len(evidence) / 5.0) * 100.0, 100.0)

        return {
            "level": level,
            "score": round(score, 2),
            "confidence": round(confidence, 2),
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 6. REGIME & PHASE DETECTION (WYCKOFF/SMC)
    # --------------------------------------------------------------------------
    def _analyze_regime_and_phase(self, dir_res: DirectionResult, str_res: StrengthResult, adv_res: AdvancedMetrics, df: pd.DataFrame) -> RegimeResult:
        latest = df.iloc[-1]
        eval_len = min(len(df), 50)
        
        atr = latest['atr_14']
        atr_sma = df['atr_14'].tail(eval_len).mean()
        is_volatile = atr > (atr_sma * TREND_CONFIG["thresholds"]["atr_expansion_mult"])
        
        adx = latest['adx']
        str_score = str_res['score']
        er = adv_res['trend_efficiency_score']
        
        # Regime Detection logic
        if adx > 25.0 and str_score > 40.0:
            if is_volatile: regime = "Volatile Trending"
            elif er > 60: regime = "Clean Trending"
            else: regime = "Trending"
        elif adx < 20.0 and not is_volatile:
            regime = "Compression / Sideways"
        else:
            regime = "Choppy / Transitional"
            
        # Wyckoff / SMC Phase
        phase = "Unknown"
        is_bull = dir_res['score'] > 0
        recent_len = min(len(df), 10)
        
        recent_choch = df['choch'].tail(recent_len).sum() if 'choch' in df.columns else 0
        recent_bos = df['bos'].tail(recent_len).sum() if 'bos' in df.columns else 0
        
        if regime == "Compression / Sideways":
            if 'ema_200' in df.columns and pd.notna(latest['ema_200']):
                phase = "Accumulation Zone" if latest['close'] > latest['ema_200'] else "Distribution Zone"
            else:
                phase = "Consolidation"
        else:
            if is_bull:
                if recent_choch < 0: phase = "Distribution Risk (CHOCH Down)"
                elif recent_bos > 0: phase = "Mature Markup"
                else: phase = "Early Markup"
            else:
                if recent_choch > 0: phase = "Accumulation Risk (CHOCH Up)"
                elif recent_bos < 0: phase = "Mature Markdown"
                else: phase = "Early Markdown"

        return {
            "regime": regime,
            "phase": phase,
            "confidence": round(max(str_score, 50.0), 2)
        }

    # --------------------------------------------------------------------------
    # 7. CONTINUATION ANALYSIS
    # --------------------------------------------------------------------------
    def _analyze_continuation(self, dir_res: DirectionResult, str_res: StrengthResult, qual_res: QualityResult, exh_res: ExhaustionResult, adv_res: AdvancedMetrics, df: pd.DataFrame) -> ContinuationResult:
        evidence: List[EvidenceItem] = []
        prob = 50.0
        macro_sign = 1 if dir_res['score'] > 0 else -1
        
        # Momentum check
        dir_impact = (abs(dir_res['score']) / 100.0) * 15.0
        str_impact = (str_res['score'] / 100.0) * 15.0
        prob += (dir_impact + str_impact)
        evidence.append({"type": "Momentum", "weight": 30.0, "value": "Direction and Strength align", "polarity": macro_sign})
        
        # Quality & Pullback Health
        if qual_res['score'] > 60 and adv_res['pullback_health'] > 60:
            prob += 20.0
            evidence.append({"type": "Health", "weight": 20.0, "value": "Healthy structure/pullbacks", "polarity": macro_sign})
        else:
            prob -= 10.0
            evidence.append({"type": "Health", "weight": 20.0, "value": "Messy structure", "polarity": -macro_sign})
            
        # Exhaustion Penalty
        exh_penalty = (exh_res['score'] / 100.0) * 30.0
        prob -= exh_penalty
        if exh_penalty > 15:
            evidence.append({"type": "Exhaustion", "weight": 30.0, "value": "Exhaustion markers active", "polarity": -macro_sign})
            
        prob = np.clip(prob, 5.0, 95.0)
        
        if prob >= 70: category = "High"
        elif prob >= 45: category = "Moderate"
        else: category = "Low"

        # Reliability driven confidence
        conf_score = adv_res['trend_reliability']

        return {
            "probability": round(prob, 2),
            "confidence": round(conf_score, 2),
            "category": category,
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 8. MULTI-TIMEFRAME (MTF) CONTEXT (WEIGHTED)
    # --------------------------------------------------------------------------
    def _analyze_mtf(self, df: pd.DataFrame) -> MTFResult:
        latest = df.iloc[-1]
        timeframes = {}
        
        local_bull = latest['close'] > latest['ema_50']
        timeframes['Local'] = "Bullish" if local_bull else "Bearish"
        
        score = 0.0
        active_weight = 0.0
        
        # Local weight setup (10% implied if no HTF, adjusted dynamically)
        local_weight = 0.10
        active_weight += local_weight
        if local_bull: score += local_weight
        
        weights = TREND_CONFIG["mtf_weights"]
        
        for suffix in self.mtf_suffixes:
            c_col, e_col = f'close{suffix}', f'ema_50{suffix}'
            if c_col in df.columns and e_col in df.columns and pd.notna(latest[c_col]) and pd.notna(latest[e_col]):
                w = weights.get(suffix, 0.0)
                active_weight += w
                if latest[c_col] > latest[e_col]:
                    timeframes[suffix.strip('_')] = "Bullish"
                    score += w
                else:
                    timeframes[suffix.strip('_')] = "Bearish"

        # Normalize Alignment Score based on available weighted data
        norm_score = (score / active_weight) * 100.0 if active_weight > 0 else (100.0 if local_bull else 0.0)
        norm_score = np.clip(norm_score, 0.0, 100.0)
        
        if active_weight == local_weight:
            alignment = "Local Only"
            context = "MTF Data not provided from Layer 1."
        elif norm_score >= 80:
            alignment = "Full Bullish Alignment"
            context = "Higher timeframes heavily confirm upward trend."
        elif norm_score <= 20:
            alignment = "Full Bearish Alignment"
            context = "Higher timeframes heavily confirm downward trend."
        elif (local_bull and norm_score < 50) or (not local_bull and norm_score >= 50):
            alignment = "LTF / HTF Conflict"
            context = "Local timeframe diverges from macro trend."
        else:
            alignment = "Mixed / Neutral"
            context = "Timeframes lack unified directional consensus."

        return {
            "alignment": alignment,
            "score": round(norm_score, 2),
            "timeframes": timeframes,
            "context": context
        }

# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "TrendAnalyzer",
    "TrendAnalysisResult",
    "DirectionResult",
    "StrengthResult",
    "QualityResult",
    "ContinuationResult",
    "ExhaustionResult",
    "RegimeResult",
    "MTFResult",
    "AdvancedMetrics",
    "EvidenceItem"
]
