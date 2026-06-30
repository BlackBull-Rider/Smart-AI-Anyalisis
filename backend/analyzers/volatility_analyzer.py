import logging
import numpy as np
import pandas as pd
from typing import Dict, List, TypedDict, Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

EPSILON = 1e-9

VOLATILITY_CONFIG = {
    "lookbacks": {
        "micro": 5,
        "short": 10,
        "medium": 20,
        "long": 50,
        "macro": 252  # 252 for institutional rolling percentiles / z-scores
    },
    "thresholds": {
        "extreme_high_percentile": 90.0,
        "high_percentile": 75.0,
        "low_percentile": 25.0,
        "extreme_low_percentile": 10.0,
        "contraction_ratio": 0.5,
        "expansion_ratio": 1.5,
        "atr_spike_mult": 2.0
    }
}

# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class EvidenceItem(TypedDict):
    type: str
    weight: float
    value: str
    polarity: int  # 1 for Expansion/High, -1 for Contraction/Low, 0 for Neutral

class VolatilityStateResult(TypedDict):
    regime: str
    score: float
    confidence: float
    evidence: List[EvidenceItem]

class ContractionExpansionResult(TypedDict):
    phase: str
    intensity: float
    confidence: float
    evidence: List[EvidenceItem]

class SqueezeResult(TypedDict):
    is_squeezing: bool
    squeeze_score: float
    duration: int
    confidence: float
    evidence: List[EvidenceItem]

class BreakoutProbabilityResult(TypedDict):
    probability: float
    confidence: float
    evidence: List[EvidenceItem]

class AdvancedVolatilityMetrics(TypedDict):
    efficiency_ratio: float
    atr_percentile: float
    regime_persistence: int
    atr_z_score: float
    realized_vol_ratio: float
    gap_frequency: int
    volatility_clustering: float

class VolatilityAnalysisResult(TypedDict):
    state: VolatilityStateResult
    cycle: ContractionExpansionResult
    squeeze: SqueezeResult
    breakout: BreakoutProbabilityResult
    advanced_metrics: AdvancedVolatilityMetrics

# ==============================================================================
# VOLATILITY ANALYZER ENGINE
# ==============================================================================

class VolatilityAnalyzer:
    """
    Institutional Volatility Analyzer.
    Evaluates volatility regimes, contraction/expansion cycles, squeeze setups,
    and advanced institutional metrics (Gaps, Clustering, Z-Scores).
    Strictly Layer-2 compliant (Zero indicator calculation).
    """

    def __init__(self):
        self.req_cols = [
            'open', 'high', 'low', 'close', 'volume', 'atr_14'
        ]
        self.opt_cols = [
            'bbw_20_2.0', 'hv_21', 'sqz_20', 'ei_14', 'chop_14'
        ]
        self._feature_cache: Dict[str, Optional[str]] = {}

    def _get_cached_col(self, df: pd.DataFrame, key: str) -> Optional[str]:
        """O(1) cached lookup for dynamic indicator columns."""
        if key not in self._feature_cache:
            self._feature_cache[key] = next((c for c in df.columns if key in c.lower()), None)
        return self._feature_cache[key]

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strict validation. Raises ValueError on missing critical data."""
        missing = [col for col in self.req_cols if col not in df.columns]
        if missing:
            logger.error(f"VolatilityAnalyzer missing critical columns: {missing}")
            raise ValueError(f"VolatilityAnalyzer requires missing columns: {missing}")
            
        min_len = VOLATILITY_CONFIG["lookbacks"]["long"]
        if len(df) < min_len:
            raise ValueError(f"VolatilityAnalyzer requires at least {min_len} bars, got {len(df)}.")

        working_df = df.copy()

        num_cols = working_df.select_dtypes(include=[np.number]).columns
        if np.isinf(working_df[num_cols]).any().any():
            working_df[num_cols] = working_df[num_cols].replace([np.inf, -np.inf], np.nan)

        if working_df[self.req_cols].isna().any().any():
            working_df[self.req_cols] = working_df[self.req_cols].ffill().bfill()
            
        if working_df[self.req_cols].isna().any().any():
            raise ValueError("Critical volatility columns contain unresolvable NaNs.")
            
        return working_df

    def analyze(self, df: pd.DataFrame) -> VolatilityAnalysisResult:
        safe_df = self._validate_data(df)
        
        # Optimize memory by slicing only the maximum required lookback window
        eval_len = VOLATILITY_CONFIG["lookbacks"]["macro"]
        working_df = safe_df.tail(min(len(safe_df), eval_len))

        adv_metrics = self._analyze_advanced_metrics(working_df)
        state = self._analyze_state(working_df, adv_metrics)
        cycle = self._analyze_cycle(working_df)
        squeeze = self._analyze_squeeze(working_df, adv_metrics)
        breakout = self._analyze_breakout(state, cycle, squeeze, adv_metrics)

        return {
            "state": state,
            "cycle": cycle,
            "squeeze": squeeze,
            "breakout": breakout,
            "advanced_metrics": adv_metrics
        }

    # --------------------------------------------------------------------------
    # 1. ADVANCED VOLATILITY METRICS (Processed First for Dependencies)
    # --------------------------------------------------------------------------
    def _analyze_advanced_metrics(self, df: pd.DataFrame) -> AdvancedVolatilityMetrics:
        latest = df.iloc[-1]
        macro_len = VOLATILITY_CONFIG["lookbacks"]["macro"]
        short_len = VOLATILITY_CONFIG["lookbacks"]["short"]
        
        recent_macro = df.tail(macro_len)
        recent_short = df.tail(short_len)
        
        # 1. Rolling ATR Percentile (252 bars for institutional stability)
        atr_rank = (recent_macro['atr_14'].rank(pct=True).iloc[-1]) * 100.0
        
        # 2. Kaufman Efficiency Ratio (ER)
        net_change = abs(latest['close'] - recent_short['close'].iloc[0])
        sum_abs_change = recent_short['close'].diff().abs().sum()
        efficiency_ratio = net_change / (sum_abs_change + EPSILON)
        
        # 3. ATR Z-Score
        atr_mean = recent_macro['atr_14'].mean()
        atr_std = recent_macro['atr_14'].std() + EPSILON
        atr_z = (latest['atr_14'] - atr_mean) / atr_std
        
        # 4. Volatility Regime Persistence (Days in current low/high state)
        if atr_rank < 25.0:
            threshold = recent_macro['atr_14'].quantile(0.25)
            persistence = (recent_macro['atr_14'] < threshold).iloc[::-1].cummin().sum()
        elif atr_rank > 75.0:
            threshold = recent_macro['atr_14'].quantile(0.75)
            persistence = (recent_macro['atr_14'] > threshold).iloc[::-1].cummin().sum()
        else:
            persistence = 0

        # 5. Realized Volatility Ratio (ATR Proxy vs Historical Volatility)
        rvr = 1.0
        hv_col = self._get_cached_col(df, 'hv_')
        if hv_col and pd.notna(latest[hv_col]) and latest[hv_col] > 0:
            annualized_atr_pct = (latest['atr_14'] / latest['close']) * np.sqrt(252) * 100.0
            rvr = annualized_atr_pct / (latest[hv_col] + EPSILON)
            
        # 6. Gap Volatility (Frequency of ATR-significant gaps)
        gaps = (df['open'] - df['close'].shift(1)).abs()
        gap_threshold = df['atr_14'].shift(1) * 0.5
        gap_freq = (gaps > gap_threshold).tail(20).sum()
        
        # 7. Volatility Clustering (Autocorrelation of ATR returns)
        atr_returns = recent_macro['atr_14'].pct_change().dropna()
        clustering_score = atr_returns.autocorr(lag=1) if len(atr_returns) > 2 else 0.0

        return {
            "efficiency_ratio": round(efficiency_ratio, 3),
            "atr_percentile": round(atr_rank, 2),
            "regime_persistence": int(persistence),
            "atr_z_score": round(atr_z, 2),
            "realized_vol_ratio": round(rvr, 2),
            "gap_frequency": int(gap_freq),
            "volatility_clustering": round(clustering_score, 2)
        }

    # --------------------------------------------------------------------------
    # 2. VOLATILITY STATE / REGIME
    # --------------------------------------------------------------------------
    def _analyze_state(self, df: pd.DataFrame, adv: AdvancedVolatilityMetrics) -> VolatilityStateResult:
        evidence: List[EvidenceItem] = []
        score = 0.0
        max_confidence = 40.0 # Base conf for ATR
        confidence_points = 40.0 
        
        latest = df.iloc[-1]
        
        # ATR Rank Evaluation (Stable 252-bar rolling)
        atr_rank = adv['atr_percentile']
        if atr_rank > VOLATILITY_CONFIG["thresholds"]["extreme_high_percentile"]:
            score += 40.0
            evidence.append({"type": "ATR_Rank", "weight": 40.0, "value": f"Extreme Volatility (Rank: {atr_rank:.1f}%)", "polarity": 1})
        elif atr_rank < VOLATILITY_CONFIG["thresholds"]["extreme_low_percentile"]:
            score -= 40.0
            evidence.append({"type": "ATR_Rank", "weight": 40.0, "value": f"Volatility Collapse (Rank: {atr_rank:.1f}%)", "polarity": -1})
        else:
            evidence.append({"type": "ATR_Rank", "weight": 40.0, "value": f"Normal Range (Rank: {atr_rank:.1f}%)", "polarity": 0})

        # Historical Volatility Integration
        max_confidence += 30.0
        hv_col = self._get_cached_col(df, 'hv_')
        if hv_col and pd.notna(latest[hv_col]):
            confidence_points += 30.0
            hv_rank = (df[hv_col].tail(252).rank(pct=True).iloc[-1]) * 100.0
            if hv_rank > 75.0:
                score += 30.0
                evidence.append({"type": "HV_Rank", "weight": 30.0, "value": "High Annualized Volatility", "polarity": 1})
            elif hv_rank < 25.0:
                score -= 30.0
                evidence.append({"type": "HV_Rank", "weight": 30.0, "value": "Low Annualized Volatility", "polarity": -1})

        # Efficiency Ratio Analysis (Replaced Noise Ratio)
        max_confidence += 30.0
        confidence_points += 30.0
        er = adv['efficiency_ratio']
        if er > 0.7:
            score -= 30.0
            evidence.append({"type": "Efficiency", "weight": 30.0, "value": f"Highly Efficient Move (ER: {er:.2f})", "polarity": -1})
        elif er < 0.3:
            score += 30.0
            evidence.append({"type": "Efficiency", "weight": 30.0, "value": f"Choppy/Inefficient (ER: {er:.2f})", "polarity": 1})

        norm_score = np.clip(score, -100.0, 100.0)
        
        if norm_score >= 60: regime = "Extreme High Volatility"
        elif norm_score >= 20: regime = "High Volatility"
        elif norm_score > -20: regime = "Normal Volatility"
        elif norm_score > -60: regime = "Low Volatility / Compression"
        else: regime = "Extreme Volatility Collapse"

        # Evidence-based confidence
        target_pol = 1 if norm_score > 0 else -1 if norm_score < 0 else 0
        aligned_w = sum(e['weight'] for e in evidence if e['polarity'] == target_pol or e['polarity'] == 0)
        signal_clarity = (aligned_w / max_confidence) if max_confidence > 0 else 0.0
        data_quality = (confidence_points / max_confidence) if max_confidence > 0 else 0.0
        
        final_confidence = np.clip((signal_clarity * 0.7 + data_quality * 0.3) * 100.0, 0.0, 100.0)

        return {
            "regime": regime,
            "score": round(norm_score, 2),
            "confidence": round(final_confidence, 2),
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 3. CONTRACTION / EXPANSION CYCLE
    # --------------------------------------------------------------------------
    def _analyze_cycle(self, df: pd.DataFrame) -> ContractionExpansionResult:
        evidence: List[EvidenceItem] = []
        intensity = 0.0
        max_conf = 0.0
        actual_conf = 0.0
        
        latest = df.iloc[-1]
        
        # ATR Slope Analysis
        max_conf += 50.0
        recent_atr = df['atr_14'].tail(VOLATILITY_CONFIG["lookbacks"]["short"])
        if len(recent_atr) > 1:
            actual_conf += 50.0
            atr_slope = recent_atr.diff().mean()
            atr_sma = df['atr_14'].mean()
            normalized_slope = (atr_slope / (atr_sma + EPSILON)) * 100.0
            
            if normalized_slope > 2.0:
                intensity += 50.0
                evidence.append({"type": "ATR_Slope", "weight": 50.0, "value": "ATR Expanding Rapidly", "polarity": 1})
            elif normalized_slope < -2.0:
                intensity -= 50.0
                evidence.append({"type": "ATR_Slope", "weight": 50.0, "value": "ATR Contracting Rapidly", "polarity": -1})

        # BB Width / Expansion Index Integration
        max_conf += 50.0
        bbw_col = self._get_cached_col(df, 'bbw')
        ei_col = self._get_cached_col(df, 'ei_')
        
        if bbw_col and pd.notna(latest[bbw_col]):
            actual_conf += 50.0
            bbw_slope = df[bbw_col].tail(5).diff().mean()
            if bbw_slope > 0:
                intensity += 50.0
                evidence.append({"type": "BB_Width", "weight": 50.0, "value": "Bands Expanding", "polarity": 1})
            elif bbw_slope < 0:
                intensity -= 50.0
                evidence.append({"type": "BB_Width", "weight": 50.0, "value": "Bands Contracting", "polarity": -1})
        elif ei_col and pd.notna(latest[ei_col]):
            actual_conf += 50.0
            if latest[ei_col] > 0:
                intensity += 50.0
                evidence.append({"type": "Expansion_Index", "weight": 50.0, "value": "Volatility Expanding", "polarity": 1})
            else:
                intensity -= 50.0
                evidence.append({"type": "Expansion_Index", "weight": 50.0, "value": "Volatility Contracting", "polarity": -1})

        norm_intensity = np.clip(intensity, -100.0, 100.0)
        
        if norm_intensity >= 60: phase = "Aggressive Expansion"
        elif norm_intensity >= 20: phase = "Expansion"
        elif norm_intensity > -20: phase = "Equilibrium"
        elif norm_intensity > -60: phase = "Contraction"
        else: phase = "Aggressive Contraction"

        confidence = (actual_conf / max_conf * 100.0) if max_conf > 0 else 0.0

        return {
            "phase": phase,
            "intensity": round(norm_intensity, 2),
            "confidence": round(confidence, 2),
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 4. SQUEEZE DETECTION
    # --------------------------------------------------------------------------
    def _analyze_squeeze(self, df: pd.DataFrame, adv: AdvancedVolatilityMetrics) -> SqueezeResult:
        evidence: List[EvidenceItem] = []
        is_squeezing = False
        duration = 0
        score = 0.0
        confidence = 100.0
        
        sqz_col = self._get_cached_col(df, 'sqz')
        
        if sqz_col and pd.notna(df[sqz_col].iloc[-1]):
            # Valid Layer-2 Data usage
            sqz_series = df[sqz_col].tail(VOLATILITY_CONFIG["lookbacks"]["medium"])
            is_squeezing = bool(sqz_series.iloc[-1])
            
            if is_squeezing:
                for val in reversed(sqz_series.values):
                    if val: duration += 1
                    else: break
                
                score = min((duration / 10.0) * 100.0, 100.0)
                evidence.append({"type": "Squeeze_Indicator", "weight": 100.0, "value": f"Active Squeeze ({duration} bars)", "polarity": -1})
            else:
                evidence.append({"type": "Squeeze_Indicator", "weight": 100.0, "value": "No Active Squeeze", "polarity": 0})
        else:
            # High quality proxy utilizing advanced metrics
            confidence = 70.0 # Proxy reduces data-confidence
            if adv['atr_percentile'] < 15.0 and adv['atr_z_score'] < -1.5:
                is_squeezing = True
                duration = adv['regime_persistence']
                score = 100.0 - adv['atr_percentile']
                evidence.append({"type": "Z_Score_Proxy", "weight": 100.0, "value": "Extreme Volatility Collapse (Proxy Squeeze)", "polarity": -1})
            else:
                evidence.append({"type": "Z_Score_Proxy", "weight": 100.0, "value": "Insufficient Compression", "polarity": 0})

        return {
            "is_squeezing": is_squeezing,
            "squeeze_score": round(score, 2),
            "duration": duration,
            "confidence": round(confidence, 2),
            "evidence": evidence
        }

    # --------------------------------------------------------------------------
    # 5. BREAKOUT PROBABILITY (PREDICTIVE, NON-DIRECTIONAL)
    # --------------------------------------------------------------------------
    def _analyze_breakout(self, state: VolatilityStateResult, cycle: ContractionExpansionResult, squeeze: SqueezeResult, adv: AdvancedVolatilityMetrics) -> BreakoutProbabilityResult:
        evidence: List[EvidenceItem] = []
        prob = 0.0
        max_weight = 100.0
        
        # 1. Squeeze Weight
        if squeeze['is_squeezing']:
            bonus = min(squeeze['duration'] * 5.0, 20.0)
            prob += 30.0 + bonus
            evidence.append({"type": "Squeeze_Energy", "weight": 50.0, "value": "Squeeze building kinetic energy", "polarity": 1})
            
        # 2. Cycle Weight (Deep contraction precedes expansion)
        if cycle['intensity'] < -40.0:
            prob += 30.0
            evidence.append({"type": "Cycle_Contraction", "weight": 30.0, "value": "Deep contraction phase", "polarity": 1})
            
        # 3. Z-Score Extremes
        if adv['atr_z_score'] < -2.0:
            prob += 20.0
            evidence.append({"type": "Z_Score", "weight": 20.0, "value": "Z-Score heavily compressed", "polarity": 1})

        prob = np.clip(prob, 5.0, 95.0)
        
        # Evidence Alignment Confidence Check
        # Confidence drops if indicators disagree (e.g. Squeezing but Z-score is high)
        active_weight = sum(e['weight'] for e in evidence if e['polarity'] == 1)
        conf = (active_weight / max_weight) * 100.0 if active_weight > 0 else (100.0 if prob < 20 else 50.0)

        return {
            "probability": round(prob, 2),
            "confidence": round(conf, 2),
            "evidence": evidence
        }

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    "VolatilityAnalyzer",
    "VolatilityAnalysisResult",
    "VolatilityStateResult",
    "ContractionExpansionResult",
    "SqueezeResult",
    "BreakoutProbabilityResult",
    "AdvancedVolatilityMetrics",
    "EvidenceItem"
]

