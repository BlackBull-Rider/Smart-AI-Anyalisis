from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EPS = 1e-12
MIN_LR = 0.15
MAX_LR = 8.0

# Centralized Feature List for Strict L3 Database Input Alignment
REQUIRED_FEATURES = (
    "close", "high", "low", "volume", "rvol_20", "vol_zscore", "adx_14", "rsi_14",
    "pattern_confidence", "breakout_pressure", "trend_angle", 
    "smart_money_candle", "liquidity_sweep_candle", "institutional_body",
    "absorption_candle", "rejection_candle", "body_strength", "wick_strength",
    "upper_wick", "lower_wick", "bullish_candle", "bearish_candle",
    "engulfing_body", "hammer_shape", "inverted_hammer_shape", "shooting_star_shape", 
    "hanging_man_shape", "doji", "dragonfly_doji", "gravestone_doji",
    "double_top_detected", "double_bottom_detected", "hs_detected", "ihs_detected",
    "triangle_detected", "wedge_detected", "cup_detected", "rounding_bottom_detected"
)

def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value: return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None

def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")

def _safe_float(value: Any, default: float = 0.0) -> float:
    n = _num(value)
    if n is None or not math.isfinite(n):
        return default
    return float(n)

def _raw(row: Mapping[str, Any], *names: str) -> Any:
    """O(1) strict lookup assuming pre-normalized dictionary."""
    for name in names:
        val = row.get(_key(name))
        if val is not None:
            return val
    return None

def _value(row: Mapping[str, Any], *names: str) -> Optional[float]:
    return _num(_raw(row, *names))

def _sigmoid(value: float) -> float:
    value = _clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + math.exp(-value))

def _lr(weight: float) -> float:
    return _clip(math.exp(_clip(weight, -1.9, 2.08)), MIN_LR, MAX_LR)

class PatternAnalyzer:
    """
    World's No. 1 Institution Grade Pattern Analyzer.
    Utilizes Synergistic Matching (Classical Macro + SMC Micro + Volume Profiling).
    Optimized for DB-friendly nested/flat L3 payloads.
    """

    MACRO_PATTERNS = {
        "ihs_detected": {"name": "Inverse Head & Shoulders", "dir": 1, "weight": 0.95},
        "hs_detected": {"name": "Head & Shoulders", "dir": -1, "weight": 0.95},
        "double_bottom_detected": {"name": "Double Bottom", "dir": 1, "weight": 0.85},
        "double_top_detected": {"name": "Double Top", "dir": -1, "weight": 0.85},
        "triple_bottom_detected": {"name": "Triple Bottom", "dir": 1, "weight": 0.90},
        "triple_top_detected": {"name": "Triple Top", "dir": -1, "weight": 0.90},
        "cup_detected": {"name": "Cup & Handle", "dir": 1, "weight": 0.80},
        "rounding_bottom_detected": {"name": "Rounding Bottom", "dir": 1, "weight": 0.75}
    }

    BILATERAL_MACRO = {
        "triangle_detected": {"name": "Triangle", "weight": 0.75},
        "wedge_detected": {"name": "Wedge", "weight": 0.75}
    }

    MICRO_PATTERNS = {
        "hammer_shape": {"name": "Hammer", "dir": 1, "weight": 0.60},
        "inverted_hammer_shape": {"name": "Inverted Hammer", "dir": 1, "weight": 0.55},
        "shooting_star_shape": {"name": "Shooting Star", "dir": -1, "weight": 0.60},
        "hanging_man_shape": {"name": "Hanging Man", "dir": -1, "weight": 0.55},
        "doji": {"name": "Doji", "dir": 0, "weight": 0.40},
        "dragonfly_doji": {"name": "Dragonfly Doji", "dir": 1, "weight": 0.50},
        "gravestone_doji": {"name": "Gravestone Doji", "dir": -1, "weight": 0.50},
        "smart_money_candle": {"name": "Smart Money Block", "dir": 0, "weight": 0.85},
        "liquidity_sweep_candle": {"name": "Liquidity Sweep", "dir": 0, "weight": 0.90},
        "absorption_candle": {"name": "Absorption", "dir": 0, "weight": 0.80},
        "rejection_candle": {"name": "Strong Rejection", "dir": 0, "weight": 0.80}
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze(self, data: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            rows = self._rows(data)
            if not rows: return self._empty()

            current = rows[-1]
            close = _value(current, "close", "cmp", "price", "last_price")

            if close is None or close <= 0:
                result = self._empty()
                result["pattern_analyzer"]["evidence"].append(
                    self._evidence("Pattern", "Current price is unavailable or invalid in the DB payload.", 0.99, 0.35)
                )
                return result

            # Coverage Tracing exactly from dataset
            found_features = sum(1 for name in REQUIRED_FEATURES if _raw(current, name) is not None)
            feature_coverage = (found_features / len(REQUIRED_FEATURES)) * 100.0

            # Core Analysis
            dominant_pattern, pattern_score, pattern_dir, institutional_modifiers = self._evaluate_patterns(current)
            
            # Status Mapping
            if pattern_dir > 0.2:
                pattern_status = "bullish"
            elif pattern_dir < -0.2:
                pattern_status = "bearish"
            elif dominant_pattern != "None":
                pattern_status = "neutral"
            else:
                pattern_status = "none"

            # Apply Institutional Synergistic Naming
            final_pattern_name = self._synergize_name(dominant_pattern, institutional_modifiers)

            # Evidence & Confidence
            evidence = self._evidence_engine(current, final_pattern_name, pattern_status, pattern_score, institutional_modifiers)
            confidence = self._calculate_confidence(current, pattern_score, evidence)

            return {
                "pattern_analyzer": {
                    "confidence": _safe_float(round(confidence, 4)),
                    "pattern": str(final_pattern_name),
                    "pattern_status": pattern_status,
                    "feature_coverage": _safe_float(round(feature_coverage, 2)),
                    "evidence": evidence,
                }
            }

        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, ArithmeticError) as e:
            self.logger.error(f"Pattern Analyzer Math/Format Error: {e}")
            result = self._empty()
            result["pattern_analyzer"]["evidence"].append(self._evidence("Pattern", "Data format or math failure in processing.", 0.96, 0.42))
            return result
        except Exception as e:
            self.logger.exception(f"Pattern Analyzer Critical Failure: {e}")
            result = self._empty()
            result["pattern_analyzer"]["evidence"].append(self._evidence("Pattern", "Execution encountered a critical failure.", 0.96, 0.42))
            return result

    def _empty(self) -> Dict[str, Any]:
        """Provides a DB friendly empty state ensuring L3 schema integrity."""
        return {
            "pattern_analyzer": {
                "confidence": 0.0,
                "pattern": "None",
                "pattern_status": "none",
                "feature_coverage": 0.0,
                "evidence": [],
            }
        }

    def _flatten_db_payload(self, data: Any) -> Dict[str, Any]:
        flat = {}
        if not isinstance(data, dict): return flat
        for k, v in data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[_key(sub_k)] = sub_v
            else:
                flat[_key(k)] = v
        return flat

    def _rows(self, data: Any) -> List[Mapping[str, Any]]:
        raw_rows = []
        if data is None: return []

        if hasattr(data, "to_dict") and hasattr(data, "columns"):
            try: raw_rows = data.to_dict(orient="records")
            except Exception: return []
        elif isinstance(data, Mapping):
            found_nested_list = False
            for name in ("features", "data", "payload", "rows", "historical_data"):
                nested = data.get(name)
                if isinstance(nested, list):
                    raw_rows = nested
                    found_nested_list = True
                    break
            if not found_nested_list: raw_rows = [self._flatten_db_payload(data)]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            raw_rows = [self._flatten_db_payload(item) if isinstance(item, dict) else item for item in data if isinstance(item, Mapping)]

        normalized_rows = []
        for row in raw_rows:
            normalized = {}
            for k, v in row.items(): normalized[_key(k)] = v
            normalized_rows.append(normalized)
            
        return normalized_rows

    def _evaluate_patterns(self, current: Mapping[str, Any]) -> Tuple[str, float, int, List[str]]:
        """Advanced ranking algorithm to identify the dominant pattern + institutional footprint."""
        
        candidates = []
        institutional_modifiers = []
        
        # Explicit handling of pattern_confidence to avoid 0.0 being falsy
        sys_conf = _value(current, "pattern_confidence")
        if sys_conf is None:
            sys_conf = 50.0
            
        rvol = _value(current, "rvol_20", "volume_ratio") or 1.0
        zscore = _value(current, "vol_zscore") or 0.0
        vol_multiplier = 1.0 + _clip((rvol - 1.0) * 0.2 + (zscore * 0.1), 0.0, 0.5)
        
        bullish_c = _value(current, "bullish_candle") == 1.0
        bearish_c = _value(current, "bearish_candle") == 1.0

        # 1. Macro Patterns
        for key, meta in self.MACRO_PATTERNS.items():
            if _value(current, key) == 1.0:
                score = meta["weight"] * sys_conf * vol_multiplier
                candidates.append((meta["name"], score, meta["dir"], "macro"))

        # 2. Bilateral Macros
        trend_angle = _value(current, "trend_angle") or 0.0
        breakout_pr = _value(current, "breakout_pressure") or 50.0
        
        for key, meta in self.BILATERAL_MACRO.items():
            if _value(current, key) == 1.0:
                dir_val = 1 if (breakout_pr > 50 or trend_angle > 0) else -1
                score = meta["weight"] * sys_conf * vol_multiplier
                candidates.append((meta["name"], score, dir_val, "macro"))

        # 3. Structural/Micro Patterns
        if _value(current, "engulfing_body") == 1.0:
            dir_val = 1 if bullish_c else -1 if bearish_c else 0
            if dir_val != 0:
                name = "Bullish Engulfing" if dir_val == 1 else "Bearish Engulfing"
                candidates.append((name, 0.75 * 100 * vol_multiplier, dir_val, "micro"))

        for key, meta in self.MICRO_PATTERNS.items():
            if _value(current, key) == 1.0:
                dir_val = meta["dir"]
                
                # Dynamic direction resolution for neutral context SMC
                if dir_val == 0:
                    if key in ["smart_money_candle", "absorption_candle"]:
                        if bullish_c: dir_val = 1
                        elif bearish_c: dir_val = -1
                        else: dir_val = 0
                        institutional_modifiers.append("SMC Block" if key == "smart_money_candle" else "Absorption")
                    
                    elif key in ["liquidity_sweep_candle", "rejection_candle"]:
                        l_wick = _value(current, "lower_wick") or 0.0
                        u_wick = _value(current, "upper_wick") or 0.0
                        if l_wick > u_wick: dir_val = 1
                        elif u_wick > l_wick: dir_val = -1
                        else: dir_val = 0
                        institutional_modifiers.append("Liquidity Sweep" if key == "liquidity_sweep_candle" else "Strong Rejection")
                
                score = meta["weight"] * 100 * vol_multiplier
                candidates.append((meta["name"], score, dir_val, "micro"))

        if not candidates:
            return "None", 0.0, 0, institutional_modifiers

        # Sort by Score Descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        dominant = candidates[0]
        
        return dominant[0], dominant[1], dominant[2], institutional_modifiers

    def _synergize_name(self, base_name: str, modifiers: List[str]) -> str:
        if base_name == "None":
            return "None"
            
        unique_mods = list(dict.fromkeys(modifiers))
        if base_name in ["Liquidity Sweep", "Smart Money Block", "Absorption", "Strong Rejection"]:
            return base_name

        if unique_mods:
            return f"{base_name} ({', '.join(unique_mods)})"
        return base_name

    def _evidence_engine(self, current: Mapping[str, Any], pattern_name: str, status: str, score: float, modifiers: List[str]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []

        if pattern_name == "None":
            evidence.append(self._evidence("Pattern", "No distinct institutional or classical patterns detected in the current window.", 0.95, 0.50))
            return evidence

        # Base Pattern Evidence
        reliability = _clip(score / 150.0, 0.60, 0.98) 
        lr = _lr(1.5 * (reliability - 0.5))
        evidence.append(self._evidence("Pattern", f"Primary pattern identified as {pattern_name} with {status} implications.", reliability, lr))

        # Institutional Confluence
        if modifiers:
            evidence.append(self._evidence("Pattern", f"Pattern carries institutional footprint validation: {', '.join(modifiers)}.", 0.90, 1.85))

        # Volume Confirmation
        rvol = _value(current, "rvol_20", "volume_ratio")
        zscore = _value(current, "vol_zscore")
        
        if rvol is not None and rvol > 1.2:
            evidence.append(self._evidence("Pattern", f"Pattern formation is heavily validated by relative volume expansion ({rvol:.2f}x average).", 0.85, 1.60))
        elif rvol is not None and rvol < 0.8:
            evidence.append(self._evidence("Pattern", f"Pattern formation lacks volume participation ({rvol:.2f}x average), risking a false signal.", 0.80, 0.65))

        if zscore is not None and zscore > 1.5:
            evidence.append(self._evidence("Pattern", f"Extreme volume anomaly (Z-Score: {zscore:.2f}) indicates major institutional participation.", 0.88, 1.75))

        # Momentum Context
        rsi = _value(current, "rsi_14")
        if rsi is not None:
            if status == "bullish" and rsi < 40:
                evidence.append(self._evidence("Pattern", f"Bullish pattern aligns with oversold/divergent RSI context ({rsi:.1f}).", 0.82, 1.45))
            elif status == "bearish" and rsi > 60:
                evidence.append(self._evidence("Pattern", f"Bearish pattern aligns with overbought/divergent RSI context ({rsi:.1f}).", 0.82, 1.45))

        return self._unique_evidence(evidence)

    def _calculate_confidence(self, current: Mapping[str, Any], score: float, evidence: Sequence[Mapping[str, Any]]) -> float:
        if score == 0.0: return 0.0

        completeness = sum(1 for name in REQUIRED_FEATURES if _raw(current, name) is not None) / len(REQUIRED_FEATURES)
        evidence_quality = sum(_num(item.get("reliability")) or 0.0 for item in evidence) / max(len(evidence), 1)
        
        contradictions = sum(1 for item in evidence if ((_num(item.get("likelihood_ratio")) or 1.0) < 0.82))
        contradiction_ratio = _clip(contradictions / max(len(evidence), 1), 0.0, 0.40)

        base_confidence = (0.50 * _clip(score / 120.0, 0.0, 1.0)) + (0.25 * completeness) + (0.25 * evidence_quality)
        final_confidence = 100.0 * base_confidence * (1.0 - contradiction_ratio)
        
        return _clip(final_confidence, 0.0, 100.0)

    def _evidence(self, category: str, message: str, reliability: float, likelihood_ratio: float) -> Dict[str, Any]:
        """Strict DB schema generation for L3 outputs."""
        return {
            "category": str(category),
            "message": str(message),
            "reliability": _safe_float(round(_clip(reliability, 0.0, 1.0), 6)),
            "likelihood_ratio": _safe_float(round(_clip(likelihood_ratio, MIN_LR, MAX_LR), 6)),
        }

    def _unique_evidence(self, evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen = set()
        for item in evidence:
            key = (item.get("category"), item.get("message"))
            if key in seen: continue
            seen.add(key)
            result.append(item)
        return result

def analyze(data: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    return PatternAnalyzer().analyze(data, *args, **kwargs)

__all__ = ["PatternAnalyzer", "analyze"]
