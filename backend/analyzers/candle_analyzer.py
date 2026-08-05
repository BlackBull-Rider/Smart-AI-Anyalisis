from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

EPS = 1e-12
MIN_LR = 0.15
MAX_LR = 8.0

REQUIRED_FEATURES = (
    "open", "high", "low", "close", "volume", 
    "body", "body_size", "body_strength", "wick_strength", "upper_wick", "lower_wick", 
    "bullish_candle", "bearish_candle", "candle_strength", "direction_strength", 
    "dominance_score", "pressure_score", "clv", 
    "doji", "dragonfly_doji", "gravestone_doji", 
    "hammer_shape", "inverted_hammer_shape", "shooting_star_shape", "hanging_man_shape", 
    "engulfing_body", "marubozu", "bullish_marubozu", "bearish_marubozu", 
    "gap_up", "gap_down", "gap_percent", "gap_filled", 
    "rvol_20", "vol_zscore", "volume_ratio", 
    "smart_money_candle", "liquidity_sweep_candle", "institutional_body", 
    "absorption_candle", "rejection_candle", "pattern_confidence", "atr_14", "atr"
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

def _truth(value: Any) -> Optional[bool]:
    n = _num(value)
    if n is not None:
        if n > 0: return True
        if n < 0: return False
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "y", "bullish", "up", "confirmed"}: return True
        if value in {"false", "no", "n", "bearish", "down", "rejected"}: return False
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
    for name in names:
        val = row.get(_key(name))
        if val is not None:
            return val
    return None

def _value(row: Mapping[str, Any], *names: str) -> Optional[float]:
    return _num(_raw(row, *names))

def _lr(weight: float) -> float:
    return _clip(math.exp(_clip(weight, -1.9, 2.08)), MIN_LR, MAX_LR)


class CandleAnalyzer:
    """
    Institution Grade Candle Analyzer.
    Analyzes candle structure, auction balance, and quality using O(1) flattened L3 payloads.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze(self, data: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            rows = self._rows(data)
            if not rows: return self._empty()

            current = self._get_valid_row(rows)
            if current is None:
                result = self._empty()
                result["candle_analyzer"]["evidence"].append(
                    self._evidence("Candle", "No valid candle data (OHLC) available in the payload.", 0.99, 0.35)
                )
                return result

            # OHLC derivation
            o = _value(current, "open") or 0.0
            h = _value(current, "high") or 0.0
            l = _value(current, "low") or 0.0
            c = _value(current, "close", "cmp", "price", "last_price") or 0.0

            if c <= 0 or h < l:
                result = self._empty()
                result["candle_analyzer"]["evidence"].append(
                    self._evidence("Candle", "Invalid or malformed OHLC parameters in payload.", 0.99, 0.35)
                )
                return result

            c_range = max(h - l, EPS)
            body = abs(c - o)
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l

            body_ratio = body / c_range
            u_wick_ratio = upper_wick / c_range
            l_wick_ratio = lower_wick / c_range
            clv = ((c - l) - (h - c)) / c_range if c_range > EPS else 0.0

            # Provided vs Derived
            bullish_flag = _value(current, "bullish_candle")
            bearish_flag = _value(current, "bearish_candle")
            
            is_bullish = bullish_flag == 1.0 if bullish_flag is not None else (c > o)
            is_bearish = bearish_flag == 1.0 if bearish_flag is not None else (c < o)

            # Directional Scoring
            dir_score = clv * 0.4
            if is_bullish: dir_score += 0.3
            if is_bearish: dir_score -= 0.3

            dir_str = _value(current, "direction_strength")
            if dir_str is not None:
                dir_score += _clip(dir_str, -1.0, 1.0) * 0.3
                
            dir_score = _clip(dir_score, -1.0, 1.0)

            # Auction Balance Status & Magnitude
            if dir_score >= 0.25:
                auction_balance_status = "bullish"
            elif dir_score <= -0.25:
                auction_balance_status = "bearish"
            else:
                auction_balance_status = "neutral"

            auction_balance = _clip(abs(dir_score) * 100.0, 0.0, 100.0)

            # Quality Scoring
            quality_score = body_ratio * 40.0
            
            atr = _value(current, "atr_14", "atr")
            if atr and atr > 0:
                range_to_atr = _clip(c_range / atr, 0.0, 3.0)
                quality_score += (range_to_atr / 2.0) * 30.0
            else:
                quality_score += 15.0

            rvol = _value(current, "rvol_20", "volume_ratio")
            zscore = _value(current, "vol_zscore")
            if rvol and rvol > 1.0:
                vol_boost = _clip((rvol - 1.0) * 10.0, 0.0, 20.0)
                quality_score += vol_boost
            
            if _value(current, "institutional_body") == 1.0 or _value(current, "smart_money_candle") == 1.0:
                quality_score += 10.0
                
            candle_quality = _clip(quality_score, 0.0, 100.0)
            
            if candle_quality > 40.0:
                candle_quality_status = auction_balance_status
            else:
                candle_quality_status = "neutral"

            # Evidence Engine
            evidence = self._evidence_engine(
                current, c_range, body_ratio, u_wick_ratio, l_wick_ratio, clv, 
                auction_balance_status, candle_quality, rvol, zscore
            )

            # Confidence Engine
            confidence = self._calculate_confidence(
                current, auction_balance, candle_quality, auction_balance_status, 
                u_wick_ratio, l_wick_ratio, rvol, evidence
            )

            return {
                "candle_analyzer": {
                    "confidence": _safe_float(round(confidence, 4)),
                    "candle_quality": _safe_float(round(candle_quality, 4)),
                    "candle_quality_status": candle_quality_status,
                    "auction_balance": _safe_float(round(auction_balance, 4)),
                    "auction_balance_status": auction_balance_status,
                    "evidence": evidence
                }
            }

        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, ArithmeticError) as e:
            self.logger.error(f"Candle Analyzer Math/Format Error: {e}")
            result = self._empty()
            result["candle_analyzer"]["evidence"].append(self._evidence("Candle", "Data format or math failure in processing.", 0.96, 0.42))
            return result
        except Exception as e:
            self.logger.exception(f"Candle Analyzer Critical Failure: {e}")
            result = self._empty()
            result["candle_analyzer"]["evidence"].append(self._evidence("Candle", "Execution encountered a critical failure.", 0.96, 0.42))
            return result

    def _empty(self) -> Dict[str, Any]:
        return {
            "candle_analyzer": {
                "confidence": 0.0,
                "candle_quality": 0.0,
                "candle_quality_status": "neutral",
                "auction_balance": 0.0,
                "auction_balance_status": "neutral",
                "evidence": []
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
            found_nested = False
            for name in ("features", "data", "payload", "rows", "historical_data"):
                nested = data.get(name)
                if isinstance(nested, list):
                    raw_rows = nested
                    found_nested = True
                    break
            if not found_nested: raw_rows = [self._flatten_db_payload(data)]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            raw_rows = [self._flatten_db_payload(item) if isinstance(item, dict) else item for item in data if isinstance(item, Mapping)]

        normalized_rows = []
        for row in raw_rows:
            normalized = {}
            for k, v in row.items(): normalized[_key(k)] = v
            normalized_rows.append(normalized)
            
        return normalized_rows

    def _get_valid_row(self, rows: List[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        for i in range(len(rows) - 1, -1, -1):
            c = _value(rows[i], "close", "cmp", "price", "last_price")
            if c is not None and c > 0:
                return rows[i]
        return None

    def _evidence_engine(
        self, current: Mapping[str, Any], c_range: float, body_ratio: float, 
        u_wick_ratio: float, l_wick_ratio: float, clv: float, 
        auction_status: str, quality: float, rvol: Optional[float], zscore: Optional[float]
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []

        # Structural Evidence
        if body_ratio >= 0.7:
            msg = f"Large real body ({body_ratio*100:.1f}% of range) indicates committed directional participation."
            evidence.append(self._evidence("Candle", msg, 0.85, 2.2))
        elif body_ratio <= 0.2:
            msg = f"Small real body ({body_ratio*100:.1f}% of range) reflects market equilibrium or indecision."
            evidence.append(self._evidence("Candle", msg, 0.80, 0.8))

        # Wick Evidence
        if u_wick_ratio >= 0.5:
            msg = f"Significant upper wick ({u_wick_ratio*100:.1f}% of range) shows strong seller rejection at higher prices."
            evidence.append(self._evidence("Candle", msg, 0.88, 1.8 if auction_status != "bullish" else 0.6))
        if l_wick_ratio >= 0.5:
            msg = f"Significant lower wick ({l_wick_ratio*100:.1f}% of range) shows strong buyer rejection at lower prices."
            evidence.append(self._evidence("Candle", msg, 0.88, 1.8 if auction_status != "bearish" else 0.6))

        # Close Location (CLV)
        if clv >= 0.7:
            evidence.append(self._evidence("Candle", "Close near the absolute high of the candle, confirming strong buyer-side auction control.", 0.85, 2.1))
        elif clv <= -0.7:
            evidence.append(self._evidence("Candle", "Close near the absolute low of the candle, confirming strong seller-side auction control.", 0.85, 2.1))

        # Precomputed Patterns
        for pat, desc in [
            ("doji", "Doji"), ("hammer_shape", "Hammer"), ("shooting_star_shape", "Shooting Star"),
            ("engulfing_body", "Engulfing"), ("marubozu", "Marubozu")
        ]:
            if _value(current, pat) == 1.0:
                evidence.append(self._evidence("Candle", f"Structure matches classical {desc} characteristics.", 0.82, 1.5))

        # Institutional Context
        if _value(current, "liquidity_sweep_candle") == 1.0:
            evidence.append(self._evidence("Candle", "Candle structural footprint implies a liquidity sweep of recent extremes.", 0.90, 2.5))
        if _value(current, "smart_money_candle") == 1.0 or _value(current, "institutional_body") == 1.0:
            evidence.append(self._evidence("Candle", "Candle proportions and placement validate institutional-level participation.", 0.88, 2.0))
        if _value(current, "absorption_candle") == 1.0:
            evidence.append(self._evidence("Candle", "Price action suggests absorption of opposing aggressive order flow.", 0.85, 1.8))

        # Volume Confirmation
        if rvol is not None:
            if rvol > 1.5:
                evidence.append(self._evidence("Candle", f"Relative volume expansion ({rvol:.2f}x) validates the structural context.", 0.87, 2.3))
            elif rvol < 0.6:
                evidence.append(self._evidence("Candle", f"Low relative volume ({rvol:.2f}x) highlights a lack of broad market participation.", 0.80, 0.6))
        
        if zscore is not None and zscore >= 2.0:
            evidence.append(self._evidence("Candle", f"Extreme volume anomaly (Z-Score {zscore:.2f}) signifies a major institutional footprint.", 0.92, 3.0))

        # Gap Context
        if _value(current, "gap_up") == 1.0:
            gap_pct = _value(current, "gap_percent") or 0.0
            evidence.append(self._evidence("Candle", f"Candle opened with an upward gap ({gap_pct:.2f}%), indicating aggressive pre-market buying.", 0.82, 1.6))
        elif _value(current, "gap_down") == 1.0:
            gap_pct = _value(current, "gap_percent") or 0.0
            evidence.append(self._evidence("Candle", f"Candle opened with a downward gap ({abs(gap_pct):.2f}%), indicating aggressive pre-market selling.", 0.82, 1.6))

        return self._unique_evidence(evidence)

    def _calculate_confidence(
        self, current: Mapping[str, Any], balance: float, quality: float, status: str, 
        u_wick: float, l_wick: float, rvol: Optional[float], evidence: Sequence[Mapping[str, Any]]
    ) -> float:
        # Base confidence from quality metrics
        base_conf = (quality * 0.5) + (balance * 0.3) + 20.0

        # Feature coverage tracking
        found_features = sum(1 for name in REQUIRED_FEATURES if _raw(current, name) is not None)
        coverage_ratio = found_features / max(len(REQUIRED_FEATURES), 1)
        base_conf *= (0.5 + 0.5 * coverage_ratio)

        # Contradiction Penalties
        penalty = 0.0
        if status == "bullish" and u_wick > 0.4:
            penalty += 15.0 # Bullish but massive top rejection
        if status == "bearish" and l_wick > 0.4:
            penalty += 15.0 # Bearish but massive bottom rejection
        if rvol is not None and rvol < 0.5 and quality > 50.0:
            penalty += 10.0 # High quality but no volume support

        # Evidence quality processing
        contradictions = sum(1 for item in evidence if ((_num(item.get("likelihood_ratio")) or 1.0) < 0.85))
        penalty += (contradictions * 5.0)

        confidence = _clip(base_conf - penalty, 0.0, 100.0)
        return confidence

    def _evidence(self, category: str, message: str, reliability: float, likelihood_ratio: float) -> Dict[str, Any]:
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
        
        # Keep concise: limit to top 8 most meaningful
        result.sort(key=lambda x: x.get("likelihood_ratio", 1.0), reverse=True)
        return result[:8]


def analyze(data: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    return CandleAnalyzer().analyze(data, *args, **kwargs)

__all__ = ["CandleAnalyzer", "analyze"]
