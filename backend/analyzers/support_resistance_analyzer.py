from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

EPS = 1e-12
MIN_LR = 0.15
MAX_LR = 8.0

# Centralized Feature List for Strict Alignment
REQUIRED_FEATURES = (
    "close", "high", "low", "volume", "atr_14", "rsi_14", "adx_14", "rvol_20",
    "confirmed_swing_low", "confirmed_swing_high", "structure_support", 
    "structure_resistance", "sm_trend_score", "sm_liquidity_score", 
    "bos_up", "bos_down", "choch_up", "choch_down", 
    "sm_discount_zone", "sm_premium_zone"
)

def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", "")
            if not value:
                return None
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

def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or abs(b) <= EPS: return None
    value = a / b
    return value if math.isfinite(value) else None

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

@dataclass
class Candidate:
    price: float
    side: str
    source: str
    reliability: float
    touches: float = 0.0
    reactions: float = 0.0
    volume_confirmation: float = 0.5
    recency: float = 0.5
    structure: float = 0.5
    smc: float = 0.5
    zone_width: float = 0.0
    crossed: bool = False
    confirmed_cross: bool = False
    reclaimed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Zone:
    side: str
    lower: float
    upper: float
    center: float
    candidates: List[Candidate] = field(default_factory=list)
    source_count: int = 0
    touches: float = 0.0
    confluence: float = 0.0
    quality: float = 0.0
    strength: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.upper - self.lower)


class SupportResistanceAnalyzer:
    """
    Final Institutional Grade S/R Analyzer.
    Optimized for strict determinism, O(1) reads, and DB-safe L3 exports.
    """
    SOURCE_RELIABILITY: Dict[str, float] = {
        "confirmed_swing": 0.98, "structure": 0.97, "demand_supply": 0.96, "institutional": 0.95,
        "smc_liquidity": 0.94, "order_block": 0.93, "fvg": 0.91, "volume_profile": 0.90,
        "cluster": 0.89, "swing": 0.88, "fractal": 0.85, "previous_period": 0.84,
        "vwap": 0.83, "rolling": 0.81, "dynamic": 0.80, "adaptive": 0.80,
        "gap": 0.77, "fibonacci": 0.76, "pivot": 0.74, "donchian": 0.72,
        "psychological": 0.67, "atr": 0.62,
    }

    LEVELS: Dict[str, Tuple[str, str]] = {
        "confirmed_swing_low": ("support", "confirmed_swing"),
        "confirmed_swing_high": ("resistance", "confirmed_swing"),
        "last_swing_low": ("support", "swing"),
        "last_swing_high": ("resistance", "swing"),
        "swing_low": ("support", "swing"),
        "swing_high": ("resistance", "swing"),
        "fractal_low": ("support", "fractal"),
        "fractal_high": ("resistance", "fractal"),
        "fractal_support": ("support", "fractal"),
        "fractal_resistance": ("resistance", "fractal"),
        "rolling_low": ("support", "rolling"),
        "rolling_high": ("resistance", "rolling"),
        "rolling_support": ("support", "rolling"),
        "rolling_resistance": ("resistance", "rolling"),
        "dynamic_support": ("support", "dynamic"),
        "dynamic_resistance": ("resistance", "dynamic"),
        "adaptive_support": ("support", "adaptive"),
        "adaptive_resistance": ("resistance", "adaptive"),
        "structure_support": ("support", "structure"),
        "structure_resistance": ("resistance", "structure"),
        "institutional_support": ("support", "institutional"),
        "institutional_resistance": ("resistance", "institutional"),
        "cluster_support": ("support", "cluster"),
        "cluster_resistance": ("resistance", "cluster"),
        "vol_support": ("support", "cluster"),
        "vol_resistance": ("resistance", "cluster"),
        "nearest_support": ("support", "dynamic"),
        "nearest_resistance": ("resistance", "dynamic"),
        "atr_support": ("support", "atr"),
        "atr_resistance": ("resistance", "atr"),
        "gap_support": ("support", "gap"),
        "gap_resistance": ("resistance", "gap"),
        "vwap": ("both", "vwap"),
        "rolling_vwap": ("both", "vwap"),
        "swing_high_vwap": ("resistance", "vwap"),
        "swing_low_vwap": ("support", "vwap"),
        "avp_poc": ("both", "volume_profile"),
        "avp_vah": ("resistance", "volume_profile"),
        "avp_val": ("support", "volume_profile"),
        "price_cluster": ("both", "cluster"),
        "smart_money_level": ("both", "smc_liquidity"),
        "internal_liquidity": ("both", "smc_liquidity"),
        "external_liquidity": ("both", "smc_liquidity"),
        "buy_side_liquidity": ("resistance", "smc_liquidity"),
        "sell_side_liquidity": ("support", "smc_liquidity"),
        "sm_buy_side_liquidity": ("resistance", "smc_liquidity"),
        "sm_sell_side_liquidity": ("support", "smc_liquidity"),
        "demand_zone": ("support", "demand_supply"),
        "supply_zone": ("resistance", "demand_supply"),
        "reaction_demand": ("support", "demand_supply"),
        "reaction_supply": ("resistance", "demand_supply"),
        "sup_zone_lower": ("support", "demand_supply"),
        "sup_zone_upper": ("support", "demand_supply"),
        "res_zone_lower": ("resistance", "demand_supply"),
        "res_zone_upper": ("resistance", "demand_supply"),
        "golden_zone_lower": ("support", "fibonacci"),
        "golden_zone_upper": ("resistance", "fibonacci"),
        "fib_0": ("both", "fibonacci"),
        "fib_236": ("both", "fibonacci"),
        "fib_382": ("both", "fibonacci"),
        "fib_500": ("both", "fibonacci"),
        "fib_618": ("both", "fibonacci"),
        "fib_786": ("both", "fibonacci"),
        "fib_100": ("both", "fibonacci"),
        "donchian_lower": ("support", "donchian"),
        "donchian_upper": ("resistance", "donchian"),
        "prev_day_low": ("support", "previous_period"),
        "prev_day_high": ("resistance", "previous_period"),
        "daily_low": ("support", "previous_period"),
        "daily_high": ("resistance", "previous_period"),
        "daily_mid": ("both", "previous_period"),
        "weekly_low": ("support", "previous_period"),
        "weekly_high": ("resistance", "previous_period"),
        "weekly_mid": ("both", "previous_period"),
        "monthly_low": ("support", "previous_period"),
        "monthly_high": ("resistance", "previous_period"),
        "monthly_mid": ("both", "previous_period"),
        "yearly_low": ("support", "previous_period"),
        "yearly_high": ("resistance", "previous_period"),
        "yearly_mid": ("both", "previous_period"),
        "pivot": ("both", "pivot"),
        "pivot_s1": ("support", "pivot"),
        "pivot_s2": ("support", "pivot"),
        "pivot_s3": ("support", "pivot"),
        "pivot_r1": ("resistance", "pivot"),
        "pivot_r2": ("resistance", "pivot"),
        "pivot_r3": ("resistance", "pivot"),
        "woodie_pivot": ("both", "pivot"),
        "camarilla_pivot": ("both", "pivot"),
        "demark_pivot": ("both", "pivot"),
        "psychological_level": ("both", "psychological"),
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
                result["support_resistance_analyzer"]["evidence"].append(
                    self._evidence("SR", "Current price is unavailable or invalid in the payload.", 0.99, 0.35)
                )
                return result

            # Coverage Tracing using unified CONSTANT
            found_features = sum(1 for name in REQUIRED_FEATURES if _raw(current, name) is not None)
            feature_coverage = (found_features / len(REQUIRED_FEATURES)) * 100.0

            atr = self._atr(current, close)
            tolerance = self._tolerance(current, close, atr)

            candidates = self._generate_candidates(rows, current, close, atr, tolerance)
            candidates = self._validate_candidates(candidates, rows, current, close, atr, tolerance)
            zones = self._cluster(candidates, close, atr, tolerance)

            support = self._select(zones, "support", close, atr)
            resistance = self._select(zones, "resistance", close, atr)

            support_status = self._status(support, "support", current, close, atr, tolerance)
            resistance_status = self._status(resistance, "resistance", current, close, atr, tolerance)

            support_price = self._level_price(support, "support", close)
            resistance_price = self._level_price(resistance, "resistance", close)

            clearance, clearance_status = self._clearance(
                support, resistance, support_status, resistance_status, close, atr, tolerance
            )

            reward, reward_status = self._reward(close, resistance, resistance_status, zones, atr)
            risk, risk_status = self._risk(close, support, support_status, zones, atr)

            evidence = self._evidence_engine(
                rows=rows, current=current, close=close, atr=atr,
                support=support, resistance=resistance,
                support_status=support_status, resistance_status=resistance_status,
                clearance=clearance, clearance_status=clearance_status,
                reward=reward, risk=risk
            )

            confidence = self._confidence(rows, current, atr, support, resistance, evidence)

            return {
                "support_resistance_analyzer": {
                    "confidence": _safe_float(round(confidence, 4)),
                    "support": _safe_float(round(support_price, 6) if support_price is not None else 0.0),
                    "support_status": support_status,
                    "resistance": _safe_float(round(resistance_price, 6) if resistance_price is not None else 0.0),
                    "resistance_status": resistance_status,
                    "clearance": _safe_float(round(clearance, 6)),
                    "clearance_status": clearance_status,
                    "reward": _safe_float(round(max(0.0, reward), 6)),
                    "reward_status": reward_status,
                    "risk": _safe_float(round(max(0.0, risk), 6)),
                    "risk_status": risk_status,
                    "feature_coverage": _safe_float(round(feature_coverage, 2)),
                    "evidence": evidence,
                }
            }

        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, ArithmeticError) as e:
            self.logger.error(f"S/R Data Math Error: {e}")
            result = self._empty()
            result["support_resistance_analyzer"]["evidence"].append(self._evidence("SR", "Data format or math failure in processing.", 0.96, 0.42))
            return result
        except Exception as e:
            self.logger.exception(f"S/R Critical Failure: {e}")
            result = self._empty()
            result["support_resistance_analyzer"]["evidence"].append(self._evidence("SR", "Execution encountered a critical failure.", 0.96, 0.42))
            return result

    def _empty(self) -> Dict[str, Any]:
        return {
            "support_resistance_analyzer": {
                "confidence": 0.0, "support": 0.0, "support_status": "unknown",
                "resistance": 0.0, "resistance_status": "unknown", "clearance": 0.0,
                "clearance_status": "unknown", "reward": 0.0, "reward_status": "low",
                "risk": 0.0, "risk_status": "high", "feature_coverage": 0.0, "evidence": [],
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

        # One-time normalization mapping
        normalized_rows = []
        for row in raw_rows:
            normalized = {}
            for k, v in row.items(): normalized[_key(k)] = v
            normalized_rows.append(normalized)
            
        return normalized_rows

    def _atr(self, row: Mapping[str, Any], close: float) -> Optional[float]:
        atr = _value(row, "atr_14", "adaptive_atr", "atr", "true_range", "tr")
        if atr is not None and atr > 0: return atr

        natr = _value(row, "natr_14", "natr")
        if natr is not None and natr > 0:
            res = close * natr / 100.0
            if res > 0: return res

        std = _value(row, "std_20", "std", "rolling_std")
        return std if std is not None and std > 0 else None

    def _tolerance(self, row: Mapping[str, Any], close: float, atr: Optional[float]) -> float:
        base = atr if atr and atr > 0 else close * 0.01
        natr = _value(row, "natr_14", "natr")
        if natr is not None and natr > 0: base = max(base, close * natr / 100.0)
        hv = _value(row, "hv_21")
        if hv is not None and hv > 0: base = max(base, close * hv / 100.0 / math.sqrt(252.0))
        return max(close * 0.001, base * 0.22)

    def _generate_candidates(self, rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], close: float, atr: Optional[float], tolerance: float) -> List[Candidate]:
        candidates: List[Candidate] = []
        trend_score = _value(current, "sm_trend_score", "trend_strength") or 0.0

        for feature, (side, source) in self.LEVELS.items():
            value = _value(current, feature)
            if value is None or value <= 0: continue

            if side == "both":
                if abs(value - close) <= tolerance:
                    side = "support" if trend_score >= 0 else "resistance"
                else:
                    side = "support" if value < close else "resistance"

            candidates.append(Candidate(price=value, side=side, source=source, reliability=self.SOURCE_RELIABILITY.get(source, 0.60), metadata={"feature": feature}))

        self._zone_candidates(candidates, current, close)
        self._order_block_candidates(candidates, current, close)
        self._fvg_candidates(candidates, current, close)
        self._psychological_candidates(candidates, current, close)

        if len(rows) > 1:
            self._historical_candidates(candidates, rows, close, atr, tolerance)

        return self._deduplicate(candidates, tolerance)

    def _zone_candidates(self, candidates: List[Candidate], row: Mapping[str, Any], close: float) -> None:
        for lower_name, upper_name, side in (("sup_zone_lower", "sup_zone_upper", "support"), ("res_zone_lower", "res_zone_upper", "resistance")):
            lower = _value(row, lower_name)
            upper = _value(row, upper_name)
            if lower is None and upper is None: continue
            if lower is None: lower = upper
            if upper is None: upper = lower
            if upper < lower: lower, upper = upper, lower
            width = upper - lower

            for value, feature in ((lower, lower_name), (upper, upper_name)):
                if value is not None and value > 0:
                    candidates.append(Candidate(value, side, "demand_supply", self.SOURCE_RELIABILITY["demand_supply"], zone_width=width, metadata={"feature": feature}))

    def _order_block_candidates(self, candidates: List[Candidate], row: Mapping[str, Any], close: float) -> None:
        high = _value(row, "order_block_high", "bullish_ob_high", "bearish_ob_high")
        low = _value(row, "order_block_low", "bullish_ob_low", "bearish_ob_low")

        if high is not None and low is not None and high >= low > 0:
            side = "support" if (high + low) / 2 <= close else "resistance"
            width = high - low
            candidates.extend((
                Candidate(low, side, "order_block", self.SOURCE_RELIABILITY["order_block"], zone_width=width, metadata={"feature": "order_block_low"}),
                Candidate(high, side, "order_block", self.SOURCE_RELIABILITY["order_block"], zone_width=width, metadata={"feature": "order_block_high"}),
            ))
            return

        for name in ("sm_fresh_ob", "sm_mitigated_ob", "order_block", "bullish_ob", "bearish_ob"):
            value = _value(row, name)
            if value is None or value <= 0: continue
            candidates.append(Candidate(value, "support" if value <= close else "resistance", "order_block", self.SOURCE_RELIABILITY["order_block"], metadata={"feature": name}))

    def _fvg_candidates(self, candidates: List[Candidate], row: Mapping[str, Any], close: float) -> None:
        high = _value(row, "fvg_high", "active_fvg_high")
        low = _value(row, "fvg_low", "active_fvg_low")

        if high is not None and low is not None and high >= low > 0:
            side = "support" if (high + low) / 2 <= close else "resistance"
            width = high - low
            candidates.extend((
                Candidate(low, side, "fvg", self.SOURCE_RELIABILITY["fvg"], zone_width=width, metadata={"feature": "fvg_low"}),
                Candidate(high, side, "fvg", self.SOURCE_RELIABILITY["fvg"], zone_width=width, metadata={"feature": "fvg_high"}),
            ))
            return

        for name in ("sm_active_fvg", "sm_mitigated_fvg", "fvg"):
            value = _value(row, name)
            if value is not None and value > 0:
                candidates.append(Candidate(value, "support" if value <= close else "resistance", "fvg", self.SOURCE_RELIABILITY["fvg"], metadata={"feature": name}))

    def _psychological_candidates(self, candidates: List[Candidate], row: Mapping[str, Any], close: float) -> None:
        supplied = _value(row, "psychological_level")
        if supplied is not None and supplied > 0:
            candidates.append(Candidate(supplied, "support" if supplied <= close else "resistance", "psychological", self.SOURCE_RELIABILITY["psychological"], metadata={"feature": "psychological_level"}))
            return

        step = 5.0 if close < 100 else 10.0 if close < 500 else 25.0 if close < 1000 else 50.0 if close < 2500 else 100.0 if close < 5000 else 250.0
        lower = math.floor(close / step) * step
        upper = math.ceil(close / step) * step

        if lower > 0: candidates.append(Candidate(lower, "support", "psychological", self.SOURCE_RELIABILITY["psychological"] * 0.72, metadata={"derived": True}))
        if upper > 0: candidates.append(Candidate(upper, "resistance", "psychological", self.SOURCE_RELIABILITY["psychological"] * 0.72, metadata={"derived": True}))

    def _historical_candidates(self, candidates: List[Candidate], rows: Sequence[Mapping[str, Any]], close: float, atr: Optional[float], tolerance: float) -> None:
        sample = rows[-min(len(rows), 300):]
        total = max(len(sample), 1)
        seen_highs, seen_lows = [], []

        for index, row in enumerate(sample):
            recency = (index + 1) / total
            high = _value(row, "confirmed_swing_high", "swing_high", "fractal_high")
            low = _value(row, "confirmed_swing_low", "swing_low", "fractal_low")

            if high is not None and high > 0:
                if not any(abs(high - h) <= tolerance for h in seen_highs):
                    candidates.append(Candidate(high, "resistance", "swing", self.SOURCE_RELIABILITY["swing"], recency=recency, metadata={"historical": True}))
                    seen_highs.append(high)
            if low is not None and low > 0:
                if not any(abs(low - l) <= tolerance for l in seen_lows):
                    candidates.append(Candidate(low, "support", "swing", self.SOURCE_RELIABILITY["swing"], recency=recency, metadata={"historical": True}))
                    seen_lows.append(low)

    def _deduplicate(self, candidates: Sequence[Candidate], tolerance: float) -> List[Candidate]:
        result: List[Candidate] = []
        for candidate in sorted(candidates, key=lambda item: (item.side, item.price)):
            if not math.isfinite(candidate.price) or candidate.price <= 0: continue

            duplicate: Optional[Candidate] = None
            for existing in reversed(result):
                if existing.side != candidate.side: continue
                if candidate.price - existing.price > tolerance: break
                if abs(candidate.price - existing.price) <= tolerance:
                    duplicate = existing
                    break

            if duplicate is None:
                result.append(candidate)
            else:
                sources = duplicate.metadata.setdefault("sources", [])
                sources.append(candidate.source)
                if candidate.reliability > duplicate.reliability: duplicate.reliability = candidate.reliability
                if candidate.zone_width > duplicate.zone_width: duplicate.zone_width = candidate.zone_width

        return result

    def _validate_candidates(self, candidates: Sequence[Candidate], rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], close: float, atr: Optional[float], tolerance: float) -> List[Candidate]:
        result: List[Candidate] = []
        for candidate in candidates:
            touches, reactions, volume, recency, crossed, confirmed_cross = self._interaction(candidate, rows, current, close, atr, tolerance)
            candidate.touches = touches
            candidate.reactions = reactions
            candidate.volume_confirmation = volume
            candidate.recency = max(candidate.recency, recency)
            candidate.crossed = crossed
            candidate.confirmed_cross = confirmed_cross
            candidate.structure = self._structure_alignment(candidate, current)
            candidate.smc = self._smc_alignment(candidate, current)
            candidate.reclaimed = self._reclaimed(candidate, rows, close, tolerance)

            if candidate.zone_width <= 0:
                supplied_width = _value(current, "zone_width")
                candidate.zone_width = supplied_width if supplied_width is not None and supplied_width > 0 else max((atr or close * 0.01) * 0.15, tolerance * 0.35)

            result.append(candidate)
        return result

    def _interaction(self, candidate: Candidate, rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], close: float, atr: Optional[float], tolerance: float) -> Tuple[float, float, float, float, bool, bool]:
        if not rows: return 0.0, 0.0, 0.5, candidate.recency, False, False

        sample = rows[-min(len(rows), 300):]
        touches = 0.0
        reaction_values: List[float] = []
        volume_values: List[float] = []
        last_touch = -1
        crossed = False
        confirmed_cross = False

        for index, row in enumerate(sample):
            high = _value(row, "high")
            low = _value(row, "low")
            row_close = _value(row, "close", "cmp", "price")

            if row_close is None: continue
            touched = False
            if high is not None and low is not None: touched = low - tolerance <= candidate.price <= high + tolerance

            if touched:
                touches += 1.0
                last_touch = index
                reaction = self._support_reaction(row, candidate.price, atr) if candidate.side == "support" else self._resistance_reaction(row, candidate.price, atr)
                if reaction > 0: reaction_values.append(reaction)
                volume_values.append(self._volume_signal(row))

            if candidate.side == "support":
                penetration = candidate.price - row_close
                if penetration > tolerance:
                    crossed = True
                    if _ratio(penetration, atr or close * 0.01) >= 0.35: confirmed_cross = True
            else:
                penetration = row_close - candidate.price
                if penetration > tolerance:
                    crossed = True
                    if _ratio(penetration, atr or close * 0.01) >= 0.35: confirmed_cross = True

        reactions = sum(reaction_values) / len(reaction_values) if reaction_values else 0.0
        volume = sum(volume_values) / len(volume_values) if volume_values else self._volume_signal(current if current else {})
        recency = (last_touch + 1) / len(sample) if last_touch >= 0 else candidate.recency

        return (touches, _clip(reactions, 0.0, 1.0), _clip(volume, 0.0, 1.0), _clip(recency, 0.0, 1.0), crossed, confirmed_cross)

    def _support_reaction(self, row: Mapping[str, Any], level: float, atr: Optional[float]) -> float:
        low, close = _value(row, "low"), _value(row, "close")
        if low is None or close is None or close <= low: return 0.0
        high = _value(row, "high")
        candle_range = _value(row, "candle_range") or (high - low if high is not None else None)
        wick_ratio = _ratio(close - low, candle_range)
        atr_bounce = _ratio(close - low, atr)
        return _clip(0.58 * (wick_ratio or 0.0) + 0.42 * _sigmoid((atr_bounce or 0.0) - 0.15), 0.0, 1.0)

    def _resistance_reaction(self, row: Mapping[str, Any], level: float, atr: Optional[float]) -> float:
        high, close = _value(row, "high"), _value(row, "close")
        if high is None or close is None or high <= close: return 0.0
        low = _value(row, "low")
        candle_range = _value(row, "candle_range") or (high - low if low is not None else None)
        wick_ratio = _ratio(high - close, candle_range)
        atr_rejection = _ratio(high - close, atr)
        return _clip(0.58 * (wick_ratio or 0.0) + 0.42 * _sigmoid((atr_rejection or 0.0) - 0.15), 0.0, 1.0)

    def _volume_signal(self, row: Mapping[str, Any]) -> float:
        values: List[float] = []
        rvol = _value(row, "rvol_20", "volume_ratio")
        if rvol is not None: values.append(_clip(rvol / 2.0, 0.0, 1.0))
        zscore = _value(row, "vol_zscore")
        if zscore is not None: values.append(_sigmoid(zscore))
        mfi = _value(row, "mfi_14")
        if mfi is not None: values.append(_clip(mfi / 100.0, 0.0, 1.0))
        cmf = _value(row, "cmf_20")
        if cmf is not None:
            if abs(cmf) > 2.0: cmf = cmf / 100.0
            values.append(_clip(0.5 + cmf, 0.0, 1.0))
        buy, sell = _value(row, "buy_volume"), _value(row, "sell_volume")
        if buy is not None and sell is not None and buy + sell > 0: values.append(_clip(buy / (buy + sell), 0.0, 1.0))
        return sum(values) / len(values) if values else 0.5

    def _structure_alignment(self, candidate: Candidate, row: Mapping[str, Any]) -> float:
        values: List[float] = []
        bos_up, bos_down = _value(row, "bos_up", "hh_breakout"), _value(row, "bos_down", "ll_breakdown")
        choch_up, choch_down = _value(row, "choch_up", "mss_bullish"), _value(row, "choch_down", "mss_bearish")

        if candidate.side == "support":
            if bos_up is not None and bos_up > 0: values.append(1.0)
            if choch_up is not None and choch_up > 0: values.append(0.9)
            if bos_down is not None and bos_down > 0: values.append(0.15)
            if choch_down is not None and choch_down > 0: values.append(0.20)
            if _truth(_raw(row, "higher_low")): values.append(0.90)
            if _truth(_raw(row, "lower_low")): values.append(0.20)
        else:
            if bos_down is not None and bos_down > 0: values.append(1.0)
            if choch_down is not None and choch_down > 0: values.append(0.9)
            if bos_up is not None and bos_up > 0: values.append(0.15)
            if choch_up is not None and choch_up > 0: values.append(0.20)
            if _truth(_raw(row, "lower_high")): values.append(0.90)
            if _truth(_raw(row, "higher_high")): values.append(0.20)

        return sum(values) / len(values) if values else 0.5

    def _smc_alignment(self, candidate: Candidate, row: Mapping[str, Any]) -> float:
        values: List[float] = []
        for name in ("sm_liquidity_score", "sm_institutional_score", "sm_smart_money_score", "sm_bos_score", "sm_choch_score", "sm_trend_score"):
            val = _value(row, name)
            if val is not None: values.append(_clip((val + 1.0) / 2.0, 0.0, 1.0))

        premium, discount = _truth(_raw(row, "sm_premium_zone")), _truth(_raw(row, "sm_discount_zone"))
        if candidate.side == "support":
            if discount: values.append(1.0)
            if premium: values.append(0.15)
        else:
            if premium: values.append(1.0)
            if discount: values.append(0.15)
        return sum(values) / len(values) if values else 0.5

    def _reclaimed(self, candidate: Candidate, rows: Sequence[Mapping[str, Any]], close: float, tolerance: float) -> bool:
        if len(rows) < 3: return False
        sample = rows[-min(len(rows), 25):-1]
        breached = False

        for row in sample:
            row_close = _value(row, "close")
            if row_close is None: continue
            if candidate.side == "support":
                if row_close < candidate.price - tolerance: breached = True
            else:
                if row_close > candidate.price + tolerance: breached = True

        if not breached: return False

        if candidate.side == "support": return close >= candidate.price - (tolerance * 0.5)
        return close <= candidate.price + (tolerance * 0.5)

    def _cluster(self, candidates: Sequence[Candidate], close: float, atr: Optional[float], tolerance: float) -> List[Zone]:
        zones: List[Zone] = []
        for side in ("support", "resistance"):
            items = sorted([c for c in candidates if c.side == side], key=lambda c: c.price)
            for candidate in items:
                target: Optional[Zone] = None
                for zone in reversed(zones):
                    if zone.side != side: continue
                    if candidate.price > zone.upper + tolerance: break
                    if candidate.price >= zone.lower - tolerance:
                        target = zone
                        break

                if target is None:
                    half = max(candidate.zone_width / 2.0, tolerance * 0.35)
                    zones.append(Zone(side=side, lower=candidate.price - half, upper=candidate.price + half, center=candidate.price, candidates=[candidate]))
                else:
                    target.candidates.append(candidate)
                    target.lower = min(target.lower, candidate.price - candidate.zone_width / 2.0)
                    target.upper = max(target.upper, candidate.price + candidate.zone_width / 2.0)

        for zone in zones: self._finalize_zone(zone, close, atr)
        return zones

    def _finalize_zone(self, zone: Zone, close: float, atr: Optional[float]) -> None:
        if not zone.candidates: return
        weights = [c.reliability * (0.55 + 0.15 * c.recency + 0.10 * c.reactions + 0.10 * c.volume_confirmation + 0.05 * c.structure + 0.05 * c.smc) for c in zone.candidates]
        total_weight = sum(weights)
        
        if total_weight > EPS: zone.center = sum(c.price * weight for c, weight in zip(zone.candidates, weights)) / total_weight
        else: zone.center = sum(c.price for c in zone.candidates) / len(zone.candidates)

        source_names = set(c.source for c in zone.candidates)
        for c in zone.candidates: source_names.update(c.metadata.get("sources", []))

        zone.source_count = len(source_names)
        zone.touches = sum(min(c.touches, 15.0) for c in zone.candidates)

        source_confluence = 1.0 - math.exp(-0.68 * zone.source_count)
        touch_confluence = 1.0 - math.exp(-0.30 * zone.touches)

        qualities = [c.reliability * (0.34 + 0.18 * c.recency + 0.16 * c.reactions + 0.13 * c.volume_confirmation + 0.10 * c.structure + 0.09 * c.smc) for c in zone.candidates]
        zone.quality = _clip(sum(qualities) / max(len(qualities), 1), 0.0, 1.0)
        zone.confluence = _clip(0.57 * source_confluence + 0.43 * touch_confluence, 0.0, 1.0)

        positive_structure = sum(1 for c in zone.candidates if c.structure >= 0.65)
        negative_cross = sum(1 for c in zone.candidates if c.confirmed_cross and not c.reclaimed)

        zone.strength = _clip(
            0.43 * zone.quality + 0.28 * zone.confluence + 0.12 * _clip(zone.touches / 8.0, 0.0, 1.0) +
            0.10 * _clip(positive_structure / max(len(zone.candidates), 1), 0.0, 1.0) +
            0.07 * _clip(sum(c.smc for c in zone.candidates) / max(len(zone.candidates), 1), 0.0, 1.0) -
            0.20 * _clip(negative_cross / max(len(zone.candidates), 1), 0.0, 1.0), 0.0, 1.0
        )

        minimum_width = max((atr or close * 0.01) * 0.08, close * 0.0005)
        if zone.upper - zone.lower < minimum_width:
            half = minimum_width / 2.0
            zone.lower, zone.upper = zone.center - half, zone.center + half

    def _select(self, zones: Sequence[Zone], side: str, close: float, atr: Optional[float]) -> Optional[Zone]:
        candidates: List[Tuple[float, Zone]] = []
        volatility_unit = max(atr or close * 0.01, EPS)

        for zone in zones:
            if zone.side != side: continue
            distance = close - zone.center if side == "support" else zone.center - close
            around = zone.lower <= close <= zone.upper
            if distance < 0 and not around: continue

            proximity = math.exp(-0.32 * abs(distance) / volatility_unit)
            activity = self._activity(zone)
            score = 0.46 * zone.strength + 0.25 * zone.confluence + 0.17 * proximity + 0.12 * activity
            if around: score += 0.08
            candidates.append((score, zone))

        if not candidates: return None
        candidates.sort(key=lambda item: (item[0], item[1].strength, item[1].source_count), reverse=True)
        return candidates[0][1]

    def _activity(self, zone: Zone) -> float:
        if not zone.candidates: return 0.0
        values = []
        for c in zone.candidates:
            val = (0.28 * c.reliability + 0.18 * c.recency + 0.18 * c.reactions + 0.14 * c.volume_confirmation + 0.12 * c.structure + 0.10 * c.smc)
            if c.confirmed_cross and not c.reclaimed: val *= 0.25
            elif c.reclaimed: val *= 0.80
            values.append(val)
        return _clip(sum(values) / len(values), 0.0, 1.0)

    def _level_price(self, zone: Optional[Zone], side: str, close: float) -> Optional[float]:
        if zone is None: return None
        if zone.lower <= close <= zone.upper: return zone.lower if side == "support" else zone.upper
        return zone.center

    def _status(self, zone: Optional[Zone], side: str, current: Mapping[str, Any], close: float, atr: Optional[float], tolerance: float) -> str:
        if zone is None: return "unknown"
        reclaimed = any(c.reclaimed for c in zone.candidates)
        unit = max(atr or close * 0.01, EPS)

        if side == "support":
            if close < zone.lower - tolerance:
                dist = (zone.lower - close) / unit
                return "cleared" if dist >= 1.0 else "broken"
            if zone.lower - tolerance <= close <= zone.upper + tolerance:
                if self._support_reaction(current, zone.center, atr) >= 0.60: return "rejected"
                return "testing"
            if reclaimed: return "reclaimed"
            return "active"
        else:
            if close > zone.upper + tolerance:
                dist = (close - zone.upper) / unit
                return "cleared" if dist >= 1.0 else "broken"
            if zone.lower - tolerance <= close <= zone.upper + tolerance:
                if self._resistance_reaction(current, zone.center, atr) >= 0.60: return "rejected"
                return "testing"
            if reclaimed: return "reclaimed"
            return "active"

    def _clearance(self, support: Optional[Zone], resistance: Optional[Zone], support_status: str, resistance_status: str, close: float, atr: Optional[float], tolerance: float) -> Tuple[float, str]:
        unit = max(atr or close * 0.01, EPS)
        res_clearance, res_status = 0.0, "not_cleared"
        sup_clearance, sup_status = 0.0, "not_cleared"

        if resistance is not None:
            if close > resistance.upper:
                norm = (close - resistance.upper) / unit
                res_clearance = _clip(norm, 0.0, 6.0)
                res_status = "cleared" if norm >= 1.0 else "partially_cleared"
                if resistance_status in {"broken", "reclaimed", "rejected"}: res_status = resistance_status
            elif close >= resistance.lower - tolerance:
                res_status = "testing_resistance"

        if support is not None:
            if close < support.lower:
                norm = (support.lower - close) / unit
                sup_clearance = _clip(norm, 0.0, 6.0)
                sup_status = "cleared" if norm >= 1.0 else "partially_cleared"
                if support_status in {"broken", "reclaimed", "rejected"}: sup_status = support_status
            elif close <= support.upper + tolerance:
                sup_status = "testing_support"

        if res_clearance > 0 and sup_clearance == 0: return res_clearance, res_status
        if sup_clearance > 0 and res_clearance == 0: return sup_clearance, sup_status
        if res_status == "testing_resistance": return 0.0, res_status
        if sup_status == "testing_support": return 0.0, sup_status
        
        return 0.0, "between_levels"

    def _reward(self, close: float, resistance: Optional[Zone], resistance_status: str, zones: Sequence[Zone], atr: Optional[float]) -> Tuple[float, str]:
        if resistance is None: return float(atr * 5.0 if atr and atr > 0 else close * 0.05), "high"
        if resistance_status in {"broken", "cleared"}:
            alternatives = [z for z in zones if (z.side == "resistance" and z.center > close and z is not resistance)]
            alternatives.sort(key=lambda z: (z.center, -z.strength))
            if not alternatives: return float(atr * 5.0 if atr and atr > 0 else close * 0.05), "high"
            resistance = alternatives[0]

        target = resistance.lower if resistance.lower > close else resistance.center
        distance = max(0.0, target - close)
        normalized = distance / max(atr or close * 0.01, EPS)

        if normalized >= 3.0 and resistance.strength >= 0.55: status = "high"
        elif normalized >= 1.75: status = "medium"
        else: status = "low"
        return distance, status

    def _risk(self, close: float, support: Optional[Zone], support_status: str, zones: Sequence[Zone], atr: Optional[float]) -> Tuple[float, str]:
        if support is None: return float(atr * 5.0 if atr and atr > 0 else close * 0.05), "high"
        if support_status in {"broken", "cleared"}:
            alternatives = [z for z in zones if (z.side == "support" and z.center < close and z is not support)]
            alternatives.sort(key=lambda z: (-z.center, -z.strength))
            if not alternatives: return float(atr * 5.0 if atr and atr > 0 else close * 0.05), "high"
            support = alternatives[0]

        unit = max(atr or close * 0.01, EPS)
        volatility_buffer = unit * (1.2 if support.strength >= 0.75 else 2.0)
        invalidation = support.lower - volatility_buffer
        
        distance = max(0.0, close - invalidation)
        normalized = distance / unit

        if normalized <= 1.25: status = "low"
        elif normalized <= 2.50: status = "medium"
        else: status = "high"
        return distance, status

    def _evidence_engine(self, rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], close: float, atr: Optional[float], support: Optional[Zone], resistance: Optional[Zone], support_status: str, resistance_status: str, clearance: float, clearance_status: str, reward: float, risk: float) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []

        if support is not None:
            evidence.append(self._evidence("SR", f"Strongest relevant support zone is centered near {support.center:.2f}, derived from {support.source_count} independent source families.", 0.65 + 0.30 * support.strength, _lr(1.35 * (support.strength - 0.45))))
            if support.touches >= 2: evidence.append(self._evidence("SR", f"Support has approximately {support.touches:.0f} supplied historical interactions.", 0.84, 1.45))
            if support.source_count >= 2: evidence.append(self._evidence("SR", f"Support has multi-source confluence from {support.source_count} independent calculations.", 0.87, 1.65))
            if support_status == "rejected": evidence.append(self._evidence("SR", "Current candle structure shows rejection from the support zone.", 0.86, 1.70))
            if support_status in {"broken", "cleared"}: evidence.append(self._evidence("SR", f"Support is currently {support_status} relative to price.", 0.91, 0.58))
        else:
            evidence.append(self._evidence("SR", "No sufficiently validated support zone exists in the supplied features.", 0.96, 0.45))

        if resistance is not None:
            evidence.append(self._evidence("SR", f"Strongest relevant resistance zone is centered near {resistance.center:.2f}, derived from {resistance.source_count} independent source families.", 0.65 + 0.30 * resistance.strength, _lr(1.35 * (resistance.strength - 0.45))))
            if resistance.touches >= 2: evidence.append(self._evidence("SR", f"Resistance has approximately {resistance.touches:.0f} supplied historical interactions.", 0.84, 1.45))
            if resistance.source_count >= 2: evidence.append(self._evidence("SR", f"Resistance has multi-source confluence from {resistance.source_count} independent calculations.", 0.87, 1.65))
            if resistance_status == "rejected": evidence.append(self._evidence("SR", "Current candle structure shows rejection from the resistance zone.", 0.86, 1.70))
            if resistance_status in {"broken", "cleared"}: evidence.append(self._evidence("SR", f"Resistance is currently {resistance_status} relative to price.", 0.91, 1.72))

        self._structure_evidence(evidence, current)
        self._volume_evidence(evidence, current)
        self._smc_evidence(evidence, current)
        self._momentum_evidence(evidence, current)
        self._breakout_evidence(evidence, current, close, atr, support, resistance)

        if atr is not None: evidence.append(self._evidence("SR", f"Level distances are normalized against ATR; ATR is approximately {atr:.4f}, or {atr / close * 100.0:.2f}% of price.", 0.88, 1.0))
        if clearance > 0: evidence.append(self._evidence("SR", f"Relevant barrier clearance measures {clearance:.2f} ATR and is classified as {clearance_status}.", 0.84, _lr(0.30 + min(clearance, 3.0) * 0.25)))
        
        if reward > 0 and risk > 0:
            rr = reward / risk
            if rr >= 2.0: evidence.append(self._evidence("SR", f"Structural reward distance is {rr:.2f} times the structural risk distance.", 0.80, 1.40))
            elif rr < 1.0: evidence.append(self._evidence("SR", f"Structural reward distance is only {rr:.2f} times the structural risk distance.", 0.82, 0.70))

        return self._unique_evidence(evidence)

    def _structure_evidence(self, evidence: List[Dict[str, Any]], row: Mapping[str, Any]) -> None:
        bos_up, bos_down = _value(row, "bos_up", "hh_breakout"), _value(row, "bos_down", "ll_breakdown")
        choch_up, choch_down = _value(row, "choch_up", "mss_bullish"), _value(row, "choch_down", "mss_bearish")

        if bos_up is not None and bos_up > 0: evidence.append(self._evidence("SR", "Supplied market structure confirms an upward BOS/breakout event.", 0.88, 1.75))
        if bos_down is not None and bos_down > 0: evidence.append(self._evidence("SR", "Supplied market structure confirms a downward BOS/breakdown event.", 0.88, 0.57))
        if choch_up is not None and choch_up > 0: evidence.append(self._evidence("SR", "Supplied structure indicates bullish CHOCH/MSS context.", 0.83, 1.52))
        if choch_down is not None and choch_down > 0: evidence.append(self._evidence("SR", "Supplied structure indicates bearish CHOCH/MSS context.", 0.83, 0.66))

    def _volume_evidence(self, evidence: List[Dict[str, Any]], row: Mapping[str, Any]) -> None:
        rvol, zscore = _value(row, "rvol_20", "volume_ratio"), _value(row, "vol_zscore")
        if rvol is not None:
            if rvol >= 1.25: evidence.append(self._evidence("SR", f"Relative volume is elevated at {rvol:.2f}x.", 0.84, 1.55))
            elif rvol < 0.75: evidence.append(self._evidence("SR", f"Relative volume is subdued at {rvol:.2f}x, weakening volume confirmation.", 0.81, 0.69))
        if zscore is not None and zscore >= 1.5: evidence.append(self._evidence("SR", f"Volume z-score is elevated at {zscore:.2f}, indicating unusual participation.", 0.80, 1.50))

    def _smc_evidence(self, evidence: List[Dict[str, Any]], row: Mapping[str, Any]) -> None:
        for name, label in (("sm_liquidity_score", "SMC liquidity"), ("sm_institutional_score", "SMC institutional"), ("sm_smart_money_score", "smart-money"), ("sm_trend_score", "SMC trend")):
            value = _value(row, name)
            if value is None or abs(value) < 0.15: continue
            evidence.append(self._evidence("SR", f"{label.capitalize()} context is directional at {value:.3f}.", 0.75, _lr(0.55 * value)))
        if _truth(_raw(row, "sm_discount_zone")): evidence.append(self._evidence("SR", "Price is marked within the supplied SMC discount-zone context.", 0.80, 1.45))
        if _truth(_raw(row, "sm_premium_zone")): evidence.append(self._evidence("SR", "Price is marked within the supplied SMC premium-zone context.", 0.80, 1.45))

    def _momentum_evidence(self, evidence: List[Dict[str, Any]], row: Mapping[str, Any]) -> None:
        rsi, macd_hist, adx = _value(row, "rsi_14"), _value(row, "macd_hist", "macd_histogram"), _value(row, "adx_14", "adx")
        if rsi is not None:
            if rsi <= 35: evidence.append(self._evidence("SR", f"RSI is {rsi:.2f}, providing oversold momentum context near support.", 0.75, 1.30))
            elif rsi >= 65: evidence.append(self._evidence("SR", f"RSI is {rsi:.2f}, providing overbought momentum context near resistance.", 0.75, 1.30))
        if macd_hist is not None:
            if macd_hist > 0: evidence.append(self._evidence("SR", f"MACD histogram is positive at {macd_hist:.4g}.", 0.72, 1.24))
            elif macd_hist < 0: evidence.append(self._evidence("SR", f"MACD histogram is negative at {macd_hist:.4g}.", 0.72, 0.81))
        if adx is not None and adx < 20: evidence.append(self._evidence("SR", f"ADX is {adx:.2f}, indicating limited trend-strength confirmation.", 0.78, 0.72))

    def _breakout_evidence(self, evidence: List[Dict[str, Any]], row: Mapping[str, Any], close: float, atr: Optional[float], support: Optional[Zone], resistance: Optional[Zone]) -> None:
        body_strength = _value(row, "body_strength", "body_percent", "body_to_range")
        rvol, adx = _value(row, "rvol_20", "volume_ratio"), _value(row, "adx_14", "adx")
        bos_up, bos_down = _value(row, "bos_up", "hh_breakout"), _value(row, "bos_down", "ll_breakdown")
        unit = max(atr or close * 0.01, EPS)

        if resistance is not None and close > resistance.upper:
            penetration = (close - resistance.upper) / unit
            if penetration >= 0.50 and (body_strength is None or body_strength >= 0.55): evidence.append(self._evidence("SR", f"Price has penetrated resistance by {penetration:.2f} ATR with supportive candle-body evidence.", 0.87, 1.85))
            else: evidence.append(self._evidence("SR", "Price is above resistance, but penetration/candle evidence is insufficient for strong breakout confirmation.", 0.83, 0.76))
            if rvol is not None and rvol < 1.0: evidence.append(self._evidence("SR", "Resistance breakout lacks relative-volume confirmation.", 0.85, 0.67))
            if bos_up is None or bos_up <= 0: evidence.append(self._evidence("SR", "No supplied bullish BOS confirmation accompanies the resistance clearance.", 0.78, 0.73))

        if support is not None and close < support.lower:
            penetration = (support.lower - close) / unit
            if penetration >= 0.50 and (body_strength is None or body_strength >= 0.55): evidence.append(self._evidence("SR", f"Price has penetrated support by {penetration:.2f} ATR with supportive candle-body evidence for a breakdown.", 0.87, 1.82))
            else: evidence.append(self._evidence("SR", "Price is below support, but penetration/candle evidence is insufficient for strong breakdown confirmation.", 0.83, 0.76))
            if rvol is not None and rvol < 1.0: evidence.append(self._evidence("SR", "Support breakdown lacks relative-volume confirmation.", 0.85, 0.67))
            if bos_down is None or bos_down <= 0: evidence.append(self._evidence("SR", "No supplied bearish BOS confirmation accompanies the support failure.", 0.78, 0.73))

    def _confidence(self, rows: Sequence[Mapping[str, Any]], current: Mapping[str, Any], atr: Optional[float], support: Optional[Zone], resistance: Optional[Zone], evidence: Sequence[Mapping[str, Any]]) -> float:
        completeness = sum(1 for name in REQUIRED_FEATURES if _raw(current, name) is not None) / len(REQUIRED_FEATURES)
        history = _clip(math.log1p(min(len(rows), 300)) / math.log1p(100.0), 0.0, 1.0)
        
        level_values = []
        for zone in (support, resistance):
            if zone is not None: level_values.append(0.45 * zone.strength + 0.25 * zone.confluence + 0.15 * zone.quality + 0.15 * _clip(zone.touches / 6.0, 0.0, 1.0))
        level_quality = sum(level_values) / len(level_values) if level_values else 0.0
        
        # Guard against evidence inflation via limiting scaling
        evidence_quality = sum(_num(item.get("reliability")) or 0.0 for item in evidence) / max(len(evidence), 1)
        contradictions = sum(1 for item in evidence if ((_num(item.get("likelihood_ratio")) or 1.0) < 0.82))
        contradiction_ratio = _clip(contradictions / max(len(evidence), 1), 0.0, 0.40)

        volume_quality = self._field_quality(current, ("volume", "avg_volume_20", "rvol_20", "volume_ratio", "vol_zscore", "mfi_14", "cmf_20", "obv", "delta_volume"))
        structure_quality = self._field_quality(current, ("swing_high", "swing_low", "confirmed_swing_high", "confirmed_swing_low", "higher_high", "higher_low", "lower_high", "lower_low", "bos_up", "bos_down", "choch_up", "choch_down", "mss_bullish", "mss_bearish"))
        smc_quality = self._field_quality(current, ("sm_bos", "sm_choch", "sm_mss", "sm_buy_side_liquidity", "sm_sell_side_liquidity", "sm_fresh_ob", "sm_active_fvg", "sm_premium_zone", "sm_discount_zone", "sm_trend_score", "sm_liquidity_score", "sm_institutional_score", "sm_smart_money_score"))
        volatility_quality = 1.0 if atr and atr > 0 else 0.40

        raw_confidence = (0.20 * completeness + 0.09 * history + 0.28 * level_quality + 0.12 * evidence_quality + 0.08 * volume_quality + 0.08 * structure_quality + 0.07 * smc_quality + 0.08 * volatility_quality)
        confidence = 100.0 * raw_confidence * (1.0 - contradiction_ratio)

        if support is None and resistance is None: confidence *= 0.30
        elif support is None or resistance is None: confidence *= 0.70

        return _clip(confidence, 0.0, 100.0)

    def _field_quality(self, row: Mapping[str, Any], names: Iterable[str]) -> float:
        names = tuple(names)
        if not names: return 0.0
        return sum(1 for name in names if _raw(row, name) is not None) / len(names)

    def _evidence(self, category: str, message: str, reliability: float, likelihood_ratio: float) -> Dict[str, Any]:
        """Strict schema generation for L3 outputs."""
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
    return SupportResistanceAnalyzer().analyze(data, *args, **kwargs)

__all__ = ["SupportResistanceAnalyzer", "analyze"]
