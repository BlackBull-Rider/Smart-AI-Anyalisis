from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import math


# ============================================================
# 23 INDEPENDENT INSTITUTIONAL INDICATOR ENGINES
# ============================================================
#
# IMPORTANT:
# - Every engine receives ONLY its own indicator series.
# - OHLCV is supplied separately only as PRICE STRUCTURE REFERENCE.
# - No indicator engine reads another indicator.
# - No prediction.
# - No future data.
# - Historical/live trace only.
# - Minimum recommended trace window: 50 completed periods.
#
# Expected input:
#
# indicator_values = [...]
#
# price_ohlcv = [
#     {
#         "open": ...,
#         "high": ...,
#         "low": ...,
#         "close": ...,
#         "volume": ...
#     },
#     ...
# ]
#
# The latest element is the current/completed period.
# ============================================================


INDICATOR_NAMES = [
    "rsi_14",
    "rsi_slope",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx_14",
    "plus_di",
    "minus_di",
    "dx",
    "roc_12",
    "cci_20",
    "mom_10",
    "trix_18",
    "ppo",
    "ppo_signal",
    "ppo_hist",
    "dpo_20",
    "stoch_k",
    "stoch_d",
    "stoch_rsi_k",
    "stoch_rsi_d",
    "williams_r",
    "ultimate_osc",
]


# ============================================================
# SAFE NUMERIC HELPERS
# ============================================================

def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _clean(values: Sequence[Any]) -> List[float]:
    return [float(x) if _f(x) is not None else math.nan for x in values]


def _valid(v: float) -> bool:
    return math.isfinite(v)


def _delta(values: Sequence[float]) -> List[Optional[float]]:
    out = [None]
    for i in range(1, len(values)):
        if _valid(values[i]) and _valid(values[i - 1]):
            out.append(values[i] - values[i - 1])
        else:
            out.append(None)
    return out


def _velocity(values: Sequence[float]) -> List[Optional[float]]:
    return _delta(values)


def _acceleration(values: Sequence[float]) -> List[Optional[float]]:
    d = _delta(values)
    out = [None]
    for i in range(1, len(d)):
        if d[i] is not None and d[i - 1] is not None:
            out.append(d[i] - d[i - 1])
        else:
            out.append(None)
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    x = [v for v in values if _valid(v)]
    return sum(x) / len(x) if x else None


def _std(values: Sequence[float]) -> Optional[float]:
    x = [v for v in values if _valid(v)]
    if len(x) < 2:
        return None
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def _zscore(values: Sequence[float], current: float) -> Optional[float]:
    m = _mean(values)
    s = _std(values)
    if m is None or s is None or s == 0:
        return 0.0
    return (current - m) / s


def _percentile(values: Sequence[float], current: float) -> Optional[float]:
    x = sorted(v for v in values if _valid(v))
    if not x:
        return None
    rank = sum(v <= current for v in x)
    return 100.0 * rank / len(x)


def _sign(x: Optional[float]) -> str:
    if x is None:
        return "unknown"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "flat"


def _direction(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    if delta > 0:
        return "rising"
    if delta < 0:
        return "falling"
    return "flat"


# ============================================================
# PRICE STRUCTURE
# ============================================================

def _price_structure(ohlcv: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Price is NOT treated as another indicator.
    It is only the external market-structure reference.
    """

    if not ohlcv:
        return {
            "available": False,
            "trace": [],
        }

    trace = []

    for i, row in enumerate(ohlcv):
        o = _f(row.get("open"))
        h = _f(row.get("high"))
        l = _f(row.get("low"))
        c = _f(row.get("close"))
        v = _f(row.get("volume"))

        if None in (o, h, l, c):
            trace.append({
                "period": i,
                "available": False,
            })
            continue

        body = c - o
        rng = max(h - l, 0.0)

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        gap = None
        if i > 0:
            pc = _f(ohlcv[i - 1].get("close"))
            if pc not in (None, 0):
                gap = o - pc

        close_location = None
        if rng > 0:
            close_location = (c - l) / rng

        trace.append({
            "period": i,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "body": body,
            "range": rng,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "gap": gap,
            "close_location": close_location,
            "candle_direction": (
                "bullish" if body > 0
                else "bearish" if body < 0
                else "doji"
            ),
        })

    return {
        "available": True,
        "trace": trace,
    }


def _price_relationship(
    indicator_delta: Optional[float],
    price_trace: Sequence[Dict[str, Any]],
    i: int,
) -> str:

    if indicator_delta is None or i >= len(price_trace):
        return "unknown"

    p = price_trace[i]

    if not p.get("available"):
        return "unknown"

    body = _f(p.get("body"))
    gap = _f(p.get("gap"))

    if body is None:
        return "unknown"

    # Indicator movement compared against actual price movement.
    if indicator_delta > 0 and body > 0:
        return "indicator_rising_with_price"

    if indicator_delta < 0 and body < 0:
        return "indicator_falling_with_price"

    if indicator_delta > 0 and body < 0:
        if gap is not None and gap < 0:
            return "indicator_rising_against_gap_down_price"

        return "indicator_rising_against_price"

    if indicator_delta < 0 and body > 0:
        if gap is not None and gap > 0:
            return "indicator_falling_against_gap_up_price"

        return "indicator_falling_against_price"

    return "mixed"


# ============================================================
# STRUCTURAL TRACE
# ============================================================

def _trace(
    values: Sequence[Any],
    ohlcv: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    values = _clean(values)

    n = len(values)

    if n == 0:
        return {
            "period": [],
            "value": [],
            "delta": [],
            "velocity": [],
            "acceleration": [],
            "price": {},
            "relationship": "no_data",
        }

    d = _delta(values)
    vel = _velocity(values)
    acc = _acceleration(values)

    price = _price_structure(ohlcv or [])
    price_trace = price.get("trace", [])

    price_values = []
    price_delta = []
    price_velocity = []
    price_acceleration = []
    relationships = []

    for i in range(n):

        if i < len(price_trace) and price_trace[i].get("available"):
            pv = price_trace[i].get("close")
        else:
            pv = None

        price_values.append(pv)

    price_delta = _delta([
        x if x is not None else math.nan
        for x in price_values
    ])

    price_velocity = price_delta

    price_acceleration = _acceleration([
        x if x is not None else math.nan
        for x in price_values
    ])

    for i in range(n):
        relationships.append(
            _price_relationship(
                d[i],
                price_trace,
                i,
            )
        )

    return {
        "period": list(range(n)),
        "value": [
            None if not _valid(v) else v
            for v in values
        ],
        "delta": d,
        "velocity": vel,
        "acceleration": acc,
        "price": {
            "value": price_values,
            "delta": price_delta,
            "velocity": price_velocity,
            "acceleration": price_acceleration,
        },
        "relationship": relationships,
    }


# ============================================================
# STRUCTURAL MEANING HELPERS
# ============================================================

def _recent_window(
    values: Sequence[Any],
    periods: int = 50,
) -> List[float]:

    x = _clean(values)

    if len(x) <= periods:
        return [v for v in x if _valid(v)]

    return [v for v in x[-periods:] if _valid(v)]


def _slope_state(delta: Sequence[Optional[float]]) -> str:

    x = [v for v in delta[-5:] if v is not None]

    if not x:
        return "unknown"

    pos = sum(v > 0 for v in x)
    neg = sum(v < 0 for v in x)

    if pos == len(x):
        return "persistent_rising"

    if neg == len(x):
        return "persistent_falling"

    if pos > neg:
        return "rising_mixed"

    if neg > pos:
        return "falling_mixed"

    return "unstable"


def _acceleration_state(
    acceleration: Sequence[Optional[float]]
) -> str:

    x = [v for v in acceleration[-5:] if v is not None]

    if not x:
        return "unknown"

    if all(v > 0 for v in x):
        return "positive_acceleration"

    if all(v < 0 for v in x):
        return "negative_acceleration"

    return "mixed_acceleration"


def _structural_phase(
    delta: Sequence[Optional[float]],
    acceleration: Sequence[Optional[float]],
) -> str:

    d = [v for v in delta[-5:] if v is not None]
    a = [v for v in acceleration[-5:] if v is not None]

    if not d:
        return "unknown"

    rising = sum(v > 0 for v in d)
    falling = sum(v < 0 for v in d)

    if rising >= 4:
        if a and sum(v > 0 for v in a) >= len(a) * 0.6:
            return "accelerating_continuation"
        return "rising_continuation"

    if falling >= 4:
        if a and sum(v < 0 for v in a) >= len(a) * 0.6:
            return "accelerating_decline"
        return "falling_continuation"

    return "transition_or_consolidation"


def _divergence_trace(
    values: Sequence[Any],
    ohlcv: Optional[Sequence[Dict[str, Any]]],
    lookback: int = 50,
) -> Dict[str, Any]:

    if not ohlcv:
        return {
            "bullish": False,
            "bearish": False,
            "type": "unavailable",
        }

    vals = _clean(values)
    price = _price_structure(ohlcv)["trace"]

    n = min(len(vals), len(price))

    if n < 10:
        return {
            "bullish": False,
            "bearish": False,
            "type": "insufficient_data",
        }

    start = max(0, n - lookback)

    segment = []

    for i in range(start, n):
        if _valid(vals[i]) and price[i].get("available"):
            segment.append(
                (
                    i,
                    vals[i],
                    price[i]["close"],
                )
            )

    if len(segment) < 10:
        return {
            "bullish": False,
            "bearish": False,
            "type": "insufficient_data",
        }

    # Split the historical window into two structural halves.
    mid = len(segment) // 2

    first = segment[:mid]
    second = segment[mid:]

    ind_first = _mean([x[1] for x in first])
    ind_second = _mean([x[1] for x in second])

    price_first = _mean([x[2] for x in first])
    price_second = _mean([x[2] for x in second])

    if None in (
        ind_first,
        ind_second,
        price_first,
        price_second,
    ):
        return {
            "bullish": False,
            "bearish": False,
            "type": "unavailable",
        }

    ind_change = ind_second - ind_first
    price_change = price_second - price_first

    bullish = price_change < 0 and ind_change > 0
    bearish = price_change > 0 and ind_change < 0

    if bullish:
        typ = "bullish_divergence"

    elif bearish:
        typ = "bearish_divergence"

    elif (
        price_change > 0
        and ind_change > 0
    ):
        typ = "bullish_alignment"

    elif (
        price_change < 0
        and ind_change < 0
    ):
        typ = "bearish_alignment"

    else:
        typ = "mixed_structure"

    return {
        "bullish": bullish,
        "bearish": bearish,
        "type": typ,
        "indicator_change": ind_change,
        "price_change": price_change,
        "lookback": lookback,
    }


def _base_result(
    name: str,
    values: Sequence[Any],
    ohlcv: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:

    trace = _trace(values, ohlcv)

    clean = [
        v for v in _clean(values)
        if _valid(v)
    ]

    d = trace["delta"]
    a = trace["acceleration"]

    current = clean[-1] if clean else None

    return {
        "indicator": name,
        "trace_window": {
            "periods": min(50, len(values)),
            "direction": "backward_from_current",
        },
        "trace": trace,
        "structure": {
            "current": current,
            "historical_percentile": (
                _percentile(clean, current)
                if current is not None
                else None
            ),
            "zscore": (
                _zscore(clean[-50:], current)
                if current is not None
                else None
            ),
            "path": _slope_state(d),
            "acceleration": _acceleration_state(a),
            "phase": _structural_phase(d, a),
        },
        "price_comparison": _divergence_trace(
            values,
            ohlcv,
            50,
        ),
    }


# ============================================================
# INDICATOR-SPECIFIC ENGINES
# ============================================================

def rsi_14_engine(values, ohlcv=None):
    r = _base_result("rsi_14", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is not None:
        if c >= 70:
            zone = "extreme_upper"
        elif c >= 55:
            zone = "upper_momentum"
        elif c >= 45:
            zone = "neutral_transition"
        elif c >= 30:
            zone = "lower_momentum"
        else:
            zone = "extreme_lower"
    else:
        zone = "unknown"

    r["institutional_state"] = {
        "zone": zone,
        "midline_reference": 50.0,
        "overbought_reference": 70.0,
        "oversold_reference": 30.0,
        "state_logic": (
            "momentum_position_and_path"
        ),
    }

    return r


def rsi_slope_engine(values, ohlcv=None):
    r = _base_result("rsi_slope", values, ohlcv)

    d = _delta(_clean(values))

    r["institutional_state"] = {
        "slope": _slope_state(d),
        "acceleration": _acceleration_state(
            _acceleration(_clean(values))
        ),
        "meaning": (
            "RSI slope is tracked as the change-rate of RSI, "
            "not as an independent overbought/oversold oscillator."
        ),
    }

    return r


def macd_engine(values, ohlcv=None):
    r = _base_result("macd", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "above_zero" if c is not None and c > 0
            else "below_zero" if c is not None and c < 0
            else "at_zero" if c is not None
            else "unknown"
        ),
        "path": r["structure"]["path"],
        "acceleration": r["structure"]["acceleration"],
        "price_alignment": r["price_comparison"]["type"],
    }

    return r


def macd_signal_engine(values, ohlcv=None):
    r = _base_result("macd_signal", values, ohlcv)

    r["institutional_state"] = {
        "signal_path": r["structure"]["path"],
        "signal_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def macd_hist_engine(values, ohlcv=None):
    r = _base_result("macd_hist", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "positive" if c is not None and c > 0
            else "negative" if c is not None and c < 0
            else "zero" if c is not None
            else "unknown"
        ),
        "expansion_contraction": (
            "expanding" if len(x) >= 2 and abs(x[-1]) > abs(x[-2])
            else "contracting" if len(x) >= 2
            else "unknown"
        ),
        "price_structure": r["price_comparison"]["type"],
    }

    return r


def adx_14_engine(values, ohlcv=None):
    r = _base_result("adx_14", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c < 20:
        zone = "weak_trend"
    elif c < 25:
        zone = "developing_trend"
    elif c < 40:
        zone = "established_trend"
    else:
        zone = "strong_trend"

    r["institutional_state"] = {
        "trend_strength_zone": zone,
        "strength_path": r["structure"]["path"],
        "strength_acceleration": r["structure"]["acceleration"],
        "price_structure": r["price_comparison"]["type"],
    }

    return r


def plus_di_engine(values, ohlcv=None):
    r = _base_result("plus_di", values, ohlcv)

    r["institutional_state"] = {
        "directional_pressure_path": r["structure"]["path"],
        "pressure_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def minus_di_engine(values, ohlcv=None):
    r = _base_result("minus_di", values, ohlcv)

    r["institutional_state"] = {
        "directional_pressure_path": r["structure"]["path"],
        "pressure_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def dx_engine(values, ohlcv=None):
    r = _base_result("dx", values, ohlcv)

    r["institutional_state"] = {
        "directional_separation_path": r["structure"]["path"],
        "separation_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def roc_12_engine(values, ohlcv=None):
    r = _base_result("roc_12", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "positive" if c is not None and c > 0
            else "negative" if c is not None and c < 0
            else "zero" if c is not None
            else "unknown"
        ),
        "rate_path": r["structure"]["path"],
        "rate_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def cci_20_engine(values, ohlcv=None):
    r = _base_result("cci_20", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= 100:
        zone = "upper_extreme"
    elif c >= 0:
        zone = "positive"
    elif c > -100:
        zone = "negative"
    else:
        zone = "lower_extreme"

    r["institutional_state"] = {
        "zone": zone,
        "zero_reference": 0.0,
        "upper_reference": 100.0,
        "lower_reference": -100.0,
        "path": r["structure"]["path"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def mom_10_engine(values, ohlcv=None):
    r = _base_result("mom_10", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "positive" if c is not None and c > 0
            else "negative" if c is not None and c < 0
            else "zero" if c is not None
            else "unknown"
        ),
        "momentum_path": r["structure"]["path"],
        "momentum_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def trix_18_engine(values, ohlcv=None):
    r = _base_result("trix_18", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "positive" if c is not None and c > 0
            else "negative" if c is not None and c < 0
            else "zero" if c is not None
            else "unknown"
        ),
        "trend_rate_path": r["structure"]["path"],
        "trend_rate_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def ppo_engine(values, ohlcv=None):
    r = _base_result("ppo", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "positive" if c is not None and c > 0
            else "negative" if c is not None and c < 0
            else "zero" if c is not None
            else "unknown"
        ),
        "percentage_momentum_path": r["structure"]["path"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def ppo_signal_engine(values, ohlcv=None):
    r = _base_result("ppo_signal", values, ohlcv)

    r["institutional_state"] = {
        "signal_path": r["structure"]["path"],
        "signal_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def ppo_hist_engine(values, ohlcv=None):
    r = _base_result("ppo_hist", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "zero_relation": (
            "positive" if c is not None and c > 0
            else "negative" if c is not None and c < 0
            else "zero" if c is not None
            else "unknown"
        ),
        "histogram_structure": (
            "expanding_positive"
            if len(x) >= 2 and x[-1] > 0 and abs(x[-1]) > abs(x[-2])
            else "contracting_positive"
            if len(x) >= 2 and x[-1] > 0
            else "expanding_negative"
            if len(x) >= 2 and x[-1] < 0 and abs(x[-1]) > abs(x[-2])
            else "contracting_negative"
            if len(x) >= 2 and x[-1] < 0
            else "unknown"
        ),
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def dpo_20_engine(values, ohlcv=None):
    r = _base_result("dpo_20", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    r["institutional_state"] = {
        "center_relation": (
            "above_center" if c is not None and c > 0
            else "below_center" if c is not None and c < 0
            else "at_center" if c is not None
            else "unknown"
        ),
        "cyclical_path": r["structure"]["path"],
        "cyclical_acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def stoch_k_engine(values, ohlcv=None):
    r = _base_result("stoch_k", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= 80:
        zone = "overbought"
    elif c <= 20:
        zone = "oversold"
    else:
        zone = "mid_range"

    r["institutional_state"] = {
        "zone": zone,
        "upper_reference": 80.0,
        "lower_reference": 20.0,
        "path": r["structure"]["path"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def stoch_d_engine(values, ohlcv=None):
    r = _base_result("stoch_d", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= 80:
        zone = "overbought"
    elif c <= 20:
        zone = "oversold"
    else:
        zone = "mid_range"

    r["institutional_state"] = {
        "zone": zone,
        "path": r["structure"]["path"],
        "acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def stoch_rsi_k_engine(values, ohlcv=None):
    r = _base_result("stoch_rsi_k", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= 80:
        zone = "overbought"
    elif c <= 20:
        zone = "oversold"
    else:
        zone = "mid_range"

    r["institutional_state"] = {
        "zone": zone,
        "path": r["structure"]["path"],
        "acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def stoch_rsi_d_engine(values, ohlcv=None):
    r = _base_result("stoch_rsi_d", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= 80:
        zone = "overbought"
    elif c <= 20:
        zone = "oversold"
    else:
        zone = "mid_range"

    r["institutional_state"] = {
        "zone": zone,
        "path": r["structure"]["path"],
        "acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def williams_r_engine(values, ohlcv=None):
    r = _base_result("williams_r", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= -20:
        zone = "upper_extreme"
    elif c <= -80:
        zone = "lower_extreme"
    else:
        zone = "mid_range"

    r["institutional_state"] = {
        "zone": zone,
        "upper_reference": -20.0,
        "lower_reference": -80.0,
        "path": r["structure"]["path"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


def ultimate_osc_engine(values, ohlcv=None):
    r = _base_result("ultimate_osc", values, ohlcv)

    x = _clean(values)
    c = x[-1] if x else None

    if c is None:
        zone = "unknown"
    elif c >= 70:
        zone = "upper_extreme"
    elif c <= 30:
        zone = "lower_extreme"
    else:
        zone = "mid_range"

    r["institutional_state"] = {
        "zone": zone,
        "upper_reference": 70.0,
        "lower_reference": 30.0,
        "path": r["structure"]["path"],
        "acceleration": r["structure"]["acceleration"],
        "price_relationship": r["price_comparison"]["type"],
    }

    return r


# ============================================================
# ENGINE REGISTRY
# ============================================================
#
# Registry does NOT normalize indicators.
# It only maps each indicator to its own independent engine.
# ============================================================

ENGINES = {
    "rsi_14": rsi_14_engine,
    "rsi_slope": rsi_slope_engine,

    "macd": macd_engine,
    "macd_signal": macd_signal_engine,
    "macd_hist": macd_hist_engine,

    "adx_14": adx_14_engine,
    "plus_di": plus_di_engine,
    "minus_di": minus_di_engine,
    "dx": dx_engine,

    "roc_12": roc_12_engine,
    "cci_20": cci_20_engine,
    "mom_10": mom_10_engine,
    "trix_18": trix_18_engine,

    "ppo": ppo_engine,
    "ppo_signal": ppo_signal_engine,
    "ppo_hist": ppo_hist_engine,

    "dpo_20": dpo_20_engine,

    "stoch_k": stoch_k_engine,
    "stoch_d": stoch_d_engine,

    "stoch_rsi_k": stoch_rsi_k_engine,
    "stoch_rsi_d": stoch_rsi_d_engine,

    "williams_r": williams_r_engine,
    "ultimate_osc": ultimate_osc_engine,
}


# ============================================================
# SINGLE ENGINE EXECUTION
# ============================================================

def run_indicator_engine(
    indicator: str,
    values: Sequence[Any],
    ohlcv: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    if indicator not in ENGINES:
        raise ValueError(
            f"Unknown indicator engine: {indicator}"
        )

    if len(values) < 50:
        raise ValueError(
            f"{indicator}: minimum 50 historical periods required"
        )

    engine = ENGINES[indicator]

    # Only the selected indicator series is passed.
    return engine(values, ohlcv)


# ============================================================
# RUN ALL 23 — STILL INDEPENDENT
# ============================================================

def run_all_indicator_engines(
    indicator_data: Dict[str, Sequence[Any]],
    ohlcv: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:

    missing = [
        name
        for name in INDICATOR_NAMES
        if name not in indicator_data
    ]

    if missing:
        raise ValueError(
            "Missing indicator inputs: "
            + ", ".join(missing)
        )

    output = {}

    for name in INDICATOR_NAMES:
        values = indicator_data[name]

        if len(values) < 50:
            raise ValueError(
                f"{name}: minimum 50 periods required"
            )

        # IMPORTANT:
        # Each engine gets ONLY its own indicator.
        output[name] = run_indicator_engine(
            name,
            values,
            ohlcv,
        )

    return output


# ============================================================
# COMPACT JSON-SAFE OUTPUT
# ============================================================

def compact_engine_output(result: Dict[str, Any]) -> Dict[str, Any]:

    trace = result.get("trace", {})

    return {
        "indicator": result.get("indicator"),

        "period": trace.get("period", []),

        "value": trace.get("value", []),

        "delta": trace.get("delta", []),

        "velocity": trace.get("velocity", []),

        "acceleration": trace.get(
            "acceleration",
            [],
        ),

        "price": trace.get(
            "price",
            {},
        ),

        "relationship": trace.get(
            "relationship",
            [],
        ),

        "institutional_state": result.get(
            "institutional_state",
            {},
        ),

        "structure": result.get(
            "structure",
            {},
        ),

        "price_comparison": result.get(
            "price_comparison",
            {},
        ),
    }


__all__ = [
    "INDICATOR_NAMES",
    "ENGINES",
    "run_indicator_engine",
    "run_all_indicator_engines",
    "compact_engine_output",
]
