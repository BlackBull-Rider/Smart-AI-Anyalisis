from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping
import time


# ============================================================
# CONTRACT
# ============================================================

ANALYZER_NAMES = (
    "momentum",
    "candle",
    "volume",
    "volatility",
    "support_resistance",
    "pattern",
    "trend",
    "market_structure",
    "liquidity",
)


VALID_STATUS = {
    "ready",
    "insufficient_data",
    "error",
    "invalid",
}

VALID_DIRECTION = {
    "bullish",
    "bearish",
    "neutral",
    "mixed",
}

VALID_STRENGTH = {
    "very_weak",
    "weak",
    "moderate",
    "strong",
    "very_strong",
}


# ============================================================
# ANALYZER REGISTRY
# ============================================================

@dataclass(frozen=True)
class AnalyzerSpec:
    name: str
    runner: Callable[[Any], Dict[str, Any]]


# ------------------------------------------------------------
# এখানে actual analyzer imports বসবে
# ------------------------------------------------------------

from backend.analyzers.momentum import analyze as momentum_analyzer
from backend.analyzers.candle import analyze as candle_analyzer
from backend.analyzers.volume import analyze as volume_analyzer
from backend.analyzers.volatility import analyze as volatility_analyzer
from backend.analyzers.support_resistance import analyze as sr_analyzer
from backend.analyzers.pattern import analyze as pattern_analyzer
from backend.analyzers.trend import analyze as trend_analyzer
from backend.analyzers.market_structure import analyze as structure_analyzer
from backend.analyzers.liquidity import analyze as liquidity_analyzer


ANALYZERS = (
    AnalyzerSpec("momentum", momentum_analyzer),
    AnalyzerSpec("candle", candle_analyzer),
    AnalyzerSpec("volume", volume_analyzer),
    AnalyzerSpec("volatility", volatility_analyzer),
    AnalyzerSpec("support_resistance", sr_analyzer),
    AnalyzerSpec("pattern", pattern_analyzer),
    AnalyzerSpec("trend", trend_analyzer),
    AnalyzerSpec("market_structure", structure_analyzer),
    AnalyzerSpec("liquidity", liquidity_analyzer),
)


# ============================================================
# VALIDATION
# ============================================================

def _validate_number(
    value: Any,
    field: str,
) -> None:

    if value is None:
        return

    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field}: expected numeric value"
        )

    if isinstance(value, float):
        if value != value or value in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError(
                f"{field}: non-finite number"
            )


def validate_analyzer_output(
    name: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:

    if not isinstance(result, Mapping):
        raise ValueError(
            f"{name}: analyzer output must be JSON object"
        )

    result = dict(result)

    status = result.get("status")

    if status not in VALID_STATUS:
        raise ValueError(
            f"{name}: invalid status={status!r}"
        )

    direction = result.get("direction")

    if direction is not None:
        if direction not in VALID_DIRECTION:
            raise ValueError(
                f"{name}: invalid direction={direction!r}"
            )

    strength = result.get("strength")

    if strength is not None:
        if strength not in VALID_STRENGTH:
            raise ValueError(
                f"{name}: invalid strength={strength!r}"
            )

    confidence = result.get("confidence")

    _validate_number(
        confidence,
        f"{name}.confidence",
    )

    if confidence is not None:
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(
                f"{name}.confidence must be between 0 and 1"
            )

    for key in (
        "evidence",
        "levels",
        "zones",
        "relationships",
    ):
        if key in result and result[key] is not None:
            if not isinstance(result[key], list):
                raise ValueError(
                    f"{name}.{key}: expected list"
                )

    return result


# ============================================================
# ANALYZER EXECUTION
# ============================================================

def run_single_analyzer(
    spec: AnalyzerSpec,
    feature_df: Any,
) -> Dict[str, Any]:

    started = time.perf_counter()

    try:

        raw = spec.runner(feature_df)

        result = validate_analyzer_output(
            spec.name,
            raw,
        )

        elapsed = time.perf_counter() - started

        return {
            "name": spec.name,
            "status": "ready",
            "elapsed_ms": round(
                elapsed * 1000,
                3,
            ),
            "output": result,
        }

    except Exception as exc:

        elapsed = time.perf_counter() - started

        return {
            "name": spec.name,
            "status": "error",
            "elapsed_ms": round(
                elapsed * 1000,
                3,
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "output": None,
        }


# ============================================================
# CONSENSUS
# ============================================================

def build_consensus(
    outputs: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:

    bullish = 0
    bearish = 0
    neutral = 0
    mixed = 0
    valid = 0

    for result in outputs.values():

        if result.get("status") != "ready":
            continue

        data = result.get("output") or {}

        direction = data.get("direction")

        if direction == "bullish":
            bullish += 1
        elif direction == "bearish":
            bearish += 1
        elif direction == "neutral":
            neutral += 1
        elif direction == "mixed":
            mixed += 1

        valid += 1

    if valid == 0:
        direction = "mixed"
        agreement = 0.0

    else:

        counts = {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "mixed": mixed,
        }

        direction = max(
            counts,
            key=counts.get,
        )

        agreement = (
            counts[direction] / valid
        )

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "mixed": mixed,
        "valid_analyzers": valid,
        "agreement": round(
            agreement,
            4,
        ),
        "direction": direction,
    }


# ============================================================
# CONFLUENCE
# ============================================================

def build_confluence(
    outputs: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:

    bullish = []
    bearish = []
    neutral = []
    conflicts = []

    for name, result in outputs.items():

        if result.get("status") != "ready":
            continue

        data = result.get("output") or {}

        direction = data.get(
            "direction"
        )

        evidence = data.get(
            "evidence",
            [],
        )

        item = {
            "analyzer": name,
            "strength": data.get("strength"),
            "confidence": data.get("confidence"),
            "evidence": evidence,
        }

        if direction == "bullish":
            bullish.append(item)

        elif direction == "bearish":
            bearish.append(item)

        elif direction == "neutral":
            neutral.append(item)

    if bullish and bearish:
        conflicts.append({
            "type": "directional_conflict",
            "bullish_analyzers": [
                x["analyzer"]
                for x in bullish
            ],
            "bearish_analyzers": [
                x["analyzer"]
                for x in bearish
            ],
        })

    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "conflicts": conflicts,
    }


# ============================================================
# LEVEL AGGREGATION
# ============================================================

def collect_levels(
    outputs: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:

    support = []
    resistance = []
    breakout = []
    invalidation = []
    target = []

    for name, result in outputs.items():

        if result.get("status") != "ready":
            continue

        data = result.get("output") or {}

        levels = data.get(
            "levels",
            [],
        )

        for level in levels:

            if not isinstance(level, Mapping):
                continue

            item = dict(level)

            item["source"] = name

            level_type = item.get(
                "type"
            )

            if level_type == "support":
                support.append(item)

            elif level_type == "resistance":
                resistance.append(item)

            elif level_type == "breakout":
                breakout.append(item)

            elif level_type == "invalidation":
                invalidation.append(item)

            elif level_type == "target":
                target.append(item)

    return {
        "support": support,
        "resistance": resistance,
        "breakout": breakout,
        "invalidation": invalidation,
        "target": target,
    }


# ============================================================
# FINAL ENGINE
# ============================================================

def run_analyzer_engine(
    feature_df: Any,
    *,
    symbol: str,
    timeframe: str,
    as_of: str,
    event_id: str,
) -> Dict[str, Any]:

    started = time.perf_counter()

    if feature_df is None:
        raise ValueError(
            "feature_df cannot be None"
        )

    outputs: Dict[
        str,
        Dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # RUN ALL 9 ANALYZERS
    # --------------------------------------------------------

    for spec in ANALYZERS:

        result = run_single_analyzer(
            spec,
            feature_df,
        )

        outputs[
            spec.name
        ] = result

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    ready = sum(
        1
        for x in outputs.values()
        if x["status"] == "ready"
    )

    failed = (
        len(outputs) - ready
    )

    if failed == 0:
        engine_status = "complete"

    elif ready > 0:
        engine_status = "partial"

    else:
        engine_status = "failed"

    # --------------------------------------------------------
    # SYNTHESIS
    # --------------------------------------------------------

    clean_outputs = {
        name: item
        for name, item in outputs.items()
    }

    consensus = build_consensus(
        clean_outputs
    )

    confluence = build_confluence(
        clean_outputs
    )

    levels = collect_levels(
        clean_outputs
    )

    elapsed = (
        time.perf_counter() - started
    )

    # --------------------------------------------------------
    # FINAL JSON
    # --------------------------------------------------------

    return {
        "schema_version":
            "analyzer_engine_output.v1",

        "event_id":
            event_id,

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "as_of":
            as_of,

        "engine_status":
            engine_status,

        "execution": {
            "total_analyzers":
                len(ANALYZERS),

            "ready":
                ready,

            "failed":
                failed,

            "elapsed_ms":
                round(
                    elapsed * 1000,
                    3,
                ),
        },

        "consensus":
            consensus,

        "confluence":
            confluence,

        "key_levels":
            levels,

        "analyzers":
            {
                name: item
                for name, item
                in outputs.items()
            },
    }
