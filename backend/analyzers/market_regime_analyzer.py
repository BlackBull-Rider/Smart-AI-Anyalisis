from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_LR = 0.15
MAX_LR = 8.0


FIELD_CONTRACTS: Dict[str, List[Tuple[str, float]]] = {
    "close": [
        ("close", 1.0), ("adj_close", 1.0), ("adjusted_close", 1.0),
        ("cmp", 1.0), ("price", 1.0), ("last_price", 1.0),
    ],
    "open": [("open", 1.0)],
    "high": [("high", 1.0)],
    "low": [("low", 1.0)],
    "volume": [("volume", 1.0), ("total_volume", 1.0)],
    "avg_volume": [
        ("avg_volume", 1.0), ("average_volume", 1.0),
        ("volume_average", 1.0), ("volume_avg", 1.0),
    ],
    "ema20": [("ema20", 1.0), ("ema_20", 1.0), ("ema20_value", 1.0)],
    "ema50": [("ema50", 1.0), ("ema_50", 1.0), ("ema50_value", 1.0)],
    "ema100": [("ema100", 1.0), ("ema_100", 1.0), ("ema100_value", 1.0)],
    "ema200": [("ema200", 1.0), ("ema_200", 1.0), ("ema200_value", 1.0)],
    "sma20": [("sma20", 1.0), ("sma_20", 1.0)],
    "sma50": [("sma50", 1.0), ("sma_50", 1.0)],
    "sma200": [("sma200", 1.0), ("sma_200", 1.0)],
    "rsi": [("rsi", 1.0), ("rsi14", 1.0), ("rsi_14", 1.0)],
    "macd": [("macd", 1.0), ("macd_line", 1.0)],
    "macd_signal": [
        ("macd_signal", 1.0),
        ("macd_signal_line", 1.0),
        ("signal_line", 1.0),
    ],
    "macd_hist": [("macd_hist", 1.0), ("macd_histogram", 1.0)],
    "adx": [("adx", 1.0), ("adx14", 1.0), ("adx_14", 1.0)],
    "atr": [("atr", 1.0), ("atr14", 1.0), ("atr_14", 1.0)],
    "atr_pct": [
        ("atr_pct", 1.0),
        ("atr_percent", 1.0),
        ("atr_percentage", 1.0),
    ],
    "volatility": [
        ("volatility", 1.0),
        ("historical_volatility", 1.0),
        ("realized_volatility", 1.0),
    ],
    "return_1d": [
        ("return_1d", 1.0),
        ("daily_return", 1.0),
        ("return_daily", 1.0),
        ("pct_change_1d", 1.0),
    ],
    "return_5d": [
        ("return_5d", 1.0),
        ("return_5_day", 1.0),
        ("weekly_return", 1.0),
        ("pct_change_5d", 1.0),
    ],
    "return_20d": [
        ("return_20d", 1.0),
        ("return_20_day", 1.0),
        ("monthly_return", 1.0),
        ("pct_change_20d", 1.0),
    ],
    "return_60d": [
        ("return_60d", 1.0),
        ("return_60_day", 1.0),
        ("quarterly_return", 1.0),
        ("pct_change_60d", 1.0),
    ],
    "advance_decline": [
        ("advance_decline", 1.0),
        ("advance_decline_ratio", 1.0),
        ("ad_ratio", 1.0),
    ],
    "breadth": [
        ("breadth", 1.0),
        ("market_breadth", 1.0),
        ("breadth_ratio", 1.0),
    ],
    "advancers": [
        ("advancers", 1.0),
        ("advancing_stocks", 1.0),
    ],
    "decliners": [
        ("decliners", 1.0),
        ("declining_stocks", 1.0),
    ],
    "new_highs": [
        ("new_highs", 1.0),
        ("new_52w_highs", 1.0),
        ("new_high_count", 1.0),
    ],
    "new_lows": [
        ("new_lows", 1.0),
        ("new_52w_lows", 1.0),
        ("new_low_count", 1.0),
    ],
    "beta": [("beta", 1.0), ("market_beta", 1.0)],
    "drawdown": [
        ("drawdown", 1.0),
        ("max_drawdown", 1.0),
        ("current_drawdown", 1.0),
    ],
    "vix": [
        ("vix", 1.0),
        ("india_vix", 1.0),
        ("market_vix", 1.0),
    ],
    "put_call_ratio": [
        ("put_call_ratio", 1.0),
        ("pcr", 1.0),
        ("oi_pcr", 1.0),
    ],
    "institutional_flow": [
        ("institutional_flow", 1.0),
        ("fii_flow", 1.0),
        ("fii_net_flow", 1.0),
        ("dii_flow", 1.0),
        ("institutional_net_flow", 1.0),
    ],
    "market_return": [
        ("market_return", 1.0),
        ("index_return", 1.0),
        ("nifty_return", 1.0),
    ],
}


DOMAIN_FEATURES: Dict[str, List[str]] = {
    "trend": [
        "close", "ema20", "ema50", "ema100", "ema200",
        "sma20", "sma50", "sma200", "adx",
        "return_20d", "return_60d",
    ],
    "momentum": [
        "rsi", "macd", "macd_signal", "macd_hist",
        "return_1d", "return_5d", "return_20d",
    ],
    "volatility": [
        "atr", "atr_pct", "volatility", "vix", "drawdown",
    ],
    "participation": [
        "volume", "avg_volume", "breadth", "advance_decline",
        "advancers", "decliners", "new_highs", "new_lows",
        "institutional_flow",
    ],
    "risk": [
        "beta", "drawdown", "vix", "put_call_ratio", "volatility",
    ],
}


DATE_ALIASES = (
    "date",
    "datetime",
    "timestamp",
    "time",
    "trade_date",
    "trading_date",
    "price_date",
    "as_of_date",
    "asof",
)


@dataclass
class EvidenceNode:
    domain: str
    direction: float
    magnitude: float
    message: str


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    try:
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("%", "")
            if not text:
                return None
            value = float(text)

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    return _num(value)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    value = _clip(value, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-value))


def _score(
    value: float,
    neutral: float,
    scale: float,
    invert: bool = False,
) -> float:
    if scale <= 0:
        return 50.0

    z = (value - neutral) / scale

    if invert:
        z = -z

    return _sigmoid(z) * 100.0


def _likelihood_ratio(magnitude: float, direction: float) -> float:
    base = 1.0 + (_clip(magnitude, 0.0, 1.0) * 3.0)

    if direction >= 0:
        ratio = base
    else:
        ratio = 1.0 / base

    return _clip(ratio, MIN_LR, MAX_LR)


def _parse_time(value: Any) -> float:
    if value is None:
        return 0.0

    numeric = _num(value)

    if numeric is not None:
        return numeric

    text = str(value).strip()

    if not text:
        return 0.0

    match = re.match(
        r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})",
        text,
    )

    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return float(year * 10000 + month * 100 + day)
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def _normalize_key(key: Any) -> str:
    text = str(key).strip().lower()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


class MarketRegimeAnalyzer:
    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    def analyze(
        self,
        data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        try:
            snapshots = self._parse_payload_chronologically(data)

            if not snapshots:
                return self._empty()

            feature_history = self._extract_feature_history(snapshots)

            latest_values, feature_trace = self._build_latest_features(
                snapshots,
                feature_history,
            )

            coverage = self._build_feature_coverage(
                latest_values,
                feature_trace,
            )

            trend = self._analyze_trend(
                latest_values,
                feature_history,
            )

            momentum = self._analyze_momentum(
                latest_values,
                feature_history,
            )

            volatility = self._analyze_volatility(
                latest_values,
                feature_history,
            )

            participation = self._analyze_participation(
                latest_values,
                feature_history,
            )

            risk = self._analyze_risk(
                latest_values,
                feature_history,
            )

            transition = self._analyze_transition(
                latest_values,
                feature_history,
            )

            regime = self._detect_regime(
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                participation=participation,
                risk=risk,
                transition=transition,
            )

            evidence_nodes: List[EvidenceNode] = []

            evidence_nodes.extend(trend["evidence"])
            evidence_nodes.extend(momentum["evidence"])
            evidence_nodes.extend(volatility["evidence"])
            evidence_nodes.extend(participation["evidence"])
            evidence_nodes.extend(risk["evidence"])
            evidence_nodes.extend(transition["evidence"])
            evidence_nodes.extend(regime["evidence"])

            evidence = self._format_evidence(evidence_nodes)

            confidence = self._calculate_confidence(
                snapshots=snapshots,
                coverage=coverage,
                trend=trend,
                momentum=momentum,
                volatility=volatility,
                participation=participation,
                risk=risk,
                transition=transition,
                evidence=evidence,
            )

            return {
                "market_regime_analyzer": {
                    "confidence": self._round(confidence),
                    "regime": self._round(regime["regime"]),
                    "regime_status": regime["regime_status"],
                    "regime_score": self._round(regime["regime_score"]),
                    "trend": self._round(trend["value"]),
                    "trend_strength": self._round(trend["strength"]),
                    "volatility": self._round(volatility["value"]),
                    "volatility_status": volatility["status"],
                    "momentum": self._round(momentum["value"]),
                    "momentum_status": momentum["status"],
                    "participation": self._round(participation["value"]),
                    "participation_status": participation["status"],
                    "risk": self._round(risk["value"]),
                    "risk_status": risk["status"],
                    "transition": self._round(transition["value"]),
                    "transition_status": transition["status"],
                    "feature_coverage_pct": self._round(
                        coverage["coverage_pct"]
                    ),
                    "feature_trace_summary": coverage["summary"],
                    "feature_trace": feature_trace,
                    "domain_coverage": coverage["domain_coverage"],
                    "evidence": evidence[:12],
                }
            }

        except Exception as exc:
            self.logger.exception(
                "Market Regime Analyzer Critical Failure: %s",
                exc,
            )

            result = self._empty()

            result["market_regime_analyzer"]["evidence"].append(
                {
                    "category": "Regime",
                    "domain": "system",
                    "message": "Execution encountered a critical failure.",
                    "reliability": 0.96,
                    "likelihood_ratio": 0.42,
                }
            )

            return result

    def _empty(self) -> Dict[str, Any]:
        feature_names = list(FIELD_CONTRACTS.keys())

        return {
            "market_regime_analyzer": {
                "confidence": 0.0,
                "regime": None,
                "regime_status": "insufficient_data",
                "regime_score": None,
                "trend": None,
                "trend_strength": None,
                "volatility": None,
                "volatility_status": "unknown",
                "momentum": None,
                "momentum_status": "unknown",
                "participation": None,
                "participation_status": "unknown",
                "risk": None,
                "risk_status": "unknown",
                "transition": None,
                "transition_status": "unknown",
                "feature_coverage_pct": 0.0,
                "feature_trace_summary": {
                    "total_features": len(feature_names),
                    "used_features": 0,
                    "available_not_used": 0,
                    "missing_features": len(feature_names),
                    "invalid_features": 0,
                    "coverage_pct": 0.0,
                    "used_feature_names": [],
                    "available_not_used_names": [],
                    "missing_feature_names": feature_names,
                    "invalid_feature_names": [],
                },
                "feature_trace": {},
                "domain_coverage": {
                    domain: {
                        "total_features": len(features),
                        "used_features": 0,
                        "coverage_pct": 0.0,
                        "used_feature_names": [],
                        "missing_feature_names": list(features),
                    }
                    for domain, features in DOMAIN_FEATURES.items()
                },
                "evidence": [],
            }
        }

    def _parse_payload_chronologically(
        self,
        data: Any,
    ) -> List[Dict[str, Any]]:

        raw_rows: List[Dict[str, Any]] = []

        if data is None:
            return []

        if hasattr(data, "to_dict") and hasattr(data, "columns"):
            try:
                rows = data.to_dict(orient="records")

                if isinstance(rows, list):
                    raw_rows.extend(
                        row for row in rows
                        if isinstance(row, Mapping)
                    )
            except Exception:
                pass

        elif isinstance(data, Mapping):

            nested_keys = (
                "features",
                "historical_data",
                "historical",
                "market_history",
                "market_data",
                "regime_history",
                "rows",
                "records",
                "data",
                "items",
            )

            found = False

            for key in nested_keys:
                value = data.get(key)

                if isinstance(value, list):
                    raw_rows.extend(
                        row for row in value
                        if isinstance(row, Mapping)
                    )
                    found = True

                elif (
                    isinstance(value, Mapping)
                    and self._looks_like_snapshot(value)
                ):
                    raw_rows.append(dict(value))
                    found = True

            if not found and self._looks_like_snapshot(data):
                raw_rows.append(dict(data))

        elif isinstance(data, Sequence) and not isinstance(
            data,
            (str, bytes, bytearray),
        ):
            raw_rows.extend(
                dict(row)
                for row in data
                if isinstance(row, Mapping)
            )

        extracted: List[Dict[str, Any]] = []

        for index, row in enumerate(raw_rows):
            normalized: Dict[str, Any] = {}

            for raw_key, value in row.items():
                key = _normalize_key(raw_key)
                normalized[key] = value

            timestamp = self._extract_timestamp(normalized)

            normalized["_t"] = timestamp
            normalized["_idx"] = index

            if len(normalized) > 2:
                extracted.append(normalized)

        extracted.sort(
            key=lambda row: (
                row.get("_t", 0.0),
                row.get("_idx", 0),
            )
        )

        return extracted

    def _looks_like_snapshot(
        self,
        value: Mapping[str, Any],
    ) -> bool:

        normalized_keys = {
            _normalize_key(key)
            for key in value.keys()
        }

        known = set()

        for aliases in FIELD_CONTRACTS.values():
            known.update(alias for alias, _ in aliases)

        return bool(normalized_keys.intersection(known))

    def _extract_timestamp(
        self,
        row: Mapping[str, Any],
    ) -> float:

        for alias in DATE_ALIASES:
            key = _normalize_key(alias)

            if key in row:
                parsed = _parse_time(row[key])

                if parsed > 0:
                    return parsed

        return 0.0

    def _extract_feature_history(
        self,
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, List[Tuple[float, float]]]:

        history: Dict[str, List[Tuple[float, float]]] = {}

        for feature, aliases in FIELD_CONTRACTS.items():

            series: List[Tuple[float, float]] = []

            for snapshot in snapshots:

                value = None

                for alias, multiplier in aliases:

                    if alias not in snapshot:
                        continue

                    raw = _num(snapshot.get(alias))

                    if raw is None:
                        continue

                    normalized = raw * multiplier

                    if not math.isfinite(normalized):
                        continue

                    value = normalized
                    break

                if value is not None:
                    series.append(
                        (
                            float(snapshot.get("_t", 0.0)),
                            float(value),
                        )
                    )

            if series:
                history[feature] = series

        return history

    def _build_latest_features(
        self,
        snapshots: List[Dict[str, Any]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Tuple[
        Dict[str, Optional[float]],
        Dict[str, Dict[str, Any]],
    ]:

        latest: Dict[str, Optional[float]] = {}
        traces: Dict[str, Dict[str, Any]] = {}

        for feature, aliases in FIELD_CONTRACTS.items():

            series = history.get(feature, [])

            latest_value = (
                series[-1][1]
                if series
                else None
            )

            latest[feature] = latest_value

            aliases_checked = [
                alias
                for alias, _ in aliases
            ]

            source_field = None
            raw_value = None
            multiplier = None
            invalid = False

            for snapshot in reversed(snapshots):

                for alias, mult in aliases:

                    if alias not in snapshot:
                        continue

                    raw = _num(snapshot.get(alias))

                    if raw is None:
                        invalid = True
                        continue

                    normalized = raw * mult

                    if not math.isfinite(normalized):
                        invalid = True
                        continue

                    source_field = alias
                    raw_value = raw
                    multiplier = mult

                    break

                if source_field is not None:
                    break

            if latest_value is not None:

                traces[feature] = {
                    "status": "used",
                    "feature": feature,
                    "source_field": source_field,
                    "raw_value": self._json_value(raw_value),
                    "multiplier": self._json_value(multiplier),
                    "normalized_value": self._round(latest_value),
                    "historical_points": len(series),
                    "latest_timestamp": self._round(
                        series[-1][0]
                    ),
                    "aliases_checked": aliases_checked,
                }

            elif invalid:

                traces[feature] = {
                    "status": "invalid",
                    "feature": feature,
                    "source_field": None,
                    "raw_value": None,
                    "multiplier": None,
                    "normalized_value": None,
                    "historical_points": 0,
                    "latest_timestamp": None,
                    "aliases_checked": aliases_checked,
                }

            else:

                traces[feature] = {
                    "status": "missing",
                    "feature": feature,
                    "source_field": None,
                    "raw_value": None,
                    "multiplier": None,
                    "normalized_value": None,
                    "historical_points": 0,
                    "latest_timestamp": None,
                    "aliases_checked": aliases_checked,
                }

        return latest, traces

    def _build_feature_coverage(
        self,
        latest: Dict[str, Optional[float]],
        traces: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        total = len(FIELD_CONTRACTS)

        used_names = [
            feature
            for feature in FIELD_CONTRACTS
            if traces.get(feature, {}).get("status") == "used"
        ]

        invalid_names = [
            feature
            for feature in FIELD_CONTRACTS
            if traces.get(feature, {}).get("status") == "invalid"
        ]

        missing_names = [
            feature
            for feature in FIELD_CONTRACTS
            if traces.get(feature, {}).get("status") == "missing"
        ]

        available_not_used_names: List[str] = []

        for feature, value in latest.items():
            if value is not None and feature not in used_names:
                available_not_used_names.append(feature)

        coverage_pct = (
            (len(used_names) / total) * 100.0
            if total
            else 0.0
        )

        domain_coverage: Dict[str, Any] = {}

        for domain, features in DOMAIN_FEATURES.items():

            used = [
                feature
                for feature in features
                if feature in used_names
            ]

            missing = [
                feature
                for feature in features
                if feature not in used_names
            ]

            domain_total = len(features)

            domain_coverage[domain] = {
                "total_features": domain_total,
                "used_features": len(used),
                "coverage_pct": self._round(
                    (len(used) / domain_total) * 100.0
                    if domain_total
                    else 0.0
                ),
                "used_feature_names": used,
                "missing_feature_names": missing,
            }

        return {
            "coverage_pct": coverage_pct,
            "summary": {
                "total_features": total,
                "used_features": len(used_names),
                "available_not_used": len(
                    available_not_used_names
                ),
                "missing_features": len(missing_names),
                "invalid_features": len(invalid_names),
                "coverage_pct": coverage_pct,
                "used_feature_names": used_names,
                "available_not_used_names": available_not_used_names,
                "missing_feature_names": missing_names,
                "invalid_feature_names": invalid_names,
            },
            "domain_coverage": domain_coverage,
        }

    def _analyze_trend(
        self,
        latest: Dict[str, Optional[float]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:

        components: List[float] = []
        evidence: List[EvidenceNode] = []

        close = latest.get("close")

        ema20 = latest.get("ema20")
        ema50 = latest.get("ema50")
        ema100 = latest.get("ema100")
        ema200 = latest.get("ema200")

        if close is not None and ema20 is not None:
            score = 60.0 if close > ema20 else 40.0
            components.append(score)

            evidence.append(
                EvidenceNode(
                    "trend",
                    1.0 if score > 50 else -1.0,
                    0.35,
                    "Price is above EMA20."
                    if score > 50
                    else "Price is below EMA20.",
                )
            )

        if ema20 is not None and ema50 is not None:
            score = 65.0 if ema20 > ema50 else 35.0
            components.append(score)

            evidence.append(
                EvidenceNode(
                    "trend",
                    1.0 if score > 50 else -1.0,
                    0.40,
                    "EMA20 is above EMA50."
                    if score > 50
                    else "EMA20 is below EMA50.",
                )
            )

        if ema50 is not None and ema100 is not None:
            components.append(
                65.0 if ema50 > ema100 else 35.0
            )

        if ema100 is not None and ema200 is not None:
            components.append(
                70.0 if ema100 > ema200 else 30.0
            )

        if (
            ema50 is not None
            and ema200 is not None
            and close is not None
        ):
            aligned_bull = close > ema50 > ema200
            aligned_bear = close < ema50 < ema200

            if aligned_bull:
                components.append(80.0)

                evidence.append(
                    EvidenceNode(
                        "trend",
                        1.0,
                        0.75,
                        "Price and medium/long-term averages show bullish alignment.",
                    )
                )

            elif aligned_bear:
                components.append(20.0)

                evidence.append(
                    EvidenceNode(
                        "trend",
                        -1.0,
                        0.75,
                        "Price and medium/long-term averages show bearish alignment.",
                    )
                )

        adx = latest.get("adx")

        if adx is not None:
            adx_strength = _clip(
                (adx - 10.0) / 30.0,
                0.0,
                1.0,
            )
        else:
            adx_strength = None

        return20 = latest.get("return_20d")
        return60 = latest.get("return_60d")

        if return20 is not None:
            components.append(
                _score(
                    return20,
                    neutral=0.0,
                    scale=8.0,
                )
            )

        if return60 is not None:
            components.append(
                _score(
                    return60,
                    neutral=0.0,
                    scale=15.0,
                )
            )

        if not components:
            return {
                "value": None,
                "strength": None,
                "status": "unknown",
                "evidence": [],
            }

        value = sum(components) / len(components)

        if adx_strength is not None:
            strength = adx_strength * 100.0
        else:
            distance = abs(value - 50.0) * 2.0
            strength = _clip(distance, 0.0, 100.0)

        if value >= 65.0:
            status = "bullish"
        elif value <= 35.0:
            status = "bearish"
        else:
            status = "sideways"

        return {
            "value": value,
            "strength": strength,
            "status": status,
            "evidence": evidence,
        }

    def _analyze_momentum(
        self,
        latest: Dict[str, Optional[float]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:

        components: List[float] = []
        evidence: List[EvidenceNode] = []

        rsi = latest.get("rsi")

        if rsi is not None:
            rsi_score = _score(
                rsi,
                neutral=50.0,
                scale=15.0,
            )

            components.append(rsi_score)

            if rsi >= 60.0:
                evidence.append(
                    EvidenceNode(
                        "momentum",
                        1.0,
                        0.55,
                        f"RSI momentum is positive ({rsi:.1f}).",
                    )
                )

            elif rsi <= 40.0:
                evidence.append(
                    EvidenceNode(
                        "momentum",
                        -1.0,
                        0.55,
                        f"RSI momentum is negative ({rsi:.1f}).",
                    )
                )

        macd = latest.get("macd")
        signal = latest.get("macd_signal")
        hist = latest.get("macd_hist")

        if macd is not None and signal is not None:
            spread = macd - signal

            scale = max(
                abs(macd),
                abs(signal),
                1.0,
            )

            components.append(
                _score(
                    spread / scale,
                    neutral=0.0,
                    scale=0.2,
                )
            )

        if hist is not None:
            components.append(
                _score(
                    hist,
                    neutral=0.0,
                    scale=max(abs(hist), 1.0),
                )
            )

        for key in (
            "return_1d",
            "return_5d",
            "return_20d",
        ):
            value = latest.get(key)

            if value is not None:
                components.append(
                    _score(
                        value,
                        neutral=0.0,
                        scale=8.0
                        if key != "return_20d"
                        else 12.0,
                    )
                )

        if not components:
            return {
                "value": None,
                "status": "unknown",
                "evidence": [],
            }

        value = sum(components) / len(components)

        if value >= 65.0:
            status = "positive"
        elif value <= 35.0:
            status = "negative"
        else:
            status = "neutral"

        return {
            "value": value,
            "status": status,
            "evidence": evidence,
        }

    def _analyze_volatility(
        self,
        latest: Dict[str, Optional[float]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:

        values: List[float] = []
        evidence: List[EvidenceNode] = []

        atr_pct = latest.get("atr_pct")

        if atr_pct is not None:
            values.append(
                _clip(
                    atr_pct,
                    0.0,
                    100.0,
                )
            )

        volatility = latest.get("volatility")

        if volatility is not None:
            values.append(
                _clip(
                    volatility,
                    0.0,
                    200.0,
                )
            )

        vix = latest.get("vix")

        if vix is not None:
            values.append(
                _clip(
                    vix,
                    0.0,
                    200.0,
                )
            )

        atr = latest.get("atr")
        close = latest.get("close")

        if (
            atr is not None
            and close is not None
            and close > 0
            and atr_pct is None
        ):
            values.append(
                _clip(
                    (atr / close) * 100.0,
                    0.0,
                    100.0,
                )
            )

        if not values:
            return {
                "value": None,
                "status": "unknown",
                "evidence": [],
            }

        raw = sum(values) / len(values)

        normalized = _clip(
            raw * 5.0,
            0.0,
            100.0,
        )

        if normalized < 30.0:
            status = "low"

            evidence.append(
                EvidenceNode(
                    "volatility",
                    1.0,
                    0.35,
                    "Volatility conditions are relatively contained.",
                )
            )

        elif normalized > 70.0:
            status = "high"

            evidence.append(
                EvidenceNode(
                    "volatility",
                    -1.0,
                    0.65,
                    "Volatility conditions are elevated.",
                )
            )

        else:
            status = "normal"

        return {
            "value": normalized,
            "status": status,
            "evidence": evidence,
        }

    def _analyze_participation(
        self,
        latest: Dict[str, Optional[float]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:

        components: List[float] = []
        evidence: List[EvidenceNode] = []

        breadth = latest.get("breadth")

        if breadth is not None:
            if 0.0 <= breadth <= 1.0:
                breadth *= 100.0

            components.append(
                _clip(breadth, 0.0, 100.0)
            )

        ad = latest.get("advance_decline")

        if ad is not None:
            components.append(
                _score(
                    ad,
                    neutral=1.0,
                    scale=0.5,
                )
            )

        advancers = latest.get("advancers")
        decliners = latest.get("decliners")

        if (
            advancers is not None
            and decliners is not None
            and advancers >= 0
            and decliners >= 0
        ):
            total = advancers + decliners

            if total > 0:
                components.append(
                    (advancers / total) * 100.0
                )

        volume = latest.get("volume")
        avg_volume = latest.get("avg_volume")

        if (
            volume is not None
            and avg_volume is not None
            and avg_volume > 0
        ):
            volume_ratio = volume / avg_volume

            components.append(
                _score(
                    volume_ratio,
                    neutral=1.0,
                    scale=0.4,
                )
            )

            if volume_ratio >= 1.5:
                evidence.append(
                    EvidenceNode(
                        "participation",
                        1.0,
                        0.45,
                        "Trading volume is materially above its reference average.",
                    )
                )

        new_highs = latest.get("new_highs")
        new_lows = latest.get("new_lows")

        if (
            new_highs is not None
            and new_lows is not None
            and new_highs >= 0
            and new_lows >= 0
        ):
            total_extremes = new_highs + new_lows

            if total_extremes > 0:
                components.append(
                    (new_highs / total_extremes) * 100.0
                )

        institutional = latest.get("institutional_flow")

        if institutional is not None:
            components.append(
                _score(
                    institutional,
                    neutral=0.0,
                    scale=max(abs(institutional), 1.0),
                )
            )

        if not components:
            return {
                "value": None,
                "status": "unknown",
                "evidence": [],
            }

        value = sum(components) / len(components)

        if value >= 65.0:
            status = "broad"

            evidence.append(
                EvidenceNode(
                    "participation",
                    1.0,
                    0.55,
                    "Market participation is supportive.",
                )
            )

        elif value <= 35.0:
            status = "weak"

            evidence.append(
                EvidenceNode(
                    "participation",
                    -1.0,
                    0.55,
                    "Market participation is weak.",
                )
            )

        else:
            status = "mixed"

        return {
            "value": value,
            "status": status,
            "evidence": evidence,
        }

    def _analyze_risk(
        self,
        latest: Dict[str, Optional[float]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:

        components: List[float] = []
        evidence: List[EvidenceNode] = []

        beta = latest.get("beta")

        if beta is not None:
            components.append(
                _score(
                    beta,
                    neutral=1.0,
                    scale=0.5,
                    invert=True,
                )
            )

        drawdown = latest.get("drawdown")

        if drawdown is not None:
            dd = abs(drawdown)

            components.append(
                _score(
                    dd,
                    neutral=10.0,
                    scale=8.0,
                    invert=True,
                )
            )

            if dd >= 20.0:
                evidence.append(
                    EvidenceNode(
                        "risk",
                        -1.0,
                        0.65,
                        f"Current drawdown is elevated ({dd:.1f}%).",
                    )
                )

        vix = latest.get("vix")

        if vix is not None:
            components.append(
                _score(
                    vix,
                    neutral=20.0,
                    scale=10.0,
                    invert=True,
                )
            )

        pcr = latest.get("put_call_ratio")

        if pcr is not None:
            components.append(
                _score(
                    pcr,
                    neutral=1.0,
                    scale=0.5,
                )
            )

        volatility = latest.get("volatility")

        if volatility is not None:
            components.append(
                _score(
                    volatility,
                    neutral=20.0,
                    scale=15.0,
                    invert=True,
                )
            )

        if not components:
            return {
                "value": None,
                "status": "unknown",
                "evidence": [],
            }

        value = sum(components) / len(components)

        if value >= 65.0:
            status = "controlled"
        elif value <= 35.0:
            status = "elevated"
        else:
            status = "moderate"

        return {
            "value": value,
            "status": status,
            "evidence": evidence,
        }

    def _analyze_transition(
        self,
        latest: Dict[str, Optional[float]],
        history: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:

        evidence: List[EvidenceNode] = []
        directional_changes: List[float] = []

        for feature in (
            "return_20d",
            "return_60d",
            "rsi",
            "adx",
        ):

            series = history.get(feature, [])

            if len(series) < 2:
                continue

            previous = series[-2][1]
            current = series[-1][1]

            delta = current - previous

            directional_changes.append(
                _clip(delta / 10.0, -1.0, 1.0)
            )

        if not directional_changes:
            return {
                "value": None,
                "status": "unknown",
                "evidence": [],
            }

        average_change = sum(
            directional_changes
        ) / len(directional_changes)

        transition_score = abs(average_change) * 100.0

        return20 = latest.get("return_20d")
        return60 = latest.get("return_60d")

        if (
            return20 is not None
            and return60 is not None
        ):
            signs_conflict = (
                return20 > 0
                and return60 < 0
            ) or (
                return20 < 0
                and return60 > 0
            )

            if signs_conflict:
                transition_score = max(
                    transition_score,
                    70.0,
                )

                evidence.append(
                    EvidenceNode(
                        "transition",
                        -1.0,
                        0.70,
                        "Shorter- and longer-horizon returns are directionally divergent, indicating a potential regime transition.",
                    )
                )

        if transition_score >= 65.0:
            status = "transitioning"
        elif transition_score >= 35.0:
            status = "watch"
        else:
            status = "stable"

        return {
            "value": transition_score,
            "status": status,
            "evidence": evidence,
        }

    def _detect_regime(
        self,
        trend: Dict[str, Any],
        momentum: Dict[str, Any],
        volatility: Dict[str, Any],
        participation: Dict[str, Any],
        risk: Dict[str, Any],
        transition: Dict[str, Any],
    ) -> Dict[str, Any]:

        trend_value = trend.get("value")
        momentum_value = momentum.get("value")
        participation_value = participation.get("value")
        risk_value = risk.get("value")
        transition_value = transition.get("value")

        available = [
            value
            for value in (
                trend_value,
                momentum_value,
                participation_value,
                risk_value,
            )
            if value is not None
        ]

        if not available:
            return {
                "regime": None,
                "regime_status": "insufficient_data",
                "regime_score": None,
                "evidence": [],
            }

        components: List[float] = []

        if trend_value is not None:
            components.append(trend_value * 0.40)

        if momentum_value is not None:
            components.append(momentum_value * 0.25)

        if participation_value is not None:
            components.append(participation_value * 0.20)

        if risk_value is not None:
            components.append(risk_value * 0.15)

        total_weight = 0.0

        if trend_value is not None:
            total_weight += 0.40

        if momentum_value is not None:
            total_weight += 0.25

        if participation_value is not None:
            total_weight += 0.20

        if risk_value is not None:
            total_weight += 0.15

        if total_weight <= 0:
            return {
                "regime": None,
                "regime_status": "insufficient_data",
                "regime_score": None,
                "evidence": [],
            }

        regime_score = sum(components) / total_weight
        evidence: List[EvidenceNode] = []

        if (
            transition_value is not None
            and transition_value >= 65.0
        ):
            status = "transition"

            evidence.append(
                EvidenceNode(
                    "regime",
                    0.0,
                    0.75,
                    "Market structure contains sufficient directional divergence to classify the regime as transitioning.",
                )
            )

        elif regime_score >= 65.0:
            status = "bull"

            evidence.append(
                EvidenceNode(
                    "regime",
                    1.0,
                    _clip(
                        (regime_score - 50.0) / 50.0,
                        0.2,
                        1.0,
                    ),
                    "Aggregate market conditions are directionally bullish.",
                )
            )

        elif regime_score <= 35.0:
            status = "bear"

            evidence.append(
                EvidenceNode(
                    "regime",
                    -1.0,
                    _clip(
                        (50.0 - regime_score) / 50.0,
                        0.2,
                        1.0,
                    ),
                    "Aggregate market conditions are directionally bearish.",
                )
            )

        else:
            status = "sideways"

            evidence.append(
                EvidenceNode(
                    "regime",
                    0.0,
                    0.40,
                    "Aggregate market conditions do not show sufficient directional dominance for a bull or bear regime.",
                )
            )

        return {
            "regime": regime_score,
            "regime_status": status,
            "regime_score": regime_score,
            "evidence": evidence,
        }

    def _calculate_confidence(
        self,
        snapshots: List[Dict[str, Any]],
        coverage: Dict[str, Any],
        trend: Dict[str, Any],
        momentum: Dict[str, Any],
        volatility: Dict[str, Any],
        participation: Dict[str, Any],
        risk: Dict[str, Any],
        transition: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> float:

        coverage_pct = coverage["coverage_pct"]

        domain_values = [
            trend.get("value"),
            momentum.get("value"),
            volatility.get("value"),
            participation.get("value"),
            risk.get("value"),
            transition.get("value"),
        ]

        domain_count = sum(
            value is not None
            for value in domain_values
        )

        domain_coverage = domain_count / 6.0

        unique_dates = len(
            {
                snapshot.get("_t")
                for snapshot in snapshots
                if snapshot.get("_t", 0.0) > 0
            }
        )

        history_score = _clip(
            unique_dates / 5.0,
            0.0,
            1.0,
        )

        evidence_score = _clip(
            len(evidence) / 10.0,
            0.0,
            1.0,
        )

        confidence = (
            (coverage_pct * 0.45)
            + (domain_coverage * 30.0)
            + (history_score * 15.0)
            + (evidence_score * 10.0)
        )

        contradictions = sum(
            1
            for item in evidence
            if "contradiction" in str(
                item.get("message", "")
            ).lower()
        )

        confidence -= contradictions * 8.0

        return _clip(
            confidence,
            0.0,
            100.0,
        )

    def _format_evidence(
        self,
        nodes: List[EvidenceNode],
    ) -> List[Dict[str, Any]]:

        output: List[Dict[str, Any]] = []
        seen = set()

        for node in nodes:

            message = str(node.message)

            if message in seen:
                continue

            seen.add(message)

            magnitude = _clip(
                node.magnitude,
                0.0,
                1.0,
            )

            reliability = _clip(
                0.70 + (magnitude * 0.30),
                0.0,
                1.0,
            )

            likelihood_ratio = _likelihood_ratio(
                magnitude,
                node.direction,
            )

            output.append(
                {
                    "category": "Regime",
                    "domain": str(node.domain),
                    "message": message,
                    "reliability": self._round(reliability),
                    "likelihood_ratio": self._round(
                        likelihood_ratio
                    ),
                }
            )

        output.sort(
            key=lambda item: abs(
                float(
                    item.get(
                        "likelihood_ratio",
                        1.0,
                    )
                ) - 1.0
            ),
            reverse=True,
        )

        return output

    def _json_value(
        self,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        number = _num(value)

        if number is not None:
            return self._round(number)

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    def _round(
        self,
        value: Any,
    ) -> Optional[float]:

        number = _num(value)

        if number is None:
            return None

        return round(number, 4)


def analyze(
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:

    return MarketRegimeAnalyzer().analyze(
        data,
        *args,
        **kwargs,
    )


__all__ = [
    "MarketRegimeAnalyzer",
    "analyze",
]
