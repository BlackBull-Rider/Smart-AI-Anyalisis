from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_LR = 0.15
MAX_LR = 8.0


# ============================================================================
# STRICT IPO FEATURE CONTRACT
# ============================================================================
#
# Each tuple:
#   (database/input alias, multiplier)
#
# Multipliers are explicit. No automatic unit guessing is performed.
#
# Percentage fields:
#   - *_pct / *_percent     -> already percentage points
#   - *_ratio / *_decimal   -> decimal fractions converted explicitly
#
# The analyzer also accepts common NSE/Yahoo/database naming variants.
# ============================================================================

FIELD_CONTRACTS: Dict[str, List[Tuple[str, float]]] = {
    # Risk
    "ipo_risk": [
        ("ipo_risk", 1.0),
        ("risk", 1.0),
        ("ipo_risk_score", 1.0),
        ("risk_score", 1.0),
    ],
    "valuation": [
        ("ipo_valuation", 1.0),
        ("valuation", 1.0),
        ("valuation_score", 1.0),
        ("ipo_valuation_score", 1.0),
    ],

    # IPO quality / listing
    "ipo_quality": [
        ("ipo_quality", 1.0),
        ("ipo_quality_score", 1.0),
        ("quality_score", 1.0),
    ],
    "listing": [
        ("listing_strength", 1.0),
        ("listing_score", 1.0),
        ("listing_performance", 1.0),
        ("listing_gain_pct", 1.0),
        ("listing_gain_percent", 1.0),
    ],

    # Subscription / demand
    "subscription": [
        ("subscription", 1.0),
        ("subscription_multiple", 1.0),
        ("subscription_times", 1.0),
        ("overall_subscription", 1.0),
        ("overall_subscription_multiple", 1.0),
    ],
    "institutional": [
        ("institutional_subscription", 1.0),
        ("institutional_subscription_multiple", 1.0),
        ("qib_subscription", 1.0),
        ("qib_subscription_multiple", 1.0),
        ("institutional_demand", 1.0),
        ("institutional_score", 1.0),
    ],
    "anchor": [
        ("anchor_subscription", 1.0),
        ("anchor_subscription_multiple", 1.0),
        ("anchor_allocation", 1.0),
        ("anchor_score", 1.0),
        ("anchor_investor_score", 1.0),
    ],
    "demand": [
        ("demand", 1.0),
        ("demand_score", 1.0),
        ("ipo_demand", 1.0),
        ("ipo_demand_score", 1.0),
        ("retail_demand", 1.0),
    ],

    # Liquidity
    "liquidity": [
        ("ipo_liquidity", 1.0),
        ("liquidity", 1.0),
        ("liquidity_score", 1.0),
        ("listing_liquidity", 1.0),
        ("trading_liquidity", 1.0),
    ],

    # Business
    "business": [
        ("business_quality", 1.0),
        ("business_score", 1.0),
        ("business_strength", 1.0),
        ("company_quality", 1.0),
        ("fundamental_quality", 1.0),
    ],

    # Additional fundamental IPO context
    "revenue_growth": [
        ("revenue_growth_yoy", 1.0),
        ("revenue_growth_pct", 1.0),
        ("revenue_growth_percent", 1.0),
        ("revenue_growth", 1.0),
    ],
    "earnings_growth": [
        ("earnings_growth_yoy", 1.0),
        ("earnings_growth_pct", 1.0),
        ("earnings_growth_percent", 1.0),
        ("earnings_growth", 1.0),
    ],
    "roe": [
        ("roe", 1.0),
        ("return_on_equity", 1.0),
        ("return_on_equity_pct", 1.0),
    ],
    "roce": [
        ("roce", 1.0),
        ("return_on_capital_employed", 1.0),
        ("return_on_capital_employed_pct", 1.0),
    ],

    # IPO pricing
    "issue_price": [
        ("issue_price", 1.0),
        ("ipo_price", 1.0),
        ("offer_price", 1.0),
        ("price_band_upper", 1.0),
        ("upper_price_band", 1.0),
    ],
    "listing_price": [
        ("listing_price", 1.0),
        ("listing_day_price", 1.0),
        ("open_price", 1.0),
        ("listing_day_open", 1.0),
    ],
    "current_price": [
        ("current_price", 1.0),
        ("cmp", 1.0),
        ("last_price", 1.0),
        ("market_price", 1.0),
    ],

    # Size / liquidity context
    "issue_size": [
        ("issue_size", 1.0),
        ("issue_size_cr", 1.0),
        ("issue_size_crore", 1.0),
        ("total_issue_size", 1.0),
    ],
    "market_cap": [
        ("market_cap", 1.0),
        ("market_capitalization", 1.0),
        ("market_cap_cr", 1.0),
    ],

    # Ownership / dilution
    "promoter_holding": [
        ("promoter_holding", 1.0),
        ("promoter_holding_pct", 1.0),
        ("promoter_holding_percent", 1.0),
    ],
    "fresh_issue_pct": [
        ("fresh_issue_pct", 1.0),
        ("fresh_issue_percent", 1.0),
    ],
    "ofs_pct": [
        ("ofs_pct", 1.0),
        ("ofs_percent", 1.0),
        ("offer_for_sale_pct", 1.0),
    ],

    # Risk context
    "debt_equity": [
        ("debt_to_equity", 1.0),
        ("debt_equity", 1.0),
        ("debt_eq", 1.0),
    ],
    "profit_margin": [
        ("net_margin", 1.0),
        ("profit_margin", 1.0),
        ("profit_margin_pct", 1.0),
        ("profit_margins", 1.0),
    ],
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class EvidenceNode:
    domain: str
    direction: float
    magnitude: float
    message: str


# ============================================================================
# SAFE HELPERS
# ============================================================================

def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    try:
        if isinstance(value, str):
            text = value.strip()

            if not text:
                return None

            text = text.replace(",", "")
            text = text.replace("₹", "")
            text = text.replace("$", "")

            # Explicitly strip a percentage sign.
            if text.endswith("%"):
                text = text[:-1].strip()

            # Handle simple "x" suffix used for subscription multiples.
            if text.lower().endswith("x"):
                text = text[:-1].strip()

            value = float(text)

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError, OverflowError):
        return None


def _str(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError, OverflowError):
        return None


def _sigmoid(x: float) -> float:
    x = _clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-x))


def _continuous_score(
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


def _lr(magnitude: float, direction: float) -> float:
    magnitude = _clip(abs(magnitude), 0.0, 1.0)

    base = 1.0 + (magnitude * 3.0)

    if direction >= 0:
        return _clip(base, MIN_LR, MAX_LR)

    return _clip(1.0 / base, MIN_LR, MAX_LR)


def _normalise_key(key: Any) -> str:
    return (
        str(key)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _parse_time(value: Any, fallback: float = 0.0) -> float:
    if value is None:
        return fallback

    if isinstance(value, bool):
        return fallback

    if isinstance(value, (int, float)):
        try:
            value = float(value)
            return value if math.isfinite(value) else fallback
        except (TypeError, ValueError):
            return fallback

    text = str(value).strip()

    if not text:
        return fallback

    # YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        try:
            return (
                float(text[0:4]) * 10000.0
                + float(text[5:7]) * 100.0
                + float(text[8:10])
            )
        except (ValueError, IndexError):
            return fallback

    # YYYY/MM/DD
    if re.match(r"^\d{4}/\d{2}/\d{2}", text):
        try:
            return (
                float(text[0:4]) * 10000.0
                + float(text[5:7]) * 100.0
                + float(text[8:10])
            )
        except (ValueError, IndexError):
            return fallback

    numeric = _num(text)

    if numeric is not None:
        return numeric

    return fallback


def _date_from_row(row: Mapping[str, Any]) -> float:
    aliases = (
        "date",
        "datetime",
        "timestamp",
        "time",
        "period",
        "report_date",
        "financial_date",
        "as_of_date",
        "updated_at",
        "created_at",
        "listing_date",
        "ipo_date",
    )

    for alias in aliases:
        key = _normalise_key(alias)

        if key in row:
            parsed = _parse_time(row[key])

            if parsed != 0.0:
                return parsed

    return 0.0


def _status(score: Optional[float], high: float, low: float) -> str:
    if score is None:
        return "unknown"

    if score >= high:
        return "high"

    if score <= low:
        return "low"

    return "neutral"


def _safe_round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None

    return _safe_float(round(value, digits))


# ============================================================================
# IPO ANALYZER
# ============================================================================

class IPOAnalyzer:
    """
    Production-grade IPO Analyzer.

    Design principles
    -----------------
    1. Database friendly.
    2. L3 Analyzer friendly.
    3. Accepts DataFrame / dict / list / wrapped payloads.
    4. Chronological historical parsing.
    5. Uses latest VALID snapshot, not blindly the last row.
    6. Never fabricates missing values.
    7. Explicit feature contracts.
    8. Full feature-level tracing.
    9. Domain-level coverage.
    10. Deterministic output.
    11. No Buy/Sell/Entry/SL/Target logic.
    12. No dependency on scoring engines.
    13. Exception safe.
    14. JSON serializable output.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ========================================================================
    # PUBLIC API
    # ========================================================================

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

            histories = self._build_feature_histories(snapshots)

            latest_snapshot = self._get_latest_valid_snapshot(
                snapshots,
                histories,
            )

            if latest_snapshot is None:
                return self._empty()

            latest_metrics, feature_trace = self._extract_latest_features(
                histories,
                latest_snapshot,
            )

            feature_trace_summary = self._build_feature_trace_summary(
                histories,
                feature_trace,
            )

            domain_coverage = self._build_domain_coverage(
                histories,
                feature_trace,
            )

            nodes: List[EvidenceNode] = []

            # =================================================================
            # RISK
            # =================================================================

            risk_score = self._calculate_risk(
                latest_metrics,
                nodes,
            )

            risk_status = (
                "high"
                if risk_score is not None and risk_score >= 60.0
                else "low"
                if risk_score is not None and risk_score <= 40.0
                else "neutral"
                if risk_score is not None
                else "unknown"
            )

            # =================================================================
            # VALUATION
            # =================================================================

            valuation_score = self._calculate_valuation(
                latest_metrics,
                nodes,
            )

            valuation_status = (
                "attractive"
                if valuation_score is not None and valuation_score >= 60.0
                else "expensive"
                if valuation_score is not None and valuation_score <= 40.0
                else "neutral"
                if valuation_score is not None
                else "unknown"
            )

            # =================================================================
            # IPO QUALITY
            # =================================================================

            ipo_quality_score = self._calculate_ipo_quality(
                latest_metrics,
                nodes,
            )

            ipo_quality_status = (
                "high"
                if ipo_quality_score is not None and ipo_quality_score >= 60.0
                else "low"
                if ipo_quality_score is not None and ipo_quality_score <= 40.0
                else "neutral"
                if ipo_quality_score is not None
                else "unknown"
            )

            # =================================================================
            # LISTING
            # =================================================================

            listing_score = self._calculate_listing(
                latest_metrics,
                histories,
                nodes,
            )

            listing_status = (
                "strong"
                if listing_score is not None and listing_score >= 60.0
                else "weak"
                if listing_score is not None and listing_score <= 40.0
                else "neutral"
                if listing_score is not None
                else "unknown"
            )

            # =================================================================
            # SUBSCRIPTION
            # =================================================================

            subscription_score = self._calculate_subscription(
                latest_metrics,
                nodes,
            )

            subscription_status = (
                "high"
                if subscription_score is not None and subscription_score >= 60.0
                else "low"
                if subscription_score is not None and subscription_score <= 40.0
                else "neutral"
                if subscription_score is not None
                else "unknown"
            )

            # =================================================================
            # INSTITUTIONAL
            # =================================================================

            institutional_score = self._calculate_institutional(
                latest_metrics,
                nodes,
            )

            institutional_status = (
                "strong"
                if institutional_score is not None and institutional_score >= 60.0
                else "weak"
                if institutional_score is not None and institutional_score <= 40.0
                else "neutral"
                if institutional_score is not None
                else "unknown"
            )

            # =================================================================
            # ANCHOR
            # =================================================================

            anchor_score = self._calculate_anchor(
                latest_metrics,
                nodes,
            )

            anchor_status = (
                "excellent"
                if anchor_score is not None and anchor_score >= 70.0
                else "poor"
                if anchor_score is not None and anchor_score <= 40.0
                else "neutral"
                if anchor_score is not None
                else "unknown"
            )

            # =================================================================
            # DEMAND
            # =================================================================

            demand_score = self._calculate_demand(
                latest_metrics,
                nodes,
            )

            demand_status = (
                "huge"
                if demand_score is not None and demand_score >= 70.0
                else "weak"
                if demand_score is not None and demand_score <= 40.0
                else "neutral"
                if demand_score is not None
                else "unknown"
            )

            # =================================================================
            # LIQUIDITY
            # =================================================================

            liquidity_score = self._calculate_liquidity(
                latest_metrics,
                nodes,
            )

            liquidity_status = (
                "high"
                if liquidity_score is not None and liquidity_score >= 60.0
                else "low"
                if liquidity_score is not None and liquidity_score <= 40.0
                else "neutral"
                if liquidity_score is not None
                else "unknown"
            )

            # =================================================================
            # BUSINESS
            # =================================================================

            business_score = self._calculate_business(
                latest_metrics,
                nodes,
            )

            business_status = (
                "excellent"
                if business_score is not None and business_score >= 60.0
                else "poor"
                if business_score is not None and business_score <= 40.0
                else "neutral"
                if business_score is not None
                else "unknown"
            )

            # =================================================================
            # OPPORTUNITY
            # =================================================================

            opportunity_score = self._calculate_opportunity(
                risk_score=risk_score,
                valuation_score=valuation_score,
                ipo_quality_score=ipo_quality_score,
                listing_score=listing_score,
                subscription_score=subscription_score,
                institutional_score=institutional_score,
                anchor_score=anchor_score,
                demand_score=demand_score,
                liquidity_score=liquidity_score,
                business_score=business_score,
            )

            opportunity_status = (
                "high"
                if opportunity_score is not None and opportunity_score >= 60.0
                else "low"
                if opportunity_score is not None and opportunity_score <= 40.0
                else "neutral"
                if opportunity_score is not None
                else "unknown"
            )

            # =================================================================
            # CROSS-DOMAIN CONTRADICTIONS
            # =================================================================

            self._detect_contradictions(
                risk_score=risk_score,
                valuation_score=valuation_score,
                demand_score=demand_score,
                institutional_score=institutional_score,
                business_score=business_score,
                subscription_score=subscription_score,
                listing_score=listing_score,
                nodes=nodes,
            )

            # =================================================================
            # EVIDENCE
            # =================================================================

            evidence = self._format_evidence(nodes)

            # =================================================================
            # CONFIDENCE
            # =================================================================

            confidence = self._calculate_confidence(
                feature_trace_summary=feature_trace_summary,
                domain_coverage=domain_coverage,
                evidence=evidence,
                snapshots=snapshots,
            )

            return {
                "ipo_analyzer": {
                    "confidence": _safe_round(confidence),

                    "risk": _safe_round(risk_score),
                    "risk_status": risk_status,

                    "valuation": _safe_round(valuation_score),
                    "valuation_status": valuation_status,

                    "ipo_quality": _safe_round(ipo_quality_score),
                    "ipo_quality_status": ipo_quality_status,

                    "listing": _safe_round(listing_score),
                    "listing_status": listing_status,

                    "subscription": _safe_round(subscription_score),
                    "subscription_status": subscription_status,

                    "institutional": _safe_round(institutional_score),
                    "institutional_status": institutional_status,

                    "anchor": _safe_round(anchor_score),
                    "anchor_status": anchor_status,

                    "demand": _safe_round(demand_score),
                    "demand_status": demand_status,

                    "liquidity": _safe_round(liquidity_score),
                    "liquidity_status": liquidity_status,

                    "business": _safe_round(business_score),
                    "business_status": business_status,

                    "opportunity": _safe_round(opportunity_score),
                    "opportunity_status": opportunity_status,

                    "feature_coverage_pct": feature_trace_summary[
                        "coverage_pct"
                    ],

                    "feature_trace_summary": feature_trace_summary,

                    "feature_trace": feature_trace,

                    "domain_coverage": domain_coverage,

                    "evidence": evidence[:12],
                }
            }

        except Exception as exc:
            self.logger.exception(
                "IPO Analyzer Critical Failure: %s",
                exc,
            )

            result = self._empty()

            result["ipo_analyzer"]["evidence"].append(
                {
                    "category": "IPO",
                    "domain": "system",
                    "message": "Execution encountered a critical failure.",
                    "reliability": 0.96,
                    "likelihood_ratio": 0.42,
                }
            )

            return result

    # ========================================================================
    # PARSER
    # ========================================================================

    def _parse_payload_chronologically(
        self,
        data: Any,
    ) -> List[Dict[str, Any]]:
        raw_rows: List[Dict[str, Any]] = []

        if data is None:
            return raw_rows

        # pandas DataFrame
        if hasattr(data, "to_dict") and hasattr(data, "columns"):
            try:
                converted = data.to_dict(orient="records")

                if isinstance(converted, list):
                    raw_rows.extend(
                        row
                        for row in converted
                        if isinstance(row, dict)
                    )
            except Exception:
                pass

        elif isinstance(data, Mapping):
            wrappers = (
                "features",
                "feature_history",
                "historical_data",
                "ipo_history",
                "ipo_data",
                "historical_ipo_data",
                "rows",
                "records",
                "data",
                "items",
            )

            found_list = False

            for wrapper in wrappers:
                value = data.get(wrapper)

                if isinstance(value, list):
                    raw_rows.extend(
                        row
                        for row in value
                        if isinstance(row, Mapping)
                    )
                    found_list = True

            dict_wrappers = (
                "ipo_snapshot",
                "ipo_snapshot_data",
                "latest_ipo",
                "company_profile",
            )

            for wrapper in dict_wrappers:
                value = data.get(wrapper)

                if isinstance(value, Mapping):
                    raw_rows.append(dict(value))
                    found_list = True

            if not found_list:
                raw_rows.append(dict(data))

        elif isinstance(data, list):
            raw_rows.extend(
                row
                for row in data
                if isinstance(row, Mapping)
            )

        # Normalize every row.
        extracted: List[Dict[str, Any]] = []

        for index, row in enumerate(raw_rows):
            normalized: Dict[str, Any] = {}

            for raw_key, value in row.items():
                if value is None:
                    continue

                key = _normalise_key(raw_key)

                normalized[key] = value

            if not normalized:
                continue

            timestamp = _date_from_row(normalized)

            normalized["_t"] = timestamp
            normalized["_idx"] = index

            # A row is retained if it contains at least one contracted
            # feature, even when the row has no explicit date.
            if self._row_contains_contract_feature(normalized):
                extracted.append(normal)

        extracted.sort(
            key=lambda row: (
                float(row.get("_t", 0.0)),
                int(row.get("_idx", 0)),
            )
        )

        return extracted

    def _row_contains_contract_feature(
        self,
        row: Mapping[str, Any],
    ) -> bool:
        aliases = {
            alias
            for contracts in FIELD_CONTRACTS.values()
            for alias, _ in contracts
        }

        return any(
            key in aliases
            for key in row.keys()
        )

    # ========================================================================
    # FEATURE HISTORY
    # ========================================================================

    def _build_feature_histories(
        self,
        snapshots: List[Dict[str, Any]],
    ) -> Dict[str, List[Tuple[float, float, str, Any]]]:
        histories: Dict[
            str,
            List[Tuple[float, float, str, Any]],
        ] = {}

        for feature, aliases in FIELD_CONTRACTS.items():
            series: List[Tuple[float, float, str, Any]] = []

            for snapshot in snapshots:
                for alias, multiplier in aliases:
                    if alias not in snapshot:
                        continue

                    raw = _num(snapshot.get(alias))

                    if raw is None:
                        continue

                    normalized = raw * multiplier

                    if not math.isfinite(normalized):
                        continue

                    series.append(
                        (
                            float(snapshot.get("_t", 0.0)),
                            float(normalized),
                            alias,
                            snapshot.get(alias),
                        )
                    )

                    break

            histories[feature] = series

        return histories

    def _get_latest_valid_snapshot(
        self,
        snapshots: List[Dict[str, Any]],
        histories: Dict[str, List[Tuple[float, float, str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        if not snapshots:
            return None

        # Valid means at least one contracted feature has a valid numeric
        # value in the row. This avoids blindly trusting the final DB row.
        for snapshot in reversed(snapshots):
            timestamp = float(snapshot.get("_t", 0.0))

            for feature_history in histories.values():
                for timestamp_value, _, _, _ in reversed(feature_history):
                    if timestamp_value == timestamp:
                        return snapshot

        return None

    # ========================================================================
    # LATEST FEATURE EXTRACTION + TRACE
    # ========================================================================

    def _extract_latest_features(
        self,
        histories: Dict[str, List[Tuple[float, float, str, Any]]],
        latest_snapshot: Dict[str, Any],
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]]]:

        latest_metrics: Dict[str, Optional[float]] = {
            feature: None
            for feature in FIELD_CONTRACTS
        }

        trace: Dict[str, Dict[str, Any]] = {}

        latest_timestamp = float(
            latest_snapshot.get("_t", 0.0)
        )

        for feature, aliases in FIELD_CONTRACTS.items():
            history = histories.get(feature, [])

            aliases_checked = [
                alias
                for alias, _ in aliases
            ]

            matching = [
                point
                for point in history
                if point[0] == latest_timestamp
            ]

            if matching:
                timestamp, normalized, source_field, raw_value = matching[-1]

                multiplier = 1.0

                for alias, mult in aliases:
                    if alias == source_field:
                        multiplier = mult
                        break

                latest_metrics[feature] = normalized

                trace[feature] = {
                    "status": "used",
                    "feature": feature,
                    "source_field": source_field,
                    "raw_value": raw_value,
                    "multiplier": multiplier,
                    "normalized_value": normalized,
                    "historical_points": len(history),
                    "latest_timestamp": timestamp,
                    "aliases_checked": aliases_checked,
                }

                continue

            # The latest snapshot may not contain a particular feature.
            # Search backwards for the latest VALID value only.
            fallback = self._latest_valid_history_point(history)

            if fallback is not None:
                timestamp, normalized, source_field, raw_value = fallback

                multiplier = 1.0

                for alias, mult in aliases:
                    if alias == source_field:
                        multiplier = mult
                        break

                latest_metrics[feature] = normalized

                trace[feature] = {
                    "status": "used",
                    "feature": feature,
                    "source_field": source_field,
                    "raw_value": raw_value,
                    "multiplier": multiplier,
                    "normalized_value": normalized,
                    "historical_points": len(history),
                    "latest_timestamp": timestamp,
                    "aliases_checked": aliases_checked,
                    "valid_history_fallback": True,
                }

                continue

            trace[feature] = {
                "status": "missing",
                "feature": feature,
                "source_field": None,
                "raw_value": None,
                "multiplier": None,
                "normalized_value": None,
                "historical_points": 0,
                "latest_timestamp": latest_timestamp,
                "aliases_checked": aliases_checked,
            }

        return latest_metrics, trace

    @staticmethod
    def _latest_valid_history_point(
        history: List[Tuple[float, float, str, Any]],
    ) -> Optional[Tuple[float, float, str, Any]]:
        if not history:
            return None

        for point in reversed(history):
            if point[1] is None:
                continue

            if math.isfinite(point[1]):
                return point

        return None

    # ========================================================================
    # TRACE / COVERAGE
    # ========================================================================

    def _build_feature_trace_summary(
        self,
        histories: Dict[str, List[Tuple[float, float, str, Any]]],
        feature_trace: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        total = len(FIELD_CONTRACTS)

        used_names = sorted(
            feature
            for feature, trace in feature_trace.items()
            if trace.get("status") == "used"
        )

        missing_names = sorted(
            feature
            for feature, trace in feature_trace.items()
            if trace.get("status") == "missing"
        )

        invalid_names = sorted(
            feature
            for feature, trace in feature_trace.items()
            if trace.get("status") == "invalid"
        )

        used = len(used_names)

        coverage = (
            (used / float(total)) * 100.0
            if total
            else 0.0
        )

        return {
            "total_features": total,
            "used_features": used,
            "available_not_used": 0,
            "missing_features": len(missing_names),
            "invalid_features": len(invalid_names),
            "coverage_pct": _safe_round(coverage),
            "used_feature_names": used_names,
            "available_not_used_names": [],
            "missing_feature_names": missing_names,
            "invalid_feature_names": invalid_names,
        }

    def _build_domain_coverage(
        self,
        histories: Dict[str, List[Tuple[float, float, str, Any]]],
        feature_trace: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        domains = {
            "risk": [
                "ipo_risk",
                "debt_equity",
            ],
            "valuation": [
                "valuation",
            ],
            "ipo_quality": [
                "ipo_quality",
                "business",
                "revenue_growth",
                "earnings_growth",
                "roe",
                "roce",
            ],
            "listing": [
                "listing",
                "listing_price",
                "issue_price",
            ],
            "subscription": [
                "subscription",
            ],
            "institutional": [
                "institutional",
            ],
            "anchor": [
                "anchor",
            ],
            "demand": [
                "demand",
                "subscription",
            ],
            "liquidity": [
                "liquidity",
                "issue_size",
                "market_cap",
            ],
            "business": [
                "business",
                "revenue_growth",
                "earnings_growth",
                "roe",
                "roce",
                "profit_margin",
            ],
            "ownership": [
                "promoter_holding",
                "fresh_issue_pct",
                "ofs_pct",
            ],
        }

        result: Dict[str, Any] = {}

        for domain, features in domains.items():
            total = len(features)

            used_names = [
                feature
                for feature in features
                if feature_trace.get(feature, {}).get("status")
                == "used"
            ]

            used = len(used_names)

            coverage = (
                (used / float(total)) * 100.0
                if total
                else 0.0
            )

            result[domain] = {
                "coverage_pct": _safe_round(coverage),
                "used": used,
                "total": total,
                "used_features": sorted(used_names),
            }

        return result

    # ========================================================================
    # DOMAIN CALCULATORS
    # ========================================================================

    def _calculate_risk(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        values: List[float] = []

        ipo_risk = m.get("ipo_risk")

        if ipo_risk is not None:
            score = _clip(ipo_risk, 0.0, 100.0)

            # Higher explicit risk score = higher risk.
            values.append(score)

            direction = -1.0 if score > 50.0 else 1.0

            nodes.append(
                EvidenceNode(
                    "risk",
                    direction,
                    abs(score - 50.0) / 50.0,
                    f"IPO risk score is {score:.1f}.",
                )
            )

        debt = m.get("debt_equity")

        if debt is not None:
            score = _continuous_score(
                debt,
                neutral=1.0,
                scale=0.8,
                invert=False,
            )

            values.append(score)

            direction = -1.0 if score > 50.0 else 1.0

            nodes.append(
                EvidenceNode(
                    "risk",
                    direction,
                    abs(score - 50.0) / 50.0,
                    f"Debt-to-equity context indicates "
                    f"{'elevated' if score > 50.0 else 'moderate/lower'} leverage risk.",
                )
            )

        if not values:
            return None

        return _clip(sum(values) / len(values), 0.0, 100.0)

    def _calculate_valuation(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        explicit = m.get("valuation")

        if explicit is not None:
            score = _clip(explicit, 0.0, 100.0)

            nodes.append(
                EvidenceNode(
                    "valuation",
                    1.0 if score >= 50.0 else -1.0,
                    abs(score - 50.0) / 50.0,
                    f"IPO valuation assessment score is {score:.1f}.",
                )
            )

            return score

        issue = m.get("issue_price")
        market = m.get("market_cap")

        if issue is None or market is None or issue <= 0:
            return None

        # Without earnings/book-value/revenue normalization, market cap and
        # issue price alone cannot establish valuation. Do not fabricate.
        return None

    def _calculate_ipo_quality(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        components: List[float] = []

        explicit = m.get("ipo_quality")

        if explicit is not None:
            components.append(_clip(explicit, 0.0, 100.0))

        business = m.get("business")

        if business is not None:
            components.append(_clip(business, 0.0, 100.0))

        revenue_growth = m.get("revenue_growth")

        if revenue_growth is not None:
            score = _continuous_score(
                revenue_growth,
                neutral=8.0,
                scale=12.0,
            )

            components.append(score)

        earnings_growth = m.get("earnings_growth")

        if earnings_growth is not None:
            score = _continuous_score(
                earnings_growth,
                neutral=10.0,
                scale=15.0,
            )

            components.append(score)

        roe = m.get("roe")

        if roe is not None:
            components.append(
                _continuous_score(
                    roe,
                    neutral=12.0,
                    scale=8.0,
                )
            )

        roce = m.get("roce")

        if roce is not None:
            components.append(
                _continuous_score(
                    roce,
                    neutral=12.0,
                    scale=8.0,
                )
            )

        if not components:
            return None

        score = _clip(
            sum(components) / len(components),
            0.0,
            100.0,
        )

        if score >= 60.0:
            nodes.append(
                EvidenceNode(
                    "ipo_quality",
                    1.0,
                    abs(score - 50.0) / 50.0,
                    "Available business-quality and growth evidence "
                    "supports a stronger IPO quality profile.",
                )
            )
        elif score <= 40.0:
            nodes.append(
                EvidenceNode(
                    "ipo_quality",
                    -1.0,
                    abs(score - 50.0) / 50.0,
                    "Available business-quality evidence indicates "
                    "a weaker IPO quality profile.",
                )
            )

        return score

    def _calculate_listing(
        self,
        m: Dict[str, Optional[float]],
        histories: Dict[str, List[Tuple[float, float, str, Any]]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        explicit = m.get("listing")

        if explicit is not None:
            score = _clip(explicit, 0.0, 100.0)

            nodes.append(
                EvidenceNode(
                    "listing",
                    1.0 if score >= 50.0 else -1.0,
                    abs(score - 50.0) / 50.0,
                    f"Listing strength score is {score:.1f}.",
                )
            )

            return score

        issue = m.get("issue_price")
        listing = m.get("listing_price")

        if issue is None or listing is None or issue <= 0:
            return None

        listing_return = ((listing / issue) - 1.0) * 100.0

        score = _continuous_score(
            listing_return,
            neutral=0.0,
            scale=10.0,
        )

        direction = 1.0 if listing_return > 0 else -1.0

        nodes.append(
            EvidenceNode(
                "listing",
                direction,
                min(abs(listing_return) / 20.0, 1.0),
                f"Listing price is {listing_return:.1f}% "
                f"relative to the issue price.",
            )
        )

        return score

    def _calculate_subscription(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        value = m.get("subscription")

        if value is None:
            return None

        if value < 0:
            return None

        # Subscription multiples are not converted into arbitrary scoring
        # values. The curve is continuous and deterministic.
        score = _continuous_score(
            value,
            neutral=5.0,
            scale=5.0,
        )

        nodes.append(
            EvidenceNode(
                "subscription",
                1.0 if value >= 5.0 else -1.0,
                min(abs(value - 5.0) / 10.0, 1.0),
                f"Overall IPO subscription is {value:.2f}x.",
            )
        )

        return score

    def _calculate_institutional(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        value = m.get("institutional")

        if value is None:
            return None

        if value < 0:
            return None

        score = _continuous_score(
            value,
            neutral=5.0,
            scale=5.0,
        )

        nodes.append(
            EvidenceNode(
                "institutional",
                1.0 if value >= 5.0 else -1.0,
                min(abs(value - 5.0) / 10.0, 1.0),
                f"Institutional/QIB subscription is {value:.2f}x.",
            )
        )

        return score

    def _calculate_anchor(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        value = m.get("anchor")

        if value is None:
            return None

        if value < 0:
            return None

        score = _continuous_score(
            value,
            neutral=1.0,
            scale=0.5,
        )

        nodes.append(
            EvidenceNode(
                "anchor",
                1.0 if value >= 1.0 else -1.0,
                min(abs(value - 1.0), 1.0),
                f"Anchor demand/allocation metric is {value:.2f}.",
            )
        )

        return score

    def _calculate_demand(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        explicit = m.get("demand")

        if explicit is not None:
            score = _clip(explicit, 0.0, 100.0)

            nodes.append(
                EvidenceNode(
                    "demand",
                    1.0 if score >= 50.0 else -1.0,
                    abs(score - 50.0) / 50.0,
                    f"IPO demand score is {score:.1f}.",
                )
            )

            return score

        subscription = m.get("subscription")
        institutional = m.get("institutional")

        components: List[float] = []

        if subscription is not None:
            components.append(
                _continuous_score(
                    subscription,
                    neutral=5.0,
                    scale=5.0,
                )
            )

        if institutional is not None:
            components.append(
                _continuous_score(
                    institutional,
                    neutral=5.0,
                    scale=5.0,
                )
            )

        if not components:
            return None

        score = sum(components) / len(components)

        nodes.append(
            EvidenceNode(
                "demand",
                1.0 if score >= 50.0 else -1.0,
                abs(score - 50.0) / 50.0,
                "Demand strength is derived from available subscription "
                "and institutional-demand evidence.",
            )
        )

        return _clip(score, 0.0, 100.0)

    def _calculate_liquidity(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        explicit = m.get("liquidity")

        if explicit is not None:
            score = _clip(explicit, 0.0, 100.0)

            nodes.append(
                EvidenceNode(
                    "liquidity",
                    1.0 if score >= 50.0 else -1.0,
                    abs(score - 50.0) / 50.0,
                    f"IPO liquidity score is {score:.1f}.",
                )
            )

            return score

        issue_size = m.get("issue_size")
        market_cap = m.get("market_cap")

        if issue_size is None or market_cap is None:
            return None

        if market_cap <= 0 or issue_size < 0:
            return None

        # This is a structural liquidity proxy only when both values exist.
        ratio = issue_size / market_cap

        score = _continuous_score(
            ratio,
            neutral=0.05,
            scale=0.05,
            invert=True,
        )

        nodes.append(
            EvidenceNode(
                "liquidity",
                1.0 if score >= 50.0 else -1.0,
                abs(score - 50.0) / 50.0,
                f"Issue-size to market-capitalization ratio is "
                f"{ratio:.2%}.",
            )
        )

        return _clip(score, 0.0, 100.0)

    def _calculate_business(
        self,
        m: Dict[str, Optional[float]],
        nodes: List[EvidenceNode],
    ) -> Optional[float]:

        components: List[float] = []

        explicit = m.get("business")

        if explicit is not None:
            components.append(_clip(explicit, 0.0, 100.0))

        revenue_growth = m.get("revenue_growth")

        if revenue_growth is not None:
            components.append(
                _continuous_score(
                    revenue_growth,
                    neutral=8.0,
                    scale=12.0,
                )
            )

        earnings_growth = m.get("earnings_growth")

        if earnings_growth is not None:
            components.append(
                _continuous_score(
                    earnings_growth,
                    neutral=10.0,
                    scale=15.0,
                )
            )

        roe = m.get("roe")

        if roe is not None:
            components.append(
                _continuous_score(
                    roe,
                    neutral=12.0,
                    scale=8.0,
                )
            )

        roce = m.get("roce")

        if roce is not None:
            components.append(
                _continuous_score(
                    roce,
                    neutral=12.0,
                    scale=8.0,
                )
            )

        margin = m.get("profit_margin")

        if margin is not None:
            components.append(
                _continuous_score(
                    margin,
                    neutral=8.0,
                    scale=6.0,
                )
            )

        if not components:
            return None

        score = _clip(
            sum(components) / len(components),
            0.0,
            100.0,
        )

        if score >= 60.0:
            nodes.append(
                EvidenceNode(
                    "business",
                    1.0,
                    abs(score - 50.0) / 50.0,
                    "Available operating and fundamental metrics "
                    "indicate stronger business quality.",
                )
            )
        elif score <= 40.0:
            nodes.append(
                EvidenceNode(
                    "business",
                    -1.0,
                    abs(score - 50.0) / 50.0,
                    "Available operating and fundamental metrics "
                    "indicate weaker business quality.",
                )
            )

        return score

    # ========================================================================
    # OPPORTUNITY
    # ========================================================================

    def _calculate_opportunity(
        self,
        *,
        risk_score: Optional[float],
        valuation_score: Optional[float],
        ipo_quality_score: Optional[float],
        listing_score: Optional[float],
        subscription_score: Optional[float],
        institutional_score: Optional[float],
        anchor_score: Optional[float],
        demand_score: Optional[float],
        liquidity_score: Optional[float],
        business_score: Optional[float],
    ) -> Optional[float]:

        components: List[Tuple[float, float]] = []

        if risk_score is not None:
            components.append((100.0 - risk_score, 1.5))

        if valuation_score is not None:
            components.append((valuation_score, 1.5))

        if ipo_quality_score is not None:
            components.append((ipo_quality_score, 1.5))

        if listing_score is not None:
            components.append((listing_score, 0.75))

        if subscription_score is not None:
            components.append((subscription_score, 1.0))

        if institutional_score is not None:
            components.append((institutional_score, 1.25))

        if anchor_score is not None:
            components.append((anchor_score, 0.75))

        if demand_score is not None:
            components.append((demand_score, 1.0))

        if liquidity_score is not None:
            components.append((liquidity_score, 0.75))

        if business_score is not None:
            components.append((business_score, 1.5))

        if not components:
            return None

        numerator = sum(
            value * weight
            for value, weight in components
        )

        denominator = sum(
            weight
            for _, weight in components
        )

        if denominator <= 0:
            return None

        return _clip(
            numerator / denominator,
            0.0,
            100.0,
        )

    # ========================================================================
    # CONTRADICTIONS
    # ========================================================================

    def _detect_contradictions(
        self,
        *,
        risk_score: Optional[float],
        valuation_score: Optional[float],
        demand_score: Optional[float],
        institutional_score: Optional[float],
        business_score: Optional[float],
        subscription_score: Optional[float],
        listing_score: Optional[float],
        nodes: List[EvidenceNode],
    ) -> None:

        if (
            valuation_score is not None
            and demand_score is not None
            and valuation_score <= 30.0
            and demand_score >= 70.0
        ):
            nodes.append(
                EvidenceNode(
                    "contradiction",
                    -1.0,
                    0.8,
                    "Contradiction: strong demand is present alongside "
                    "an unattractive valuation assessment.",
                )
            )

        if (
            risk_score is not None
            and business_score is not None
            and risk_score >= 70.0
            and business_score >= 70.0
        ):
            nodes.append(
                EvidenceNode(
                    "contradiction",
                    -1.0,
                    0.75,
                    "Contradiction: strong business quality coexists "
                    "with elevated IPO risk.",
                )
            )

        if (
            institutional_score is not None
            and subscription_score is not None
            and institutional_score >= 70.0
            and subscription_score <= 30.0
        ):
            nodes.append(
                EvidenceNode(
                    "contradiction",
                    -1.0,
                    0.65,
                    "Contradiction: institutional demand is strong "
                    "while overall subscription remains weak.",
                )
            )

        if (
            listing_score is not None
            and demand_score is not None
            and listing_score <= 30.0
            and demand_score >= 70.0
        ):
            nodes.append(
                EvidenceNode(
                    "contradiction",
                    -1.0,
                    0.65,
                    "Contradiction: strong measured demand exists "
                    "despite weak listing performance.",
                )
            )

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def _format_evidence(
        self,
        nodes: List[EvidenceNode],
    ) -> List[Dict[str, Any]]:

        output: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for node in nodes:
            if node.message in seen:
                continue

            seen.add(node.message)

            magnitude = _clip(
                abs(node.magnitude),
                0.0,
                1.0,
            )

            reliability = _clip(
                0.70 + (magnitude * 0.30),
                0.0,
                1.0,
            )

            likelihood_ratio = _lr(
                magnitude,
                node.direction,
            )

            output.append(
                {
                    "category": "IPO",
                    "domain": node.domain,
                    "message": str(node.message),
                    "reliability": _safe_round(
                        reliability,
                        6,
                    ),
                    "likelihood_ratio": _safe_round(
                        likelihood_ratio,
                        6,
                    ),
                }
            )

        output.sort(
            key=lambda item: abs(
                float(item.get("likelihood_ratio", 1.0)) - 1.0
            ),
            reverse=True,
        )

        return output

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _calculate_confidence(
        self,
        *,
        feature_trace_summary: Dict[str, Any],
        domain_coverage: Dict[str, Any],
        evidence: List[Dict[str, Any]],
        snapshots: List[Dict[str, Any]],
    ) -> float:

        feature_coverage = float(
            feature_trace_summary.get(
                "coverage_pct",
                0.0,
            )
        )

        domain_values = [
            float(value.get("coverage_pct", 0.0))
            for value in domain_coverage.values()
            if isinstance(value, Mapping)
        ]

        domain_coverage_pct = (
            sum(domain_values) / len(domain_values)
            if domain_values
            else 0.0
        )

        history_depth = min(
            len(snapshots) / 5.0,
            1.0,
        ) * 20.0

        evidence_component = min(
            len(evidence),
            10,
        ) * 1.0

        confidence = (
            feature_coverage * 0.45
            + domain_coverage_pct * 0.30
            + history_depth
            + evidence_component
        )

        contradictions = sum(
            1
            for item in evidence
            if item.get("domain") == "contradiction"
        )

        confidence -= contradictions * 5.0

        return _clip(
            confidence,
            0.0,
            100.0,
        )

    # ========================================================================
    # FALLBACK
    # ========================================================================

    def _empty(self) -> Dict[str, Any]:
        total_features = len(FIELD_CONTRACTS)

        missing = sorted(FIELD_CONTRACTS.keys())

        return {
            "ipo_analyzer": {
                "confidence": 0.0,

                "risk": None,
                "risk_status": "unknown",

                "valuation": None,
                "valuation_status": "unknown",

                "ipo_quality": None,
                "ipo_quality_status": "unknown",

                "listing": None,
                "listing_status": "unknown",

                "subscription": None,
                "subscription_status": "unknown",

                "institutional": None,
                "institutional_status": "unknown",

                "anchor": None,
                "anchor_status": "unknown",

                "demand": None,
                "demand_status": "unknown",

                "liquidity": None,
                "liquidity_status": "unknown",

                "business": None,
                "business_status": "unknown",

                "opportunity": None,
                "opportunity_status": "unknown",

                "feature_coverage_pct": 0.0,

                "feature_trace_summary": {
                    "total_features": total_features,
                    "used_features": 0,
                    "available_not_used": 0,
                    "missing_features": total_features,
                    "invalid_features": 0,
                    "coverage_pct": 0.0,
                    "used_feature_names": [],
                    "available_not_used_names": [],
                    "missing_feature_names": missing,
                    "invalid_feature_names": [],
                },

                "feature_trace": {
                    feature: {
                        "status": "missing",
                        "feature": feature,
                        "source_field": None,
                        "raw_value": None,
                        "multiplier": None,
                        "normalized_value": None,
                        "historical_points": 0,
                        "latest_timestamp": 0.0,
                        "aliases_checked": [
                            alias
                            for alias, _ in FIELD_CONTRACTS[feature]
                        ],
                    }
                    for feature in FIELD_CONTRACTS
                },

                "domain_coverage": {},

                "evidence": [],
            }
        }


# ============================================================================
# MODULE-LEVEL API
# ============================================================================

def analyze(
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Public module-level analyzer entry point.
    """
    return IPOAnalyzer().analyze(
        data,
        *args,
        **kwargs,
    )


__all__ = [
    "IPOAnalyzer",
    "analyze",
    "FIELD_CONTRACTS",
]
