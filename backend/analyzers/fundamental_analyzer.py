from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MIN_LR = 0.15
MAX_LR = 8.0
TRACE_VERSION = "2.0"

# ============================================================================
# STRICT FUNDAMENTAL FIELD CONTRACT
# ============================================================================
#
# multiplier converts the database/provider value into the analyzer's
# canonical representation.
#
# Percentage fields:
#   roe = 21       -> 21.0 %
#   operating_margins = 0.16 -> 16.0 %
#   revenue_growth_yoy = 0.15 -> 15.0 %
#
# Ratio fields:
#   debt_to_equity = 0.85 -> 0.85
#   trailing_pe = 22 -> 22.0
#
# No automatic unit guessing is performed during extraction.
# ============================================================================

FIELD_CONTRACTS: Dict[str, List[Tuple[str, float]]] = {
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
    "roa": [
        ("roa", 1.0),
        ("return_on_assets", 1.0),
        ("return_on_assets_pct", 1.0),
    ],
    "op_margin": [
        ("operating_margin", 1.0),
        ("operating_margins", 100.0),
        ("operating_margin_pct", 1.0),
    ],
    "net_margin": [
        ("net_margin", 1.0),
        ("profit_margins", 100.0),
        ("net_margin_pct", 1.0),
    ],
    "gross_margin": [
        ("gross_margin", 1.0),
        ("gross_margins", 100.0),
        ("gross_margin_pct", 1.0),
    ],
    "ebitda_margin": [
        ("ebitda_margin", 1.0),
        ("ebitda_margins", 100.0),
        ("ebitda_margin_pct", 1.0),
    ],
    "rev_growth": [
        ("revenue_growth_yoy", 100.0),
        ("revenue_growth", 100.0),
        ("revenue_growth_yoy_pct", 1.0),
    ],
    "earn_growth": [
        ("earnings_growth", 100.0),
        ("earnings_growth_pct", 1.0),
    ],
    "earn_growth_qtr": [
        ("earnings_quarterly_growth", 100.0),
        ("earnings_quarterly_growth_pct", 1.0),
    ],
    "debt_eq": [
        ("debt_to_equity", 1.0),
        ("debt_equity", 1.0),
    ],
    "current_ratio": [
        ("current_ratio", 1.0),
    ],
    "quick_ratio": [
        ("quick_ratio", 1.0),
    ],
    "trailing_pe": [
        ("trailing_pe", 1.0),
        ("pe_ratio", 1.0),
    ],
    "forward_pe": [
        ("forward_pe", 1.0),
    ],
    "pb": [
        ("pb_ratio", 1.0),
        ("price_to_book", 1.0),
    ],
    "ps": [
        ("price_to_sales_trailing12_months", 1.0),
        ("price_to_sales", 1.0),
    ],
    "ev_ebitda": [
        ("enterprise_to_ebitda", 1.0),
        ("ev_ebitda", 1.0),
    ],
    "peg": [
        ("peg_ratio", 1.0),
    ],
    "div_yield": [
        ("dividend_yield", 1.0),
        ("trailing_annual_dividend_yield", 1.0),
    ],
    "promoter_holding": [
        ("promoter_holding", 1.0),
        ("promoter_holding_pct", 1.0),
    ],
    "inst_holding": [
        ("held_percent_institutions", 100.0),
        ("institutional_holding", 1.0),
        ("institutional_holding_pct", 1.0),
        ("fii_holding", 1.0),
    ],
    "insider_holding": [
        ("held_percent_insiders", 100.0),
        ("insider_holding", 1.0),
        ("insider_holding_pct", 1.0),
    ],
    "pledge": [
        ("promoter_pledge", 1.0),
        ("promoter_pledge_pct", 1.0),
        ("promoter_pledged", 1.0),
    ],
    "payout_ratio": [
        ("payout_ratio", 1.0),
        ("payout_ratio_pct", 1.0),
    ],
    "eps_ttm": [
        ("trailing_eps", 1.0),
        ("eps_trailing_twelve_months", 1.0),
        ("eps", 1.0),
    ],
    "eps_fwd": [
        ("forward_eps", 1.0),
        ("eps_forward", 1.0),
    ],
    "revenue": [
        ("total_revenue", 1.0),
        ("revenue", 1.0),
    ],
    "debt": [
        ("total_debt", 1.0),
        ("debt", 1.0),
    ],
    "cash": [
        ("total_cash", 1.0),
        ("cash", 1.0),
    ],
    "ocf": [
        ("operating_cash_flow", 1.0),
        ("ocf", 1.0),
    ],
    "fcf": [
        ("free_cash_flow", 1.0),
        ("fcf", 1.0),
    ],
    "capex": [
        ("capital_expenditure", 1.0),
        ("purchase_of_ppe", 1.0),
        ("capex", 1.0),
    ],
    "ebitda": [
        ("ebitda", 1.0),
    ],
    "book_value": [
        ("book_value", 1.0),
        ("book_value_per_share", 1.0),
    ],
    "sector": [
        ("sector", 1.0),
        ("sector_key", 1.0),
    ],
    "industry": [
        ("industry", 1.0),
        ("industry_key", 1.0),
    ],
    "risk_overall": [
        ("overall_risk", 1.0),
    ],
    "risk_audit": [
        ("audit_risk", 1.0),
    ],
    "risk_board": [
        ("board_risk", 1.0),
    ],
    "risk_comp": [
        ("compensation_risk", 1.0),
    ],
    "beta": [
        ("beta", 1.0),
    ],
}


NUMERIC_FEATURES = {
    key
    for key in FIELD_CONTRACTS
    if key not in {"sector", "industry"}
}

DOMAIN_FEATURES: Dict[str, List[str]] = {
    "profitability": [
        "roe",
        "roce",
        "roa",
        "op_margin",
        "net_margin",
        "gross_margin",
        "ebitda_margin",
    ],
    "financial": [
        "debt_eq",
        "cash",
        "debt",
        "current_ratio",
        "quick_ratio",
    ],
    "growth": [
        "rev_growth",
        "earn_growth",
        "earn_growth_qtr",
    ],
    "valuation": [
        "trailing_pe",
        "forward_pe",
        "pb",
        "ps",
        "ev_ebitda",
        "peg",
    ],
    "cash_flow": [
        "ocf",
        "fcf",
        "ebitda",
    ],
    "governance": [
        "risk_overall",
        "risk_audit",
        "risk_board",
        "pledge",
    ],
    "management": [
        "risk_comp",
        "promoter_holding",
        "inst_holding",
        "insider_holding",
    ],
    "capital_allocation": [
        "roce",
        "debt_eq",
        "capex",
        "div_yield",
        "fcf",
    ],
}


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
            text = (
                value.strip()
                .replace(",", "")
                .replace("%", "")
            )
            if not text:
                return None
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
        return value or None

    return str(value).strip() or None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    value = _safe_float(value)
    return round(value, digits) if value is not None else None


def _lr(magnitude: float, direction: float) -> float:
    magnitude = _clip(abs(magnitude), 0.0, 1.0)

    base = 1.0 + magnitude * 3.0

    if direction >= 0:
        return _clip(base, MIN_LR, MAX_LR)

    return _clip(1.0 / base, MIN_LR, MAX_LR)


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


def _parse_time(value: Any) -> float:
    if value is None:
        return 0.0

    if isinstance(value, bool):
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return 0.0

    # YYYY-MM-DD / YYYY-MM-DD HH:MM:SS
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)

    if match:
        year, month, day = map(int, match.groups())
        return float(year * 10000 + month * 100 + day)

    # YYYY/MM/DD
    match = re.match(r"^(\d{4})/(\d{2})/(\d{2})", text)

    if match:
        year, month, day = map(int, match.groups())
        return float(year * 10000 + month * 100 + day)

    try:
        return float(text)
    except ValueError:
        return 0.0


def _get_date_score(row: Mapping[str, Any]) -> Any:
    date_aliases = (
        "date",
        "datetime",
        "timestamp",
        "period",
        "report_date",
        "financial_date",
        "fiscal_date",
        "as_of_date",
        "calendar_date",
        "year",
    )

    normalized = {
        str(k).strip().lower().replace("-", "_").replace(" ", "_"): v
        for k, v in row.items()
    }

    for alias in date_aliases:
        if alias in normalized:
            return normalized[alias]

    return 0.0


def _is_financial_sector(
    sector: Optional[str],
    industry: Optional[str],
) -> bool:
    text = f"{sector or ''} {industry or ''}".lower()

    return bool(
        re.search(
            r"\b(bank|banks|nbfc|insurance|financial|financials|"
            r"capital_market|asset_management)\b",
            text,
        )
    )


def _get_sector_neutral_multiples(
    sector: Optional[str],
    industry: Optional[str],
) -> Tuple[float, float, float]:
    text = f"{sector or ''} {industry or ''}".lower()

    if _is_financial_sector(sector, industry):
        return 14.0, 1.5, 10.0

    if re.search(r"\b(tech|software|it|information_technology)\b", text):
        return 30.0, 5.0, 18.0

    if re.search(r"\b(fmcg|consumer|consumer_goods)\b", text):
        return 35.0, 6.0, 22.0

    if re.search(r"\b(energy|oil|gas|metal|mining)\b", text):
        return 12.0, 1.2, 7.0

    if re.search(r"\b(pharma|health|healthcare)\b", text):
        return 25.0, 4.0, 15.0

    return 20.0, 3.0, 12.0


# ============================================================================
# ANALYZER
# ============================================================================

class FundamentalAnalyzer:
    """
    Production-grade Fundamental Analyzer.

    Design goals
    ------------
    1. Database friendly:
       Accepts dict, list[dict], pandas DataFrame and wrapped L3 payloads.

    2. L3 friendly:
       Returns deterministic JSON-serializable primitives only.

    3. Chronological:
       Historical rows are sorted before any trend/CAGR calculation.

    4. Missing-data safe:
       Missing data remains None and does not silently become zero.

    5. Traceable:
       Every contracted feature has an automatic feature_trace entry.

    6. No decision engine contamination:
       This layer produces fundamental evidence and metrics.
       It does not produce broker orders or trade execution instructions.

    7. Confidence is coverage-aware:
       Sparse fundamental data cannot receive the same confidence as a
       well-covered multi-period dataset.

    8. Provenance:
       Trace records the source database/provider field, raw value,
       multiplier and normalized value.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

        self.field_contracts = FIELD_CONTRACTS
        self.domain_features = DOMAIN_FEATURES

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------

    def analyze(
        self,
        data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        try:
            snapshots = self._parse_payload_chronologically(data)

            if not snapshots:
                return self._empty(
                    reason="No valid fundamental snapshots supplied."
                )

            latest_snapshot = self._get_latest_valid_snapshot(snapshots)

            if latest_snapshot is None:
                latest_snapshot = snapshots[-1]

            # --------------------------------------------------------------
            # EXTRACT ALL FEATURES + TRACE
            # --------------------------------------------------------------

            metrics_history: Dict[
                str,
                List[Tuple[float, float]],
            ] = {}

            feature_trace: Dict[str, Dict[str, Any]] = {}

            for feature in NUMERIC_FEATURES:
                aliases = self.field_contracts.get(feature, [])

                series: List[Tuple[float, float]] = []

                # Historical extraction.
                for snapshot in snapshots:
                    found = False

                    for alias, multiplier in aliases:
                        if alias not in snapshot:
                            continue

                        raw_value = _num(snapshot.get(alias))

                        if raw_value is None:
                            continue

                        normalized = raw_value * multiplier

                        if not math.isfinite(normalized):
                            continue

                        series.append(
                            (
                                float(snapshot.get("_t", 0.0)),
                                normalized,
                            )
                        )

                        found = True
                        break

                    if found:
                        continue

                if series:
                    metrics_history[feature] = series

                # Latest-valid trace.
                trace = self._trace_feature(
                    feature=feature,
                    snapshot=latest_snapshot,
                    series=series,
                )

                feature_trace[feature] = trace

            # String features.
            sector = self._extract_str(
                latest_snapshot,
                "sector",
            )

            industry = self._extract_str(
                latest_snapshot,
                "industry",
            )

            feature_trace["sector"] = self._trace_string_feature(
                "sector",
                latest_snapshot,
            )

            feature_trace["industry"] = self._trace_string_feature(
                "industry",
                latest_snapshot,
            )

            latest_metrics: Dict[str, Optional[float]] = {
                feature: None
                for feature in NUMERIC_FEATURES
            }

            trend_deltas: Dict[str, Optional[float]] = {
                feature: None
                for feature in NUMERIC_FEATURES
            }

            for feature, series in metrics_history.items():
                if not series:
                    continue

                latest_metrics[feature] = series[-1][1]

                if len(series) >= 2:
                    trend_deltas[feature] = (
                        series[-1][1] - series[0][1]
                    )

            is_financial = _is_financial_sector(
                sector,
                industry,
            )

            pe_neutral, pb_neutral, ev_neutral = (
                _get_sector_neutral_multiples(
                    sector,
                    industry,
                )
            )

            used_features: set[str] = set()

            nodes: List[EvidenceNode] = []

            # --------------------------------------------------------------
            # PROFITABILITY
            # --------------------------------------------------------------

            prof_score = 0.0
            prof_weight = 0.0

            roe = latest_metrics.get("roe")

            if roe is not None:
                used_features.add("roe")
                prof_weight += 1.5

                score = _continuous_score(
                    roe,
                    neutral=12.0,
                    scale=6.0,
                )

                prof_score += score * 1.5

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                message = f"ROE is {roe:.1f}%."

                delta = trend_deltas.get("roe")

                if delta is not None and abs(delta) > 2.0:
                    message += (
                        " Structural ROE "
                        f"{'expansion' if delta > 0 else 'compression'} "
                        "observed."
                    )

                    magnitude = _clip(
                        magnitude + 0.2,
                        0.0,
                        1.0,
                    )

                nodes.append(
                    EvidenceNode(
                        "profitability",
                        direction,
                        magnitude,
                        message,
                    )
                )

            roce = latest_metrics.get("roce")

            if roce is not None:
                used_features.add("roce")
                prof_weight += 1.0

                prof_score += _continuous_score(
                    roce,
                    neutral=12.0,
                    scale=6.0,
                )

            roa = latest_metrics.get("roa")

            if roa is not None:
                used_features.add("roa")

                if is_financial:
                    prof_weight += 1.5

                    score = _continuous_score(
                        roa,
                        neutral=1.2,
                        scale=0.6,
                    )

                    prof_score += score * 1.5

                    magnitude = _clip(
                        abs(score - 50.0) / 50.0,
                        0.1,
                        1.0,
                    )

                    direction = (
                        1.0 if score >= 50.0 else -1.0
                    )

                    nodes.append(
                        EvidenceNode(
                            "profitability",
                            direction,
                            magnitude,
                            f"ROA stands at {roa:.2f}%.",
                        )
                    )

            op_margin = latest_metrics.get("op_margin")

            if op_margin is not None:
                used_features.add("op_margin")
                prof_weight += 1.0

                score = _continuous_score(
                    op_margin,
                    neutral=10.0,
                    scale=8.0,
                )

                prof_score += score

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                message = (
                    f"Operating margin is {op_margin:.1f}%."
                )

                delta = trend_deltas.get("op_margin")

                if delta is not None and abs(delta) > 1.5:
                    message += (
                        " Margins are "
                        f"{'expanding' if delta > 0 else 'contracting'}."
                    )

                nodes.append(
                    EvidenceNode(
                        "profitability",
                        direction,
                        magnitude,
                        message,
                    )
                )

            net_margin = latest_metrics.get("net_margin")

            if net_margin is not None:
                used_features.add("net_margin")
                prof_weight += 1.0

                prof_score += _continuous_score(
                    net_margin,
                    neutral=6.0,
                    scale=5.0,
                )

            gross_margin = latest_metrics.get("gross_margin")

            if gross_margin is not None:
                used_features.add("gross_margin")
                prof_weight += 0.5

                prof_score += (
                    _continuous_score(
                        gross_margin,
                        neutral=30.0,
                        scale=15.0,
                    )
                    * 0.5
                )

            ebitda_margin = latest_metrics.get("ebitda_margin")

            if ebitda_margin is not None:
                used_features.add("ebitda_margin")
                prof_weight += 0.5

                prof_score += (
                    _continuous_score(
                        ebitda_margin,
                        neutral=15.0,
                        scale=10.0,
                    )
                    * 0.5
                )

            prof_final = (
                prof_score / prof_weight
                if prof_weight > 0
                else None
            )

            prof_status = self._status(
                prof_final,
                positive="high",
                negative="low",
            )

            # --------------------------------------------------------------
            # FINANCIAL HEALTH
            # --------------------------------------------------------------

            fin_score = 0.0
            fin_weight = 0.0

            de = latest_metrics.get("debt_eq")

            if de is not None:
                # Preserve canonical ratio.
                # Some DB sources report 85 for 85%.
                # Only extreme values are normalized.
                if de > 10.0:
                    de = de / 100.0
                    latest_metrics["debt_eq"] = de

                used_features.add("debt_eq")
                fin_weight += 1.5

                if is_financial:
                    neutral = 4.0
                    scale = 2.0
                else:
                    neutral = 1.0
                    scale = 0.8

                score = _continuous_score(
                    de,
                    neutral=neutral,
                    scale=scale,
                    invert=True,
                )

                fin_score += score * 1.5

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                message = (
                    f"Debt-to-Equity ratio is {de:.2f}."
                )

                delta = trend_deltas.get("debt_eq")

                if delta is not None and not is_financial:
                    if delta < -0.2:
                        message += (
                            " Active deleveraging phase evident."
                        )
                    elif delta > 0.2:
                        message += (
                            " Increasing leverage profile."
                        )

                nodes.append(
                    EvidenceNode(
                        "financial",
                        direction,
                        magnitude,
                        message,
                    )
                )

            cash = latest_metrics.get("cash")
            debt = latest_metrics.get("debt")

            if (
                cash is not None
                and debt is not None
                and not is_financial
            ):
                used_features.add("cash")
                used_features.add("debt")

                fin_weight += 1.0

                if debt <= 0:
                    fin_score += 100.0

                    nodes.append(
                        EvidenceNode(
                            "financial",
                            1.0,
                            1.0,
                            "Zero reported debt minimizes "
                            "balance-sheet leverage risk.",
                        )
                    )
                else:
                    cash_coverage = cash / debt

                    score = _continuous_score(
                        cash_coverage,
                        neutral=0.4,
                        scale=0.3,
                    )

                    fin_score += score

                    if cash_coverage >= 1.0:
                        nodes.append(
                            EvidenceNode(
                                "financial",
                                1.0,
                                0.8,
                                "Net cash positive position "
                                "relative to existing debt.",
                            )
                        )

            cr = latest_metrics.get("current_ratio")

            if cr is not None and not is_financial:
                used_features.add("current_ratio")
                fin_weight += 1.0

                score = _continuous_score(
                    cr,
                    neutral=1.2,
                    scale=0.5,
                )

                fin_score += score

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                nodes.append(
                    EvidenceNode(
                        "financial",
                        direction,
                        magnitude,
                        f"Current ratio is {cr:.2f}.",
                    )
                )

            qr = latest_metrics.get("quick_ratio")

            if qr is not None and not is_financial:
                used_features.add("quick_ratio")
                fin_weight += 0.5

                fin_score += (
                    _continuous_score(
                        qr,
                        neutral=0.8,
                        scale=0.4,
                    )
                    * 0.5
                )

            fin_final = (
                fin_score / fin_weight
                if fin_weight > 0
                else None
            )

            fin_status = self._status(
                fin_final,
                positive="healthy",
                negative="weak",
            )

            # --------------------------------------------------------------
            # GROWTH
            # --------------------------------------------------------------

            grow_score = 0.0
            grow_weight = 0.0

            rg = latest_metrics.get("rev_growth")

            if rg is not None:
                used_features.add("rev_growth")
                grow_weight += 1.0

                score = _continuous_score(
                    rg,
                    neutral=5.0,
                    scale=8.0,
                )

                grow_score += score

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                nodes.append(
                    EvidenceNode(
                        "growth",
                        direction,
                        magnitude,
                        "Revenue is "
                        f"{'expanding' if direction > 0 else 'contracting'} "
                        f"YoY ({rg:.1f}%).",
                    )
                )

            eg_yoy = latest_metrics.get("earn_growth")
            eg_qtr = latest_metrics.get("earn_growth_qtr")

            eg = (
                eg_yoy
                if eg_yoy is not None
                else eg_qtr
            )

            if eg is not None:
                used_features.add(
                    "earn_growth"
                    if eg_yoy is not None
                    else "earn_growth_qtr"
                )

                grow_weight += 1.5

                score = _continuous_score(
                    eg,
                    neutral=8.0,
                    scale=12.0,
                )

                grow_score += score * 1.5

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                nodes.append(
                    EvidenceNode(
                        "growth",
                        direction,
                        magnitude,
                        "Earnings are "
                        f"{'growing' if direction > 0 else 'declining'} "
                        f"({eg:.1f}%).",
                    )
                )

            grow_final = (
                grow_score / grow_weight
                if grow_weight > 0
                else None
            )

            grow_status = self._status(
                grow_final,
                positive="strong",
                negative="declining",
            )

            # --------------------------------------------------------------
            # VALUATION
            # --------------------------------------------------------------

            val_score = 0.0
            val_weight = 0.0
            is_unprofitable = False

            pe_ttm = latest_metrics.get("trailing_pe")
            pe_fwd = latest_metrics.get("forward_pe")

            if pe_ttm is not None and pe_ttm < 0:
                is_unprofitable = True
                used_features.add("trailing_pe")
                val_weight += 2.0

                nodes.append(
                    EvidenceNode(
                        "valuation",
                        -1.0,
                        0.9,
                        "Negative trailing P/E highlights "
                        "unprofitability, structurally impairing "
                        "earnings-based valuation.",
                    )
                )

            else:
                active_pe = (
                    pe_ttm
                    if pe_ttm is not None and pe_ttm > 0
                    else pe_fwd
                )

                if active_pe is not None and active_pe > 0:
                    feature_name = (
                        "trailing_pe"
                        if pe_ttm is not None and pe_ttm > 0
                        else "forward_pe"
                    )

                    used_features.add(feature_name)
                    val_weight += 1.5

                    score = _continuous_score(
                        active_pe,
                        neutral=pe_neutral,
                        scale=pe_neutral * 0.4,
                        invert=True,
                    )

                    val_score += score * 1.5

                    magnitude = _clip(
                        abs(score - 50.0) / 50.0,
                        0.1,
                        1.0,
                    )

                    direction = (
                        1.0
                        if score >= 50.0
                        else -1.0
                    )

                    pe_type = (
                        "Trailing"
                        if feature_name == "trailing_pe"
                        else "Forward"
                    )

                    nodes.append(
                        EvidenceNode(
                            "valuation",
                            direction,
                            magnitude,
                            f"{pe_type} P/E "
                            f"({active_pe:.1f}x) vs Sector "
                            f"Baseline ({pe_neutral}x).",
                        )
                    )

            pb = latest_metrics.get("pb")

            if pb is not None and pb > 0:
                used_features.add("pb")
                val_weight += 1.0

                score = _continuous_score(
                    pb,
                    neutral=pb_neutral,
                    scale=pb_neutral * 0.4,
                    invert=True,
                )

                if is_unprofitable:
                    score = min(score, 40.0)

                val_score += score

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                nodes.append(
                    EvidenceNode(
                        "valuation",
                        direction,
                        magnitude,
                        f"P/B ratio ({pb:.1f}x) vs "
                        f"Sector Baseline ({pb_neutral}x).",
                    )
                )

            ev = latest_metrics.get("ev_ebitda")

            if (
                ev is not None
                and ev > 0
                and not is_financial
            ):
                used_features.add("ev_ebitda")
                val_weight += 1.0

                score = _continuous_score(
                    ev,
                    neutral=ev_neutral,
                    scale=ev_neutral * 0.4,
                    invert=True,
                )

                if is_unprofitable:
                    score = min(score, 40.0)

                val_score += score

            peg = latest_metrics.get("peg")

            if peg is not None and peg > 0:
                used_features.add("peg")
                val_weight += 1.0

                val_score += _continuous_score(
                    peg,
                    neutral=1.2,
                    scale=0.8,
                    invert=True,
                )

            ps = latest_metrics.get("ps")

            if ps is not None and ps > 0:
                used_features.add("ps")
                val_weight += 0.5

                val_score += (
                    _continuous_score(
                        ps,
                        neutral=2.5,
                        scale=2.0,
                        invert=True,
                    )
                    * 0.5
                )

            val_final = (
                val_score / val_weight
                if val_weight > 0
                else None
            )

            val_status = self._status(
                val_final,
                positive="attractive",
                negative="expensive",
            )

            # --------------------------------------------------------------
            # CASH FLOW
            # --------------------------------------------------------------

            cf_score = 0.0
            cf_weight = 0.0

            ocf = latest_metrics.get("ocf")
            ebitda = latest_metrics.get("ebitda")
            fcf = latest_metrics.get("fcf")

            if ocf is not None:
                used_features.add("ocf")
                cf_weight += 1.0

                positive = ocf > 0

                cf_score += 100.0 if positive else 0.0

                nodes.append(
                    EvidenceNode(
                        "cash_flow",
                        1.0 if positive else -1.0,
                        0.8,
                        (
                            "Operating Cash Flow is positive."
                            if positive
                            else
                            "Operating Cash Flow indicates cash burn."
                        ),
                    )
                )

                if ebitda is not None and ebitda > 0:
                    used_features.add("ebitda")
                    cf_weight += 1.0

                    conversion = ocf / ebitda

                    score = _continuous_score(
                        conversion,
                        neutral=0.7,
                        scale=0.3,
                    )

                    cf_score += score

                    if score > 75:
                        nodes.append(
                            EvidenceNode(
                                "cash_flow",
                                1.0,
                                0.7,
                                "High Earnings Quality: "
                                f"Superior cash conversion "
                                f"({conversion:.2f}).",
                            )
                        )

                    elif score < 25 and ocf > 0:
                        nodes.append(
                            EvidenceNode(
                                "cash_flow",
                                -1.0,
                                0.7,
                                "Poor Earnings Quality: "
                                f"Weak cash conversion "
                                f"({conversion:.2f}).",
                            )
                        )

            if fcf is not None:
                used_features.add("fcf")
                cf_weight += 1.0

                positive = fcf > 0

                cf_score += 100.0 if positive else 0.0

                nodes.append(
                    EvidenceNode(
                        "cash_flow",
                        1.0 if positive else -1.0,
                        0.8,
                        (
                            "Free Cash Flow generation is positive."
                            if positive
                            else
                            "Free Cash Flow remains negative."
                        ),
                    )
                )

            cf_final = (
                cf_score / cf_weight
                if cf_weight > 0
                else None
            )

            cf_status = self._status(
                cf_final,
                positive="strong",
                negative="negative",
            )

            # --------------------------------------------------------------
            # GOVERNANCE
            # --------------------------------------------------------------

            gov_score = 0.0
            gov_weight = 0.0

            risk_overall = latest_metrics.get("risk_overall")

            if risk_overall is not None:
                used_features.add("risk_overall")
                gov_weight += 2.0

                score = _continuous_score(
                    risk_overall,
                    neutral=5.0,
                    scale=2.5,
                    invert=True,
                )

                gov_score += score * 2.0

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                nodes.append(
                    EvidenceNode(
                        "governance",
                        direction,
                        magnitude,
                        "Overall quantitative governance "
                        f"risk score is {risk_overall}.",
                    )
                )

            risk_audit = latest_metrics.get("risk_audit")

            if risk_audit is not None:
                used_features.add("risk_audit")
                gov_weight += 0.5

                gov_score += (
                    _continuous_score(
                        risk_audit,
                        neutral=5.0,
                        scale=2.5,
                        invert=True,
                    )
                    * 0.5
                )

            risk_board = latest_metrics.get("risk_board")

            if risk_board is not None:
                used_features.add("risk_board")
                gov_weight += 0.5

                gov_score += (
                    _continuous_score(
                        risk_board,
                        neutral=5.0,
                        scale=2.5,
                        invert=True,
                    )
                    * 0.5
                )

            pledge = latest_metrics.get("pledge")

            if pledge is not None:
                used_features.add("pledge")
                gov_weight += 1.0

                score = _continuous_score(
                    pledge,
                    neutral=10.0,
                    scale=8.0,
                    invert=True,
                )

                gov_score += score

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                if pledge > 5.0:
                    nodes.append(
                        EvidenceNode(
                            "governance",
                            -1.0,
                            magnitude,
                            f"Promoter pledge stands at "
                            f"{pledge:.1f}%, creating structural "
                            "overhang.",
                        )
                    )

            gov_final = (
                gov_score / gov_weight
                if gov_weight > 0
                else None
            )

            gov_status = self._status(
                gov_final,
                positive="strong",
                negative="poor",
            )

            # --------------------------------------------------------------
            # MANAGEMENT / OWNERSHIP
            # --------------------------------------------------------------

            mgmt_score = 0.0
            mgmt_weight = 0.0

            for risk_value, description, key in (
                (
                    latest_metrics.get("risk_comp"),
                    "Compensation",
                    "risk_comp",
                ),
                (
                    latest_metrics.get("risk_board"),
                    "Board",
                    "risk_board",
                ),
            ):
                if risk_value is None:
                    continue

                used_features.add(key)
                mgmt_weight += 1.0

                score = _continuous_score(
                    risk_value,
                    neutral=5.0,
                    scale=2.5,
                    invert=True,
                )

                mgmt_score += score

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = 1.0 if score >= 50.0 else -1.0

                nodes.append(
                    EvidenceNode(
                        "management",
                        direction,
                        magnitude,
                        f"Management {description.lower()} "
                        f"risk score is {risk_value}.",
                    )
                )

            promoter = latest_metrics.get("promoter_holding")

            if promoter is not None:
                used_features.add("promoter_holding")
                mgmt_weight += 1.0

                score = _continuous_score(
                    promoter,
                    neutral=40.0,
                    scale=15.0,
                )

                mgmt_score += score

                if promoter > 50.0:
                    nodes.append(
                        EvidenceNode(
                            "management",
                            1.0,
                            0.6,
                            "High promoter skin-in-the-game "
                            f"({promoter:.1f}%).",
                        )
                    )

            inst = latest_metrics.get("inst_holding")

            if inst is not None:
                used_features.add("inst_holding")
                mgmt_weight += 1.0

                score = _continuous_score(
                    inst,
                    neutral=20.0,
                    scale=10.0,
                )

                mgmt_score += score

                if inst > 25.0:
                    nodes.append(
                        EvidenceNode(
                            "management",
                            1.0,
                            0.6,
                            "Significant institutional "
                            f"validation ({inst:.1f}%).",
                        )
                    )

            insider = latest_metrics.get("insider_holding")

            if insider is not None:
                used_features.add("insider_holding")
                mgmt_weight += 0.5

                mgmt_score += (
                    _continuous_score(
                        insider,
                        neutral=5.0,
                        scale=5.0,
                    )
                    * 0.5
                )

            mgmt_final = (
                mgmt_score / mgmt_weight
                if mgmt_weight > 0
                else None
            )

            mgmt_status = self._status(
                mgmt_final,
                positive="excellent",
                negative="poor",
            )

            # --------------------------------------------------------------
            # CAPITAL ALLOCATION
            # --------------------------------------------------------------

            cap_score = 0.0
            cap_weight = 0.0

            if (
                roce is not None
                and de is not None
                and not is_financial
            ):
                used_features.add("roce")
                used_features.add("debt_eq")

                cap_weight += 2.0

                allocation_efficiency = (
                    roce / max(de, 0.2)
                )

                score = _continuous_score(
                    allocation_efficiency,
                    neutral=15.0,
                    scale=10.0,
                )

                cap_score += score * 2.0

                magnitude = _clip(
                    abs(score - 50.0) / 50.0,
                    0.1,
                    1.0,
                )

                direction = (
                    1.0 if score >= 50.0 else -1.0
                )

                nodes.append(
                    EvidenceNode(
                        "capital_allocation",
                        direction,
                        magnitude,
                        (
                            "High ROCE relative to leverage "
                            "suggests superior capital deployment."
                            if direction > 0
                            else
                            "Poor ROCE relative to leverage "
                            "indicates inefficient allocation."
                        ),
                    )
                )

            capex = latest_metrics.get("capex")

            if capex is not None and ocf is not None and ocf > 0:
                used_features.add("capex")
                cap_weight += 1.0

                reinvestment = abs(capex) / ocf

                cap_score += _continuous_score(
                    reinvestment,
                    neutral=0.5,
                    scale=0.3,
                )

            div_yield = latest_metrics.get("div_yield")

            if div_yield is not None:
                used_features.add("div_yield")
                cap_weight += 1.0

                if div_yield > 0.5:
                    if fcf is not None and fcf > 0:
                        cap_score += 100.0

                        nodes.append(
                            EvidenceNode(
                                "capital_allocation",
                                1.0,
                                0.6,
                                "Sustainable dividend yield "
                                f"({div_yield:.1f}%) backed by "
                                "FCF realization.",
                            )
                        )
                    else:
                        cap_score += 40.0
                else:
                    cap_score += 50.0

            cap_final = (
                cap_score / cap_weight
                if cap_weight > 0
                else None
            )

            cap_status = self._status(
                cap_final,
                positive="efficient",
                negative="inefficient",
            )

            # --------------------------------------------------------------
            # SYNTHESIS
            # --------------------------------------------------------------

            business_score = None

            if (
                prof_final is not None
                and grow_final is not None
                and fin_final is not None
            ):
                business_score = _clip(
                    prof_final * 0.4
                    + grow_final * 0.4
                    + fin_final * 0.2,
                    0.0,
                    100.0,
                )

            elif (
                prof_final is not None
                and grow_final is not None
            ):
                business_score = _clip(
                    prof_final * 0.6
                    + grow_final * 0.4,
                    0.0,
                    100.0,
                )

            elif prof_final is not None:
                business_score = prof_final

            bus_status = self._status(
                business_score,
                positive="excellent",
                negative="weak",
            )

            wealth_creation = None

            if (
                prof_final is not None
                and grow_final is not None
                and val_final is not None
            ):
                wealth_creation = _clip(
                    prof_final * 0.4
                    + grow_final * 0.4
                    + val_final * 0.2,
                    0.0,
                    100.0,
                )

            wealth_status = self._status(
                wealth_creation,
                positive="high",
                negative="low",
            )

            longevity = None

            if (
                fin_final is not None
                and prof_final is not None
            ):
                longevity_score = (
                    fin_final * 0.6
                    + prof_final * 0.4
                )

                if gov_final is not None:
                    longevity_score = (
                        longevity_score * 0.8
                        + gov_final * 0.2
                    )

                longevity = _clip(
                    longevity_score,
                    0.0,
                    100.0,
                )

            longevity_status = self._status(
                longevity,
                positive="high",
                negative="low",
            )

            # --------------------------------------------------------------
            # BUSINESS STABILITY
            # --------------------------------------------------------------

            business_stability = None

            margin_series = metrics_history.get(
                "op_margin",
                [],
            )

            if (
                fin_final is not None
                and len(margin_series) >= 2
            ):
                changes = [
                    abs(
                        margin_series[i][1]
                        - margin_series[i - 1][1]
                    )
                    for i in range(1, len(margin_series))
                ]

                average_change = (
                    sum(changes) / len(changes)
                    if changes
                    else 0.0
                )

                stability_score = _continuous_score(
                    average_change,
                    neutral=3.0,
                    scale=2.0,
                    invert=True,
                )

                business_stability = _clip(
                    fin_final * 0.5
                    + stability_score * 0.5,
                    0.0,
                    100.0,
                )

            stability_status = self._status(
                business_stability,
                positive="stable",
                negative="volatile",
            )

            # --------------------------------------------------------------
            # EXPECTED CAGR
            # --------------------------------------------------------------

            expected_cagr = None
            cagr_components: List[float] = []

            eps_series = metrics_history.get(
                "eps_ttm",
                [],
            )

            if (
                len(eps_series) >= 2
                and eps_series[0][1] > 0
                and eps_series[-1][1] > 0
            ):
                first_t = eps_series[0][0]
                last_t = eps_series[-1][0]

                years = self._years_between(
                    first_t,
                    last_t,
                )

                if years > 0:
                    try:
                        hist_cagr = (
                            math.pow(
                                eps_series[-1][1]
                                / eps_series[0][1],
                                1.0 / years,
                            )
                            - 1.0
                        ) * 100.0

                        if math.isfinite(hist_cagr):
                            cagr_components.append(
                                hist_cagr
                            )
                    except (ValueError, OverflowError):
                        pass

            if roe is not None and roe > 0:
                payout = latest_metrics.get(
                    "payout_ratio"
                )

                if (
                    payout is None
                    and div_yield is not None
                    and pe_ttm is not None
                    and pe_ttm > 0
                ):
                    payout = (
                        div_yield / 100.0
                    ) * pe_ttm

                if payout is not None:
                    if payout > 2.0:
                        payout /= 100.0

                    payout = _clip(
                        payout,
                        0.0,
                        1.0,
                    )
                else:
                    payout = 0.2

                sgr = roe * (1.0 - payout)

                if math.isfinite(sgr):
                    cagr_components.append(sgr)

            if cagr_components:
                expected_cagr = _clip(
                    sum(cagr_components)
                    / len(cagr_components),
                    -20.0,
                    35.0,
                )

            cagr_status = self._status_cagr(
                expected_cagr
            )

            # --------------------------------------------------------------
            # MOAT
            # --------------------------------------------------------------

            moat_score = None

            roe_series = metrics_history.get(
                "roe",
                [],
            )

            gross_series = metrics_history.get(
                "gross_margin",
                [],
            )

            if (
                len(roe_series) >= 3
                and len(gross_series) >= 2
            ):
                recent_roe = [
                    value
                    for _, value
                    in roe_series[-3:]
                ]

                recent_gross = [
                    value
                    for _, value
                    in gross_series[-3:]
                ]

                min_roe = min(recent_roe)
                average_gross = (
                    sum(recent_gross)
                    / len(recent_gross)
                )

                if (
                    min_roe > 15.0
                    and average_gross > 35.0
                ):
                    moat_score = _clip(
                        (
                            min_roe / 25.0
                        ) * 50.0
                        + (
                            average_gross / 50.0
                        ) * 50.0,
                        0.0,
                        100.0,
                    )

                    nodes.append(
                        EvidenceNode(
                            "moat",
                            1.0,
                            moat_score / 100.0,
                            "Sustained multi-period high ROE "
                            "combined with robust Gross Margins "
                            "provides structural evidence of an "
                            "economic moat.",
                        )
                    )

            moat_status = (
                "wide"
                if moat_score is not None
                and moat_score > 60.0
                else
                "none"
                if moat_score is not None
                else
                "unknown"
            )

            # Explicitly unevaluable.
            innovation_score = None
            leadership_score = None

            # --------------------------------------------------------------
            # CROSS-DOMAIN CONTRADICTIONS
            # --------------------------------------------------------------

            if (
                val_final is not None
                and grow_final is not None
                and val_final > 70.0
                and grow_final < 30.0
            ):
                nodes.append(
                    EvidenceNode(
                        "fundamental",
                        -1.0,
                        0.8,
                        "Contradiction: Attractive valuation "
                        "coupled with deteriorating growth "
                        "signals a potential value trap.",
                    )
                )

            if (
                cf_final is not None
                and prof_final is not None
                and prof_final > 70.0
                and cf_final < 30.0
            ):
                nodes.append(
                    EvidenceNode(
                        "fundamental",
                        -1.0,
                        0.9,
                        "Contradiction: High accounting "
                        "profitability but weak cash flows "
                        "indicates severe earnings quality risk.",
                    )
                )

            # --------------------------------------------------------------
            # EVIDENCE
            # --------------------------------------------------------------

            evidence_output: List[Dict[str, Any]] = []
            seen_messages: set[str] = set()

            for node in nodes:
                if node.message in seen_messages:
                    continue

                seen_messages.add(node.message)

                reliability = _safe_float(
                    round(
                        _clip(
                            0.7
                            + node.magnitude * 0.3,
                            0.0,
                            1.0,
                        ),
                        6,
                    )
                )

                likelihood_ratio = _safe_float(
                    round(
                        _lr(
                            node.magnitude,
                            node.direction,
                        ),
                        6,
                    )
                )

                evidence_output.append(
                    {
                        "category": "Fundamental",
                        "domain": node.domain,
                        "message": str(node.message),
                        "reliability": reliability,
                        "likelihood_ratio": likelihood_ratio,
                    }
                )

            evidence_output.sort(
                key=lambda item: abs(
                    item.get(
                        "likelihood_ratio",
                        1.0,
                    ) - 1.0
                ),
                reverse=True,
            )

            # --------------------------------------------------------------
            # FEATURE COVERAGE / TRACE
            # --------------------------------------------------------------

            trace_summary = self._build_feature_trace_summary(
                feature_trace=feature_trace,
                used_features=used_features,
            )

            feature_coverage_pct = (
                trace_summary["coverage_pct"]
            )

            # --------------------------------------------------------------
            # DOMAIN COVERAGE
            # --------------------------------------------------------------

            domain_scores = {
                "profitability": prof_final,
                "financial": fin_final,
                "growth": grow_final,
                "valuation": val_final,
                "cash_flow": cf_final,
                "governance": gov_final,
                "management": mgmt_final,
                "capital_allocation": cap_final,
            }

            domain_trace = self._build_domain_trace(
                feature_trace,
                used_features,
            )

            domain_coverage = (
                sum(
                    1
                    for value in domain_scores.values()
                    if value is not None
                )
                / float(len(domain_scores))
            )

            unique_dates = len(
                {
                    snapshot.get("_t")
                    for snapshot in snapshots
                    if snapshot.get("_t") is not None
                }
            )

            historical_depth = _clip(
                unique_dates / 5.0,
                0.0,
                1.0,
            )

            # --------------------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------------------

            evidence_bonus = min(
                len(evidence_output),
                10,
            ) * 1.0

            confidence = (
                domain_coverage * 50.0
                + (feature_coverage_pct / 100.0) * 20.0
                + historical_depth * 20.0
                + evidence_bonus
            )

            contradictions = sum(
                1
                for evidence in evidence_output
                if "Contradiction"
                in evidence["message"]
            )

            confidence -= contradictions * 10.0

            final_confidence = _clip(
                confidence,
                0.0,
                100.0,
            )

            # --------------------------------------------------------------
            # FINAL VALUES
            # --------------------------------------------------------------

            eps_value = (
                latest_metrics.get("eps_ttm")
                if latest_metrics.get("eps_ttm")
                is not None
                else latest_metrics.get("eps_fwd")
            )

            debt_value = latest_metrics.get("debt")

            return {
                "fundamental_analyzer": {
                    "trace_version": TRACE_VERSION,

                    "confidence": _round(
                        final_confidence
                    ),

                    "feature_coverage_pct": _round(
                        feature_coverage_pct,
                        2,
                    ),

                    "feature_trace_summary": trace_summary,

                    "feature_trace": feature_trace,

                    "domain_trace": domain_trace,

                    "roe": _safe_float(
                        latest_metrics.get("roe")
                    ),
                    "roce": _safe_float(
                        latest_metrics.get("roce")
                    ),
                    "margin": _safe_float(
                        latest_metrics.get("op_margin")
                    ),
                    "eps": _safe_float(eps_value),
                    "revenue": _safe_float(
                        latest_metrics.get("revenue")
                    ),
                    "debt": _safe_float(debt_value),

                    "profitability": _round(
                        prof_final
                    ),
                    "profitability_status": prof_status,

                    "business": _round(
                        business_score
                    ),
                    "business_status": bus_status,

                    "financial": _round(
                        fin_final
                    ),
                    "financial_status": fin_status,

                    "growth": _round(
                        grow_final
                    ),
                    "growth_status": grow_status,

                    "valuation": _round(
                        val_final
                    ),
                    "valuation_status": val_status,

                    "cash_flow": _round(
                        cf_final
                    ),
                    "cash_flow_status": cf_status,

                    "management": _round(
                        mgmt_final
                    ),
                    "management_status": mgmt_status,

                    "capital_allocation": _round(
                        cap_final
                    ),
                    "capital_allocation_status": cap_status,

                    "moat": _round(
                        moat_score
                    ),
                    "moat_status": moat_status,

                    "governance": _round(
                        gov_final
                    ),
                    "governance_status": gov_status,

                    "innovation": innovation_score,
                    "leadership": leadership_score,

                    "business_stability": _round(
                        business_stability
                    ),
                    "business_stability_status":
                        stability_status,

                    "wealth_creation": _round(
                        wealth_creation
                    ),
                    "wealth_creation_status":
                        wealth_status,

                    "longevity": _round(
                        longevity
                    ),
                    "longevity_status":
                        longevity_status,

                    "expected_cagr": _round(
                        expected_cagr
                    ),
                    "expected_cagr_status":
                        cagr_status,

                    "evidence": evidence_output[:8],
                }
            }

        except Exception as exc:
            self.logger.exception(
                "Fundamental Analyzer Critical Failure: %s",
                exc,
            )

            result = self._empty(
                reason="Execution encountered a critical failure."
            )

            return result

    # ----------------------------------------------------------------------
    # EMPTY RESULT
    # ----------------------------------------------------------------------

    def _empty(
        self,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:

        message = reason or "No valid fundamental data."

        return {
            "fundamental_analyzer": {
                "trace_version": TRACE_VERSION,

                "feature_coverage_pct": 0.0,

                "feature_trace_summary": {
                    "total_features": len(
                        NUMERIC_FEATURES
                    ),
                    "used_features": 0,
                    "available_not_used": 0,
                    "missing_features": len(
                        NUMERIC_FEATURES
                    ),
                    "coverage_pct": 0.0,
                },

                "feature_trace": {},

                "domain_trace": {},

                "confidence": 0.0,

                "roe": None,
                "roce": None,
                "margin": None,
                "eps": None,
                "revenue": None,
                "debt": None,

                "profitability": None,
                "profitability_status": "unknown",

                "business": None,
                "business_status": "unknown",

                "financial": None,
                "financial_status": "unknown",

                "growth": None,
                "growth_status": "unknown",

                "valuation": None,
                "valuation_status": "unknown",

                "cash_flow": None,
                "cash_flow_status": "unknown",

                "management": None,
                "management_status": "unknown",

                "capital_allocation": None,
                "capital_allocation_status": "unknown",

                "moat": None,
                "moat_status": "unknown",

                "governance": None,
                "governance_status": "unknown",

                "innovation": None,
                "leadership": None,

                "business_stability": None,
                "business_stability_status": "unknown",

                "wealth_creation": None,
                "wealth_creation_status": "unknown",

                "longevity": None,
                "longevity_status": "unknown",

                "expected_cagr": None,
                "expected_cagr_status": "unknown",

                "evidence": [
                    {
                        "category": "Fundamental",
                        "domain": "system",
                        "message": message,
                        "reliability": 0.96,
                        "likelihood_ratio": 0.42,
                    }
                ],
            }
        }

    # ----------------------------------------------------------------------
    # PARSING
    # ----------------------------------------------------------------------

    def _parse_payload_chronologically(
        self,
        data: Any,
    ) -> List[Dict[str, Any]]:

        raw_rows: List[Dict[str, Any]] = []

        if data is None:
            return raw_rows

        # pandas DataFrame support without importing pandas.
        if (
            hasattr(data, "to_dict")
            and hasattr(data, "columns")
        ):
            try:
                records = data.to_dict(
                    orient="records"
                )

                if isinstance(records, list):
                    raw_rows.extend(
                        row
                        for row in records
                        if isinstance(row, dict)
                    )

            except Exception:
                self.logger.debug(
                    "Unable to convert DataFrame payload.",
                    exc_info=True,
                )

        elif isinstance(data, Mapping):

            list_wrappers = (
                "features",
                "historical_data",
                "fundamental_history",
                "rows",
                "fundamental_data",
                "history",
                "records",
                "snapshots",
                "data",
            )

            found_list = False

            for key in list_wrappers:
                value = data.get(key)

                if isinstance(value, list):
                    raw_rows.extend(
                        item
                        for item in value
                        if isinstance(item, dict)
                    )

                    found_list = True

            dict_wrappers = (
                "fundamental_snapshot",
                "company_profile",
                "latest_fundamentals",
                "fundamentals",
            )

            for key in dict_wrappers:
                value = data.get(key)

                if isinstance(value, dict):
                    raw_rows.append(value)
                    found_list = True

            if not found_list:
                raw_rows.append(dict(data))

        elif isinstance(data, Sequence) and not isinstance(
            data,
            (str, bytes, bytearray),
        ):
            raw_rows.extend(
                item
                for item in data
                if isinstance(item, dict)
            )

        extracted_snapshots: List[
            Dict[str, Any]
        ] = []

        for index, row in enumerate(raw_rows):

            extracted: Dict[str, Any] = {}

            timestamp = _parse_time(
                _get_date_score(row)
            )

            extracted["_t"] = timestamp
            extracted["_idx"] = index

            for raw_key, value in row.items():

                if value is None:
                    continue

                normalized_key = (
                    str(raw_key)
                    .strip()
                    .lower()
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                extracted[normalized_key] = value

            if len(extracted) > 2:
                extracted_snapshots.append(
                    extracted
                )

        extracted_snapshots.sort(
            key=lambda item: (
                float(item.get("_t", 0.0)),
                int(item.get("_idx", 0)),
            )
        )

        return extracted_snapshots

    # ----------------------------------------------------------------------
    # LATEST VALID SNAPSHOT
    # ----------------------------------------------------------------------

    def _get_latest_valid_snapshot(
        self,
        rows: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

        valid_keys = {
            alias
            for contracts in FIELD_CONTRACTS.values()
            for alias, _ in contracts
        }

        for row in reversed(rows):

            numeric_or_string_count = 0

            for key in row:
                if key in valid_keys:
                    numeric_or_string_count += 1

            if numeric_or_string_count >= 1:
                return row

        return None

    # ----------------------------------------------------------------------
    # FEATURE TRACE
    # ----------------------------------------------------------------------

    def _trace_feature(
        self,
        feature: str,
        snapshot: Mapping[str, Any],
        series: List[Tuple[float, float]],
    ) -> Dict[str, Any]:

        contracts = self.field_contracts.get(
            feature,
            [],
        )

        aliases_checked = [
            alias
            for alias, _ in contracts
        ]

        for alias, multiplier in contracts:

            if alias not in snapshot:
                continue

            raw_value = snapshot.get(alias)

            numeric_value = _num(raw_value)

            if numeric_value is None:
                return {
                    "status": "available_invalid",
                    "feature": feature,
                    "source_field": alias,
                    "raw_value": raw_value,
                    "multiplier": multiplier,
                    "normalized_value": None,
                    "historical_points": len(series),
                    "aliases_checked": aliases_checked,
                }

            normalized_value = (
                numeric_value * multiplier
            )

            if not math.isfinite(
                normalized_value
            ):
                return {
                    "status": "available_invalid",
                    "feature": feature,
                    "source_field": alias,
                    "raw_value": raw_value,
                    "multiplier": multiplier,
                    "normalized_value": None,
                    "historical_points": len(series),
                    "aliases_checked": aliases_checked,
                }

            return {
                "status": "used",
                "feature": feature,
                "source_field": alias,
                "raw_value": raw_value,
                "multiplier": multiplier,
                "normalized_value": normalized_value,
                "historical_points": len(series),
                "latest_timestamp": snapshot.get("_t"),
                "aliases_checked": aliases_checked,
            }

        return {
            "status": "missing",
            "feature": feature,
            "source_field": None,
            "raw_value": None,
            "multiplier": None,
            "normalized_value": None,
            "historical_points": len(series),
            "latest_timestamp": snapshot.get("_t"),
            "aliases_checked": aliases_checked,
        }

    def _trace_string_feature(
        self,
        feature: str,
        snapshot: Mapping[str, Any],
    ) -> Dict[str, Any]:

        contracts = self.field_contracts.get(
            feature,
            [],
        )

        aliases_checked = [
            alias
            for alias, _ in contracts
        ]

        for alias, multiplier in contracts:

            if alias not in snapshot:
                continue

            value = _str(
                snapshot.get(alias)
            )

            if value is None:
                return {
                    "status": "available_invalid",
                    "feature": feature,
                    "source_field": alias,
                    "raw_value": snapshot.get(alias),
                    "multiplier": multiplier,
                    "normalized_value": None,
                }

            return {
                "status": "used",
                "feature": feature,
                "source_field": alias,
                "raw_value": value,
                "multiplier": multiplier,
                "normalized_value": value,
                "latest_timestamp": snapshot.get("_t"),
                "aliases_checked": aliases_checked,
            }

        return {
            "status": "missing",
            "feature": feature,
            "source_field": None,
            "raw_value": None,
            "multiplier": None,
            "normalized_value": None,
            "latest_timestamp": snapshot.get("_t"),
            "aliases_checked": aliases_checked,
        }

    # ----------------------------------------------------------------------
    # TRACE SUMMARY
    # ----------------------------------------------------------------------

    def _build_feature_trace_summary(
        self,
        feature_trace: Mapping[str, Dict[str, Any]],
        used_features: set[str],
    ) -> Dict[str, Any]:

        total_features = len(NUMERIC_FEATURES)

        used = 0
        available_not_used = 0
        missing = 0
        invalid = 0

        missing_features: List[str] = []
        available_features: List[str] = []
        used_feature_names: List[str] = []

        for feature in sorted(NUMERIC_FEATURES):

            trace = feature_trace.get(
                feature,
                {},
            )

            status = trace.get(
                "status",
                "missing",
            )

            if (
                status == "used"
                or feature in used_features
            ):
                used += 1
                used_feature_names.append(feature)

            elif status == "missing":
                missing += 1
                missing_features.append(feature)

            elif status == "available_invalid":
                invalid += 1
                available_not_used += 1
                available_features.append(feature)

            else:
                available_not_used += 1
                available_features.append(feature)

        coverage = (
            used / total_features * 100.0
            if total_features
            else 0.0
        )

        return {
            "total_features": total_features,
            "used_features": used,
            "available_not_used": available_not_used,
            "missing_features": missing,
            "invalid_features": invalid,
            "coverage_pct": round(
                coverage,
                2,
            ),
            "used_feature_names": used_feature_names,
            "available_not_used_names": available_features,
            "missing_feature_names": missing_features,
        }

    # ----------------------------------------------------------------------
    # DOMAIN TRACE
    # ----------------------------------------------------------------------

    def _build_domain_trace(
        self,
        feature_trace: Mapping[str, Dict[str, Any]],
        used_features: set[str],
    ) -> Dict[str, Dict[str, Any]]:

        result: Dict[str, Dict[str, Any]] = {}

        for domain, features in self.domain_features.items():

            total = len(features)

            used = sum(
                1
                for feature in features
                if feature in used_features
            )

            available = sum(
                1
                for feature in features
                if feature_trace.get(
                    feature,
                    {},
                ).get("status")
                in {
                    "used",
                    "available_invalid",
                }
            )

            missing = max(
                total - available,
                0,
            )

            coverage = (
                used / total * 100.0
                if total
                else 0.0
            )

            result[domain] = {
                "total_features": total,
                "used_features": used,
                "available_features": available,
                "missing_features": missing,
                "coverage_pct": round(
                    coverage,
                    2,
                ),
                "features": {
                    feature: feature_trace.get(
                        feature,
                        {
                            "status": "missing"
                        },
                    )
                    for feature in features
                },
            }

        return result

    # ----------------------------------------------------------------------
    # EXTRACTION HELPERS
    # ----------------------------------------------------------------------

    def _extract_series(
        self,
        rows: List[Dict[str, Any]],
        domain_key: str,
    ) -> List[Tuple[float, float]]:

        contracts = self.field_contracts.get(
            domain_key,
            [],
        )

        series: List[
            Tuple[float, float]
        ] = []

        for row in rows:

            for alias, multiplier in contracts:

                if alias not in row:
                    continue

                value = _num(
                    row.get(alias)
                )

                if value is None:
                    continue

                normalized = (
                    value * multiplier
                )

                if not math.isfinite(normalized):
                    continue

                series.append(
                    (
                        float(
                            row.get(
                                "_t",
                                0.0,
                            )
                        ),
                        normalized,
                    )
                )

                break

        return series

    def _extract_str(
        self,
        snapshot: Mapping[str, Any],
        domain_key: str,
    ) -> Optional[str]:

        contracts = self.field_contracts.get(
            domain_key,
            [],
        )

        for alias, _ in contracts:

            if alias not in snapshot:
                continue

            value = _str(
                snapshot.get(alias)
            )

            if value is not None:
                return value

        return None

    # ----------------------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------------------

    @staticmethod
    def _status(
        value: Optional[float],
        positive: str,
        negative: str,
    ) -> str:

        if value is None:
            return "unknown"

        if value > 60.0:
            return positive

        if value < 40.0:
            return negative

        return "neutral"

    @staticmethod
    def _status_cagr(
        value: Optional[float],
    ) -> str:

        if value is None:
            return "unknown"

        if value > 15.0:
            return "high"

        if value < 5.0:
            return "low"

        return "neutral"

    # ----------------------------------------------------------------------
    # DATE MATH
    # ----------------------------------------------------------------------

    @staticmethod
    def _years_between(
        first_timestamp: float,
        last_timestamp: float,
    ) -> float:

        if first_timestamp <= 0:
            return 1.0

        if last_timestamp <= 0:
            return 1.0

        # YYYYMMDD representation.
        if (
            first_timestamp >= 10000000
            and last_timestamp >= 10000000
        ):
            first_year = int(
                first_timestamp // 10000
            )

            last_year = int(
                last_timestamp // 10000
            )

            year_delta = last_year - first_year

            return max(
                float(year_delta),
                1.0,
            )

        # Fallback for ordinary numeric timestamps.
        difference = (
            last_timestamp
            - first_timestamp
        )

        if difference <= 0:
            return 1.0

        return max(
            difference / 10000.0,
            1.0,
        )


# ============================================================================
# MODULE-LEVEL API
# ============================================================================

def analyze(
    data: Any,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:

    return FundamentalAnalyzer().analyze(
        data,
        *args,
        **kwargs,
    )


__all__ = [
    "FundamentalAnalyzer",
    "FIELD_CONTRACTS",
    "DOMAIN_FEATURES",
    "analyze",
]
