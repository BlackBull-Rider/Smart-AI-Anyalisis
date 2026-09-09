"""
GREEN BULL RIDER V6
Fundamental Analyzer
Production-grade, database-safe fundamental analysis layer.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional


logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """
    Deterministic fundamental-data analyzer.

    Responsibilities:
    - Extract latest valid fundamental values safely.
    - Normalize numeric values.
    - Never fabricate missing values.
    - Produce traceable evidence.
    - Remain scoring/decision agnostic.
    """

    NAME = "fundamental"

    FIELD_ALIASES = {
        "market_cap": (
            "market_cap",
            "marketcap",
            "market_capitalization",
            "market_capitalisation",
        ),
        "enterprise_value": (
            "enterprise_value",
            "enterprisevalue",
            "ev",
        ),
        "revenue": (
            "revenue",
            "total_revenue",
            "sales",
            "total_sales",
        ),
        "net_income": (
            "net_income",
            "netincome",
            "profit_after_tax",
            "pat",
            "net_profit",
        ),
        "ebitda": (
            "ebitda",
            "operating_ebitda",
        ),
        "operating_profit": (
            "operating_profit",
            "operating_income",
            "ebit",
        ),
        "total_assets": (
            "total_assets",
            "assets",
        ),
        "total_debt": (
            "total_debt",
            "debt",
            "total_borrowings",
            "borrowings",
        ),
        "cash": (
            "cash",
            "cash_and_equivalents",
            "cash_and_cash_equivalents",
        ),
        "equity": (
            "equity",
            "total_equity",
            "stockholders_equity",
            "shareholders_equity",
        ),
        "eps": (
            "eps",
            "diluted_eps",
            "basic_eps",
        ),
        "book_value": (
            "book_value",
            "book_value_per_share",
            "bvps",
        ),
        "pe": (
            "pe",
            "pe_ratio",
            "trailing_pe",
            "price_earnings",
        ),
        "forward_pe": (
            "forward_pe",
            "forward_pe_ratio",
        ),
        "pb": (
            "pb",
            "pb_ratio",
            "price_to_book",
        ),
        "ps": (
            "ps",
            "ps_ratio",
            "price_to_sales",
        ),
        "roe": (
            "roe",
            "return_on_equity",
        ),
        "roa": (
            "roa",
            "return_on_assets",
        ),
        "roce": (
            "roce",
            "return_on_capital_employed",
        ),
        "debt_to_equity": (
            "debt_to_equity",
            "debt_equity",
            "debt_to_equity_ratio",
        ),
        "current_ratio": (
            "current_ratio",
        ),
        "quick_ratio": (
            "quick_ratio",
            "acid_test_ratio",
        ),
        "gross_margin": (
            "gross_margin",
            "gross_profit_margin",
        ),
        "operating_margin": (
            "operating_margin",
            "operating_profit_margin",
        ),
        "net_margin": (
            "net_margin",
            "net_profit_margin",
        ),
        "revenue_growth": (
            "revenue_growth",
            "sales_growth",
            "revenue_growth_pct",
        ),
        "earnings_growth": (
            "earnings_growth",
            "profit_growth",
            "net_income_growth",
            "eps_growth",
        ),
        "free_cash_flow": (
            "free_cash_flow",
            "fcf",
        ),
        "operating_cash_flow": (
            "operating_cash_flow",
            "cash_from_operations",
            "ocf",
        ),
        "dividend_yield": (
            "dividend_yield",
            "dividend_yield_pct",
        ),
        "payout_ratio": (
            "payout_ratio",
            "dividend_payout_ratio",
        ),
    }

    DATE_FIELDS = (
        "date",
        "report_date",
        "period_end",
        "fiscal_date",
        "as_of_date",
        "timestamp",
        "updated_at",
    )

    def __init__(self, db: Any = None, connection: Any = None) -> None:
        self.db = db
        self.connection = connection

    def analyze(
        self,
        symbol: str,
        data: Optional[Any] = None,
        row: Optional[Mapping[str, Any]] = None,
        rows: Optional[Iterable[Mapping[str, Any]]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        try:
            source = self._resolve_source(data=data, row=row, rows=rows)

            if not source:
                return self._empty(symbol, "NO_VALID_FUNDAMENTAL_DATA")

            latest = self._latest_valid_row(source)

            if not latest:
                return self._empty(symbol, "NO_VALID_FUNDAMENTAL_ROW")

            values = self._extract_values(latest)
            metrics = self._calculate_metrics(values)
            quality = self._quality(metrics)

            return {
                "analyzer": self.NAME,
                "symbol": symbol,
                "status": "OK",
                "score": None,
                "rating": quality["rating"],
                "confidence": quality["confidence"],
                "fundamental": {
                    "metrics": metrics,
                    "quality": quality,
                },
                "evidence": self._build_evidence(
                    values=values,
                    metrics=metrics,
                ),
                "source": {
                    "date": self._get_date(latest),
                },
            }

        except Exception as exc:
            logger.exception(
                "Fundamental analysis failed for %s",
                symbol,
            )
            return self._empty(
                symbol,
                "ANALYSIS_ERROR",
                error=str(exc),
            )

    def _resolve_source(
        self,
        data: Any = None,
        row: Optional[Mapping[str, Any]] = None,
        rows: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> list[Mapping[str, Any]]:
        if row is not None:
            return [row]

        if rows is not None:
            return [
                item
                for item in rows
                if isinstance(item, Mapping)
            ]

        if data is None:
            return []

        if isinstance(data, Mapping):
            return [data]

        if hasattr(data, "to_dict"):
            try:
                records = data.to_dict("records")
                if isinstance(records, list):
                    return [
                        item
                        for item in records
                        if isinstance(item, Mapping)
                    ]
            except Exception:
                pass

        if isinstance(data, Iterable) and not isinstance(
            data,
            (str, bytes),
        ):
            return [
                item
                for item in data
                if isinstance(item, Mapping)
            ]

        return []

    def _latest_valid_row(
        self,
        rows: list[Mapping[str, Any]],
    ) -> Optional[Mapping[str, Any]]:
        valid = [
            row
            for row in rows
            if self._row_has_valid_data(row)
        ]

        if not valid:
            return None

        dated = [
            row
            for row in valid
            if self._parse_date(self._get_date(row)) is not None
        ]

        if dated:
            return max(
                dated,
                key=lambda item: self._parse_date(
                    self._get_date(item)
                ),
            )

        return valid[-1]

    def _row_has_valid_data(
        self,
        row: Mapping[str, Any],
    ) -> bool:
        for aliases in self.FIELD_ALIASES.values():
            for key in aliases:
                if key in row and self._number(row[key]) is not None:
                    return True
        return False

    def _extract_values(
        self,
        row: Mapping[str, Any],
    ) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {}

        for name, aliases in self.FIELD_ALIASES.items():
            result[name] = self._first_number(
                row,
                aliases,
            )

        return result

    def _calculate_metrics(
        self,
        values: Mapping[str, Optional[float]],
    ) -> Dict[str, Optional[float]]:
        metrics = dict(values)

        debt = values.get("total_debt")
        equity = values.get("equity")
        assets = values.get("total_assets")
        net_income = values.get("net_income")
        revenue = values.get("revenue")
        ebitda = values.get("ebitda")
        cash = values.get("cash")
        ocf = values.get("operating_cash_flow")
        fcf = values.get("free_cash_flow")

        if metrics.get("debt_to_equity") is None:
            metrics["debt_to_equity"] = self._safe_divide(
                debt,
                equity,
            )

        if metrics.get("roe") is None:
            metrics["roe"] = self._safe_divide(
                net_income,
                equity,
                multiplier=100.0,
            )

        if metrics.get("roa") is None:
            metrics["roa"] = self._safe_divide(
                net_income,
                assets,
                multiplier=100.0,
            )

        if metrics.get("net_margin") is None:
            metrics["net_margin"] = self._safe_divide(
                net_income,
                revenue,
                multiplier=100.0,
            )

        if metrics.get("operating_margin") is None:
            metrics["operating_margin"] = self._safe_divide(
                ebitda if ebitda is not None else values.get(
                    "operating_profit"
                ),
                revenue,
                multiplier=100.0,
            )

        if metrics.get("free_cash_flow") is None:
            if ocf is not None:
                capex = values.get("capital_expenditure")
                if capex is not None:
                    metrics["free_cash_flow"] = ocf - abs(capex)

        if values.get("enterprise_value") is None:
            market_cap = values.get("market_cap")
            if market_cap is not None:
                if debt is not None or cash is not None:
                    metrics["enterprise_value"] = (
                        market_cap
                        + (debt or 0.0)
                        - (cash or 0.0)
                    )

        return metrics

    def _quality(
        self,
        metrics: Mapping[str, Optional[float]],
    ) -> Dict[str, Any]:
        checks: list[bool] = []

        roe = metrics.get("roe")
        roa = metrics.get("roa")
        roce = metrics.get("roce")
        de = metrics.get("debt_to_equity")
        current = metrics.get("current_ratio")
        revenue_growth = metrics.get("revenue_growth")
        earnings_growth = metrics.get("earnings_growth")
        fcf = metrics.get("free_cash_flow")
        net_margin = metrics.get("net_margin")

        if roe is not None:
            checks.append(roe > 15.0)

        if roa is not None:
            checks.append(roa > 8.0)

        if roce is not None:
            checks.append(roce > 15.0)

        if de is not None:
            checks.append(de < 1.0)

        if current is not None:
            checks.append(current >= 1.0)

        if revenue_growth is not None:
            checks.append(revenue_growth > 0.0)

        if earnings_growth is not None:
            checks.append(earnings_growth > 0.0)

        if fcf is not None:
            checks.append(fcf > 0.0)

        if net_margin is not None:
            checks.append(net_margin > 0.0)

        if not checks:
            return {
                "rating": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "coverage": 0.0,
            }

        positive = sum(checks)
        coverage = len(checks) / 9.0
        confidence = min(
            100.0,
            (coverage * 60.0)
            + ((positive / len(checks)) * 40.0),
        )

        ratio = positive / len(checks)

        if ratio >= 0.75:
            rating = "STRONG"
        elif ratio >= 0.55:
            rating = "HEALTHY"
        elif ratio >= 0.35:
            rating = "MIXED"
        else:
            rating = "WEAK"

        return {
            "rating": rating,
            "confidence": round(confidence, 2),
            "coverage": round(coverage * 100.0, 2),
            "positive_checks": positive,
            "total_checks": len(checks),
        }

    def _build_evidence(
        self,
        values: Mapping[str, Optional[float]],
        metrics: Mapping[str, Optional[float]],
    ) -> list[Dict[str, Any]]:
        evidence: list[Dict[str, Any]] = []

        for field in (
            "revenue",
            "net_income",
            "eps",
            "roe",
            "roce",
            "debt_to_equity",
            "revenue_growth",
            "earnings_growth",
            "free_cash_flow",
            "current_ratio",
        ):
            value = metrics.get(field)

            if value is None:
                continue

            evidence.append(
                {
                    "type": field,
                    "value": self._clean(value),
                    "source": "fundamental_data",
                }
            )

        return evidence

    def _empty(
        self,
        symbol: str,
        reason: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = {
            "analyzer": self.NAME,
            "symbol": symbol,
            "status": "EMPTY",
            "score": None,
            "rating": "INSUFFICIENT_DATA",
            "confidence": 0.0,
            "fundamental": {
                "metrics": {},
                "quality": {
                    "rating": "INSUFFICIENT_DATA",
                    "confidence": 0.0,
                    "coverage": 0.0,
                },
            },
            "evidence": [],
            "source": {},
            "reason": reason,
        }

        if error:
            result["error"] = error

        return result

    @staticmethod
    def _first_number(
        row: Mapping[str, Any],
        aliases: Iterable[str],
    ) -> Optional[float]:
        for key in aliases:
            if key in row:
                value = FundamentalAnalyzer._number(row[key])
                if value is not None:
                    return value
        return None

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            value = float(value)
            return value if math.isfinite(value) else None

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return None

            negative = text.startswith("(") and text.endswith(")")
            text = text.strip("()")
            text = text.replace(",", "")
            text = text.replace("%", "")

            try:
                number = float(text)
            except (TypeError, ValueError):
                return None

            if negative:
                number = -number

            return number if math.isfinite(number) else None

        return None

    @staticmethod
    def _safe_divide(
        numerator: Optional[float],
        denominator: Optional[float],
        multiplier: float = 1.0,
    ) -> Optional[float]:
        if numerator is None or denominator is None:
            return None

        if denominator == 0:
            return None

        value = (numerator / denominator) * multiplier

        return value if math.isfinite(value) else None

    @staticmethod
    def _get_date(
        row: Mapping[str, Any],
    ) -> Any:
        for key in FundamentalAnalyzer.DATE_FIELDS:
            if key in row and row[key] not in (None, ""):
                return row[key]
        return None

    @staticmethod
    def _parse_date(value: Any) -> Any:
        if value is None:
            return None

        try:
            import pandas as pd

            parsed = pd.to_datetime(
                value,
                errors="coerce",
                utc=True,
            )

            if pd.isna(parsed):
                return None

            return parsed
        except Exception:
            return None

    @staticmethod
    def _clean(value: Any) -> Any:
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return round(value, 6)

        return value


def analyze_fundamental(
    symbol: str,
    data: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return FundamentalAnalyzer().analyze(
        symbol=symbol,
        data=data,
        **kwargs,
    )
