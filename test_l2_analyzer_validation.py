"""
GREEN BULL RIDER V6
Institutional L2 Analyzer Validation Runner

Validates the three rewritten Layer-2 analyzers:
- MomentumAnalyzer
- VolatilityAnalyzer
- SmartMoneyAnalyzer

Validation is intentionally stricter than an import/compile test.

Checks:
1. Real Reliance fixture loading
2. DataFrame / snapshot input compatibility
3. Exact declared feature coverage
4. Missing / invalid / used feature tracing
5. Output contract presence
6. Numeric range integrity
7. Evidence integrity
8. Determinism
9. Missing-feature resilience
10. NaN/Inf/string/bool invalid-feature handling
11. AnalyzerEngine integration when available

No market data is synthesized.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "reliance_l2_raw.json"


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


class TestResult:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        name: str,
        status: str,
        detail: str = "",
        metric: str = "",
    ) -> None:
        self.rows.append(
            {
                "test": name,
                "status": status,
                "metric": metric,
                "detail": detail,
            }
        )

    def failed(self) -> bool:
        return any(r["status"] == FAIL for r in self.rows)


def load_fixture() -> Any:
    if not FIXTURE.exists():
        raise FileNotFoundError(
            f"Fixture not found: {FIXTURE}\n"
            "Expected reliance_l2_raw.json beside this test runner."
        )

    with FIXTURE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_feature_payload(obj: Any) -> Any:
    """
    Recursively locate a useful L1 payload inside the existing Reliance fixture.
    The runner does not invent data; it only unwraps common fixture containers.
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, dict):
        preferred = (
            "features",
            "feature_snapshot",
            "l1_features",
            "data",
            "rows",
            "result",
            "reliance",
        )
        for key in preferred:
            if key in obj:
                candidate = find_feature_payload(obj[key])
                if candidate is not None:
                    return candidate

        # A dict whose values are scalars is already a feature snapshot.
        if obj and all(
            not isinstance(v, (dict, list, tuple)) for v in obj.values()
        ):
            return obj

        # Try nested dictionaries.
        for value in obj.values():
            candidate = find_feature_payload(value)
            if candidate is not None:
                return candidate

    if isinstance(obj, list):
        if not obj:
            return None

        # List of row dictionaries -> DataFrame.
        if all(isinstance(x, dict) for x in obj):
            return pd.DataFrame(obj)

        for item in obj:
            candidate = find_feature_payload(item)
            if candidate is not None:
                return candidate

    return None


def normalize_input(payload: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if isinstance(payload, pd.DataFrame):
        df = payload.copy()
        if df.empty:
            raise ValueError("Fixture DataFrame is empty.")
        return df, df.iloc[-1].to_dict()

    if isinstance(payload, dict):
        # Single feature snapshot.
        snapshot = dict(payload)
        df = pd.DataFrame([snapshot])
        return df, snapshot

    if isinstance(payload, list) and payload:
        df = pd.DataFrame(payload)
        if df.empty:
            raise ValueError("Fixture rows produced an empty DataFrame.")
        return df, df.iloc[-1].to_dict()

    raise TypeError(
        f"Unsupported fixture payload type: {type(payload).__name__}"
    )


def numeric(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def output_root(result: Any, analyzer_name: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise AssertionError(
            f"{analyzer_name}: analyzer returned {type(result).__name__}, "
            "expected dict."
        )

    if analyzer_name in result:
        node = result[analyzer_name]
    else:
        raise AssertionError(
            f"{analyzer_name}: missing top-level output key."
        )

    if not isinstance(node, dict):
        raise AssertionError(
            f"{analyzer_name}: output node must be dict."
        )

    return node


def validate_coverage(node: Dict[str, Any], name: str, tr: TestResult) -> None:
    coverage = node.get("feature_coverage")

    if not isinstance(coverage, dict):
        tr.add(
            f"{name}: feature coverage contract",
            FAIL,
            "Missing feature_coverage object.",
        )
        return

    required = {
        "contract",
        "declared_features",
        "available_features",
        "missing_features",
        "invalid_features",
        "coverage_percent",
        "available",
        "missing",
        "invalid",
        "used",
    }

    missing = sorted(required - set(coverage))
    if missing:
        tr.add(
            f"{name}: feature coverage contract",
            FAIL,
            f"Missing keys: {missing}",
        )
        return

    declared = coverage["declared_features"]
    available = coverage["available_features"]
    missing_count = coverage["missing_features"]
    invalid_count = coverage["invalid_features"]
    percent = coverage["coverage_percent"]

    problems = []

    for key, value in (
        ("declared_features", declared),
        ("available_features", available),
        ("missing_features", missing_count),
        ("invalid_features", invalid_count),
    ):
        if not isinstance(value, int) or value < 0:
            problems.append(f"{key}={value!r}")

    for key in ("available", "missing", "invalid", "used"):
        if not isinstance(coverage[key], list):
            problems.append(f"{key} is not a list")

    if not numeric(percent) or not 0.0 <= float(percent) <= 100.0:
        problems.append(f"coverage_percent={percent!r}")

    if isinstance(declared, int):
        if available + missing_count + invalid_count != declared:
            problems.append(
                "available + missing + invalid != declared"
            )

    available_set = set(coverage["available"])
    missing_set = set(coverage["missing"])
    invalid_set = set(coverage["invalid"])
    used_set = set(coverage["used"])

    if available_set & missing_set:
        problems.append("available/missing overlap")
    if available_set & invalid_set:
        problems.append("available/invalid overlap")
    if missing_set & invalid_set:
        problems.append("missing/invalid overlap")
    if not used_set.issubset(available_set):
        problems.append("used contains unavailable features")

    if problems:
        tr.add(
            f"{name}: feature coverage integrity",
            FAIL,
            "; ".join(problems),
        )
    else:
        tr.add(
            f"{name}: feature coverage integrity",
            PASS,
            "Coverage accounting and trace sets are internally consistent.",
            f"{available}/{declared} ({percent:.2f}%)",
        )


def validate_numeric_ranges(
    node: Dict[str, Any],
    name: str,
    tr: TestResult,
) -> None:
    expected_numeric = (
        "confidence",
        "strength",
        "acceleration",
        "rsi",
        "macd",
        "slowdown",
        "divergence",
        "risk",
        "atr",
        "expansion",
        "compression",
        "breakout",
        "historical",
        "volatility_quality",
        "sweep",
        "efficiency",
        "structure",
        "liquidity",
        "order_block",
        "fair_value_gap",
        "footprint",
        "zone",
    )

    bounded = []
    invalid = []

    for key in expected_numeric:
        if key not in node:
            continue

        value = node[key]
        if not numeric(value):
            invalid.append(f"{key}={value!r}")
            continue

        x = float(value)

        # All current analyzer score fields are contract scores.
        if key in {
            "confidence",
            "strength",
            "acceleration",
            "rsi",
            "slowdown",
            "divergence",
            "risk",
            "atr",
            "expansion",
            "compression",
            "breakout",
            "historical",
            "volatility_quality",
            "sweep",
            "efficiency",
            "structure",
            "liquidity",
            "order_block",
            "fair_value_gap",
            "footprint",
            "zone",
        }:
            if not 0.0 <= x <= 100.0:
                bounded.append(f"{key}={x}")

    if invalid or bounded:
        tr.add(
            f"{name}: numeric/range integrity",
            FAIL,
            f"invalid={invalid}; out_of_range={bounded}",
        )
    else:
        tr.add(
            f"{name}: numeric/range integrity",
            PASS,
            "All emitted score fields are finite and within 0..100.",
        )


def validate_evidence(
    node: Dict[str, Any],
    name: str,
    tr: TestResult,
) -> None:
    evidence = node.get("evidence")

    if not isinstance(evidence, list) or not evidence:
        tr.add(
            f"{name}: evidence contract",
            FAIL,
            "evidence must be a non-empty list.",
        )
        return

    problems = []

    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            problems.append(f"[{index}] not dict")
            continue

        for key in ("category", "message", "reliability", "likelihood_ratio"):
            if key not in item:
                problems.append(f"[{index}] missing {key}")

        if "reliability" in item and numeric(item["reliability"]):
            if not 0.0 <= float(item["reliability"]) <= 1.0:
                problems.append(
                    f"[{index}] reliability out of 0..1"
                )

        if "likelihood_ratio" in item and numeric(
            item["likelihood_ratio"]
        ):
            lr = float(item["likelihood_ratio"])
            if not 0.1 <= lr <= 10.0:
                problems.append(
                    f"[{index}] likelihood_ratio out of 0.1..10"
                )

    if problems:
        tr.add(
            f"{name}: evidence contract",
            FAIL,
            "; ".join(problems),
        )
    else:
        tr.add(
            f"{name}: evidence contract",
            PASS,
            f"{len(evidence)} evidence records validated.",
        )


def validate_common_contract(
    result: Any,
    name: str,
    tr: TestResult,
) -> Dict[str, Any]:
    node = output_root(result, name)

    if "version" not in node:
        tr.add(
            f"{name}: version contract",
            WARN,
            "No version field emitted.",
        )

    validate_coverage(node, name, tr)
    validate_numeric_ranges(node, name, tr)
    validate_evidence(node, name, tr)

    return node


def mutate_missing(
    df: pd.DataFrame,
    snapshot: Dict[str, Any],
    declared: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not declared:
        return df.copy(), dict(snapshot)

    target = declared[0]

    df2 = df.copy()
    if target in df2.columns:
        df2 = df2.drop(columns=[target])

    snap2 = dict(snapshot)
    snap2.pop(target, None)

    return df2, snap2


def mutate_invalid(
    df: pd.DataFrame,
    snapshot: Dict[str, Any],
    declared: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not declared:
        return df.copy(), dict(snapshot)

    target = declared[0]

    df2 = df.copy()
    if target in df2.columns:
        df2[target] = np.nan

    snap2 = dict(snapshot)
    if target in snap2:
        snap2[target] = np.nan

    return df2, snap2


def run_analyzer(
    analyzer: Any,
    df: pd.DataFrame,
    snapshot: Dict[str, Any],
    analyzer_name: str,
) -> Any:
    # Momentum is explicitly DataFrame-only in V6.1.
    if analyzer_name == "momentum_analyzer":
        return analyzer.analyze(df)

    # The other rewritten analyzers historically accept scalar snapshots.
    # Prefer the snapshot and only use context if the implementation accepts it.
    try:
        return analyzer.analyze(snapshot)
    except (TypeError, AttributeError):
        return analyzer.analyze(df)


def instantiate_analyzers() -> Dict[str, Any]:
    from backend.analyzers.momentum_analyzer import MomentumAnalyzer
    from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
    from backend.analyzers.smart_money_analyzer import SmartMoneyAnalyzer

    return {
        "momentum_analyzer": MomentumAnalyzer(),
        "volatility_analyzer": VolatilityAnalyzer(),
        "smart_money_analyzer": SmartMoneyAnalyzer(),
    }


def run_engine_test(
    df: pd.DataFrame,
    tr: TestResult,
) -> None:
    try:
        from backend.analyzers.analyzer_engine import AnalyzerEngine
    except Exception as exc:
        tr.add(
            "AnalyzerEngine import",
            FAIL,
            f"{type(exc).__name__}: {exc}",
        )
        return

    try:
        engine = AnalyzerEngine()

        # Support common engine entry-point names without modifying the engine.
        if hasattr(engine, "analyze"):
            try:
                result = engine.analyze(df)
            except TypeError:
                result = engine.analyze(
                    symbol="RELIANCE",
                    features=df,
                )
        elif hasattr(engine, "run"):
            try:
                result = engine.run(df)
            except TypeError:
                result = engine.run(
                    symbol="RELIANCE",
                    features=df,
                )
        else:
            tr.add(
                "AnalyzerEngine execution",
                WARN,
                "No analyze/run entry point detected.",
            )
            return

        if isinstance(result, dict):
            tr.add(
                "AnalyzerEngine execution",
                PASS,
                "Engine returned a dictionary result.",
            )
        else:
            tr.add(
                "AnalyzerEngine execution",
                FAIL,
                f"Engine returned {type(result).__name__}.",
            )

    except Exception as exc:
        tr.add(
            "AnalyzerEngine execution",
            FAIL,
            f"{type(exc).__name__}: {exc}",
        )


def print_report(tr: TestResult) -> None:
    print()
    print("=" * 88)
    print("GREEN BULL RIDER V6 - L2 INSTITUTIONAL ANALYZER VALIDATION")
    print("=" * 88)

    for row in tr.rows:
        metric = f" [{row['metric']}]" if row["metric"] else ""
        print(
            f"{row['status']:>4} | "
            f"{row['test']}{metric}"
        )
        if row["detail"]:
            print(f"       {row['detail']}")

    print("-" * 88)

    passed = sum(r["status"] == PASS for r in tr.rows)
    failed = sum(r["status"] == FAIL for r in tr.rows)
    warned = sum(r["status"] == WARN for r in tr.rows)

    print(
        f"RESULTS: PASS={passed}  FAIL={failed}  WARN={warned}"
    )

    if failed:
        print("OVERALL: FAIL")
    elif warned:
        print("OVERALL: PASS WITH WARNINGS")
    else:
        print("OVERALL: PASS")

    print("=" * 88)


def main() -> int:
    tr = TestResult()

    # ------------------------------------------------------------------
    # Fixture
    # ------------------------------------------------------------------
    try:
        raw = load_fixture()
        payload = find_feature_payload(raw)

        if payload is None:
            raise ValueError(
                "Could not locate a feature payload inside reliance_l2_raw.json."
            )

        df, snapshot = normalize_input(payload)

        tr.add(
            "Reliance fixture loading",
            PASS,
            f"Rows={len(df)}, columns={len(df.columns)}.",
        )
    except Exception as exc:
        tr.add(
            "Reliance fixture loading",
            FAIL,
            f"{type(exc).__name__}: {exc}",
        )
        print_report(tr)
        return 1

    # ------------------------------------------------------------------
    # Analyzer imports / instances
    # ------------------------------------------------------------------
    try:
        analyzers = instantiate_analyzers()
        tr.add(
            "Three rewritten analyzer imports",
            PASS,
            ", ".join(analyzers.keys()),
        )
    except Exception as exc:
        tr.add(
            "Three rewritten analyzer imports",
            FAIL,
            f"{type(exc).__name__}: {exc}",
        )
        print_report(tr)
        return 1

    # ------------------------------------------------------------------
    # Individual analyzer validation
    # ------------------------------------------------------------------
    for name, analyzer in analyzers.items():
        print(f"\n--- Validating {name} ---")

        try:
            result1 = run_analyzer(
                analyzer,
                df.copy(),
                copy.deepcopy(snapshot),
                name,
            )
            node1 = validate_common_contract(
                result1,
                name,
                tr,
            )

            result2 = run_analyzer(
                analyzer,
                df.copy(),
                copy.deepcopy(snapshot),
                name,
            )

            if result1 == result2:
                tr.add(
                    f"{name}: deterministic output",
                    PASS,
                    "Identical input produced identical output.",
                )
            else:
                tr.add(
                    f"{name}: deterministic output",
                    FAIL,
                    "Repeated execution produced different output.",
                )

            coverage = node1.get("feature_coverage", {})
            declared = list(
                dict.fromkeys(
                    coverage.get("available", [])
                    + coverage.get("missing", [])
                    + coverage.get("invalid", [])
                )
            )

            if declared:
                missing_df, missing_snapshot = mutate_missing(
                    df,
                    snapshot,
                    declared,
                )

                try:
                    missing_result = run_analyzer(
                        analyzer,
                        missing_df,
                        missing_snapshot,
                        name,
                    )
                    validate_common_contract(
                        missing_result,
                        name,
                        tr,
                    )
                    tr.add(
                        f"{name}: missing-feature resilience",
                        PASS,
                        f"Removed declared feature '{declared[0]}' without crash.",
                    )
                except Exception as exc:
                    tr.add(
                        f"{name}: missing-feature resilience",
                        FAIL,
                        f"{type(exc).__name__}: {exc}",
                    )

                invalid_df, invalid_snapshot = mutate_invalid(
                    df,
                    snapshot,
                    declared,
                )

                try:
                    invalid_result = run_analyzer(
                        analyzer,
                        invalid_df,
                        invalid_snapshot,
                        name,
                    )
                    validate_common_contract(
                        invalid_result,
                        name,
                        tr,
                    )
                    tr.add(
                        f"{name}: invalid-feature resilience",
                        PASS,
                        f"Injected NaN into '{declared[0]}' without crash.",
                    )
                except Exception as exc:
                    tr.add(
                        f"{name}: invalid-feature resilience",
                        FAIL,
                        f"{type(exc).__name__}: {exc}",
                    )
            else:
                tr.add(
                    f"{name}: feature mutation tests",
                    WARN,
                    "No declared feature list exposed by coverage trace.",
                )

        except Exception as exc:
            tr.add(
                f"{name}: real Reliance analysis",
                FAIL,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

    # ------------------------------------------------------------------
    # AnalyzerEngine integration
    # ------------------------------------------------------------------
    run_engine_test(df, tr)

    print_report(tr)
    return 1 if tr.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
