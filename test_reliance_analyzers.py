from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from backend.config.settings import settings
from backend.analyzers.analyzer_engine import AnalyzerEngine


SYMBOL = "RELIANCE"
L1_ROWS = 300


def load_l1_features() -> tuple[pd.DataFrame, Path]:
    db_path = Path(settings.database_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}"
        )

    conn = sqlite3.connect(str(db_path))

    try:
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """,
            conn,
        )

        table_names = set(tables["name"].astype(str))

        if "feature_history" not in table_names:
            raise RuntimeError(
                f"feature_history table not found in {db_path}"
            )

        columns = pd.read_sql_query(
            "PRAGMA table_info(feature_history)",
            conn,
        )

        column_names = set(
            columns["name"].astype(str)
        )

        if "symbol" not in column_names:
            raise RuntimeError(
                "feature_history has no symbol column."
            )

        order_parts = []

        if "timestamp" in column_names:
            order_parts.append(
                "CASE WHEN timestamp IS NOT NULL "
                "THEN timestamp END DESC"
            )

        if "date" in column_names:
            order_parts.append(
                "CASE WHEN date IS NOT NULL "
                "THEN date END DESC"
            )

        order_parts.append("rowid DESC")

        order_by = ", ".join(order_parts)

        query = f"""
            SELECT *
            FROM feature_history
            WHERE symbol = ?
            ORDER BY {order_by}
            LIMIT ?
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(SYMBOL, L1_ROWS),
        )

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            f"No L1 feature rows found for {SYMBOL}."
        )

    # AnalyzerEngine expects chronological L1 history.
    df = df.iloc[::-1].reset_index(drop=True)

    return df, db_path


def compact_result(result: dict) -> dict:
    """
    Keep the actual L2 analyzer result while removing
    extremely verbose diagnostic trace blocks from console output.

    Full raw result is saved separately.
    """

    if not isinstance(result, dict):
        return result

    output = dict(result)

    for key in (
        "feature_trace",
        "feature_coverage",
        "feature_trace_summary",
    ):
        output.pop(key, None)

    return output


def save_raw_results(results: dict) -> Path:
    path = (
        Path(settings.project_root)
        / "reliance_l2_raw.json"
    )

    path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    print("=" * 60)
    print("RELIANCE — ANALYZER ENGINE TEST")
    print("=" * 60)

    print()
    print("Source   : settings.database_path")
    print("L1 Table : feature_history")
    print("Pipeline : NOT CALLED")
    print()

    features, db_path = load_l1_features()

    print(f"Database : {db_path}")
    print(f"L1 Rows  : {len(features)}")
    print("L1 Data  : OK")

    print()
    print("=" * 60)
    print("RUNNING 12 ANALYZERS")
    print("=" * 60)
    print()

    engine = AnalyzerEngine()

    payload = {
        "symbol": SYMBOL,
        "in_memory_df": features,
    }

    # IMPORTANT:
    # AnalyzerEngine.run() accepts ONE dictionary payload.
    results = engine.run(payload)

    if not isinstance(results, dict):
        raise RuntimeError(
            "AnalyzerEngine returned invalid result."
        )

    raw_path = save_raw_results(results)

    analyzer_names = [
        "trend",
        "momentum",
        "volatility",
        "volume",
        "pattern",
        "support_resistance",
        "smart_money",
        "market_regime",
        "fundamental",
        "ipo",
        "candle",
        "institutional",
    ]

    successful = 0
    failed = 0

    print(
        f"{'Analyzer':<24}"
        f"{'Status':<12}"
    )
    print("-" * 40)

    for name in analyzer_names:
        result = results.get(name, {})

        status = result.get(
            "engine_status",
            "MISSING",
        )

        if status == "SUCCESS":
            successful += 1
        else:
            failed += 1

        print(
            f"{name:<24}"
            f"{status:<12}"
        )

    print()
    print("=" * 60)
    print("L2 ENGINE SUMMARY")
    print("=" * 60)

    print(f"Analyzers       : {len(analyzer_names)}")
    print(f"Successful      : {successful}")
    print(f"Failed          : {failed}")

    engine_audit = results.get(
        "_engine",
        {},
    )

    print(
        f"L1 Rows         : "
        f"{engine_audit.get('l1_rows', len(features))}"
    )

    print(
        "Technical       : "
        f"{engine_audit.get('technical_source', 'L1_LIVE')}"
    )

    print(
        "Non-technical   : "
        f"{engine_audit.get('non_technical_source', 'DATABASE_LATEST_VALID_PER_FIELD')}"
    )

    print(
        "Zero padding    : "
        f"{engine_audit.get('zero_padding', False)}"
    )

    print(
        "Recalculation   : "
        f"{engine_audit.get('indicator_recalculation', False)}"
    )

    print("Pipeline        : NOT CALLED")

    print()
    print(
        f"FULL RAW L2 JSON: {raw_path}"
    )

    print()
    print("=" * 60)
    print("L2 ANALYZER OUTPUT")
    print("=" * 60)

    for name in analyzer_names:
        result = results.get(name, {})

        print()
        print("-" * 60)
        print(name.upper())
        print("-" * 60)

        print(
            json.dumps(
                compact_result(result),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if successful == len(analyzer_names):
        print(
            "ANALYZER ENGINE HEALTHY"
        )
    else:
        print(
            f"ANALYZER ENGINE PARTIAL "
            f"({successful}/{len(analyzer_names)} SUCCESS)"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
