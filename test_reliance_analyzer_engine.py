import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from backend.analyzers.analyzer_engine import AnalyzerEngine
from backend.config.settings import settings


SYMBOL = "RELIANCE"

ANALYZER_ORDER = [
    "fundamental",
    "institutional",
    "trend",
    "momentum",
    "smart_money",
    "volatility",
    "volume",
    "ipo",
    "pattern",
    "support_resistance",
    "market_regime",
    "candle",
]


def database_path():
    path = Path(settings.database_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}"
        )

    return path


def get_connection():
    conn = sqlite3.connect(
        str(database_path()),
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table):
    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    return [
        str(row["name"]).strip()
        for row in rows
    ]


def find_l1_feature_history_table(conn):
    tables = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    candidates = []

    for row in tables:
        table = str(row["name"])
        columns = table_columns(conn, table)
        normalized = {
            c.lower().replace("-", "_").replace(" ", "_")
            for c in columns
        }

        if "symbol" not in normalized:
            continue

        score = 0

        name = table.lower()

        if "feature" in name:
            score += 100

        if "history" in name:
            score += 80

        if "indicator" in name:
            score += 50

        if "technical" in name:
            score += 30

        feature_columns = {
            "ema_20",
            "ema_50",
            "ema_200",
            "rsi",
            "macd",
            "atr",
            "volume",
            "close",
            "high",
            "low",
        }

        score += len(
            normalized.intersection(
                feature_columns
            )
        )

        if score:
            candidates.append(
                (
                    score,
                    table,
                    columns,
                )
            )

    if not candidates:
        raise RuntimeError(
            "No L1 feature-history table found."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1], candidates[0][2]


def find_order_column(columns):
    normalized = {
        c.lower(): c
        for c in columns
    }

    for name in (
        "date",
        "datetime",
        "timestamp",
        "updated_at",
        "created_at",
        "period_end",
    ):
        if name in normalized:
            return normalized[name]

    return None


def load_l1_feature_history():
    conn = None

    try:
        conn = get_connection()

        table, columns = (
            find_l1_feature_history_table(
                conn
            )
        )

        normalized_columns = {
            c.lower(): c
            for c in columns
        }

        symbol_column = normalized_columns.get(
            "symbol"
        )

        if symbol_column is None:
            raise RuntimeError(
                f"L1 table '{table}' has no symbol column."
            )

        order_column = find_order_column(
            columns
        )

        select_sql = ", ".join(
            f'"{column}"'
            for column in columns
        )

        order_sql = ""

        if order_column:
            order_sql = (
                f' ORDER BY "{order_column}" ASC'
            )

        query = f"""
            SELECT {select_sql}
            FROM "{table}"
            WHERE UPPER(TRIM("{symbol_column}")) = ?
            {order_sql}
        """

        rows = conn.execute(
            query,
            (SYMBOL,),
        ).fetchall()

        if not rows:
            raise RuntimeError(
                f"No L1 feature-history rows found "
                f"for {SYMBOL} in '{table}'."
            )

        data = [
            dict(row)
            for row in rows
        ]

        df = pd.DataFrame(data)

        if df.empty:
            raise RuntimeError(
                "L1 feature-history DataFrame is empty."
            )

        return table, df

    finally:
        if conn is not None:
            conn.close()


def compact_feature_coverage(result):
    total = 0
    available = 0
    missing = 0
    invalid = 0

    print("=" * 72)
    print("FEATURE COVERAGE SUMMARY")
    print("=" * 72)
    print()

    for name in ANALYZER_ORDER:
        analyzer_result = result.get(
            name,
            {},
        )

        coverage = analyzer_result.get(
            "feature_coverage",
            {},
        )

        summary = coverage.get(
            "summary",
            {},
        )

        required_total = int(
            summary.get(
                "required_total",
                0,
            )
        )

        required_received = int(
            summary.get(
                "required_received",
                0,
            )
        )

        required_valid = int(
            summary.get(
                "required_valid",
                0,
            )
        )

        required_missing = int(
            summary.get(
                "required_missing",
                max(
                    required_total
                    - required_received,
                    0,
                ),
            )
        )

        required_invalid = max(
            required_received
            - required_valid,
            0,
        )

        total += required_total
        available += required_received
        missing += required_missing
        invalid += required_invalid

        coverage_pct = (
            required_received
            / required_total
            * 100.0
            if required_total
            else 100.0
        )

        print(
            f"{name:<24}"
            f"{required_received:>4}/"
            f"{required_total:<4}"
            f" {coverage_pct:>7.2f}%"
        )

    coverage_pct = (
        available / total * 100.0
        if total
        else 0.0
    )

    print()
    print(
        f"TOTAL FEATURES: {total}"
    )
    print(
        f"AVAILABLE:      {available}"
    )
    print(
        f"MISSING:        {missing}"
    )
    print(
        f"INVALID:        {invalid}"
    )
    print(
        f"COVERAGE:       {coverage_pct:.2f}%"
    )


def build_l3_payload(result):
    payload = {}

    for name in ANALYZER_ORDER:
        analyzer_result = result.get(
            name,
            {},
        )

        if not isinstance(
            analyzer_result,
            dict,
        ):
            continue

        analyzer_key = (
            f"{name}_analyzer"
        )

        analyzer_output = (
            analyzer_result.get(
                analyzer_key
            )
        )

        if isinstance(
            analyzer_output,
            dict,
        ):
            payload[analyzer_key] = (
                analyzer_output
            )
        else:
            payload[analyzer_key] = {}

    return payload


def save_json(path, payload):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            payload,
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def run_test():
    print("=" * 72)
    print("GREEN BULL RIDER V6")
    print("RELIANCE L2 ANALYZER ENGINE TEST")
    print("=" * 72)
    print()

    try:
        db_path = database_path()

        print(
            f"DATABASE: {db_path}"
        )
        print(
            f"SYMBOL:   {SYMBOL}"
        )
        print()

        table, df = (
            load_l1_feature_history()
        )

        print(
            f"L1 TABLE: {table}"
        )
        print(
            f"L1 ROWS:  {len(df)}"
        )
        print(
            f"L1 COLS:  {len(df.columns)}"
        )

    except Exception as exc:
        print(
            f"L1 INPUT ERROR: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    print()
    print(
        "RUNNING L2 ANALYZER ENGINE..."
    )

    try:
        engine = AnalyzerEngine()

        result = engine.run(
            {
                "symbol": SYMBOL,
                "in_memory_df": df,
            }
        )

    except Exception as exc:
        print(
            f"L2 ENGINE ERROR: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    print()
    print("=" * 72)
    print("L2 ANALYZER OUTPUT")
    print("=" * 72)
    print()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print()

    compact_feature_coverage(
        result
    )

    print()
    print("=" * 72)
    print("ANALYZER STATUS")
    print("=" * 72)
    print()

    passed = 0

    for name in ANALYZER_ORDER:
        status = result.get(
            name,
            {},
        ).get(
            "engine_status",
            "FAILED",
        )

        label = (
            "PASS"
            if status == "SUCCESS"
            else "FAIL"
        )

        if label == "PASS":
            passed += 1

        print(
            f"{name:<24}{label}"
        )

    print()
    print(
        f"{passed}/{len(ANALYZER_ORDER)} PASS"
    )

    if passed != len(
        ANALYZER_ORDER
    ):
        print(
            "L2 TEST FAILED"
        )
        return 1

    l3_payload = (
        build_l3_payload(result)
    )

    output_file = (
        Path("reliance_l3_payload.json")
    )

    save_json(
        output_file,
        l3_payload,
    )

    print()
    print("=" * 72)
    print("L3 PAYLOAD")
    print("=" * 72)
    print()

    print(
        json.dumps(
            l3_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print()
    print(
        f"L3 PAYLOAD SAVED: "
        f"{output_file}"
    )

    print()
    print("=" * 72)
    print("L2 TEST PASSED")
    print(
        "12/12 ANALYZERS PASS"
    )
    print(
        "L3 PAYLOAD READY"
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(
        run_test()
    )
