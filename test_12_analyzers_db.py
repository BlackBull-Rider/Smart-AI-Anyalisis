from __future__ import annotations

import math
import sqlite3
import traceback
from pathlib import Path

from backend.config.settings import settings


ANALYZERS = [
    ("candle", "backend.analyzers.candle_analyzer", "CandleAnalyzer"),
    ("fundamental", "backend.analyzers.fundamental_analyzer", "FundamentalAnalyzer"),
    ("institutional", "backend.analyzers.institutional_analyzer", "InstitutionalAnalyzer"),
    ("ipo", "backend.analyzers.ipo_analyzer", "IPOAnalyzer"),
    ("market_regime", "backend.analyzers.market_regime_analyzer", "MarketRegimeAnalyzer"),
    ("momentum", "backend.analyzers.momentum_analyzer", "MomentumAnalyzer"),
    ("pattern", "backend.analyzers.pattern_analyzer", "PatternAnalyzer"),
    ("smart_money", "backend.analyzers.smart_money_analyzer", "SmartMoneyAnalyzer"),
    ("support_resistance", "backend.analyzers.support_resistance_analyzer", "SupportResistanceAnalyzer"),
    ("trend", "backend.analyzers.trend_analyzer", "TrendAnalyzer"),
    ("volatility", "backend.analyzers.volatility_analyzer", "VolatilityAnalyzer"),
    ("volume", "backend.analyzers.volume_analyzer", "VolumeAnalyzer"),
]


SOURCE_TABLES = (
    "feature_history",
    "financial_data",
    "fundamental_data",
    "fundamental_snapshot",
    "historical_data",
    "ipo_data",
    "macro_environment",
    "shareholding_data",
    "corporate_actions",
    "earnings_history",
    "analyst_data",
    "market_signals",
    "indicators",
    "latest_indicators",
)


def json_safe(value):
    if value is None:
        return True

    if isinstance(value, (str, bool, int)):
        return True

    if isinstance(value, float):
        return math.isfinite(value)

    if isinstance(value, dict):
        return all(
            isinstance(k, str) and json_safe(v)
            for k, v in value.items()
        )

    if isinstance(value, (list, tuple)):
        return all(json_safe(v) for v in value)

    return False


def table_columns(conn, table):
    try:
        return [
            row[1]
            for row in conn.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
        ]
    except Exception:
        return []


def find_column(columns, names):
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for name in names:
        column = lookup.get(str(name).lower())
        if column:
            return column

    return None


def get_tables(conn):
    return [
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()
    ]


def find_symbol_column(conn, table):
    columns = table_columns(conn, table)

    return find_column(
        columns,
        (
            "symbol",
            "ticker",
            "stock",
            "stock_symbol",
            "security_symbol",
            "instrument",
            "instrument_symbol",
        ),
    )


def find_date_column(conn, table):
    columns = table_columns(conn, table)

    return find_column(
        columns,
        (
            "date",
            "datetime",
            "timestamp",
            "trade_date",
            "as_of_date",
            "observation_date",
            "report_date",
            "financial_date",
            "period_end",
            "period_date",
            "created_at",
            "updated_at",
        ),
    )


def fetch_rows(conn, table, symbol=None, limit=500):
    columns = table_columns(conn, table)

    if not columns:
        return []

    symbol_col = find_symbol_column(
        conn,
        table,
    )

    date_col = find_date_column(
        conn,
        table,
    )

    sql = f'SELECT * FROM "{table}"'
    params = []

    if symbol_col and symbol:
        sql += f' WHERE "{symbol_col}" = ?'
        params.append(symbol)

    if date_col:
        sql += f' ORDER BY "{date_col}" DESC'

    sql += f" LIMIT {int(limit)}"

    try:
        cursor = conn.execute(
            sql,
            params,
        )

        names = [
            description[0]
            for description in cursor.description
        ]

        rows = [
            dict(zip(names, row))
            for row in cursor.fetchall()
        ]

        # Analyzer history must be chronological.
        rows.reverse()

        return rows

    except Exception:
        return []


def fetch_latest_valid_rows(
    conn,
    table,
    symbol=None,
    limit=100,
):
    rows = fetch_rows(
        conn,
        table,
        symbol,
        limit,
    )

    if not rows:
        return []

    # Preserve chronological order but remove
    # completely empty records.
    valid = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        has_value = any(
            value is not None
            and str(value).strip() != ""
            for value in row.values()
        )

        if has_value:
            valid.append(row)

    return valid


def detect_symbol(conn):
    preferred_tables = (
        "stock_master",
        "historical_data",
        "feature_history",
        "fundamental_data",
        "fundamental_snapshot",
        "financial_data",
        "ipo_data",
    )

    tables = set(get_tables(conn))

    for table in preferred_tables:
        if table not in tables:
            continue

        symbol_col = find_symbol_column(
            conn,
            table,
        )

        if not symbol_col:
            continue

        try:
            row = conn.execute(
                f'''
                SELECT "{symbol_col}"
                FROM "{table}"
                WHERE "{symbol_col}" IS NOT NULL
                  AND TRIM(CAST("{symbol_col}" AS TEXT)) <> ''
                LIMIT 1
                '''
            ).fetchone()

            if row and row[0]:
                return str(row[0]).strip()

        except Exception:
            continue

    return None


def build_l3_payload(conn, symbol):
    tables = set(get_tables(conn))

    data = {}

    # ---------------------------------------------------------------
    # PRIMARY L3 SOURCES
    # ---------------------------------------------------------------

    for table in SOURCE_TABLES:
        if table not in tables:
            data[table] = []
            continue

        data[table] = fetch_latest_valid_rows(
            conn,
            table,
            symbol,
            1000 if table in {
                "feature_history",
                "historical_data",
                "financial_data",
                "fundamental_data",
            } else 300,
        )

    # ---------------------------------------------------------------
    # EXPLICIT HISTORY ALIASES
    # ---------------------------------------------------------------

    data["feature_history"] = data.get(
        "feature_history",
        [],
    )

    data["feature_history_data"] = data[
        "feature_history"
    ]

    data["features_history"] = data[
        "feature_history"
    ]

    data["financial_history"] = data.get(
        "financial_data",
        [],
    )

    data["fundamental_history"] = data.get(
        "fundamental_data",
        [],
    )

    data["fundamental_snapshot_history"] = data.get(
        "fundamental_snapshot",
        [],
    )

    data["historical_history"] = data.get(
        "historical_data",
        [],
    )

    data["ipo_history"] = data.get(
        "ipo_data",
        [],
    )

    data["macro_history"] = data.get(
        "macro_environment",
        [],
    )

    data["shareholding_history"] = data.get(
        "shareholding_data",
        [],
    )

    data["earnings_history"] = data.get(
        "earnings_history",
        [],
    )

    # ---------------------------------------------------------------
    # EXISTING L3 ALIASES
    # ---------------------------------------------------------------

    historical = data["historical_data"]

    data["history"] = historical
    data["rows"] = historical

    # Keep "features" pointing at feature history first.
    # Fall back to historical data only if feature_history
    # genuinely does not exist.
    if data["feature_history"]:
        data["features"] = data["feature_history"]
    else:
        data["features"] = historical

    data["indicator_history"] = data[
        "indicators"
    ]

    # ---------------------------------------------------------------
    # LATEST VALID SNAPSHOT
    # ---------------------------------------------------------------

    def latest_valid(rows):
        if not isinstance(rows, list):
            return None

        for row in reversed(rows):
            if not isinstance(row, dict):
                continue

            for value in row.values():
                if value is None:
                    continue

                if isinstance(value, float):
                    if not math.isfinite(value):
                        continue

                if isinstance(value, str):
                    if not value.strip():
                        continue

                return row

        return None

    data["latest_feature"] = latest_valid(
        data["feature_history"]
    )

    data["latest_financial"] = latest_valid(
        data["financial_data"]
    )

    data["latest_fundamental"] = latest_valid(
        data["fundamental_data"]
    )

    data["latest_fundamental_snapshot"] = latest_valid(
        data["fundamental_snapshot"]
    )

    data["latest_historical"] = latest_valid(
        data["historical_data"]
    )

    data["latest_ipo"] = latest_valid(
        data["ipo_data"]
    )

    data["latest_macro"] = latest_valid(
        data["macro_environment"]
    )

    data["latest_shareholding"] = latest_valid(
        data["shareholding_data"]
    )

    data["latest_earnings"] = latest_valid(
        data["earnings_history"]
    )

    data["latest_market_signal"] = latest_valid(
        data["market_signals"]
    )

    # ---------------------------------------------------------------
    # SOURCE METADATA
    # ---------------------------------------------------------------

    data["_db_sources"] = {
        table: {
            "available": table in tables,
            "row_count": len(
                data.get(table, [])
            ),
            "symbol_filtered": True,
            "chronological": True,
        }
        for table in SOURCE_TABLES
    }

    data["_l3_source_priority"] = [
        "feature_history",
        "financial_data",
        "fundamental_data",
        "fundamental_snapshot",
        "historical_data",
        "ipo_data",
        "macro_environment",
        "shareholding_data",
        "corporate_actions",
        "earnings_history",
        "analyst_data",
        "market_signals",
    ]

    data["symbol"] = symbol

    return data


def extract_block(result, analyzer_name):
    if not isinstance(result, dict):
        return None

    expected = f"{analyzer_name}_analyzer"

    if expected in result:
        return result[expected]

    if analyzer_name in result:
        return result[analyzer_name]

    expected_lower = expected.lower()

    for key, value in result.items():
        if str(key).lower() == expected_lower:
            return value

    return None


def validate_output(analyzer_name, result):
    errors = []

    if not isinstance(result, dict):
        return ["result_not_dict"]

    if not json_safe(result):
        errors.append(
            "not_json_serializable"
        )

    block = extract_block(
        result,
        analyzer_name,
    )

    if block is None:
        errors.append(
            f"missing_{analyzer_name}_analyzer_block"
        )
        return errors

    if not isinstance(block, dict):
        errors.append(
            "analyzer_block_not_dict"
        )
        return errors

    if "confidence" not in block:
        errors.append(
            "missing_confidence"
        )

    if "evidence" not in block:
        errors.append(
            "missing_evidence"
        )

    if "feature_coverage_pct" not in block:
        errors.append(
            "missing_feature_coverage_pct"
        )

    if "feature_trace" not in block:
        errors.append(
            "missing_feature_trace"
        )

    if "feature_trace_summary" not in block:
        errors.append(
            "missing_feature_trace_summary"
        )

    return errors


def main():
    print("=" * 78)
    print("GREEN BULL RIDER V6")
    print(
        "12 ANALYZERS — "
        "SETTINGS → DATABASE → FULL L3 INPUT"
    )
    print("=" * 78)

    # ---------------------------------------------------------------
    # SETTINGS DATABASE
    # ---------------------------------------------------------------

    db_path = Path(
        settings.database_path
    ).expanduser()

    print("\nSETTINGS DATABASE PATH:")
    print(db_path)

    if not db_path.exists():
        print(
            "\nERROR: configured database does not exist"
        )
        return 2

    print("\nDATABASE EXISTS: YES")

    # ---------------------------------------------------------------
    # READ ONLY CONNECTION
    # ---------------------------------------------------------------

    uri = f"file:{db_path}?mode=ro"

    try:
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=30.0,
            check_same_thread=False,
        )

    except Exception as exc:
        print(
            "\nDATABASE CONNECTION FAILED"
        )
        print(
            type(exc).__name__,
            exc,
        )
        return 3

    try:
        tables = get_tables(conn)

        print("\nDATABASE TABLES:")
        print(", ".join(tables))

        # -----------------------------------------------------------
        # SYMBOL
        # -----------------------------------------------------------

        symbol = detect_symbol(conn)

        if not symbol:
            print(
                "\nERROR: Could not detect a valid symbol."
            )
            return 4

        print("\nTEST SYMBOL:")
        print(symbol)

        # -----------------------------------------------------------
        # BUILD FULL L3 PAYLOAD
        # -----------------------------------------------------------

        payload = build_l3_payload(
            conn,
            symbol,
        )

        print("\nL3 INPUT COUNTS:")

        for table in SOURCE_TABLES:
            count = len(
                payload.get(
                    table,
                    [],
                )
            )

            print(
                f"{table:22s}: {count}"
            )

        # -----------------------------------------------------------
        # LATEST VALID SNAPSHOTS
        # -----------------------------------------------------------

        print("\nLATEST VALID SNAPSHOTS:")

        latest_map = (
            ("feature_history", "latest_feature"),
            ("financial_data", "latest_financial"),
            ("fundamental_data", "latest_fundamental"),
            (
                "fundamental_snapshot",
                "latest_fundamental_snapshot",
            ),
            (
                "historical_data",
                "latest_historical",
            ),
            ("ipo_data", "latest_ipo"),
            (
                "macro_environment",
                "latest_macro",
            ),
            (
                "shareholding_data",
                "latest_shareholding",
            ),
            (
                "earnings_history",
                "latest_earnings",
            ),
            (
                "market_signals",
                "latest_market_signal",
            ),
        )

        for table, key in latest_map:
            snapshot = payload.get(key)

            print(
                f"{table:22s}: "
                f"{'YES' if snapshot else 'NO'}"
            )

        print(
            "\nFULL L3 SOURCE PAYLOAD READY"
        )

        # -----------------------------------------------------------
        # ANALYZERS
        # -----------------------------------------------------------

        print("\n" + "=" * 78)
        print("12 ANALYZER RESULTS")
        print("=" * 78)

        passed = 0
        failed = 0

        for analyzer_name, module_name, class_name in ANALYZERS:

            print(
                f"\n[{analyzer_name.upper()}]"
            )

            try:
                module = __import__(
                    module_name,
                    fromlist=[class_name],
                )

                analyzer_cls = getattr(
                    module,
                    class_name,
                )

                analyzer = analyzer_cls()

                result = analyzer.analyze(
                    payload
                )

                errors = validate_output(
                    analyzer_name,
                    result,
                )

                block = extract_block(
                    result,
                    analyzer_name,
                )

                if errors:
                    failed += 1

                    print("STATUS: FAIL")

                    for error in errors:
                        print(
                            "  ERROR:",
                            error,
                        )

                else:
                    passed += 1

                    print("STATUS: PASS")

                    print(
                        "confidence:",
                        block.get(
                            "confidence"
                        ),
                    )

                    print(
                        "feature_coverage_pct:",
                        block.get(
                            "feature_coverage_pct"
                        ),
                    )

                    summary = block.get(
                        "feature_trace_summary"
                    )

                    if isinstance(
                        summary,
                        dict,
                    ):
                        print(
                            "used_features:",
                            summary.get(
                                "used_features"
                            ),
                        )

                        print(
                            "missing_features:",
                            summary.get(
                                "missing_features"
                            ),
                        )

                    evidence = block.get(
                        "evidence"
                    )

                    print(
                        "evidence_count:",
                        len(evidence)
                        if isinstance(
                            evidence,
                            list,
                        )
                        else "INVALID",
                    )

            except Exception as exc:
                failed += 1

                print(
                    "STATUS: EXCEPTION"
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

                traceback.print_exc(
                    limit=8
                )

        # -----------------------------------------------------------
        # FINAL SUMMARY
        # -----------------------------------------------------------

        print("\n" + "=" * 78)
        print("FINAL SUMMARY")
        print("=" * 78)

        print(
            "TOTAL:",
            len(ANALYZERS),
        )

        print(
            "PASS :",
            passed,
        )

        print(
            "FAIL :",
            failed,
        )

        if failed == 0:
            print(
                "\nALL_12_ANALYZERS_DB_TEST_OK"
            )
            return 0

        print(
            "\nANALYZER_INTEGRATION_FAILURE"
        )

        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
