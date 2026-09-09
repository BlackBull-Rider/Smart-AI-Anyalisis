from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from backend.config.settings import settings

from backend.analyzers.candle_analyzer import (
    analyze_candles,
    CANDLE_FEATURE_COLUMNS,
    REQUIRED_PRICE_COLUMNS as CANDLE_REQUIRED_PRICE_COLUMNS,
)

from backend.analyzers.volume_analyzer import (
    analyze_volume,
    REQUIRED_PRICE_COLUMNS as VOLUME_REQUIRED_PRICE_COLUMNS,
    REQUIRED_VOLUME_COLUMNS,
)


# ============================================================
# CONFIG
# ============================================================

DB_PATH = settings.database_path

OUTPUT_DIR = (
    settings.project_root
    / "logs"
    / "analyzer_tests"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATAFRAME_ROWS = 250

SYMBOL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else None
)

TIMEFRAME = (
    sys.argv[2]
    if len(sys.argv) > 2
    else None
)


# ============================================================
# ANALYZER TYPE
# ============================================================

AnalyzerFunction = Callable[
    [pd.DataFrame],
    dict[str, Any],
]


# ============================================================
# ANALYZER REGISTRY
#
# Every analyzer receives the SAME DataFrame.
#
# Analyzer itself decides what it needs.
# Test runner does not fabricate analyzer input.
# ============================================================

ANALYZERS: dict[
    str,
    AnalyzerFunction,
] = {
    "candle": analyze_candles,
    "volume": analyze_volume,
}


# ============================================================
# DATABASE
# ============================================================

def connect() -> sqlite3.Connection:

    if not DB_PATH.exists():

        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(
        str(DB_PATH)
    )

    conn.row_factory = sqlite3.Row

    return conn


def list_tables(
    conn: sqlite3.Connection,
) -> list[str]:

    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    return [
        str(row["name"])
        for row in rows
    ]


def table_columns(
    conn: sqlite3.Connection,
    table: str,
) -> list[str]:

    rows = conn.execute(
        f'PRAGMA table_info("{table}")'
    ).fetchall()

    return [
        str(row["name"])
        for row in rows
    ]


# ============================================================
# COLUMN RESOLUTION
# ============================================================

def resolve_column(
    columns: list[str],
    wanted: str,
) -> str | None:

    exact = {
        column.lower(): column
        for column in columns
    }

    if wanted.lower() in exact:

        return exact[wanted.lower()]

    aliases = {

        "symbol": (
            "symbol",
            "ticker",
            "asset",
        ),

        "timeframe": (
            "timeframe",
            "interval",
            "tf",
        ),

        "timestamp": (
            "timestamp",
            "datetime",
            "date",
            "time",
            "ts",
        ),

        "open": (
            "open",
            "open_price",
        ),

        "high": (
            "high",
            "high_price",
        ),

        "low": (
            "low",
            "low_price",
        ),

        "close": (
            "close",
            "close_price",
        ),

        "volume": (
            "volume",
            "vol",
        ),
    }

    for alias in aliases.get(
        wanted.lower(),
        (),
    ):

        if alias in exact:

            return exact[alias]

    return None


# ============================================================
# MARKET TABLE DISCOVERY
#
# Prefer feature_history because that is the real feature
# DataFrame source for the analyzer pipeline.
# ============================================================

def find_market_table(
    conn: sqlite3.Connection,
) -> tuple[str, list[str]]:

    tables = list_tables(conn)

    # --------------------------------------------------------
    # First preference: feature_history
    # --------------------------------------------------------

    preferred = [
        "feature_history",
        "features",
        "market_data",
        "ohlcv",
    ]

    for preferred_table in preferred:

        if preferred_table not in tables:

            continue

        columns = table_columns(
            conn,
            preferred_table,
        )

        lowered = {
            column.lower()
            for column in columns
        }

        if {
            "open",
            "high",
            "low",
            "close",
        }.issubset(lowered):

            return (
                preferred_table,
                columns,
            )

    # --------------------------------------------------------
    # Generic discovery
    # --------------------------------------------------------

    candidates: list[
        tuple[int, str, list[str]]
    ] = []

    all_expected_features = (
        list(CANDLE_FEATURE_COLUMNS)
        + list(REQUIRED_VOLUME_COLUMNS)
    )

    for table in tables:

        columns = table_columns(
            conn,
            table,
        )

        lowered = {
            column.lower()
            for column in columns
        }

        if not {
            "open",
            "high",
            "low",
            "close",
        }.issubset(lowered):

            continue

        feature_hits = sum(
            1
            for feature
            in all_expected_features
            if feature.lower()
            in lowered
        )

        candidates.append(
            (
                feature_hits,
                table,
                columns,
            )
        )

    if not candidates:

        raise RuntimeError(
            "Could not find a market/feature table "
            "containing OHLC columns."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, table, columns = candidates[0]

    return table, columns


# ============================================================
# SYMBOL / TIMEFRAME DISCOVERY
# ============================================================

def available_symbols(
    conn: sqlite3.Connection,
    table: str,
    symbol_column: str,
) -> list[str]:

    rows = conn.execute(
        f'''
        SELECT DISTINCT "{symbol_column}"
        FROM "{table}"
        WHERE "{symbol_column}" IS NOT NULL
        ORDER BY "{symbol_column}"
        '''
    ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


def available_timeframes(
    conn: sqlite3.Connection,
    table: str,
    timeframe_column: str,
) -> list[str]:

    rows = conn.execute(
        f'''
        SELECT DISTINCT "{timeframe_column}"
        FROM "{table}"
        WHERE "{timeframe_column}" IS NOT NULL
        ORDER BY "{timeframe_column}"
        '''
    ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


# ============================================================
# FEATURE COLUMN COLLECTION
#
# IMPORTANT:
# Only real database columns are selected.
# No fake columns are created.
# ============================================================

def collect_existing_columns(
    columns: list[str],
    requested_columns: tuple[str, ...],
) -> list[str]:

    selected: list[str] = []

    for requested in requested_columns:

        actual = resolve_column(
            columns,
            requested,
        )

        if actual is not None:

            selected.append(
                actual
            )

    return list(
        dict.fromkeys(
            selected
        )
    )


# ============================================================
# LIVE DATABASE -> DATAFRAME
# ============================================================

def load_analyzer_datafrem(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    symbol: str,
    timeframe: str | None,
    rows_limit: int = DATAFRAME_ROWS,
) -> pd.DataFrame:

    symbol_col = resolve_column(
        columns,
        "symbol",
    )

    timeframe_col = resolve_column(
        columns,
        "timeframe",
    )

    timestamp_col = resolve_column(
        columns,
        "timestamp",
    )

    if symbol_col is None:

        raise RuntimeError(
            f"{table}: symbol column not found."
        )

    if timestamp_col is None:

        raise RuntimeError(
            f"{table}: timestamp/date column not found."
        )

    # --------------------------------------------------------
    # Required analyzer input columns
    #
    # OHLC
    # + candle features
    # + volume features
    # + optional candle cross-confirmation columns
    # --------------------------------------------------------

    requested_columns: list[str] = []

    requested_columns.extend(
        CANDLE_REQUIRED_PRICE_COLUMNS
    )

    requested_columns.extend(
        VOLUME_REQUIRED_PRICE_COLUMNS
    )

    requested_columns.extend(
        CANDLE_FEATURE_COLUMNS
    )

    requested_columns.extend(
        REQUIRED_VOLUME_COLUMNS
    )

    # --------------------------------------------------------
    # Optional candle columns required by Volume Analyzer
    # --------------------------------------------------------

    volume_optional_candle = (
        "bullish_candle",
        "bearish_candle",
        "neutral_candle",
        "body_strength",
        "body_percent",
        "body_size",
        "real_body",
        "candle_range",
        "pressure_score",
        "dominance_score",
        "balance_score",
        "close_position",
        "inside_bar",
        "outside_bar",
        "engulfing_body",
        "expansion_candle",
        "compression_candle",
        "impulse_candle",
        "indecision_candle",
    )

    requested_columns.extend(
        volume_optional_candle
    )

    # --------------------------------------------------------
    # Resolve only columns that really exist.
    # --------------------------------------------------------

    selected_columns = (
        collect_existing_columns(
            columns,
            tuple(
                dict.fromkeys(
                    requested_columns
                )
            ),
        )
    )

    # Timestamp is mandatory.
    if timestamp_col not in selected_columns:

        selected_columns.insert(
            0,
            timestamp_col,
        )

    if not selected_columns:

        raise RuntimeError(
            "No analyzer input columns found."
        )

    # --------------------------------------------------------
    # SQL filters
    # --------------------------------------------------------

    where_parts = [
        f'"{symbol_col}" = ?'
    ]

    params: list[Any] = [
        symbol
    ]

    if (
        timeframe is not None
        and timeframe_col is not None
    ):

        where_parts.append(
            f'"{timeframe_col}" = ?'
        )

        params.append(
            timeframe
        )

    where_sql = " AND ".join(
        where_parts
    )

    select_sql = ", ".join(
        f'"{column}"'
        for column in selected_columns
    )

    # --------------------------------------------------------
    # Latest N rows
    #
    # DESC fetch for efficiency,
    # then ASC restore for analyzers.
    # --------------------------------------------------------

    query = f'''
        SELECT {select_sql}
        FROM "{table}"
        WHERE {where_sql}
        ORDER BY "{timestamp_col}" DESC
        LIMIT ?
    '''

    params.append(
        rows_limit
    )

    rows = conn.execute(
        query,
        params,
    ).fetchall()

    if not rows:

        raise RuntimeError(
            f"No data found for "
            f"{symbol} "
            f"{timeframe or ''}."
        )

    records = [
        dict(row)
        for row in rows
    ]

    datafrem = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Latest-first -> chronological
    # --------------------------------------------------------

    datafrem = (
        datafrem
        .sort_values(
            by=timestamp_col
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Normalize OHLC only.
    #
    # Database aliases remain untouched elsewhere.
    # --------------------------------------------------------

    canonical_ohlc = {
        "open": resolve_column(
            columns,
            "open",
        ),

        "high": resolve_column(
            columns,
            "high",
        ),

        "low": resolve_column(
            columns,
            "low",
        ),

        "close": resolve_column(
            columns,
            "close",
        ),

        "volume": resolve_column(
            columns,
            "volume",
        ),
    }

    rename_map = {}

    for canonical, actual in (
        canonical_ohlc.items()
    ):

        if (
            actual is not None
            and actual in datafrem.columns
            and actual != canonical
        ):

            rename_map[
                actual
            ] = canonical

    if rename_map:

        datafrem = datafrem.rename(
            columns=rename_map
        )

    # --------------------------------------------------------
    # Timestamp -> DataFrame index
    # --------------------------------------------------------

    if timestamp_col in datafrem.columns:

        datafrem = datafrem.set_index(
            timestamp_col
        )

    return datafrem


# ============================================================
# SCHEMA REPORT
# ============================================================

def schema_report(
    datafrem: pd.DataFrame,
) -> dict[str, Any]:

    available = {
        str(column).lower()
        for column in datafrem.columns
    }

    candle_missing = [
        column
        for column
        in CANDLE_FEATURE_COLUMNS
        if column.lower()
        not in available
    ]

    volume_missing = [
        column
        for column
        in REQUIRED_VOLUME_COLUMNS
        if column.lower()
        not in available
    ]

    return {
        "candle": {
            "expected": len(
                CANDLE_FEATURE_COLUMNS
            ),
            "present": (
                len(CANDLE_FEATURE_COLUMNS)
                - len(candle_missing)
            ),
            "missing": candle_missing,
        },

        "volume": {
            "expected": len(
                REQUIRED_VOLUME_COLUMNS
            ),
            "present": (
                len(REQUIRED_VOLUME_COLUMNS)
                - len(volume_missing)
            ),
            "missing": volume_missing,
        },

        "dataframe_columns": len(
            datafrem.columns
        ),
    }


# ============================================================
# JSON SAFE
# ============================================================

def _is_nan_value(
    value: Any,
) -> bool:

    try:

        result = pd.isna(
            value
        )

        if isinstance(
            result,
            bool,
        ):

            return result

    except (
        TypeError,
        ValueError,
    ):

        pass

    return False


def json_safe(
    value: Any,
) -> Any:

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    if _is_nan_value(
        value
    ):

        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):

        return value

    return str(value)


# ============================================================
# SINGLE ANALYZER TEST
# ============================================================

def run_single_analyzer_test(
    name: str,
    analyzer: AnalyzerFunction,
    datafrem: pd.DataFrame,
) -> dict[str, Any]:

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"ANALYZER: {name}"
    )

    print(
        "=" * 78
    )

    print(
        f"DataFrame rows : {len(datafrem)}"
    )

    print(
        f"DataFrame cols : {len(datafrem.columns)}"
    )

    print(
        f"First period   : {datafrem.index[0]}"
    )

    print(
        f"Last period    : {datafrem.index[-1]}"
    )

    result = analyzer(
        datafrem.copy()
    )

    if not isinstance(
        result,
        dict,
    ):

        raise TypeError(
            f"{name} analyzer must return dict."
        )

    print(
        "STATUS         : PASSED"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 78
    )

    print(
        "GREEN BULL RIDER V6"
    )

    print(
        "LIVE DATABASE ANALYZER TEST RUNNER"
    )

    print(
        "=" * 78
    )

    print(
        f"\nDatabase:\n{DB_PATH}"
    )

    conn = connect()

    try:

        # ----------------------------------------------------
        # FIND REAL TABLE
        # ----------------------------------------------------

        table, columns = (
            find_market_table(
                conn
            )
        )

        print(
            f"\nDetected market table: {table}"
        )

        print(
            f"Database columns: {len(columns)}"
        )

        # ----------------------------------------------------
        # SYMBOL
        # ----------------------------------------------------

        symbol_col = resolve_column(
            columns,
            "symbol",
        )

        timeframe_col = resolve_column(
            columns,
            "timeframe",
        )

        if symbol_col is None:

            raise RuntimeError(
                "Symbol column not found."
            )

        symbol = SYMBOL

        if symbol is None:

            symbols = available_symbols(
                conn,
                table,
                symbol_col,
            )

            if not symbols:

                raise RuntimeError(
                    "No symbols found."
                )

            print(
                "\nAvailable symbols:"
            )

            for number, value in enumerate(
                symbols[:50],
                start=1,
            ):

                print(
                    f"{number}. {value}"
                )

            symbol = input(
                "\nEnter symbol: "
            ).strip()

            if not symbol:

                raise RuntimeError(
                    "Symbol is required."
                )

        # ----------------------------------------------------
        # TIMEFRAME
        # ----------------------------------------------------

        timeframe = TIMEFRAME

        if (
            timeframe is None
            and timeframe_col is not None
        ):

            timeframes = (
                available_timeframes(
                    conn,
                    table,
                    timeframe_col,
                )
            )

            if timeframes:

                print(
                    "\nAvailable timeframes:"
                )

                for number, value in enumerate(
                    timeframes,
                    start=1,
                ):

                    print(
                        f"{number}. {value}"
                    )

                timeframe_input = input(
                    "\nEnter timeframe "
                    "(blank = all): "
                ).strip()

                timeframe = (
                    timeframe_input
                    or None
                )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 78
        )

        print(
            "TEST TARGET"
        )

        print(
            "-" * 78
        )

        print(
            f"Symbol    : {symbol}"
        )

        print(
            f"Timeframe : {timeframe or 'ALL'}"
        )

        print(
            f"Rows      : {DATAFRAME_ROWS}"
        )

        # ----------------------------------------------------
        # LOAD ONE SHARED DATAFRAME
        # ----------------------------------------------------

        datafrem = load_analyzer_datafrem(
            conn=conn,
            table=table,
            columns=columns,
            symbol=symbol,
            timeframe=timeframe,
            rows_limit=DATAFRAME_ROWS,
        )

        print(
            "\n"
            + "-" * 78
        )

        print(
            "DATAFRAME LOADED"
        )

        print(
            "-" * 78
        )

        print(
            f"Rows      : {len(datafrem)}"
        )

        print(
            f"Columns   : {len(datafrem.columns)}"
        )

        print(
            f"From      : {datafrem.index[0]}"
        )

        print(
            f"To        : {datafrem.index[-1]}"
        )

        # ----------------------------------------------------
        # SCHEMA REPORT
        # ----------------------------------------------------

        schemas = schema_report(
            datafrem
        )

        print(
            "\n"
            + "-" * 78
        )

        print(
            "ANALYZER INPUT SCHEMA"
        )

        print(
            "-" * 78
        )

        print(
            "CANDLE:"
        )

        print(
            f"  Present : "
            f"{schemas['candle']['present']}"
            f"/"
            f"{schemas['candle']['expected']}"
        )

        if schemas["candle"]["missing"]:

            print(
                "  Missing:"
            )

            for column in (
                schemas["candle"]["missing"]
            ):

                print(
                    f"    - {column}"
                )

        print(
            "VOLUME:"
        )

        print(
            f"  Present : "
            f"{schemas['volume']['present']}"
            f"/"
            f"{schemas['volume']['expected']}"
        )

        if schemas["volume"]["missing"]:

            print(
                "  Missing:"
            )

            for column in (
                schemas["volume"]["missing"]
            ):

                print(
                    f"    - {column}"
                )

        # ----------------------------------------------------
        # RUN ANALYZERS
        # ----------------------------------------------------

        results: dict[
            str,
            dict[str, Any],
        ] = {}

        failures: dict[
            str,
            str,
        ] = {}

        for name, analyzer in (
            ANALYZERS.items()
        ):

            try:

                results[name] = (
                    run_single_analyzer_test(
                        name=name,
                        analyzer=analyzer,
                        datafrem=datafrem,
                    )
                )

            except Exception as exc:

                failures[name] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    "\nSTATUS         : FAILED"
                )

                print(
                    f"ERROR          : "
                    f"{failures[name]}"
                )

        # ----------------------------------------------------
        # FINAL JSON
        # ----------------------------------------------------

        output = {
            "test_runner": (
                "live_database_analyzer_test"
            ),

            "version": "1.1.0",

            "database": str(
                DB_PATH
            ),

            "table": table,

            "symbol": symbol,

            "timeframe": timeframe,

            "dataframe": {
                "rows": len(
                    datafrem
                ),

                "columns": len(
                    datafrem.columns
                ),

                "start": str(
                    datafrem.index[0]
                ),

                "end": str(
                    datafrem.index[-1]
                ),
            },

            "input_schema": schemas,

            "analyzer_count": len(
                ANALYZERS
            ),

            "successful": len(
                results
            ),

            "failed": len(
                failures
            ),

            "failures": failures,

            "results": results,
        }

        output = json_safe(
            output
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        safe_timeframe = (
            timeframe
            or "all"
        )

        output_file = (
            OUTPUT_DIR
            / (
                f"{symbol}_"
                f"{safe_timeframe}_"
                "all_analyzer_test.json"
            )
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as fp:

            json.dump(
                output,
                fp,
                ensure_ascii=False,
                indent=2,
            )

        # ----------------------------------------------------
        # FINAL REPORT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 78
        )

        print(
            "FINAL TEST REPORT"
        )

        print(
            "=" * 78
        )

        print(
            f"Symbol    : {symbol}"
        )

        print(
            f"Timeframe : {timeframe or 'ALL'}"
        )

        print(
            f"Rows      : {len(datafrem)}"
        )

        print(
            f"Analyzers : {len(ANALYZERS)}"
        )

        print(
            f"Passed    : {len(results)}"
        )

        print(
            f"Failed    : {len(failures)}"
        )

        print(
            f"\nJSON:\n{output_file}"
        )

        if failures:

            print(
                "\nFAILED ANALYZERS:"
            )

            for name, error in (
                failures.items()
            ):

                print(
                    f"  - {name}: {error}"
                )

            raise SystemExit(1)

        print(
            "\nALL REGISTERED ANALYZERS PASSED."
        )

    finally:

        conn.close()


if __name__ == "__main__":
    main()
