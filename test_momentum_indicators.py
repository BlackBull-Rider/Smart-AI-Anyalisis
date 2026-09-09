from __future__ import annotations

import sqlite3

from backend.config.settings import settings
from backend.analyzers.momentum_analyzer import (
    INDICATOR_NAMES,
    run_indicator_engine,
)


def main() -> None:
    db = settings.database_path

    print(f"DB: {db}")

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Discover feature_history schema
    # ------------------------------------------------------------------
    cols = [
        r["name"]
        for r in con.execute("PRAGMA table_info(feature_history)")
    ]

    print("feature_history columns:", cols)

    required = {"symbol"}
    missing = required - set(cols)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Find time column automatically.
    date_col = next(
        (
            x for x in (
                "date",
                "timestamp",
                "datetime",
                "period",
                "created_at",
            )
            if x in cols
        ),
        None,
    )

    if date_col is None:
        raise RuntimeError(
            "No date/timestamp column found in feature_history"
        )

    # ------------------------------------------------------------------
    # Find indicator columns
    # ------------------------------------------------------------------
    missing_indicators = [
        x for x in INDICATOR_NAMES
        if x not in cols
    ]

    if missing_indicators:
        raise RuntimeError(
            "Missing indicator columns:\n"
            + "\n".join(missing_indicators)
        )

    # ------------------------------------------------------------------
    # Pick an actual symbol from the database
    # ------------------------------------------------------------------
    symbol = con.execute(
        """
        SELECT symbol
        FROM feature_history
        WHERE symbol IS NOT NULL
        GROUP BY symbol
        ORDER BY COUNT(*) DESC
        LIMIT 1
        """
    ).fetchone()

    if symbol is None:
        raise RuntimeError("No symbols found in feature_history")

    symbol = symbol["symbol"]

    print(f"Symbol: {symbol}")
    print(f"Date column: {date_col}")

    # ------------------------------------------------------------------
    # REAL latest 100 feature-history periods
    # ------------------------------------------------------------------
    indicator_sql = ", ".join(
        f'"{x}"' for x in INDICATOR_NAMES
    )

    rows = con.execute(
        f"""
        SELECT
            "{date_col}" AS period,
            {indicator_sql}
        FROM feature_history
        WHERE symbol = ?
        ORDER BY "{date_col}" DESC
        LIMIT 100
        """,
        (symbol,),
    ).fetchall()

    rows = list(reversed(rows))

    if len(rows) < 50:
        raise RuntimeError(
            f"Only {len(rows)} periods found; "
            "minimum required is 50"
        )

    print(f"Periods loaded: {len(rows)}")

    # ------------------------------------------------------------------
    # Run ALL 23 engines independently
    # ------------------------------------------------------------------
    passed = 0

    for indicator in INDICATOR_NAMES:

        values = [
            row[indicator]
            for row in rows
        ]

        try:
            result = run_indicator_engine(
                indicator=indicator,
                values=values,
            )

            assert result["indicator"] == indicator
            assert len(result["trace"]["value"]) == len(rows)
            assert len(result["trace"]["delta"]) == len(rows)
            assert len(result["trace"]["acceleration"]) == len(rows)

            print(f"[PASS] {indicator}")
            passed += 1

        except Exception as exc:
            print(f"[FAIL] {indicator}: {exc}")

    con.close()

    print()
    print("=" * 60)
    print(f"RESULT: {passed}/{len(INDICATOR_NAMES)} engines passed")
    print("=" * 60)

    if passed != len(INDICATOR_NAMES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
