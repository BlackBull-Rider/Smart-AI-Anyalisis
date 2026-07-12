"""
backend/pipeline/trend_pipeline.py

Trend Pipeline

Flow

market.db
    ↓
feature_history
    ↓
financial_data
    ↓
TrendFeatureAdapter
    ↓
TrendAnalyzer
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from backend.adapters.trend_feature_adapter import TrendFeatureAdapter
from backend.analyzers.trend_analyzer import TrendAnalyzer


DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"


class TrendPipeline:

    def __init__(self):

        self.adapter = TrendFeatureAdapter()
        self.analyzer = TrendAnalyzer()

    def _connect(self):

        return sqlite3.connect(DB_PATH)

    def load_feature_history(
        self,
        symbol: str,
        limit: int = 500,
    ) -> pd.DataFrame:

        query = """
        SELECT *
        FROM feature_history
        WHERE symbol=?
        ORDER BY date ASC
        LIMIT ?
        """

        with self._connect() as conn:

            return pd.read_sql(
                query,
                conn,
                params=[symbol.upper(), limit],
                parse_dates=["date"],
            )

    def load_financial(
        self,
        symbol: str,
    ) -> pd.Series:

        query = """
        SELECT *
        FROM financial_data
        WHERE symbol=?
        LIMIT 1
        """

        with self._connect() as conn:

            df = pd.read_sql(
                query,
                conn,
                params=[symbol.upper()],
            )

        if df.empty:

            return pd.Series(dtype=float)

        return df.iloc[0]

    def run(
        self,
        symbol: str,
    ):

        feature_df = self.load_feature_history(symbol)

        financial = self.load_financial(symbol)

        adapted_df = self.adapter.adapt(
            feature_df,
            financial,
        )

        result = self.analyzer.analyze(
            adapted_df,
        )

        return result


def run(symbol: str):

    pipeline = TrendPipeline()

    return pipeline.run(symbol)


if __name__ == "__main__":

    import json

    result = run("RELIANCE")

    print(
        json.dumps(
            result,
            indent=4,
            default=str,
        )
    )
