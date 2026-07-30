"""
GREEN BULL RIDER V6
Module: backend/repository/stock_repository.py

Repository Layer (100% Future-Proof & Dynamic Schema Evolution)

Responsibilities
----------------
- Read/Write Database
- Dynamic Table & Column Creation (Auto-DDL)
- Provider to DB Auto-Mapping
- JSON Serialization for Engine Outputs
- Zero Magic Numbers & Zero Placeholders
"""

from __future__ import annotations

import logging
import re
import json
import pandas as pd
import numpy as np

from backend.db.connection import db

logger = logging.getLogger(__name__)

class DynamicDataMapper:
    """
    Universal Data Mapper: Intercepts raw provider data and maps it to standardized DB columns.
    """
    def __init__(self):
        self.MASTER_ALIASES = {
            "pe_ratio": ["trailingpe", "pe", "peratio", "priceearnings"],
            "pb_ratio": ["pricetobook", "pb", "pbratio", "price_to_book"],
            "debt_to_equity": ["debttoequity", "debtequity", "debt_equity"],
            "revenue_growth_yoy": ["revenuegrowth", "salesgrowth", "revenue_growth", "sales_growth"],
            "profit_growth_yoy": ["earningsgrowth", "profitgrowth", "netincomegrowth", "profit_growth"],
            "eps_growth_yoy": ["epsgrowth", "eps_growth"],
            "fcf_growth_yoy": ["freecashflowgrowth", "fcf_growth"],
            "total_revenue": ["sales", "operatingrevenue", "revenue"],
            "capital_expenditure": ["capex", "capitalexpenditure"],
            "shareholder_equity": ["total_equity", "equity", "totalequity"],
            "accounts_receivable": ["receivables", "accountsreceivable"],
            "issue_size": ["ipo_size", "issuesize"],
            "free_cash_flow": ["freecashflow", "fcf"],
            "operating_cash_flow": ["operatingcashflow", "cfo"],
            "market_cap": ["marketcap", "market_capitalization"],
            "dividend_yield": ["dividendyield"],
            "current_ratio": ["currentratio"],
            "quick_ratio": ["quickratio"],
            "total_assets": ["assets", "totalassets"],
            "total_liabilities": ["liabilities", "totalliabilities"],
            "total_debt": ["totaldebt", "debt"],
            "retained_earnings": ["retainedearnings"],
            "ebitda": ["normalizedebitda", "ebitda"],
            "fii_holding": ["fii", "foreigninstitutional"],
            "dii_holding": ["dii", "domesticinstitutional"],
            "promoter_holding": ["promoter", "heldpercentinsiders"],
            "institutional_holding": ["heldpercentinstitutions"]
        }

    def _clean_key(self, key: str) -> str:
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str(key))
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        name = re.sub(r'[^a-z0-9_]', '_', name)
        return re.sub(r'_+', '_', name).strip('_')

    def map_data(self, raw_data: dict) -> dict:
        mapped_data = {}
        for raw_key, value in raw_data.items():
            clean_key = self._clean_key(raw_key)
            target_col = clean_key
            
            for standard_col, aliases in self.MASTER_ALIASES.items():
                if clean_key in [self._clean_key(a) for a in aliases] or clean_key == standard_col:
                    target_col = standard_col
                    break
                    
            mapped_data[target_col] = value
        return mapped_data


class StockRepository:

    def __init__(self):
        self.mapper = DynamicDataMapper()

    # =====================================================================
    # DYNAMIC SCHEMA EVOLUTION ENGINE (GOD MODE BULK INSERT)
    # =====================================================================

    def dynamic_bulk_insert(self, table: str, rows: list[dict], pk_cols: list[str]) -> None:
        """
        100% Future-Proof Insert: 
        - Maps Aliases
        - Converts Dicts/Lists to JSON strings
        - Creates Tables if missing
        - Adds Columns if missing
        - Inserts Data safely (SQL Syntax Quoted)
        """
        if not rows:
            return

        # 1. Clean, Map, and Serialize
        cleaned_rows = []
        for row in rows:
            mapped = self.mapper.map_data(row)
            clean_row = {}
            for k, v in mapped.items():
                if isinstance(v, (dict, list, tuple, set)) or hasattr(v, 'tolist'):
                    try:
                        val = v.tolist() if hasattr(v, 'tolist') else (list(v) if isinstance(v, set) else v)
                        clean_row[k] = json.dumps(val, default=str)
                    except Exception:
                        clean_row[k] = str(v)
                elif pd.api.types.is_scalar(v):
                    if pd.isna(v):
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                else:
                    clean_row[k] = str(v)
            cleaned_rows.append(clean_row)

        # 2. Infer Data Types
        schema_types = {}
        for row in cleaned_rows:
            for k, v in row.items():
                if k not in schema_types or schema_types[k] == "TEXT":
                    if v is not None:
                        if isinstance(v, int): schema_types[k] = "INTEGER"
                        elif isinstance(v, float): schema_types[k] = "REAL"
                        else: schema_types[k] = "TEXT"
                        
        for pk in pk_cols:
            if pk not in schema_types:
                schema_types[pk] = "TEXT"

        # 3. Auto-DDL (Create Table or Add Columns)
        db_cols_info = db.fetchall(f"PRAGMA table_info({table})")
        existing_cols = {c['name'].lower() for c in db_cols_info}

        if not existing_cols:
            # ১৪৬ নম্বর লাইনটিকে এভাবে লেখ:
            quoted_cols = [f'"{pk}"' for pk in pk_cols]
            pk_str = f"PRIMARY KEY({', '.join(quoted_cols)})"
            create_sql = f"CREATE TABLE {table} ({', '.join(col_defs)}, {pk_str})"
            db.execute(create_sql)
            logger.info(f"Dynamically created table: {table}")
        else:
            for k, col_type in schema_types.items():
                if k not in existing_cols:
                    db.execute(f'ALTER TABLE {table} ADD COLUMN "{k}" {col_type}')
                    logger.info(f"Dynamically added column: {k} to table {table}")

        # 4. Execute Bulk Insert
        all_columns = list(schema_types.keys())
        placeholders = ",".join(["?"] * len(all_columns))
        cols_str = ",".join([f'"{c}"' for c in all_columns])
        sql = f"INSERT OR REPLACE INTO {table} ({cols_str}) VALUES ({placeholders})"

        values = []
        for row in cleaned_rows:
            values.append(tuple(row.get(col, None) for col in all_columns))

        db.executemany(sql, values)

    # =====================================================================
    # ENGINE OUTPUT PERSISTENCE (HISTORY TABLES)
    # =====================================================================

    def save_features(self, symbol: str, dataframe: pd.DataFrame) -> int:
        """Indicator Engine Output -> feature_history"""
        if dataframe.empty: return 0
        df = dataframe.copy()
        
        if "date" not in [c.lower() for c in df.columns] and "date" in df.index.names:
            df = df.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        df["symbol"] = symbol
        last_date = self.get_last_feature_date(symbol)
        if last_date:
            df = df[df["date"] > last_date]

        if df.empty: return 0
        self.dynamic_bulk_insert("feature_history", df.to_dict("records"), ["symbol", "date"])
        return len(df)

    def save_analyzer_history(self, symbol: str, date: str, analyzer_results: dict) -> None:
        """Analyzer Engine Output -> analyzer_history"""
        row = {"symbol": symbol, "date": date}
        row.update(analyzer_results)
        self.dynamic_bulk_insert("analyzer_history", [row], ["symbol", "date"])

    def save_scoring_history(self, symbol: str, date: str, scoring_results: dict) -> None:
        """Scoring Engine Output -> scoring_history"""
        row = {"symbol": symbol, "date": date}
        row.update(scoring_results)
        self.dynamic_bulk_insert("scoring_history", [row], ["symbol", "date"])

    def save_decision_history(self, symbol: str, date: str, decision_results: dict) -> None:
        """Decision Engine Output -> decision_history"""
        row = {"symbol": symbol, "date": date}
        row.update(decision_results)
        self.dynamic_bulk_insert("decision_history", [row], ["symbol", "date"])

    def save_ai_history(self, symbol: str, date: str, ai_output: dict) -> None:
        """AI LLM Output -> ai_history"""
        row = {"symbol": symbol, "date": date}
        row.update(ai_output)
        self.dynamic_bulk_insert("ai_history", [row], ["symbol", "date"])

    # =====================================================================
    # MASTER / RAW DATA TABLES
    # =====================================================================

    def save_history(self, dataframe: pd.DataFrame) -> None:
        """Raw OHLCV -> historical_data"""
        if dataframe.empty: return
        df = dataframe.copy()
        if "date" not in [c.lower() for c in df.columns]: df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns: df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
        self.dynamic_bulk_insert("historical_data", df.to_dict("records"), ["symbol", "date"])

    def save_company_profile(self, data: dict) -> None:
        self.dynamic_bulk_insert("company_profile", [data], ["symbol"])

    def save_fundamental(self, data: dict) -> None:
        self.dynamic_bulk_insert("fundamental_data", [data], ["symbol"])

    def save_ipo_data(self, data: dict) -> None:
        self.dynamic_bulk_insert("ipo_data", [data], ["symbol"])

    def save_financials(self, rows: list[dict]) -> None:
        if not rows: return
        self.dynamic_bulk_insert("financial_data", rows, ["symbol", "fiscal_year", "fiscal_quarter"])

    def save_shareholding(self, rows) -> None:
        if rows is None: return
        if isinstance(rows, dict): rows = [rows]
        if not isinstance(rows, list): rows = list(rows)
        if not rows: return
        self.dynamic_bulk_insert("shareholding_data", rows, ["symbol", "quarter"])

    def save_corporate_actions(self, rows: list[dict]) -> None:
        self.dynamic_bulk_insert("corporate_actions", rows, ["symbol", "action_date", "action_type"])

    def save_earnings(self, rows: list[dict]) -> None:
        self.dynamic_bulk_insert("earnings_history", rows, ["symbol", "quarter"])

    def save_analyst_data(self, data: dict) -> None:
        self.dynamic_bulk_insert("analyst_data", [data], ["symbol"])

    # =====================================================================
    # RETRIEVAL (GET) METHODS
    # =====================================================================

    def get_active_symbols(self) -> list[dict]:
        rows = db.fetchall("SELECT symbol, company_name, exchange FROM stock_master WHERE UPPER(status)='ACTIVE' ORDER BY symbol")
        return [dict(row) for row in rows]

    def get_symbol(self, symbol: str) -> dict | None:
        row = db.fetchone("SELECT * FROM stock_master WHERE symbol=?", (symbol.upper(),))
        return dict(row) if row else None

    def symbol_exists(self, symbol: str) -> bool:
        row = db.fetchone("SELECT 1 FROM stock_master WHERE symbol=? LIMIT 1", (symbol.upper(),))
        return row is not None

    def get_last_history_date(self, symbol: str):
        row = db.fetchone("SELECT MAX(date) AS last_date FROM historical_data WHERE symbol=?", (symbol.upper(),))
        return row["last_date"] if row else None

    def get_last_feature_date(self, symbol: str):
        row = db.fetchone("SELECT MAX(date) AS last_date FROM feature_history WHERE symbol=?", (symbol.upper(),))
        return row["last_date"] if row else None

    def get_company_profile(self, symbol: str) -> dict | None:
        row = db.fetchone("SELECT * FROM company_profile WHERE symbol=?", (symbol.upper(),))
        return dict(row) if row else None

    def get_fundamental(self, symbol: str) -> dict | None:
        row = db.fetchone("SELECT * FROM fundamental_data WHERE symbol=?", (symbol.upper(),))
        return dict(row) if row else None

    def get_ipo_data(self, symbol: str) -> dict | None:
        row = db.fetchone("SELECT * FROM ipo_data WHERE symbol=?", (symbol.upper(),))
        return dict(row) if row else None

    def get_financials(self, symbol: str) -> list[dict]:
        rows = db.fetchall("SELECT * FROM financial_data WHERE symbol=? ORDER BY fiscal_year DESC, fiscal_quarter DESC", (symbol.upper(),))
        return [dict(row) for row in rows]

    def get_shareholding(self, symbol: str) -> list[dict]:
        rows = db.fetchall("SELECT * FROM shareholding_data WHERE symbol=? ORDER BY quarter DESC", (symbol.upper(),))
        return [dict(row) for row in rows]

    def get_corporate_actions(self, symbol: str) -> list[dict]:
        rows = db.fetchall("SELECT * FROM corporate_actions WHERE symbol=? ORDER BY action_date DESC", (symbol.upper(),))
        return [dict(row) for row in rows]

    def get_earnings(self, symbol: str) -> list[dict]:
        rows = db.fetchall("SELECT * FROM earnings_history WHERE symbol=? ORDER BY quarter DESC", (symbol.upper(),))
        return [dict(row) for row in rows]

repository = StockRepository()
