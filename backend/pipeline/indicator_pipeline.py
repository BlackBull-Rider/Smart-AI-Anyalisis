"""
GREEN BULL AI
Indicator Pipeline (Ultra-Fast MULTIPROCESSING Architecture)
"""
from __future__ import annotations

import logging
import sqlite3
import time
import warnings
import os
from pathlib import Path
import concurrent.futures

import pandas as pd
import numpy as np

from backend.indicators.indicator_engine import run as indicator_engine

warnings.simplefilter(action='ignore')
pd.set_option('future.no_silent_downcasting', True)

DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"
LOG_FILE = Path.home() / "Green-Bull-Data-Engine" / "pipeline_errors.log"

# ==========================================================
# STRICT FILE LOGGING
# ==========================================================
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode='a',
    level=logging.ERROR,
    format='[%(asctime)s] SYMBOL: %(symbol)s\n%(message)s\n' + '-'*60
)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ==========================================================
# WORKER FUNCTION (Runs in separate memory space - completely parallel)
# ==========================================================
def worker_process(symbol: str, feat_date: str, hist_date: str):
    # Fast Skip Logic
    if feat_date and hist_date and str(feat_date)[:10] >= str(hist_date)[:10]:
        return symbol, -1, None, {}

    try:
        # Engine Run (Independent Read connection inside engine)
        features = indicator_engine(symbol)

        if features is None or features.empty:
            return symbol, 0, None, {}
            
        if "date" in features.columns:
            features = features.drop_duplicates(subset=["date"], keep="last")

        if feat_date:
            features = features[pd.to_datetime(features["date"]) > pd.to_datetime(feat_date)]

        if features.empty:
            return symbol, 0, None, {}

        # Clean NaNs and Infinities
        features = features.replace([float("inf"), float("-inf"), np.inf, -np.inf], pd.NA).fillna(pd.NA)
        
        rows = len(features)
        last_date = str(features["date"].iloc[-1])[:10] if "date" in features.columns else "-"
        
        meta = {
            "features_len": len(features.columns),
            "candles_len": rows,
            "last_date": last_date
        }
        
        return symbol, rows, features, meta
    except Exception as e:
        return symbol, -2, str(e), {}


class IndicatorPipeline:
    TABLE = "feature_history"

    def __init__(self):
        self.stats = {"processed": 0, "success": 0, "failed": 0, "skipped": 0, "rows": 0}
        self.conn = sqlite3.connect(DB_PATH, timeout=60.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        # Leave 1 core free for OS and Main process to avoid freezing
        self.max_workers = max(1, (os.cpu_count() or 4) - 1)

    def fetch_all_dates(self):
        symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol", self.conn)["symbol"].tolist()
        
        hist_df = pd.read_sql_query("SELECT symbol, MAX(date) as max_date FROM historical_data GROUP BY symbol", self.conn)
        hist_dict = dict(zip(hist_df['symbol'], hist_df['max_date']))
        
        try:
            feat_df = pd.read_sql_query(f"SELECT symbol, MAX(date) as max_date FROM {self.TABLE} GROUP BY symbol", self.conn)
            feat_dict = dict(zip(feat_df['symbol'], feat_df['max_date']))
        except sqlite3.OperationalError:
            feat_dict = {}
            
        return symbols, hist_dict, feat_dict

    def save(self, df: pd.DataFrame) -> int:
        if df.empty: return 0
        df.to_sql(self.TABLE, self.conn, if_exists="append", index=False)
        return len(df)

    def run(self) -> None:
        self.stats["started"] = time.perf_counter()
        
        print("\n" + "="*50)
        print(" GREEN BULL AI - MASTER PIPELINE (MULTIPROCESSING)")
        print(f" CPU Cores Engaged: {self.max_workers}")
        print("="*50 + "\n")

        symbols, hist_dict, feat_dict = self.fetch_all_dates()
        total = len(symbols)

        # Using ProcessPoolExecutor to bypass Python's GIL
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for index, symbol in enumerate(symbols, start=1):
                f_date = feat_dict.get(symbol)
                h_date = hist_dict.get(symbol)
                
                # Pre-skip validation (Ultra Fast)
                if f_date and h_date and str(f_date)[:10] >= str(h_date)[:10]:
                    self.stats["skipped"] += 1
                    self.stats["processed"] += 1
                    continue
                    
                # Submit to independent processes
                start_time = time.perf_counter()
                future = executor.submit(worker_process, symbol, f_date, h_date)
                futures[future] = (index, start_time)

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                index, start_time = futures[future]
                elapsed = time.perf_counter() - start_time
                self.stats["processed"] += 1
                
                sym, rows, features_df, meta = future.result()
                
                if rows == -1:
                    self.stats["skipped"] += 1
                elif rows == -2:
                    self.stats["failed"] += 1
                    logger.error(f"Error in {sym}: {features_df}") # features_df holds error msg here
                elif rows == 0:
                    self.stats["failed"] += 1
                else:
                    # Write to DB strictly on Main Core to avoid SQL locking
                    self.save(features_df)
                    
                    self.stats["success"] += 1
                    self.stats["rows"] += rows
                    
                    print(
                        f"    Symbol  : {sym}\n"
                        f"    Date    : {meta['last_date']}\n"
                        f"    Features: {meta['features_len']}\n"
                        f"    Candles : {meta['candles_len']}",
                        flush=True
                    )
                    print(f"[{index:04d}/{total}] {sym:<14} + {rows} rows {elapsed:.2f}s", flush=True)

        self.conn.commit()
        self.stats["finished"] = time.perf_counter()
        elapsed = self.stats["finished"] - self.stats["started"]

        print("\n" + "="*50)
        print(" PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f" Processed : {self.stats['processed']}")
        print(f" Success   : {self.stats['success']}")
        print(f" Skipped   : {self.stats['skipped']}")
        print(f" Failed    : {self.stats['failed']}")
        print(f" Rows Added: {self.stats['rows']}")
        print(f" Time      : {elapsed:.2f}s")
        print("="*50 + "\n")
        self.conn.close()

if __name__ == "__main__":
    IndicatorPipeline().run()
