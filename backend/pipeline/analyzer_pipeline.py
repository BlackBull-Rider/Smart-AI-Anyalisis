"""
GREEN BULL AI
Master Analyzer Pipeline (Layer 2) - OPTIMIZED & JSON SAFE
Only calculates JSON for the required missing dates and handles NaNs safely.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import sys
import time
import os
import math
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

DB_PATH = Path.home() / "Green-Bull-Data-Engine" / "database" / "market.db"

def process_analyzer_worker(symbol: str, last_date: str | None, db_path: str):
    import warnings
    import pandas as pd
    import numpy as np
    import sqlite3
    import json
    import math
    warnings.simplefilter("ignore")
    
    from backend.adapters.master_feature_adapter import MasterFeatureAdapter
    
    # ---------------------------------------------------------
    # JSON SAFETY HELPERS
    # ---------------------------------------------------------
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(v) for v in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif pd.isna(obj): 
            return None
        return obj

    def json_encoder(obj):
        if isinstance(obj, (np.float32, np.float64)):
            val = float(obj)
            return None if math.isnan(val) or math.isinf(val) else val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    # ---------------------------------------------------------

    analyzers = {}
    try:
        from backend.analyzers.trend_analyzer import TrendAnalyzer
        analyzers['trend'] = TrendAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.momentum_analyzer import MomentumAnalyzer
        analyzers['momentum'] = MomentumAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.volatility_analyzer import VolatilityAnalyzer
        analyzers['volatility'] = VolatilityAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.volume_analyzer import VolumeAnalyzer
        analyzers['volume'] = VolumeAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.support_resistance_analyzer import SupportResistanceAnalyzer
        analyzers['support_resistance'] = SupportResistanceAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.smart_money_analyzer import SmartMoneyAnalyzer
        analyzers['smart_money'] = SmartMoneyAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.pattern_analyzer import PatternAnalyzer
        analyzers['pattern'] = PatternAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.market_regime_analyzer import MarketRegimeAnalyzer
        analyzers['market_regime'] = MarketRegimeAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.fundamental_analyzer import FundamentalAnalyzer
        analyzers['fundamental'] = FundamentalAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.ipo_analyzer import IPOAnalyzer
        analyzers['ipo'] = IPOAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.candle_analyzer import CandleAnalyzer
        analyzers['candle'] = CandleAnalyzer()
    except Exception: pass
    try:
        from backend.analyzers.institutional_analyzer import InstitutionalAnalyzer
        analyzers['institutional'] = InstitutionalAnalyzer()
    except Exception: pass

    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        
        if last_date is None:
            fetch_limit = 250 
            calc_limit = 1    
        else:
            missing = conn.execute(
                "SELECT COUNT(*) FROM feature_history WHERE symbol=? AND date > ?",
                (symbol, last_date)
            ).fetchone()[0]

            if missing == 0:
                conn.close()
                return symbol, None, 0

            fetch_limit = max(250, 249 + missing) 
            calc_limit = missing 

        query = "SELECT * FROM feature_history WHERE symbol=? ORDER BY date DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(symbol, fetch_limit), parse_dates=["date"])
        conn.close()

        if df.empty:
            return symbol, None, 0

        df = df.sort_values("date")
        df.set_index("date", inplace=True)
        
        adapter = MasterFeatureAdapter()
        adapted_df = adapter.adapt(df)

        results = []
        for i in range(calc_limit):
            target_idx = len(adapted_df) - calc_limit + i + 1
            if target_idx < 10: 
                continue
                
            sub_df = adapted_df.iloc[:target_idx]
            current_date = sub_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            
            row_data = {"symbol": symbol, "date": current_date}
            
            for key, engine in analyzers.items():
                try:
                    res = engine.analyze(sub_df)
                    # Safe JSON conversion
                    cleaned_res = clean_nans(res)
                    row_data[key] = json.dumps(cleaned_res, default=json_encoder) 
                except Exception:
                    row_data[key] = "{}" 
            
            results.append(row_data)

        if not results:
            return symbol, None, 0

        final_df = pd.DataFrame(results)
        return symbol, final_df, len(final_df)
        
    except Exception as e:
        return symbol, str(e), -1


class AnalyzerPipeline:
    def __init__(self):
        self.db_path = str(DB_PATH)
        self.table = "analyzer_history"
        self.max_workers = min(4, os.cpu_count() or 4) 
        self.batch_size = 30
        self.stats = {"processed": 0, "success": 0, "failed": 0, "skipped": 0, "rows": 0}
        self.total_symbols = 0
        self._last_render = 0.0
        self._rolling_metrics = deque(maxlen=120)
        self._create_table_if_not_exists()
        sys.stdout.write("\033[?25l\033[2J")
        sys.stdout.flush()

    def _create_table_if_not_exists(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                symbol TEXT, date TEXT, trend TEXT, momentum TEXT, volatility TEXT, volume TEXT,
                support_resistance TEXT, smart_money TEXT, pattern TEXT, market_regime TEXT,
                fundamental TEXT, ipo TEXT, candle TEXT, institutional TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)
        conn.commit()
        conn.close()

    def get_symbols_and_dates(self):
        conn = sqlite3.connect(self.db_path)
        symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM feature_history ORDER BY symbol", conn)["symbol"].tolist()
        try:
            last_dates_raw = conn.execute(f"SELECT symbol, MAX(date) FROM {self.table} GROUP BY symbol").fetchall()
            last_dates = {row[0]: str(row[1]) for row in last_dates_raw if row[1]}
        except sqlite3.OperationalError:
            last_dates = {}
        conn.close()
        return symbols, last_dates

    def flush_buffer(self, buffer: list):
        if not buffer: return
        combined_df = pd.concat(buffer, ignore_index=True)
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        combined_df.to_sql(self.table, conn, if_exists="append", index=False)
        conn.close()

    def render_ui(self, symbol="", force=False):
        now = time.perf_counter()
        if not force and (now - self._last_render < 0.5): return
        self._last_render = now
        self._rolling_metrics.append((now, self.stats["processed"]))
        
        while self._rolling_metrics and (now - self._rolling_metrics[0][0]) > 10.0:
            self._rolling_metrics.popleft()
        
        rate = 0.0
        if len(self._rolling_metrics) >= 2:
            dt = self._rolling_metrics[-1][0] - self._rolling_metrics[0][0]
            if dt > 0: rate = (self._rolling_metrics[-1][1] - self._rolling_metrics[0][1]) / dt

        pct = self.stats["processed"] / max(1, self.total_symbols)
        bar = "█" * int(20 * pct) + "░" * (20 - int(20 * pct))
        eta_sec = (self.total_symbols - self.stats["processed"]) / rate if rate > 0 else 0
        mem = f"{psutil.Process(os.getpid()).memory_info().rss / (1024**3):.2f} GB" if HAS_PSUTIL else "N/A"
        cpu = f"{psutil.cpu_percent():.1f}%" if HAS_PSUTIL else "N/A"

        msg = f"""\033[H\033[K============================================================
\033[K GREEN BULL AI: LAYER 2 (ANALYZER PIPELINE)
\033[K============================================================
\033[K Progress       {bar} {pct*100:.1f}%
\033[K Current Symbol {symbol}
\033[K
\033[K Completed      {self.stats['processed']} / {self.total_symbols}
\033[K JSON Written   {self.stats['rows']:,} rows
\033[K Speed          {rate:.1f} symbols/sec
\033[K Active Engines 12 Modules (Smart Money, Volatility...)
\033[K Memory         {mem} | CPU: {cpu}
\033[K Skipped        {self.stats['skipped']} (Up to date)
\033[K Failed         {self.stats['failed']}
\033[K ETA            {int(eta_sec//3600):02d}:{int((eta_sec%3600)//60):02d}:{int(eta_sec%60):02d}
\033[K============================================================"""
        sys.stdout.write(msg + "\033[J\n")
        sys.stdout.flush()

    def run(self):
        symbols, last_dates = self.get_symbols_and_dates()
        self.total_symbols = len(symbols)
        if self.total_symbols == 0: return
        buffer = []
        self.render_ui("BOOTING...", force=True)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_analyzer_worker, sym, last_dates.get(sym), self.db_path) for sym in symbols]
            for fut in as_completed(futures):
                sym, result, count = fut.result()
                self.stats["processed"] += 1
                if count > 0:
                    self.stats["success"] += 1
                    self.stats["rows"] += count
                    buffer.append(result)
                elif count == 0: self.stats["skipped"] += 1
                else: self.stats["failed"] += 1

                if len(buffer) >= self.batch_size:
                    self.flush_buffer(buffer)
                    buffer.clear()
                self.render_ui(sym)

        if buffer: self.flush_buffer(buffer)
        self.render_ui("DONE", force=True)
        sys.stdout.write("\033[?25h\n")
        print("\nLayer 2 Execution Finished.")

def run() -> None:
    AnalyzerPipeline().run()

if __name__ == "__main__":
    run()
