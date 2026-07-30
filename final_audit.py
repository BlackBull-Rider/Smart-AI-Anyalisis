import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from backend.analyzers.analyzer_engine import AnalyzerEngine

console = Console()
engine = AnalyzerEngine()

def run_audit():
    symbol = "RELIANCE"
    conn = engine._get_db_connection()
    try:
        # ইঞ্জিনের ঠিক সেইম লজিক দিয়ে ডেটা টানা এবং মার্জ করা
        df = engine._fetch_feature_history(conn, symbol)
        ctx_dicts = [
            engine._fetch_single_row(conn, "fundamental_data", symbol),
            engine._fetch_single_row(conn, "financial_data", symbol),
            engine._fetch_single_row(conn, "company_profile", symbol),
            engine._fetch_single_row(conn, "ipo_data", symbol),
            engine._fetch_single_row(conn, "corporate_actions", symbol),
            engine._fetch_single_row(conn, "shareholding_data", symbol),
            engine._fetch_single_row(conn, "analyst_data", symbol),
            engine._fetch_single_row(conn, "earnings_history", symbol),
            engine._fetch_single_row(conn, "fundamental_snapshot", symbol),
            engine._fetch_single_row(conn, "macro_environment")
        ]
        
        for ctx in ctx_dicts:
            for k, v in ctx.items():
                if k in ['symbol', 'date', 'updated_at']: continue
                if pd.notna(v): df[k] = v
                elif k not in df.columns: df[k] = np.nan

        # 🔴 এবার ইঞ্জিন যেভাবে ম্যাপ করে, ঠিক সেইভাবে এখানেও ম্যাপ করছি
        for target_col, aliases in engine.ALIAS_MAP.items():
            if target_col not in df.columns or df[target_col].isna().all():
                for alias in aliases:
                    if alias in df.columns and not df[alias].isna().all():
                        df[target_col] = df[alias]
                        break
            if target_col not in df.columns:
                df[target_col] = np.nan
    finally:
        conn.close()

    # অডিট টেবিল
    table = Table(title=f"Final Audit for {symbol}", show_header=True, header_style="bold magenta")
    table.add_column("Key")
    table.add_column("Status")
    table.add_column("Value (Sample)")

    for k in engine.ALIAS_MAP.keys():
        val = df[k].iloc[-1] if not df.empty and k in df.columns else np.nan
        if pd.isna(val):
            table.add_row(k, "[red]❌ MISSING[/red]", "NaN")
        else:
            table.add_row(k, "[green]✅ VALID[/green]", str(val)[:15])
    
    console.print(table)

if __name__ == "__main__":
    run_audit()
