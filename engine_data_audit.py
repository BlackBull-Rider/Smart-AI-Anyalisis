import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from backend.analyzers.analyzer_engine import analyzer_engine

console = Console()
symbol = "RELIANCE"

def audit_engine_data():
    engine = analyzer_engine
    conn = engine._get_db_connection()
    
    try:
        # ১. ডাটাবেস থেকে সব ডেটা টানা
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
        
        # ২. মার্জ করা
        for ctx in ctx_dicts:
            for k, v in ctx.items():
                if k in ['symbol', 'date', 'updated_at']: continue
                if pd.notna(v): df[k] = v
                elif k not in df.columns: df[k] = np.nan
        
        # ৩. ইঞ্জিনের ALIAS_MAP ব্যবহার করে ম্যাপ করা
        for target_col, aliases in engine.ALIAS_MAP.items():
            if target_col not in df.columns or pd.isna(df[target_col].iloc[-1] if not df.empty else np.nan):
                for alias in aliases:
                    if alias in df.columns and pd.notna(df[alias].iloc[-1] if not df.empty else np.nan):
                        df[target_col] = df[alias]
                        break
            if target_col not in df.columns:
                df[target_col] = np.nan
                
        # ৪. রেজাল্ট প্রিন্ট করা
        table = Table(title=f"🚀 ENGINE DATA EXTRACTION AUDIT FOR {symbol} 🚀", show_header=True, header_style="bold cyan")
        table.add_column("Analyzer Target Key")
        table.add_column("Engine Status")
        table.add_column("Extracted Value")
        
        found_count = 0
        missing_count = 0
        
        for k in engine.ALIAS_MAP.keys():
            val = df[k].iloc[-1] if not df.empty and k in df.columns else np.nan
            if pd.isna(val):
                table.add_row(k, "[red]❌ MISSING[/red]", "NaN")
                missing_count += 1
            else:
                val_str = str(val)[:15]
                table.add_row(k, "[green]✅ SUCCESS[/green]", val_str)
                found_count += 1
                
        console.print(table)
        console.print(f"\n[bold green]✅ Total Successfully Extracted & Mapped: {found_count}[/bold green]")
        console.print(f"[bold red]❌ Total Missing (Not in DB or Empty): {missing_count}[/bold red]")
        console.print(f"[bold yellow]📊 Total Engine Keys Checked: {found_count + missing_count}[/bold yellow]\n")
        
    finally:
        conn.close()

if __name__ == "__main__":
    audit_engine_data()
