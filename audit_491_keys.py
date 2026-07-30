import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from backend.analyzers.analyzer_engine import analyzer_engine

console = Console()

def main():
    symbol = "RELIANCE"
    console.print(f"\n[bold red]🔥 ULTIMATE 491+ DOT-TO-DOT LIVE DATA AUDIT FOR {symbol} 🔥[/bold red]\n")
    
    conn = analyzer_engine._get_db_connection()
    try:
        # ১. ডেটাবেস থেকে হিস্ট্রি টানা হচ্ছে (কোনো ALIAS ছাড়া)
        df = analyzer_engine._fetch_feature_history(conn, symbol)
        
        ctx_dicts = [
            analyzer_engine._fetch_single_row(conn, "fundamental_data", symbol),
            analyzer_engine._fetch_single_row(conn, "financial_data", symbol),
            analyzer_engine._fetch_single_row(conn, "company_profile", symbol),
            analyzer_engine._fetch_single_row(conn, "ipo_data", symbol),
            analyzer_engine._fetch_single_row(conn, "corporate_actions", symbol),
            analyzer_engine._fetch_single_row(conn, "shareholding_data", symbol),
            analyzer_engine._fetch_single_row(conn, "analyst_data", symbol),
            analyzer_engine._fetch_single_row(conn, "earnings_history", symbol),
            analyzer_engine._fetch_single_row(conn, "fundamental_snapshot", symbol),
            analyzer_engine._fetch_single_row(conn, "macro_environment")
        ]
        
        # ২. CAREFUL MERGE: ফান্ডামেন্টালের NaN দিয়ে টেকনিক্যালের আসল ডেটা (যেমন vwap) মোছা যাবে না
        for ctx in ctx_dicts:
            for k, v in ctx.items():
                if k in ['symbol', 'date', 'updated_at']:
                    continue
                if pd.notna(v):
                    df[k] = v
                elif k not in df.columns:
                    df[k] = np.nan

        # ৩. LATE MAPPING: এবার সব মার্জ হওয়ার পর ALIAS_MAP রান করবে (যেটা ইঞ্জিনে হয়)
        for target_col, aliases in analyzer_engine.ALIAS_MAP.items():
            if target_col not in df.columns or pd.isna(df[target_col].iloc[-1] if not df.empty else np.nan):
                for alias in aliases:
                    if alias in df.columns and pd.notna(df[alias].iloc[-1] if not df.empty else np.nan):
                        df[target_col] = df[alias]
                        break
            if target_col not in df.columns:
                df[target_col] = np.nan

    finally:
        conn.close()

    if df.empty:
        console.print("[bold red]❌ DataFrame is completely empty! No data found in DB.[/bold red]")
        return

    latest_data = df.iloc[-1]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("No.", style="dim", width=4)
    table.add_column("Analyzer Target Key", style="cyan")
    table.add_column("DB Columns Searched", style="yellow")
    table.add_column("Live Value Received", style="bold white")
    table.add_column("Status", style="bold")

    missing_count = 0
    valid_count = 0
    
    for idx, (target_key, aliases) in enumerate(analyzer_engine.ALIAS_MAP.items(), 1):
        val = latest_data.get(target_key, np.nan)
        
        if pd.isna(val):
            status = "[red]❌ NaN (Missing)[/red]"
            val_str = "NaN"
            missing_count += 1
        else:
            status = "[green]✅ Valid Data[/green]"
            val_str = str(val)
            valid_count += 1
            
        table.add_row(str(idx), target_key, ", ".join(aliases), val_str, status)

    console.print(table)
    
    console.print(f"\n[bold blue]📊 AUDIT SUMMARY FOR {symbol}:[/bold blue]")
    console.print(f"Total Keys Checked : {len(analyzer_engine.ALIAS_MAP)}")
    console.print(f"Valid Data Found   : [green]{valid_count}[/green]")
    console.print(f"Missing Data (NaN) : [red]{missing_count}[/red]")
    
    if missing_count > 0:
        console.print("\n[bold yellow]⚠️ WARNING: The missing (NaN) keys above are exactly why Analyzers crash. Fix these specific columns in your database![/bold yellow]")

if __name__ == "__main__":
    main()
