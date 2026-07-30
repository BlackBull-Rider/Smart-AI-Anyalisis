import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from backend.analyzers.analyzer_engine import analyzer_engine

console = Console()
engine = analyzer_engine

def run_deep_audit():
    symbol = "RELIANCE"
    conn = engine._get_db_connection()
    try:
        # ডেটা ফেচিং এবং ম্যাপ করা
        df = engine._fetch_feature_history(conn, symbol)
        # (এখানে ম্যাপ লজিকটা ইঞ্জিনের ভেতর থেকেই আসবে)
        
        # টেবিল সেটআপ
        table = Table(title=f"Historical Data Audit for {symbol}", show_header=True, header_style="bold magenta")
        table.add_column("Key")
        table.add_column("Status")
        table.add_column("Historical Data Found?")

        for k in engine.ALIAS_MAP.keys():
            if k not in df.columns:
                table.add_row(k, "[red]❌ NOT IN DB[/red]", "None")
                continue
            
            # পুরো হিস্ট্রিতে ডেটা আছে কি না?
            has_history = df[k].notna().any()
            # লেটেস্ট রো-তে আছে কি না?
            is_present_now = pd.notna(df[k].iloc[-1])

            if is_present_now:
                table.add_row(k, "[green]✅ VALID (Latest)[/green]", "Yes")
            elif has_history:
                table.add_row(k, "[yellow]⚠️ STALE (Stopped)[/yellow]", "Yes (Stopped Updating)")
            else:
                table.add_row(k, "[red]❌ EMPTY (Never Saved)[/red]", "No")
        
        console.print(table)
    finally:
        conn.close()

if __name__ == "__main__":
    run_deep_audit()
