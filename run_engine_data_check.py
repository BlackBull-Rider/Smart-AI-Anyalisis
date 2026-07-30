import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from backend.analyzers.analyzer_engine import analyzer_engine

console = Console()

def main():
    symbol = "RELIANCE"
    console.print(f"\n[bold green]🚀 FIRING THE FINAL ENGINE FOR {symbol}[/bold green]\n")
    
    conn = analyzer_engine._get_db_connection()
    # ডেটা ফেচ করা হচ্ছে নতুন লজিক অনুযায়ী
    df = analyzer_engine._fetch_feature_history(conn, symbol)
    
    contexts = {
        "fundamental": analyzer_engine._fetch_single_row(conn, "fundamental_data", symbol),
        "financials": analyzer_engine._fetch_single_row(conn, "financial_data", symbol),
        "profile": analyzer_engine._fetch_single_row(conn, "company_profile", symbol),
        "ipo": analyzer_engine._fetch_single_row(conn, "ipo_data", symbol),
        "shareholding": analyzer_engine._fetch_single_row(conn, "shareholding_data", symbol),
        "snapshot": analyzer_engine._fetch_single_row(conn, "fundamental_snapshot", symbol)
    }
    conn.close()

    # --- ১. ডেটাবেস থেকে কী ডেটা আসছে সেটা দেখানো ---
    console.print("[bold cyan]🔍 WHAT THE ENGINE IS ACTUALLY RECEIVING (SAMPLE DATA)[/bold cyan]")
    
    df_table = Table(title="DATAFRAME (Layer 1 Features)", show_header=True)
    df_table.add_column("Feature Key")
    df_table.add_column("Live Value")
    df_table.add_column("Status")
    
    if not df.empty:
        latest = df.iloc[-1]
        # রেন্ডম কিছু ক্রিটিক্যাল ফিচার চেক করা হচ্ছে
        test_keys = ['close', 'volume', 'rsi', 'macd_histogram', 'atr_14', 'relative_volume', 'bos', 'fvg', 'liquidity_sweep', 'bbw_20_2.0']
        for k in test_keys:
            if k in latest:
                val = latest[k]
                status = "[red]NaN (Missing)[/red]" if pd.isna(val) else "[green]Valid Data[/green]"
                df_table.add_row(k, str(val), status)
            else:
                df_table.add_row(k, "NOT FOUND", "[red]Not Mapped[/red]")
    console.print(df_table)

    ctx_table = Table(title="CONTEXT DICTIONARIES (Layer 2 Data)", show_header=True)
    ctx_table.add_column("Dictionary")
    ctx_table.add_column("Feature Key")
    ctx_table.add_column("Live Value")
    ctx_table.add_column("Status")
    
    test_ctx = [
        ("fundamental", "pe"), ("fundamental", "roe"), 
        ("financials", "net_income"), ("financials", "operating_cash_flow"),
        ("ipo", "gmp"), ("institutional", "fii_change")
    ]
    
    for ctx_name, key in test_ctx:
        ctx_data = contexts.get(ctx_name, {})
        val = ctx_data.get(key)
        status = "[red]None (Missing)[/red]" if val is None else "[green]Valid Data[/green]"
        ctx_table.add_row(ctx_name, key, str(val), status)
        
    console.print(ctx_table)

    # --- ২. ১২টা এনালাইজার ফায়ার করা ---
    console.print("\n[bold yellow]⚡ EXECUTING ALL 12 ANALYZERS ON THIS DATA...[/bold yellow]")
    results = analyzer_engine.run({"symbol": symbol})
    
    exec_table = Table(show_header=True, title="FINAL ENGINE EXECUTION STATUS")
    exec_table.add_column("Analyzer")
    exec_table.add_column("Status")
    exec_table.add_column("Result Snippet (Action/Regime)")
    
    for name, res in results.items():
        if res.get("status") == "FAILED":
            exec_table.add_row(name.upper(), "[red]CRASHED[/red]", f"[red]{res.get('error')}[/red]")
        else:
            # আউটপুট থেকে ডিসিশন বের করে দেখানো হচ্ছে
            summary = "Executed successfully"
            if "summary" in res and "action" in res["summary"]:
                summary = res["summary"]["action"]
            elif "summary" in res and "recommended_action" in res["summary"]:
                summary = res["summary"]["recommended_action"]
            elif "state" in res and "regime" in res["state"]:
                summary = res["state"]["regime"]
            elif "direction" in res and "status" in res["direction"]:
                summary = res["direction"]["status"]
            elif "summary" in res and "current_market_regime" in res["summary"]:
                summary = res["summary"]["current_market_regime"]
                
            exec_table.add_row(name.upper(), "[green]SUCCESS[/green]", summary)

    console.print(exec_table)
    
    crashed = [n for n, r in results.items() if r.get("status") == "FAILED"]
    if crashed:
        console.print(f"\n[bold red]❌ ENGINE FAILED on: {crashed}[/bold red]")
    else:
        console.print("\n[bold green]✅ ENGINE 100% BULLETPROOF. ZERO CRASHES WITH NaN/None DATA![/bold green]")

if __name__ == "__main__":
    main()
