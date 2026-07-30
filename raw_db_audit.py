import sqlite3
from rich.console import Console
from rich.table import Table

console = Console()
db_path = "database/market.db"
symbol = "RELIANCE"

def run_raw_audit():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # সব টেবিল বের করছি
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    table_ui = Table(title=f"RAW DATA AUDIT FOR {symbol} (Ignoring Any Mapping)", show_header=True)
    table_ui.add_column("Table")
    table_ui.add_column("Column Name")
    table_ui.add_column("Status")
    table_ui.add_column("Total Non-Null Rows")

    for t in tables:
        # টেবিলের কলাম লিস্ট নিচ্ছি
        cursor.execute(f"PRAGMA table_info({t})")
        cols = [r[1].lower() for r in cursor.fetchall()]
        
        for col in cols:
            if col in ['symbol', 'date', 'updated_at', 'fiscal_year', 'fiscal_quarter']: continue
            
            # সরাসরি SQL কুয়েরি: কোনো ম্যাপ নেই, শুধু দেখাচ্ছে ডেটা আছে কি না
            try:
                cursor.execute(f"SELECT COUNT({col}) FROM {t} WHERE symbol = ? AND {col} IS NOT NULL", (symbol,))
                count = cursor.fetchone()[0]
                
                if count > 0:
                    table_ui.add_row(t, col, "[green]FOUND[/green]", str(count))
                else:
                    # যদি ডেটা না পায়, শুধু তখনই লাল রঙে দেখাব
                    table_ui.add_row(t, col, "[red]EMPTY[/red]", "0")
            except:
                continue
                
    console.print(table_ui)
    conn.close()

if __name__ == "__main__":
    run_raw_audit()
