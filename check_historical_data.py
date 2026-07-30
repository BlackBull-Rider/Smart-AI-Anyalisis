import sqlite3
from rich.console import Console
from rich.table import Table
from backend.analyzers.analyzer_engine import analyzer_engine

console = Console()
symbol = "RELIANCE"

console.print(f"\n[bold magenta]🔍 DEEP HISTORICAL SCAN FOR MISSING COLUMNS ({symbol}) 🔍[/bold magenta]")
console.print("[dim]Checking from the beginning of time in the database...[/dim]\n")

conn = sqlite3.connect('database/market.db')
cursor = conn.cursor()

# ১. ডাটাবেসের সব টেবিল আর কলাম বের করা হচ্ছে
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

col_table_map = {}
for t in tables:
    cursor.execute(f"PRAGMA table_info({t})")
    cols = [r[1].lower() for r in cursor.fetchall()]
    for c in cols:
        if c not in col_table_map:
            col_table_map[c] = []
        col_table_map[c].append(t)

table_ui = Table(show_header=True, header_style="bold cyan")
table_ui.add_column("Analyzer Target Key")
table_ui.add_column("DB Column Searched")
table_ui.add_column("Table Found In")
table_ui.add_column("Historical Data Found?", style="bold")

# MTF বাদ দেওয়ার লিস্ট
mtf_suffixes = ['_5m', '_15m', '_1H', '_4H', '_D', '_W', '_M']
missing_count = 0

for target_key, aliases in analyzer_engine.ALIAS_MAP.items():
    # MTF কলামগুলো ইগনোর করা হচ্ছে
    if any(target_key.endswith(sfx) for sfx in mtf_suffixes):
        continue

    total_rows_found = 0
    schema_missing = True

    for alias in aliases:
        tables_with_col = col_table_map.get(alias, [])
        if tables_with_col:
            schema_missing = False
            for t in tables_with_col:
                cursor.execute(f"PRAGMA table_info({t})")
                t_cols = [r[1].lower() for r in cursor.fetchall()]
                
                # SQL কাউন্ট কমান্ড - চেক করছে কোনোদিন ডেটা ছিল কিনা
                try:
                    if 'symbol' in t_cols:
                        cursor.execute(f"SELECT COUNT({alias}) FROM {t} WHERE symbol=? AND {alias} IS NOT NULL", (symbol,))
                    else:
                        cursor.execute(f"SELECT COUNT({alias}) FROM {t} WHERE {alias} IS NOT NULL")
                    
                    count = cursor.fetchone()[0]
                    total_rows_found += count
                except Exception:
                    pass

    # শুধু সেগুলোই প্রিন্ট করব যেগুলো পুরোপুরি ফাঁকা বা স্কিমাতে নেই
    if schema_missing:
        table_ui.add_row(target_key, ", ".join(aliases), "[red]NONE[/red]", "[red]❌ NOT IN SCHEMA[/red]")
        missing_count += 1
    elif total_rows_found == 0:
        best_alias = aliases[0]
        best_table = col_table_map.get(best_alias, ["UNKNOWN"])[0]
        table_ui.add_row(target_key, best_alias, best_table, "[red]❌ EMPTY (0 Rows in History)[/red]")
        missing_count += 1

console.print(table_ui)
console.print(f"\n[bold yellow]Total Fully Empty/Missing Keys (excluding MTF): {missing_count}[/bold yellow]")
console.print("[dim]Note: If a column shows '0 Rows in History', it means your data scraper/calculator NEVER saved any data for this column.[/dim]\n")

conn.close()
