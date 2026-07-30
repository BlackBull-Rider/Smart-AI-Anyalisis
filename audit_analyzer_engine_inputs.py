import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from backend.analyzers.analyzer_engine import analyzer_engine

console = Console()

# 🔴 ১. ডিকশনারির জন্য লাইভ ট্র্যাকার
class TrackedDict(dict):
    def __init__(self, name, *args, **kwargs):
        self._name = name
        self.accessed_keys = set()
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        self.accessed_keys.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed_keys.add(key)
        return super().get(key, default)

# 🔴 ২. Pandas DataFrame এর জন্য লাইভ ট্র্যাকার (Monkey Patching)
original_getitem = pd.DataFrame.__getitem__
original_getattr = pd.DataFrame.__getattr__
original_get = pd.DataFrame.get

ACCESSED_DF_COLS = set()

def patched_getitem(self, key):
    if isinstance(key, str): ACCESSED_DF_COLS.add(key)
    elif isinstance(key, list): ACCESSED_DF_COLS.update(key)
    return original_getitem(self, key)

def patched_getattr(self, name):
    if name in self.columns: ACCESSED_DF_COLS.add(name)
    return original_getattr(self, name)
    
def patched_get(self, key, default=None):
    if isinstance(key, str): ACCESSED_DF_COLS.add(key)
    return original_get(self, key, default)

def main():
    symbol = "RELIANCE"
    console.print(f"[bold red]🔥 TRUE DOT-TO-DOT AUDIT (LIVE MEMORY TRACKING) FOR {symbol} 🔥[/bold red]\n")
    
    conn = analyzer_engine._get_db_connection()
    raw_df = analyzer_engine._fetch_feature_history(conn, symbol)
    
    # Tracked Dictionaries
    contexts = {
        "fundamental": TrackedDict("fundamental", analyzer_engine._fetch_single_row(conn, "fundamental_data", symbol)),
        "financials": TrackedDict("financials", analyzer_engine._fetch_single_row(conn, "financial_data", symbol)),
        "profile": TrackedDict("profile", analyzer_engine._fetch_single_row(conn, "company_profile", symbol)),
        "ipo": TrackedDict("ipo", analyzer_engine._fetch_single_row(conn, "ipo_data", symbol)),
        "corporate_actions": TrackedDict("corporate_actions", analyzer_engine._fetch_single_row(conn, "corporate_actions", symbol)),
        "shareholding": TrackedDict("shareholding", analyzer_engine._fetch_single_row(conn, "shareholding_data", symbol)),
        "analyst": TrackedDict("analyst", analyzer_engine._fetch_single_row(conn, "analyst_data", symbol)),
        "earnings": TrackedDict("earnings", analyzer_engine._fetch_single_row(conn, "earnings_history", symbol)),
        "macro": TrackedDict("macro", analyzer_engine._fetch_single_row(conn, "macro_environment")),
        "snapshot": TrackedDict("snapshot", analyzer_engine._fetch_single_row(conn, "fundamental_snapshot", symbol))
    }
    conn.close()

    # Apply Patch to intercept Pandas operations
    pd.DataFrame.__getitem__ = patched_getitem
    pd.DataFrame.__getattr__ = patched_getattr
    pd.DataFrame.get = patched_get

    global_report = {}

    for name, engine_instance in analyzer_engine.engines.items():
        ACCESSED_DF_COLS.clear()
        for ctx in contexts.values(): ctx.accessed_keys.clear()
            
        try:
            # ইঞ্জিন চালানো হচ্ছে রিয়েল ডেটা দিয়ে
            analyzer_df = raw_df.copy() if not raw_df.empty else pd.DataFrame()
            
            if name == "volatility":
                engine_instance.analyze(analyzer_df)
            elif name == "fundamental":
                engine_instance.analyze(analyzer_df, fundamental=contexts["fundamental"], financials=contexts["financials"], profile=contexts["profile"], earnings=contexts["earnings"], snapshot=contexts["snapshot"], analyst=contexts["analyst"])
            elif name == "ipo":
                engine_instance.analyze(analyzer_df, ipo=contexts["ipo"], corporate_actions=contexts["corporate_actions"])
            elif name == "institutional":
                engine_instance.analyze(analyzer_df, shareholding=contexts["shareholding"], snapshot=contexts["snapshot"])
            else:
                engine_instance.analyze(analyzer_df)
        except Exception as e:
            pass # We just want to see what was accessed before any potential error

        stats = {"total": 0, "found": 0, "missing": 0}
        table = Table(title=f"{name.upper()} ANALYZER (LIVE TOUCHED DATA)", box=box.SIMPLE, show_header=True)
        table.add_column("SOURCE"); table.add_column("ACTUAL FIELD TOUCHED"); table.add_column("STATUS")

        # ১. Dataframe চেক
        for col in ACCESSED_DF_COLS:
            if col in ['iloc', 'loc', 'index', 'columns', 'shape', 'empty', 'values', 'T']: continue
            stats["total"] += 1
            if col in raw_df.columns:
                # চেক করা হচ্ছে ইঞ্জিন এটাকে প্লেসহোল্ডার (0.0) দিয়ে ফিল করেছে কিনা
                if (raw_df[col] == 0.0).all():
                    stats["missing"] += 1
                    table.add_row("df", col, "[yellow]PLACEHOLDER (0.0)[/yellow]")
                else:
                    stats["found"] += 1
                    table.add_row("df", col, "[green]REAL DATA FOUND[/green]")
            else:
                stats["missing"] += 1
                table.add_row("df", col, "[red]MISSING[/red]")
                
        # ২. Dictionary/Context চেক
        for ctx_name, ctx_obj in contexts.items():
            for key in ctx_obj.accessed_keys:
                stats["total"] += 1
                if key in ctx_obj and ctx_obj[key] != 0.0: 
                    stats["found"] += 1
                    table.add_row(ctx_name, key, "[green]REAL DATA FOUND[/green]")
                else:
                    stats["missing"] += 1
                    table.add_row(ctx_name, key, "[red]MISSING / PLACEHOLDER[/red]")

        if stats["total"] > 0:
            console.print(table)
            global_report[name] = stats

    # Revert Pandas Patch
    pd.DataFrame.__getitem__ = original_getitem
    pd.DataFrame.__getattr__ = original_getattr
    pd.DataFrame.get = original_get

    console.print(f"\n{'='*52}\n🔥 TRUE DOT-TO-DOT FINAL SUMMARY 🔥\n{'='*52}")
    sum_table = Table(box=box.SIMPLE)
    sum_table.add_column("Analyzer")
    sum_table.add_column("Total Fields Touched")
    sum_table.add_column("Real Data Found")
    sum_table.add_column("Missing / Placeholder (0.0)")
    
    for n, r in global_report.items():
        sum_table.add_row(n.upper(), str(r['total']), f"[green]{r['found']}[/green]", f"[red]{r['missing']}[/red]")
    console.print(sum_table)

if __name__ == "__main__":
    main()
