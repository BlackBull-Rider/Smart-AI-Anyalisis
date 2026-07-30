import ast
import pandas as pd
import numpy as np
import inspect
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

# তোর প্রজেক্টের মডিউল ইমপোর্ট
from backend.db.connection import db
from backend.config.settings import settings

# ==============================================================================
# AUDIT CONFIGURATION
# ==============================================================================
console = Console()
ROOT_DIR = settings.project_root
ANALYZER_DIR = ROOT_DIR / "backend" / "analyzers"
ENGINE_FILE = ANALYZER_DIR / "analyzer_engine.py"

# ==============================================================================
# AST PARSER (Static Analysis)
# ==============================================================================
class AnalyzerVisitor(ast.NodeVisitor):
    def __init__(self):
        self.requirements = {
            "df": set(), "fundamental": set(), "financials": set(),
            "profile": set(), "ipo": set(), "shareholding": set(),
            "corporate_actions": set(), "earnings": set(), "macro": set()
        }
        self.current_var = None

    def visit_Name(self, node):
        if node.id in self.requirements:
            self.current_var = node.id
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if self.current_var and isinstance(node.slice, ast.Constant):
            self.requirements[self.current_var].add(str(node.slice.value).lower())
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self.current_var == "df" and isinstance(node.value, ast.Name) and node.value.id == "df":
            self.requirements["df"].add(str(node.attr).lower())
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
            if isinstance(node.func.value, ast.Name):
                var = node.func.value.id
                if var in self.requirements and node.args and isinstance(node.args[0], ast.Constant):
                    self.requirements[var].add(str(node.args[0].value).lower())
        self.generic_visit(node)

def get_analyzer_requirements(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    visitor = AnalyzerVisitor()
    visitor.visit(tree)
    return {k: v for k, v in visitor.requirements.items() if v}

# ==============================================================================
# ALIAS MAP EXTRACTION (Using provided Engine)
# ==============================================================================
def extract_alias_map():
    if not ENGINE_FILE.exists(): return {}
    with open(ENGINE_FILE, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == 'ALIAS_MAP':
                    return ast.literal_eval(node.value)
    return {}

# ==============================================================================
# DATA LOADER (Using provided Connection Layer)
# ==============================================================================
def load_data():
    data = {}
    tables = {
        "historical_data": "df", "fundamental_data": "fundamental",
        "financial_data": "financials", "company_profile": "profile",
        "ipo_data": "ipo", "shareholding_data": "shareholding",
        "corporate_actions": "corporate_actions", "earnings_history": "earnings",
        "macro_environment": "macro"
    }
    
    for table, key in tables.items():
        try:
            # db.fetchone ব্যবহার করা হলো
            row = db.fetchone(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1")
            if row:
                row_dict = dict(row)
                if key == "df":
                    # df-এর জন্য সব ডাটা ফেচ করা
                    rows = db.fetchall(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 200")
                    data[key] = pd.DataFrame([dict(r) for r in rows])
                else:
                    data[key] = row_dict
            else:
                data[key] = {}
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fetch {table}: {e}[/yellow]")
            data[key] = {}
    return data

# ==============================================================================
# AUDIT ENGINE
# ==============================================================================
def main():
    console.print(f"[bold blue]Using DB: {settings.database_path}[/bold blue]")
    
    data = load_data()
    alias_map = extract_alias_map()
    global_report = {}
    
    analyzer_files = list(ANALYZER_DIR.glob("*_analyzer.py"))
    
    for f_path in analyzer_files:
        reqs = get_analyzer_requirements(f_path)
        analyzer_name = f_path.stem.replace("_analyzer", "").upper()
        
        console.print(f"\n{'='*52}\n{analyzer_name} ANALYZER\n{'='*52}")
        
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("FIELD")
        table.add_column("VALUE")
        table.add_column("STATUS")
        
        stats = {"total": 0, "present": 0, "missing": 0, "null": 0, "nan": 0}
        analyzer_results = []
        
        for source, fields in reqs.items():
            for field in fields:
                stats["total"] += 1
                val = None
                status = "MISSING"
                actual_key = field
                
                # Alias resolve
                if field in alias_map:
                    for alias in alias_map[field]:
                        if alias in data.get(source, {}) or (source == 'df' and isinstance(data.get('df'), pd.DataFrame) and alias in data['df'].columns):
                            actual_key = alias
                            break
                
                # Retrieve
                if source == 'df' and isinstance(data.get('df'), pd.DataFrame):
                    if actual_key in data['df'].columns:
                        val = data['df'][actual_key].iloc[-1]
                        status = "OK"
                else:
                    if actual_key in data.get(source, {}):
                        val = data[source][actual_key]
                        status = "OK"
                
                # Audit
                if status == "OK":
                    if pd.isna(val) and not isinstance(val, (dict, list)): status = "NaN"
                    elif val is None: status = "NULL"
                    
                if status == "OK": stats["present"] += 1
                elif status == "MISSING": stats["missing"] += 1
                elif status == "NaN": stats["nan"] += 1
                elif status == "NULL": stats["null"] += 1
                
                table.add_row(field, str(val)[:15], status)
                analyzer_results.append({"field": field, "value": str(val), "status": status})
                
        console.print(table)
        global_report[analyzer_name] = {"stats": stats, "details": analyzer_results}

    # Summary
    console.print(f"\n{'='*52}\nGLOBAL SUMMARY\n{'='*52}")
    sum_table = Table(box=box.SIMPLE)
    sum_table.add_column("Analyzer")
    sum_table.add_column("Req")
    sum_table.add_column("Pres")
    sum_table.add_column("Miss")
    for n, r in global_report.items():
        sum_table.add_row(n, str(r['stats']['total']), str(r['stats']['present']), str(r['stats']['missing']))
    console.print(sum_table)

    with open("audit_report.json", "w") as f:
        json.dump(global_report, f, indent=4)
    
    console.print("\n✅ Audit Complete using project settings & connection.")

if __name__ == "__main__":
    main()
