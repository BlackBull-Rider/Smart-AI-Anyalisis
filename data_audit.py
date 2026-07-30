import json
import sqlite3
import pandas as pd
from backend.repository.stock_repository import repository
from backend.db.connection import db

def run_data_audit(symbol="RELIANCE"):
    print(f"\n🔍 RUNNING DATA AUDIT FOR: {symbol}")
    print("="*50)
    
    audit_report = {}

    # ১. Feature History (L1 Indicators)
    rows = db.fetchall("SELECT * FROM feature_history WHERE symbol=? ORDER BY date DESC LIMIT 1", (symbol,))
    if rows:
        cols = list(rows[0].keys())
        audit_report["1_Feature_History"] = {"status": "Available", "total_columns": len(cols), "columns": cols}
    else:
        audit_report["1_Feature_History"] = {"status": "MISSING ❌"}

    # ২. Fundamental Data
    fund = repository.get_fundamental_snapshot(symbol) or repository.get_fundamental(symbol)
    if fund:
        missing_fundamentals = [k for k, v in fund.items() if v is None or v == 0.0]
        audit_report["2_Fundamental"] = {
            "status": "Available", 
            "total_keys": len(fund),
            "missing_or_zero_keys": missing_fundamentals,
            "sample_data": fund
        }
    else:
        audit_report["2_Fundamental"] = {"status": "MISSING ❌"}

    # ৩. Financials (Income, Balance Sheet, Cashflow)
    fin = repository.get_financials(symbol)
    if fin:
        audit_report["3_Financials"] = {"status": "Available", "total_records": len(fin), "latest_record": fin[0]}
    else:
        audit_report["3_Financials"] = {"status": "MISSING ❌"}

    # ৪. Shareholding (FII, DII, Promoter)
    share = repository.get_shareholding(symbol)
    if share:
        audit_report["4_Shareholding"] = {"status": "Available", "total_quarters": len(share), "latest_quarter": share[0]}
    else:
        audit_report["4_Shareholding"] = {"status": "MISSING ❌"}

    # ৫. Earnings
    earn = repository.get_earnings(symbol)
    if earn:
        audit_report["5_Earnings"] = {"status": "Available", "total_records": len(earn), "latest_earnings": earn[0]}
    else:
        audit_report["5_Earnings"] = {"status": "MISSING ❌"}
        
    # ৬. IPO & Corporate Actions
    ipo = repository.get_ipo_data(symbol)
    corp = repository.get_corporate_actions(symbol)
    audit_report["6_IPO_Data"] = {"status": "Available" if ipo else "MISSING ❌"}
    audit_report["7_Corporate_Actions"] = {"status": "Available" if corp else "MISSING ❌", "total_records": len(corp) if corp else 0}

    # সেভ টু ফাইল
    output_file = f"{symbol}_Data_Audit.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=4, ensure_ascii=False, default=str)
        
    print(f"✅ Data Audit Complete! Open '{output_file}' to see exactly what data the analyzers are getting.")

if __name__ == "__main__":
    run_data_audit("RELIANCE")
