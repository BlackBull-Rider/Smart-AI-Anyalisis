import os

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ১. Replace ts_col="timestamp" with ts_col="trade_date"
    if 'ts_col="timestamp"' in content:
        content = content.replace('ts_col="timestamp"', 'ts_col="trade_date"')
        
    # ২. Replace ts = rec.get("timestamp") with ts = rec.get("trade_date")
    if 'ts = rec.get("timestamp")' in content:
        content = content.replace('ts = rec.get("timestamp")', 'ts = rec.get("trade_date")')

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Successfully patched Market Pipeline!")
    print("  - Changed incremental sync column to 'trade_date' to match universe.db")
except Exception as e:
    print(f"Failed to patch file: {e}")
