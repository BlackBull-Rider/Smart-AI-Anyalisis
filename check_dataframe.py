import pandas as pd
from backend.providers.yahoo_provider import YahooProvider
from datetime import datetime

provider = YahooProvider()

# এবার ফিক্সড ডেট দিয়ে চেক করি
df = provider.get_history("RELIANCE", start_date="2026-07-10", end_date="2026-07-20")

print(f"\n--- 🧪 FORCED DATE CHECK (2026) ---")
if df is not None and not df.empty:
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 2 rows (Testing if we get 2026 data):\n{df.head(2)}")
else:
    print("❌ Still no data for 2026. Yahoo might have issues or date calculation is flawed.")
