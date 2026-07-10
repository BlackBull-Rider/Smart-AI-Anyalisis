import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ১. Method Dictionary আপডেট করা 
    old_methods = "method_names = ['fetch_market_data', 'fetch_data', 'get_historical_data', 'fetch_historical_data', 'get_data', 'fetch', 'download']"
    new_methods = "method_names = ['fetch_market_data', 'fetch_data', 'get_historical_data', 'get_history', 'equity_history', 'history', 'get_data', 'fetch', 'download']"
    content = content.replace(old_methods, new_methods)

    # ২. date_val প্যারামিটার সাপোর্ট অ্যাড করা
    old_param = "elif 'start' in sig.parameters: kwargs['start'] = start_dt"
    new_param = "elif 'start' in sig.parameters: kwargs['start'] = start_dt\n                    elif 'date_val' in sig.parameters: kwargs['date_val'] = start_dt\n                    elif 'date' in sig.parameters: kwargs['date'] = start_dt"
    
    if old_param in content:
        content = content.replace(old_param, new_param)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("✅ Expanded Smart Auto-Discovery to support 'get_history', 'equity_history' etc.")
    print("✅ Added 'date_val' to Smart Argument Mapper.")
except Exception as e:
    print(f"Failed to patch file: {e}")
