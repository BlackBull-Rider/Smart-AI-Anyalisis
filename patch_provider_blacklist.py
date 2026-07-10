import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # মেথডগুলোকে ব্ল্যাকলিস্ট করার লজিক অ্যাড করা
    old_methods = "method_names = ['fetch_market_data', 'fetch_data', 'get_historical_data', 'get_history', 'equity_history', 'history', 'get_data', 'fetch', 'download']"
    # 'download_equity_bhavcopy' কে বাদ দিয়ে মেথড লিস্ট আপডেট
    new_methods = "method_names = ['fetch_market_data', 'fetch_data', 'get_historical_data', 'get_history', 'equity_history', 'history', 'get_data', 'fetch']"
    
    content = content.replace(old_methods, new_methods)
    
    # মেথড ফিল্টার লজিক অ্যাড করা
    filter_logic = "\n                # Blacklist specific methods that are not for single-symbol fetch\n                if name in ['download_equity_bhavcopy', 'download_bhavcopy', 'upload_data']:\n                    continue"
    
    if filter_logic not in content:
        content = content.replace("for name in method_names:", "for name in method_names:" + filter_logic)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("✅ Blacklisted 'download_equity_bhavcopy' and refined Auto-Discovery.")
except Exception as e:
    print(f"Failed to patch: {e}")
