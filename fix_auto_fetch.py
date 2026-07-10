import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find our previous fetch fallback
    pattern = r"data = provider\.fetch_market_data\(\s*symbol=sym,\s*start_date=start_dt,\s*end_date=context\.config\.end_date\s*\)"
    
    new_block = """import inspect
                method_names = ['fetch_data', 'get_historical_data', 'fetch_historical_data', 'get_data', 'fetch', 'download']
                fetch_func = None
                for name in method_names:
                    if hasattr(provider, name):
                        fetch_func = getattr(provider, name)
                        break
                
                # যদি উপরের নামগুলো না মেলে, তবে প্রথম পাবলিক মেথডটা নেবে
                if not fetch_func:
                    public_methods = [m for m in dir(provider) if callable(getattr(provider, m)) and not m.startswith('_') and m not in ['ping', 'health_check', 'supports_health_check']]
                    fetch_func = getattr(provider, public_methods[0]) if public_methods else None
                
                if fetch_func:
                    sig = inspect.signature(fetch_func)
                    kwargs = {}
                    
                    # Smart Argument Mapping
                    if 'symbol' in sig.parameters: kwargs['symbol'] = sym
                    elif 'ticker' in sig.parameters: kwargs['ticker'] = sym
                    
                    if 'start_date' in sig.parameters: kwargs['start_date'] = start_dt
                    elif 'from_date' in sig.parameters: kwargs['from_date'] = start_dt
                    elif 'start' in sig.parameters: kwargs['start'] = start_dt
                    
                    if 'end_date' in sig.parameters: kwargs['end_date'] = context.config.end_date
                    elif 'to_date' in sig.parameters: kwargs['to_date'] = context.config.end_date
                    elif 'end' in sig.parameters: kwargs['end'] = context.config.end_date
                    
                    data = fetch_func(**kwargs)
                else:
                    data = None
                    self.logger.error(f"No valid fetch method found in {provider.__class__.__name__}")"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, new_block, content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Added Smart Auto-Discovery for Provider Fetch Method!")
    else:
        print("⚠️ Pattern not found.")
except Exception as e:
    print(f"Error: {e}")
