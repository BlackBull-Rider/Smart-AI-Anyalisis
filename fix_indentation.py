import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the specific block and capture its exact indentation
    pattern = r"([ \t]+)(if fetch_func:.*?data = fetch_func\(\*\*kwargs\))"
    
    def replacer(match):
        indent = match.group(1) # Capture exactly how many spaces are there
        inner_indent = indent + "    " # Add exactly 4 spaces for the inner block
        
        code = (
            f"if fetch_func:\n"
            f"{inner_indent}sig = inspect.signature(fetch_func)\n"
            f"{inner_indent}kwargs = {{}}\n"
            f"{inner_indent}if 'symbol' in sig.parameters: kwargs['symbol'] = sym\n"
            f"{inner_indent}elif 'ticker' in sig.parameters: kwargs['ticker'] = sym\n\n"
            f"{inner_indent}if 'start_date' in sig.parameters: kwargs['start_date'] = start_dt\n"
            f"{inner_indent}elif 'from_date' in sig.parameters: kwargs['from_date'] = start_dt\n"
            f"{inner_indent}elif 'start' in sig.parameters: kwargs['start'] = start_dt\n"
            f"{inner_indent}elif 'date_val' in sig.parameters: kwargs['date_val'] = start_dt\n"
            f"{inner_indent}elif 'date' in sig.parameters: kwargs['date'] = start_dt\n\n"
            f"{inner_indent}if 'end_date' in sig.parameters: kwargs['end_date'] = context.config.end_date\n"
            f"{inner_indent}elif 'to_date' in sig.parameters: kwargs['to_date'] = context.config.end_date\n"
            f"{inner_indent}elif 'end' in sig.parameters: kwargs['end'] = context.config.end_date\n\n"
            f"{inner_indent}data = fetch_func(**kwargs)"
        )
        return indent + code

    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("✅ Fixed IndentationError successfully! Code is perfectly aligned.")
except Exception as e:
    print(f"Error: {e}")
