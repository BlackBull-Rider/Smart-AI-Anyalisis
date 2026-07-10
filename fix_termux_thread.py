import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to find the ThreadPoolExecutor block and replace it with a standard sequential loop
    pattern = r"with ThreadPoolExecutor\(.*?self\.error_handler\.handle\(e, context\)"
    
    replacement = """for batch in batches:
            try:
                self._execute_batch_with_retry(context, batch)
            except Exception as e:
                self.logger.error(f"Batch coordination failed: {e}")
                errors.append(e)
                self.error_handler.handle(e, context)"""

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ Successfully removed ThreadPoolExecutor.")
    print("✅ Enforced sequential Main-Thread execution for Termux compatibility.")
except Exception as e:
    print(f"Failed to patch file: {e}")
