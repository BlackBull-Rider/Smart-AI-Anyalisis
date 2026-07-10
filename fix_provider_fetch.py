import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the fetch_batch call block
    pattern = r"raw_data = provider\.fetch_batch\([^)]+\)"
    
    replacement = """if hasattr(provider, 'fetch_batch'):
            raw_data = provider.fetch_batch(
                symbols=batch,
                since_map=since_map,
                end_date=context.config.end_date,
                timeframe=context.config.timeframe
            )
        else:
            self.logger.info(f"Provider {context.config.provider_name} does not support fetch_batch. Falling back to sequential fetch.")
            raw_data = []
            for sym in batch:
                start_dt = since_map.get(sym, context.config.start_date)
                try:
                    data = provider.fetch_market_data(
                        symbol=sym,
                        start_date=start_dt,
                        end_date=context.config.end_date
                    )
                    if data:
                        if isinstance(data, list):
                            raw_data.extend(data)
                        else:
                            raw_data.append(data)
                except Exception as ex:
                    self.logger.error(f"Failed to fetch {sym} from provider: {ex}")"""

    new_content = re.sub(pattern, replacement, content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ Successfully patched Market Pipeline!")
    print("  - Added Fallback Loop for providers (like NSEProvider) that only support single-symbol fetch.")
except Exception as e:
    print(f"Failed to patch file: {e}")
