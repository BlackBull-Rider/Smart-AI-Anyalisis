import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ১. Fix record_latency (Adding the missing 'component' argument)
    content = content.replace(
        'metrics_engine.record_latency("pipeline.pipeline_latency", context.stats.pipeline_latency)',
        'metrics_engine.record_latency("pipeline.pipeline_latency", "pipeline", context.stats.pipeline_latency)'
    )

    # ২. Force 1 worker for SQLite stability in Termux (Using Regex to ensure it matches)
    content = re.sub(
        r'with ThreadPoolExecutor\(max_workers=[^)]+\) as tp:',
        'with ThreadPoolExecutor(max_workers=1) as tp:',
        content
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Fixed MetricsEngine arguments.")
    print("✅ Enforced single-thread DB access for Termux stability.")
except Exception as e:
    print(f"Failed to patch file: {e}")
