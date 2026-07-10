import os

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ১. ফিক্স MetricsEngine 'labels' Error
    content = content.replace('metrics_engine.increment("pipeline.success_count", labels=labels)', 'metrics_engine.increment("pipeline.success_count")')
    content = content.replace('metrics_engine.record_latency("pipeline.pipeline_latency", context.stats.pipeline_latency, labels=labels)', 'metrics_engine.record_latency("pipeline.pipeline_latency", context.stats.pipeline_latency)')
    content = content.replace('metrics_engine.increment("pipeline.failure_count", labels=labels)', 'metrics_engine.increment("pipeline.failure_count")')

    # ২. ফিক্স SQLite Concurrency 'unable to open database file' Error in Termux
    old_worker_logic = "optimal_workers = min(cpu_count * 2, max(1, len(batches)))"
    new_worker_logic = "optimal_workers = 1  # Forced to 1 for SQLite stability on Termux"
    content = content.replace(old_worker_logic, new_worker_logic)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("✅ Successfully patched Market Pipeline!")
    print("  - Removed invalid 'labels' argument from MetricsEngine.")
    print("  - Set max_workers=1 to prevent SQLite database lock in Termux.")
except Exception as e:
    print(f"Failed to patch file: {e}")
