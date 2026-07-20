import time

from backend.indicators import indicator_engine
from backend.pipeline.ai_pipeline import FeatureWorker, pipeline_queues

worker = FeatureWorker(
    pipeline_queues.symbol_queue,
    pipeline_queues.feature_queue,
)

df = indicator_engine.run("RELIANCE")

last3 = df.tail(3)

print("=" * 60)

for _, row in last3.iterrows():

    start = time.perf_counter()

    feature_dict = worker._sanitize_row(row, "RELIANCE")

    payload = {
        "symbol": "RELIANCE",
        "timestamp": feature_dict["date"],
        "features": feature_dict,
    }

    pipeline_queues.feature_queue.put(payload)

    end = time.perf_counter()

    print(
        feature_dict["date"],
        "->",
        len(feature_dict),
        "features",
        f"{(end-start)*1000:.3f} ms"
    )

print("=" * 60)
