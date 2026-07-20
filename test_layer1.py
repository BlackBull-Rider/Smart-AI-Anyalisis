import logging
import queue
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from backend.pipeline.ai_pipeline import (
    FeatureLayerManager,
    enqueue_for_ai,
    pipeline_queues,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("TestLayer1")


def run_test():
    logger.info("=== STARTING MULTI SYMBOL TEST ===")

    manager = FeatureLayerManager()
    manager.start()

    symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "LT",
        "ITC",
        "HINDUNILVR",
        "BAJFINANCE",
    ]

    for s in symbols:
        logger.info(f"Injecting: {s}")
        enqueue_for_ai(s)

    received = 0

    try:
        while received < len(symbols):
            payload = pipeline_queues.feature_queue.get(timeout=600)

            logger.info(
                f"{payload.symbol} -> {len(payload.features)} features"
            )

            pipeline_queues.feature_queue.task_done()
            received += 1

        logger.info("✅ ALL SYMBOLS PROCESSED SUCCESSFULLY")

    except queue.Empty:
        logger.error("❌ Timeout waiting for payload.")

    finally:
        logger.info("Initiating graceful shutdown...")
        manager.stop()
        manager.join()
        logger.info("=== TEST COMPLETE ===")


if __name__ == "__main__":
    run_test()
