"""
GREEN BULL RIDER V6
Module: backend/pipeline/ai_pipeline.py

Enterprise AI Pipeline - Layer 1 (Feature Engineering)
Python 3.13 Compatible
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.config.settings import settings
from backend.repository.stock_repository import repository
from backend.indicators import indicator_engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturePayload:
    """
    Strict contract for Feature Data passed from Layer-1 to Layer-2 (Analyzer).
    Analyzers MUST accept this exact structure to prevent mismatches.
    """
    symbol: str
    timestamp: str
    df: pd.DataFrame


class AIPipelineQueues:
    """
    Shared centralized message broker for the entire AI Pipeline (Layers 1-5).
    Single source of truth for inter-layer data transfer.
    """
    def __init__(self) -> None:
        maxsize = settings.performance.batch_size
        self.symbol_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=maxsize)
        self.feature_queue: queue.Queue[Optional[FeaturePayload]] = queue.Queue(maxsize=maxsize)
        self.analyzer_queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.score_queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.decision_queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.ai_queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self.db_write_queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)


pipeline_queues = AIPipelineQueues()


def enqueue_for_ai(symbol: str) -> bool:
    """
    Public API Entrypoint for integrating directly with MarketPipeline.
    """
    try:
        pipeline_queues.symbol_queue.put(symbol, timeout=5.0)
        logger.info("Successfully enqueued symbol %s for AI Pipeline processing.", symbol)
        return True
    except queue.Full:
        logger.error("Symbol queue is full. Dropped symbol %s.", symbol)
        return False


class FeatureWorker(threading.Thread):
    """
    Layer-1 Worker: Polls symbols, computes 571 indicators, sanitizes the latest row,
    wraps into FeaturePayload (using candle timestamp), and pushes to the feature queue.
    """

    def __init__(
        self,
        input_queue: queue.Queue[Optional[str]],
        output_queue: queue.Queue[Optional[FeaturePayload]],
        daemon: bool = True
    ) -> None:
        super().__init__(daemon=daemon)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.max_retries = settings.retry.retries
        self.base_delay = settings.retry.base_delay
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.logger.info("FeatureWorker (Layer-1) started.")
        
        while not self._stop_event.is_set():
            try:
                symbol = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if symbol is None:
                self.logger.info("Sentinel received. Stopping FeatureWorker.")
                self.input_queue.task_done()
                break

            try:
                self._execute_with_retry(symbol)
            except Exception as e:
                self.logger.error(
                    "Failed to process symbol %s after exhaustive retries: %s", 
                    symbol, e, exc_info=True
                )
            finally:
                self.input_queue.task_done()

        self.logger.info("FeatureWorker (Layer-1) stopped.")

    def _execute_with_retry(self, symbol: str) -> None:
        attempts = 0
        while attempts <= self.max_retries and not self._stop_event.is_set():
            try:
                success = self._process_symbol(symbol)
                if success:
                    return
                break
            except Exception as e:
                attempts += 1
                if attempts > self.max_retries:
                    raise e
                delay = self.base_delay * (2 ** (attempts - 1))
                self.logger.warning(
                    "Attempt %d failed for %s: %s. Retrying in %.2fs...", 
                    attempts, symbol, e, delay
                )
                time.sleep(delay)

    def _process_symbol(self, symbol: str) -> bool:
        self.logger.debug("Processing symbol: %s", symbol)

        start = time.perf_counter()
        df: pd.DataFrame = indicator_engine.run(symbol)
        self.logger.info("indicator_engine.run(%s) took %.2f sec", symbol, time.perf_counter() - start)

        if df is None or df.empty:
            self.logger.warning("Indicator engine returned empty DataFrame for %s", symbol)
            return False

        latest_date = df.index[-1] if not df.index.empty else pd.Timestamp.now()
        timestamp_str = latest_date.isoformat() if isinstance(latest_date, pd.Timestamp) else str(latest_date)
        
        payload = FeaturePayload(
            symbol=symbol,
            timestamp=timestamp_str,
            df=df
        )

        try:
            self.output_queue.put(payload, timeout=5.0)
            self.logger.info("Successfully pushed FeaturePayload for %s to output queue.", symbol)
            return True
        except queue.Full:
            self.logger.error("Feature queue full. Dropped payload for %s.", symbol)
            return False

    def _sanitize_row(self, row: pd.Series, symbol: str) -> Dict[str, Any]:
        row_dict = row.to_dict()
        clean_dict: Dict[str, Any] = {"symbol": symbol}
        
        for key, value in row_dict.items():
            if pd.isna(value):
                clean_dict[key] = None
            elif isinstance(value, pd.Timestamp):
                clean_dict[key] = value.isoformat()
            elif hasattr(value, "item"):
                clean_dict[key] = value.item()
            else:
                clean_dict[key] = value
                
        return clean_dict


class FeatureLayerManager:
    """
    Lifecycle manager for Layer-1 workers. Handles thread pooling, 
    graceful shutdown via sentinels, and join operations using shared queues.
    """

    def __init__(self) -> None:
        self.input_queue = pipeline_queues.symbol_queue
        self.output_queue = pipeline_queues.feature_queue
        self.workers: List[FeatureWorker] = []
        self.num_workers = 4
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self) -> None:
        self.logger.info("Starting FeatureLayerManager with %d workers.", self.num_workers)
        for _ in range(self.num_workers):
            worker = FeatureWorker(
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                daemon=True
            )
            self.workers.append(worker)
            worker.start()

    def stop(self) -> None:
        self.logger.info("Stopping FeatureLayerManager. Injecting sentinels...")
        for worker in self.workers:
            worker.stop()
        
        for _ in self.workers:
            try:
                self.input_queue.put(None, timeout=2.0)
            except queue.Full:
                self.logger.warning("Queue full while injecting sentinel.")

    def join(self) -> None:
        self.logger.info("Waiting for FeatureWorker threads to join...")
        for worker in self.workers:
            worker.join()
        self.logger.info("All FeatureWorker threads joined successfully.")



class DedicatedDBWriter(threading.Thread):
    """
    Single Dedicated Writer Thread. 
    Listens to db_write_queue and writes MISSING DATES to the database 
    using StockRepository without blocking the AI Workers.
    """
    def __init__(self, daemon: bool = True) -> None:
        super().__init__(daemon=daemon)
        self.input_queue = pipeline_queues.db_write_queue
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.logger.info("DedicatedDBWriter started.")
        
        while not self._stop_event.is_set():
            try:
                task = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                self.logger.info("Sentinel received. Stopping DB Writer.")
                self.input_queue.task_done()
                break

            task_type, payload = task
            try:
                if task_type == "FEATURE":
                    inserted = repository.save_features(payload.symbol, payload.df)
                    if inserted > 0:
                        self.logger.info("Writer saved %d missing rows for %s to feature_history.", inserted, payload.symbol)
            except Exception as e:
                self.logger.error("Writer failed to save DB data for %s: %s", payload.symbol, e, exc_info=True)
            finally:
                self.input_queue.task_done()
                
        self.logger.info("DedicatedDBWriter stopped.")
