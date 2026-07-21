#!/usr/bin/env python3
"""
GREEN BULL RIDER V6
Historical Sync Runner
"""

from backend.pipeline.market_pipeline import MarketPipeline
from backend.repository.stock_repository import repository


def progress(done: int, total: int, symbol: str) -> None:
    print(f"\rHistory : {done}/{total} [{symbol}]", end="", flush=True)


def main() -> None:
    symbols = [
        row["symbol"]
        for row in repository.get_active_symbols()
    ]

    print(f"Starting Historical Sync ({len(symbols)} symbols)")

    with MarketPipeline() as pipeline:
        report = pipeline.run_history(
            symbols=symbols,
            is_incremental=False,
            progress_callback=progress,
        )

    print("\n")
    print(report)


if __name__ == "__main__":
    main()
