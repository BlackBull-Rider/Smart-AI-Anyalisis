import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # We will replace the PipelineFactory class with one that injects absolute paths
    factory_pattern = r"class PipelineFactory:.*"
    
    new_factory = """class PipelineFactory:
    @classmethod
    def create_production_pipeline(cls) -> PipelineRunner:
        import os
        # Force Absolute Path for Termux Compatibility
        db_abs = os.path.abspath("backend/database/universe.db")
        db_url = f"sqlite:///{db_abs}"
        
        try:
            from backend.data.providers.nse import NSEConfig
            nse_cfg = NSEConfig(db_path=db_abs)
        except Exception:
            nse_cfg = None

        try:
            from backend.data.providers.yfinance import YahooFinanceConfig
            yf_cfg = YahooFinanceConfig()
        except Exception:
            yf_cfg = None

        try:
            from backend.data.market_reader import ReaderConfig
            reader_cfg = ReaderConfig(db_url=db_url)
        except Exception:
            reader_cfg = None

        try:
            from backend.data.market_writer import WriterConfig
            writer_cfg = WriterConfig(db_url=db_url)
        except Exception:
            writer_cfg = None

        ProviderRegistry.register("nse", NSEProvider(nse_cfg) if nse_cfg else NSEProvider())
        ProviderRegistry.register("yfinance", YahooFinanceProvider(yf_cfg) if yf_cfg else YahooFinanceProvider())
        
        event_bus = EventBus()
        scheduler = Scheduler()
        
        reader = MarketReader(reader_cfg) if reader_cfg else MarketReader()
        writer = MarketWriter(writer_cfg) if writer_cfg else MarketWriter()
        validator = MarketValidator()
        uow = UnitOfWork()
        compression_service = CompressionService()
        
        health_checker = PipelineHealth(db_session_cls=DatabaseSession)
        bootstrap = PipelineBootstrap(health_checker=health_checker)
        
        monitor = PipelineMonitor(event_bus=event_bus)
        hooks = PipelineHooks(monitor=monitor)
        lifecycle = PipelineLifecycle(hooks=hooks)
        
        error_handler = PipelineErrorHandler(event_bus=event_bus)
        retry_policy = PipelineRetryPolicy(monitor=monitor)
        
        executor = PipelineExecutor(
            reader=reader,
            validator=validator,
            writer=writer,
            uow=uow,
            monitor=monitor,
            event_bus=event_bus,
            compression_service=compression_service
        )
        
        coordinator = PipelineCoordinator(
            executor=executor,
            retry_policy=retry_policy,
            error_handler=error_handler
        )
        
        return PipelineRunner(
            coordinator=coordinator,
            lifecycle=lifecycle,
            bootstrap=bootstrap,
            scheduler=scheduler,
            event_bus=event_bus
        )

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import datetime
    pipeline = PipelineFactory.create_production_pipeline()

    config = PipelineConfig(
        start_date="2020-01-01",
        end_date=datetime.date.today().isoformat(),
        timeframe="1D",
        exchange="NSE",
        symbols=None,
    )

    result = pipeline.run(config)

    print("=" * 60)
    print("GREEN BULL RIDER V6 - MARKET PIPELINE")
    print("=" * 60)
    print(f"Success : {result.success}")

    if result.summary:
        print(f"Processed : {result.summary.total_processed}")
        print(f"Successful : {result.summary.successful}")
        print(f"Failed : {result.summary.failed}")
        print(f"Execution Time : {result.summary.execution_time:.2f}s")
"""

    new_content = re.sub(factory_pattern, new_factory, content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ Successfully injected Absolute Paths into PipelineFactory.")
except Exception as e:
    print(f"Failed to patch file: {e}")
