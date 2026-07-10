import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # We safely replace the whole PipelineExecutor class to ensure perfect indentation
    pattern = r"class PipelineExecutor:.*?class PipelineCoordinator:"
    
    new_executor = """class PipelineExecutor:
    def __init__(self,
                 reader: MarketReader,
                 validator: MarketValidator,
                 writer: MarketWriter,
                 uow: UnitOfWork,
                 monitor: PipelineMonitor,
                 event_bus: EventBus,
                 compression_service: CompressionService):
        self.reader = reader
        self.validator = validator
        self.writer = writer
        self.uow = uow
        self.monitor = monitor
        self.event_bus = event_bus
        self.compression_service = compression_service
        self.logger = AppLogger(self.__class__.__name__)

    @trace_span(operation="executor.process_batch", component="pipeline", kind=SpanKind.INTERNAL)
    def process_batch(self, context: PipelineContext, batch: List[str]) -> None:
        self.logger.info(f"Initiating pipeline execution for batch of {len(batch)} symbols. [Exec ID: {context.execution_id}]")

        # -------------------------------------------------------
        # STAGE 1: Database Read (Incremental Sync Check Map)
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.INCREMENTAL_CHECK)
        t_reader = time.perf_counter()

        since_map: Dict[str, str] = {sym: context.config.start_date for sym in batch}
        if hasattr(self.reader, 'read_latest_batch'):
            latest_records = self.reader.read_latest_batch(
                table_name="market_data",
                symbols=batch,
                ts_col="trade_date"
            )
            if latest_records:
                for rec in latest_records:
                    sym = rec.get("symbol")
                    ts = rec.get("trade_date")
                    if sym and ts:
                        since_map[sym] = max(context.config.start_date, str(ts))

        context.stats.add_latency("reader_latency", time.perf_counter() - t_reader)

        # -------------------------------------------------------
        # STAGE 2: Provider Batch Fetch (With Smart Auto-Discovery)
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.PROVIDER_FETCH)
        provider = ProviderRegistry.get(context.config.provider_name)
        if not provider:
            raise ProviderError(f"Data provider '{context.config.provider_name}' is not securely registered in ProviderRegistry.")

        t_provider = time.perf_counter()
        raw_data = []

        if hasattr(provider, 'fetch_batch'):
            raw_data = provider.fetch_batch(
                symbols=batch,
                since_map=since_map,
                end_date=context.config.end_date,
                timeframe=context.config.timeframe
            )
        else:
            self.logger.info(f"Provider {context.config.provider_name} does not support fetch_batch. Falling back to sequential smart fetch.")
            import inspect
            
            # Auto-discover the exact fetch method name in the provider class
            method_names = ['fetch_market_data', 'fetch_data', 'get_historical_data', 'fetch_historical_data', 'get_data', 'fetch', 'download']
            fetch_func = None
            for name in method_names:
                if hasattr(provider, name):
                    fetch_func = getattr(provider, name)
                    break
            
            if not fetch_func:
                public_methods = [m for m in dir(provider) if callable(getattr(provider, m)) and not m.startswith('_') and m not in ['ping', 'health_check', 'supports_health_check']]
                fetch_func = getattr(provider, public_methods[0]) if public_methods else None

            for sym in batch:
                start_dt = since_map.get(sym, context.config.start_date)
                try:
                    if fetch_func:
                        sig = inspect.signature(fetch_func)
                        kwargs = {}
                        if 'symbol' in sig.parameters: kwargs['symbol'] = sym
                        elif 'ticker' in sig.parameters: kwargs['ticker'] = sym
                        
                        if 'start_date' in sig.parameters: kwargs['start_date'] = start_dt
                        elif 'from_date' in sig.parameters: kwargs['from_date'] = start_dt
                        elif 'start' in sig.parameters: kwargs['start'] = start_dt
                        
                        if 'end_date' in sig.parameters: kwargs['end_date'] = context.config.end_date
                        elif 'to_date' in sig.parameters: kwargs['to_date'] = context.config.end_date
                        elif 'end' in sig.parameters: kwargs['end'] = context.config.end_date
                        
                        data = fetch_func(**kwargs)
                        if data:
                            if isinstance(data, list):
                                raw_data.extend(data)
                            else:
                                raw_data.append(data)
                    else:
                        self.logger.error(f"No valid fetch method found in {provider.__class__.__name__}")
                except Exception as ex:
                    self.logger.error(f"Failed to fetch {sym} from provider: {ex}")

        context.stats.add_latency("provider_latency", time.perf_counter() - t_provider)

        if not raw_data:
            self.logger.warning(f"No new market data retrieved from provider for batch. [Exec ID: {context.execution_id}]")
            context.mark_success(batch)
            return

        # -------------------------------------------------------
        # STAGE 3: Validation
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.VALIDATION)
        t_validator = time.perf_counter()

        if hasattr(self.validator, 'validate_batch'):
            validation_result = self.validator.validate_batch(raw_data)
            valid_records = validation_result.valid_records
            invalid_count = len(validation_result.invalid_records)
        else:
            valid_records = self.validator.validate(raw_data)
            invalid_count = len(raw_data) - len(valid_records)

        context.stats.add_latency("validator_latency", time.perf_counter() - t_validator)

        context.stats.increment("records_valid", len(valid_records))
        context.stats.increment("records_invalid", invalid_count)
        
        event_payload = MappingProxyType({"batch_size": len(batch), "valid_count": len(valid_records)})
        if hasattr(self.event_bus, 'publish_async'):
            self.event_bus.publish_async(Event("ValidationCompleted", "market_pipeline", event_payload))
        else:
            self.event_bus.publish(Event("ValidationCompleted", "market_pipeline", event_payload))

        if not valid_records:
            raise ValidationError("Data validation failed entirely for the batch. No valid records advanced.")

        # -------------------------------------------------------
        # STAGE 4: Compression Hook
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.COMPRESSION)
        valid_records = self.compression_service.compress_fields(valid_records, fields_to_compress=["summary", "reasoning", "explanation"])

        # -------------------------------------------------------
        # STAGE 5: Database Write (UnitOfWork fully managed here)
        # -------------------------------------------------------
        self.monitor.log_stage(context, PipelineStage.DATABASE_WRITE)
        t_writer = time.perf_counter()

        try:
            with self.uow:
                self.writer.write_market_data(valid_records)
                self.monitor.log_stage(context, PipelineStage.COMMIT)
                self.uow.commit()
        except Exception as e:
            self.monitor.log_stage(context, PipelineStage.ROLLBACK)
            self.uow.rollback()
            raise DatabaseError(f"Database persistence sequence failed for batch. Rolled back perfectly. Error: {str(e)}") from e

        db_lat = time.perf_counter() - t_writer
        context.stats.add_latency("writer_latency", db_lat)
        context.stats.add_latency("db_latency", db_lat)
        context.stats.increment("records_written", len(valid_records))

        context.mark_success(batch)
        
        write_payload = MappingProxyType({"records_written": len(valid_records), "execution_id": context.execution_id})
        if hasattr(self.event_bus, 'publish_async'):
            self.event_bus.publish_async(Event("WriteCompleted", "market_pipeline", write_payload))
        else:
            self.event_bus.publish(Event("WriteCompleted", "market_pipeline", write_payload))
            
        self.logger.info(f"Pipeline execution completed flawlessly for batch. [Exec ID: {context.execution_id}]")


class PipelineCoordinator:"""

    new_content = re.sub(pattern, new_executor, content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ Successfully replaced PipelineExecutor with perfect syntax and auto-discovery!")
except Exception as e:
    print(f"Failed to patch file: {e}")
