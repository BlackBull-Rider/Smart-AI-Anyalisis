"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/data/market_validator.py
Description: Enterprise Validation Layer and Data Quality Gatekeeper.
             Implements strict, high-performance, vectorized validation for all incoming
             market data (Historical, Live, Corporate Actions, Symbols, Master Data).
             Guarantees zero corrupted data enters the master database with dynamic
             tick-size checks, circuit-limit validations, and safe weekend removal.
             Python 3.13 Compatible. Compile-Safe. Runtime-Safe. Production Locked.
"""

import sys
import json
import time
import uuid
import logging
import datetime
import threading
import unicodedata
from enum import Enum
from contextvars import ContextVar
from collections import defaultdict
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# =========================================================================
# ENTERPRISE CONTEXT VARIABLES
# =========================================================================
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="SYSTEM")
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")
span_id_ctx: ContextVar[str] = ContextVar("span_id", default="")

# =========================================================================
# TELEMETRY & OBSERVABILITY ENGINE
# =========================================================================
class SpanKind(Enum):
    CLIENT = "CLIENT"
    INTERNAL = "INTERNAL"
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"

def trace_span(operation: str, component: str, kind: SpanKind):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            new_span = uuid.uuid4().hex[:8]
            t_id = trace_id_ctx.get() or uuid.uuid4().hex
            t_trace = trace_id_ctx.set(t_id)
            t_span = span_id_ctx.set(new_span)
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                SafeMetrics.record_latency(f"{component}_{operation}_ms", "validator", duration_ms)
                trace_id_ctx.reset(t_trace)
                span_id_ctx.reset(t_span)
        return wrapper
    return decorator

class SafeMetrics:
    _lock = threading.Lock()
    _stats: Dict[str, float] = defaultdict(float)

    @staticmethod
    def increment(name: str, namespace: str = "validator", amount: int = 1) -> None:
        with SafeMetrics._lock: SafeMetrics._stats[f"{namespace}.{name}"] += amount
        try:
            from backend.core.metrics import metrics_engine
            if hasattr(metrics_engine, 'increment'): metrics_engine.increment(name, namespace=namespace, amount=amount)
        except Exception: pass

    @staticmethod
    def record_latency(name: str, namespace: str, duration: float) -> None:
        with SafeMetrics._lock:
            k = f"{namespace}.{name}_rolling"
            if k not in SafeMetrics._stats: SafeMetrics._stats[k] = duration
            else: SafeMetrics._stats[k] = (SafeMetrics._stats[k] * 0.9) + (duration * 0.1)
        try:
            from backend.core.metrics import metrics_engine
            if hasattr(metrics_engine, 'record_latency'): metrics_engine.record_latency(name, namespace, duration)
        except Exception: pass

    @staticmethod
    def gauge(name: str, namespace: str, value: float) -> None:
        with SafeMetrics._lock: SafeMetrics._stats[f"{namespace}.{name}"] = value
        try:
            from backend.core.metrics import metrics_engine
            if hasattr(metrics_engine, 'gauge'): metrics_engine.gauge(name, namespace, value)
        except Exception: pass

class AuditAction(Enum):
    SYSTEM = "SYSTEM"
    VALIDATE = "VALIDATE"
    REJECT = "REJECT"

class AuditSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class AuditEngine:
    @staticmethod
    def record_event(operation: str, action: AuditAction, severity: AuditSeverity, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            from backend.core.audit import AuditEngine as CoreAudit
            if hasattr(CoreAudit, 'record_event'):
                CoreAudit.record_event(operation=operation, action=action, severity=severity, message=message, metadata=metadata)
        except Exception: pass

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def _log(self, level: int, msg: str, **kwargs):
        payload = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": logging.getLevelName(level),
            "trace_id": trace_id_ctx.get(),
            "span_id": span_id_ctx.get(),
            "request_id": request_id_ctx.get(),
            "component": "market_validator",
            "thread_id": threading.get_ident(),
            "message": msg
        }
        payload.update(kwargs)
        self.logger.log(level, json.dumps(payload, default=str))

    def debug(self, msg: str, **kwargs): self._log(logging.DEBUG, msg, **kwargs)
    def info(self, msg: str, **kwargs): self._log(logging.INFO, msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log(logging.WARNING, msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log(logging.ERROR, msg, **kwargs)
    def critical(self, msg: str, **kwargs): self._log(logging.CRITICAL, msg, **kwargs)

_logger = StructuredLogger("MarketValidator")

# =========================================================================
# EXCEPTIONS
# =========================================================================
class ValidationError(Exception): """Base exception for all validation failures."""
class SchemaError(ValidationError): """Raised on structural / column definition mismatches."""
class DataQualityError(ValidationError): """Raised when statistical or quality invariants fail."""
class MarketDataError(ValidationError): """Raised on fundamental market data rules violation."""
class OHLCError(MarketDataError): """Raised specifically on Open/High/Low/Close impossibilities."""
class CorporateActionError(ValidationError): """Raised on invalid CA parameters."""
class TimestampError(ValidationError): """Raised on impossible, future, or unordered dates."""
class SymbolError(ValidationError): """Raised on un-mappable or explicitly invalid symbols."""

# =========================================================================
# DOMAIN MODELS
# =========================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Immutable validation artifact returned to downstream systems."""
    is_valid: bool
    clean_dataframe: pd.DataFrame
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, float]
    rows_input: int
    rows_output: int
    rows_removed: int
    duplicate_count: int
    missing_count: int
    validation_duration_ms: float

@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: str
    aliases: List[str] = field(default_factory=list)
    nullable: bool = False
    is_numeric: bool = False
    is_datetime: bool = False

@dataclass(frozen=True)
class ValidationConfig:
    """Controls pipeline execution domains."""
    check_ohlc: bool = False
    check_weekends: bool = False
    check_corporate_actions: bool = False
    check_tick_size: bool = False
    tick_size: float = 0.05
    check_circuit_limits: bool = False
    circuit_limit_pct: float = 0.20
    check_vwap: bool = False
    check_delivery: bool = False

# =========================================================================
# SCHEMAS
# =========================================================================

class EnterpriseSchemas:
    OHLCV = [
        ColumnSchema(name="symbol", dtype="string", aliases=["ticker", "sym", "instrument"], nullable=False),
        ColumnSchema(name="date", dtype="datetime64[ns, UTC]", aliases=["datetime", "timestamp", "time", "trade_date"], nullable=False, is_datetime=True),
        ColumnSchema(name="open", dtype="float64", aliases=["price_open", "o"], nullable=False, is_numeric=True),
        ColumnSchema(name="high", dtype="float64", aliases=["price_high", "h"], nullable=False, is_numeric=True),
        ColumnSchema(name="low", dtype="float64", aliases=["price_low", "l"], nullable=False, is_numeric=True),
        ColumnSchema(name="close", dtype="float64", aliases=["price_close", "c", "adj_close"], nullable=False, is_numeric=True),
        ColumnSchema(name="volume", dtype="int64", aliases=["vol", "traded_qty", "tottrdqty"], nullable=False, is_numeric=True),
    ]

    INTRADAY = OHLCV + [
        ColumnSchema(name="vwap", dtype="float64", aliases=["avg_price"], nullable=True, is_numeric=True),
    ]
    
    FNO_DATA = OHLCV + [
        ColumnSchema(name="expiry_date", dtype="datetime64[ns, UTC]", aliases=["expiry"], nullable=False, is_datetime=True),
        ColumnSchema(name="strike_price", dtype="float64", aliases=["strike"], nullable=True, is_numeric=True),
        ColumnSchema(name="option_type", dtype="string", aliases=["type", "ce_pe"], nullable=True),
        ColumnSchema(name="open_interest", dtype="int64", aliases=["oi"], nullable=True, is_numeric=True),
    ]

    LIVE_QUOTE = [
        ColumnSchema(name="symbol", dtype="string", aliases=["ticker"], nullable=False),
        ColumnSchema(name="timestamp", dtype="datetime64[ns, UTC]", aliases=["time", "date"], nullable=False, is_datetime=True),
        ColumnSchema(name="last_price", dtype="float64", aliases=["ltp", "price", "close"], nullable=False, is_numeric=True),
        ColumnSchema(name="volume", dtype="int64", aliases=["vol"], nullable=False, is_numeric=True),
    ]

    CORPORATE_ACTION = [
        ColumnSchema(name="symbol", dtype="string", aliases=["ticker"], nullable=False),
        ColumnSchema(name="purpose", dtype="string", aliases=["action", "type"], nullable=False),
        ColumnSchema(name="ex_date", dtype="datetime64[ns, UTC]", aliases=["exdate", "execution_date"], nullable=False, is_datetime=True),
        ColumnSchema(name="record_date", dtype="datetime64[ns, UTC]", aliases=["recorddate"], nullable=True, is_datetime=True),
    ]

    DEALS = [
        ColumnSchema(name="symbol", dtype="string", aliases=["ticker"], nullable=False),
        ColumnSchema(name="date", dtype="datetime64[ns, UTC]", aliases=["deal_date"], nullable=False, is_datetime=True),
        ColumnSchema(name="client_name", dtype="string", aliases=["client"], nullable=False),
        ColumnSchema(name="buy_sell", dtype="string", aliases=["deal_type", "type"], nullable=False),
        ColumnSchema(name="quantity", dtype="int64", aliases=["qty"], nullable=False, is_numeric=True),
        ColumnSchema(name="price", dtype="float64", aliases=["avg_price"], nullable=False, is_numeric=True),
    ]

    MASTER = [
        ColumnSchema(name="symbol", dtype="string", aliases=["ticker"], nullable=False),
        ColumnSchema(name="company_name", dtype="string", aliases=["name"], nullable=False),
        ColumnSchema(name="series", dtype="string", aliases=["eq_series"], nullable=True),
        ColumnSchema(name="isin", dtype="string", aliases=["isin_code"], nullable=True),
    ]


# =========================================================================
# CORE VALIDATOR ENGINE
# =========================================================================

class MarketValidator:
    """
    Enterprise Data Quality Gatekeeper.
    Performs purely vectorized, highly defensive evaluations and sanitizations 
    on structural, statistical, and domain-specific invariants.
    Thread-safe and immutable-oriented.
    """

    # --- 1. DATA INGESTION & COERCION ---

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        """Robustly coerces diverse data structures into a Pandas DataFrame."""
        if isinstance(data, pd.DataFrame): return data.copy()
        
        if isinstance(data, list):
            if not data: return pd.DataFrame()
            first_elem = data[0]
            if is_dataclass(first_elem): return pd.DataFrame([asdict(x) for x in data])
            if isinstance(first_elem, tuple) and hasattr(first_elem, '_asdict'): return pd.DataFrame([x._asdict() for x in data])
            if isinstance(first_elem, dict): return pd.DataFrame(data)
            if hasattr(first_elem, '__dict__'): return pd.DataFrame([vars(x) for x in data])

        if isinstance(data, dict): return pd.DataFrame([data])
            
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return pd.DataFrame(parsed if isinstance(parsed, list) else [parsed])
            except json.JSONDecodeError as e:
                raise DataQualityError(f"Failed to parse JSON string: {e}")

        try:
            if hasattr(data, '__iter__'):
                first = next(iter(data), None)
                if first is not None and hasattr(first, 'keys'): # sqlite3.Row
                    return pd.DataFrame([dict(row) for row in data])
        except Exception: pass

        raise DataQualityError(f"Unsupported data type for validation: {type(data)}")

    # --- 2. STRUCTURAL NORMALIZATION ---

    @staticmethod
    def _normalize_columns(df: pd.DataFrame, schema: List[ColumnSchema]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Vectorized renaming, dropping, and basic type assertion based on schema aliases."""
        errors, warnings = [], []
        df.columns = df.columns.astype(str).str.strip().str.lower()
        
        alias_map = {}
        for col_schema in schema:
            alias_map[col_schema.name.lower()] = col_schema.name
            for alias in col_schema.aliases:
                alias_map[alias.lower()] = col_schema.name
                
        df.rename(columns=alias_map, inplace=True)
        
        expected_cols = {c.name for c in schema}
        actual_cols = set(df.columns)
        
        missing = expected_cols - actual_cols
        required_missing = [c.name for c in schema if c.name in missing and not c.nullable]
        if required_missing: errors.append(f"Missing mandatory columns: {required_missing}")
            
        unexpected = actual_cols - expected_cols
        if unexpected:
            warnings.append(f"Unexpected columns dropped: {unexpected}")
            df.drop(columns=list(unexpected), inplace=True)
            
        ordered_cols = [c.name for c in schema if c.name in df.columns]
        return df[ordered_cols], errors, warnings

    # --- 3. STRING & SYMBOL CLEANING ---

    @staticmethod
    def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized string normalization: Unicode normalization, trim, collapse spaces."""
        str_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in str_cols:
            mask = df[col].notna()
            if not mask.any(): continue
            df[col] = df[col].astype(str)
            df.loc[mask, col] = (
                df.loc[mask, col]
                .apply(lambda x: unicodedata.normalize('NFKC', str(x)))
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
                .str.upper()
            )
            df.loc[df[col] == '', col] = np.nan
        return df

    @staticmethod
    def _validate_symbols(df: pd.DataFrame, symbol_col: str = "symbol") -> Tuple[pd.DataFrame, List[str]]:
        errors = []
        if symbol_col not in df.columns: return df, errors

        # Robust NSE/BSE Symbol Regex handling special characters (e.g., M&M, BAJAJ-AUTO, 360ONE)
        valid_symbol_regex = r"^[A-Z0-9&_\-]+(\.NS|\.BO|\.SZ)?$"
        invalid_mask = ~df[symbol_col].str.match(valid_symbol_regex, na=False)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_examples = df.loc[invalid_mask, symbol_col].dropna().unique()[:5]
            errors.append(f"Rejected {invalid_count} rows due to invalid symbol format. Examples: {list(invalid_examples)}")
            df = df[~invalid_mask]
            
        return df, errors

    # --- 4. NUMERIC & OHLC VALIDATION ---

    @staticmethod
    def _validate_and_clean_numerics(df: pd.DataFrame, schema: List[ColumnSchema]) -> Tuple[pd.DataFrame, List[str]]:
        errors = []
        numeric_cols = [c.name for c in schema if c.is_numeric and c.name in df.columns]
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                errors.append(f"Replaced {inf_mask.sum()} infinite values with NaN in column '{col}'")
                df.loc[inf_mask, col] = np.nan
                
        float_cols = df.select_dtypes(include=['float64']).columns
        if not float_cols.empty: df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
        
        int_cols = df.select_dtypes(include=['int64']).columns
        if not int_cols.empty: df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
        
        return df, errors

    @staticmethod
    def _validate_ohlcv_logic(df: pd.DataFrame, config: ValidationConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Strict structural physics, limits, and spike checks for OHLC data."""
        errors, warnings = [], []
        req_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not req_cols.issubset(set(df.columns)): return df, errors, warnings

        invalid_mask = pd.Series(False, index=df.index)
        
        # 1. Negative Value Checks
        neg_mask = (df['open'] < 0) | (df['high'] < 0) | (df['low'] < 0) | (df['close'] < 0) | (df['volume'] < 0)
        if neg_mask.any():
            errors.append(f"Dropped {neg_mask.sum()} rows with negative OHLCV values.")
            invalid_mask |= neg_mask

        # 2. Price Physics Checks
        phys_mask = (df['high'] < df['open']) | (df['high'] < df['close']) | \
                    (df['low'] > df['open']) | (df['low'] > df['close']) | \
                    (df['high'] < df['low'])
        if phys_mask.any():
            errors.append(f"Dropped {phys_mask.sum()} rows with impossible candle physics (e.g. High < Low).")
            invalid_mask |= phys_mask

        # 3. Null Checks on Critical Fields
        null_mask = df[['open', 'high', 'low', 'close']].isnull().all(axis=1)
        if null_mask.any():
            errors.append(f"Dropped {null_mask.sum()} completely empty candles.")
            invalid_mask |= null_mask

        # 4. Tick Size Validation
        if config.check_tick_size:
            mod_res = df['close'] % config.tick_size
            tick_mask = ~(np.isclose(mod_res, 0, atol=1e-3) | np.isclose(mod_res, config.tick_size, atol=1e-3))
            if tick_mask.any():
                warnings.append(f"Flagged {tick_mask.sum()} rows failing {config.tick_size} tick size checks.")

        # 5. Circuit Limit Validations
        if config.check_circuit_limits:
            upper = 1.0 + config.circuit_limit_pct
            lower = 1.0 - config.circuit_limit_pct
            limit_mask = (df['high'] > df['open'] * upper) | (df['low'] < df['open'] * lower)
            if limit_mask.any():
                warnings.append(f"Flagged {limit_mask.sum()} rows exceeding {config.circuit_limit_pct * 100}% circuit limits (Check for Splits).")

        # 6. VWAP Validation
        if config.check_vwap and 'vwap' in df.columns:
            vwap_mask = (df['vwap'] < df['low']) | (df['vwap'] > df['high'])
            if vwap_mask.any():
                errors.append(f"Dropped {vwap_mask.sum()} rows with VWAP outside High-Low range.")
                invalid_mask |= vwap_mask

        # 7. Delivery Percentage
        if config.check_delivery and 'delivery_pct' in df.columns:
            del_mask = (df['delivery_pct'] < 0) | (df['delivery_pct'] > 100)
            if del_mask.any():
                errors.append(f"Dropped {del_mask.sum()} rows with invalid Delivery %.")
                invalid_mask |= del_mask

        # Filter
        df_clean = df[~invalid_mask].copy()
        
        # Volume safe cast to Integer
        if 'volume' in df_clean.columns:
            df_clean['volume'] = df_clean['volume'].fillna(0).astype('int64')
            
        return df_clean, errors, warnings

    # --- 5. DATE VALIDATION & TIMEZONE ---

    @staticmethod
    def _validate_and_clean_dates(df: pd.DataFrame, schema: List[ColumnSchema], config: ValidationConfig) -> Tuple[pd.DataFrame, List[str]]:
        errors = []
        date_cols = [c.name for c in schema if c.is_datetime and c.name in df.columns]
        now_utc = pd.Timestamp.utcnow()
        
        drop_mask = pd.Series(False, index=df.index)

        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
                
                old_mask = df[col] < pd.Timestamp("1990-01-01", tz="UTC")
                if old_mask.any():
                    errors.append(f"Flagged {old_mask.sum()} rows in '{col}' with dates before 1990.")
                    df.loc[old_mask, col] = pd.NaT

                fut_mask = df[col] > now_utc
                if fut_mask.any():
                    errors.append(f"Flagged {fut_mask.sum()} rows in '{col}' with impossible future dates.")
                    df.loc[fut_mask, col] = pd.NaT

                if config.check_weekends:
                    weekend_mask = df[col].dt.dayofweek >= 5
                    if weekend_mask.any():
                        errors.append(f"Dropped {weekend_mask.sum()} rows corresponding to weekends in '{col}'.")
                        drop_mask |= weekend_mask

            except Exception as e:
                errors.append(f"Critical date parsing failure on column '{col}': {e}")
                
        if drop_mask.any():
            df = df.loc[~drop_mask].copy()
            
        return df, errors

    # --- 6. CORPORATE ACTION VALIDATION ---

    @staticmethod
    def _validate_corporate_actions_logic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        errors = []
        if 'purpose' not in df.columns or 'ex_date' not in df.columns:
            return df, errors

        if 'record_date' in df.columns:
            mask = df['ex_date'].notna() & df['record_date'].notna()
            # If record date exists, ex-date strictly should not be absurdly after it.
            invalid_dates = mask & (df['ex_date'] > df['record_date'] + pd.Timedelta(days=1))
            if invalid_dates.any():
                errors.append(f"Dropped {invalid_dates.sum()} Corporate Actions with invalid Ex/Record Date ordering.")
                df = df[~invalid_dates]

        return df, errors

    # --- 7. QUALITY METRICS ---

    @staticmethod
    def _calculate_quality_metrics(df: pd.DataFrame, schema: List[ColumnSchema]) -> Dict[str, float]:
        stats = {}
        if df.empty: return stats
        total_rows = len(df)
        
        missing_pct = (df.isnull().sum() / total_rows) * 100
        for col, pct in missing_pct.items():
            stats[f"missing_pct_{col}"] = round(float(pct), 2)
            
        numeric_cols = [c.name for c in schema if c.is_numeric and c.name in df.columns]
        for col in numeric_cols:
            if df[col].notna().sum() < 2: continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                stats[f"zero_variance_{col}"] = 1.0
                continue
            lower_bound, upper_bound = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            stats[f"outlier_pct_{col}"] = round(float((outliers / total_rows) * 100), 2)

        return stats


    # =========================================================================
    # PUBLIC API FACADE
    # =========================================================================

    @classmethod
    @trace_span(operation="validate_dataframe", component="validator", kind=SpanKind.INTERNAL)
    def validate_dataframe(cls, data: Any, schema: List[ColumnSchema], config: ValidationConfig = ValidationConfig()) -> ValidationResult:
        """Master Pipeline Method. Ensures strict structural and domain checks."""
        t0 = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []
        
        try: df = cls._to_dataframe(data)
        except Exception as e:
            SafeMetrics.increment("validation_critical_failures")
            return ValidationResult(False, pd.DataFrame(), [str(e)], [], {}, 0, 0, 0, 0, 0, 0.0)

        rows_input = len(df)
        if rows_input == 0:
            return ValidationResult(False, df, ["Input data is completely empty."], [], {}, 0, 0, 0, 0, 0, 0.0)

        # Pipeline Execution
        df, s_errs, s_warns = cls._normalize_columns(df, schema)
        errors.extend(s_errs); warnings.extend(s_warns)

        df = cls._clean_strings(df)
        df, sym_errs = cls._validate_symbols(df)
        errors.extend(sym_errs)

        df, d_errs = cls._validate_and_clean_dates(df, schema, config)
        errors.extend(d_errs)

        df, n_errs = cls._validate_and_clean_numerics(df, schema)
        errors.extend(n_errs)

        # Domain Logic Execution
        if config.check_ohlc:
            df, o_errs, o_warns = cls._validate_ohlcv_logic(df, config)
            errors.extend(o_errs); warnings.extend(o_warns)
            
        if config.check_corporate_actions:
            df, ca_errs = cls._validate_corporate_actions_logic(df)
            errors.extend(ca_errs)

        # Duplicate Removal Strategy (Must sort to keep latest valid entry)
        duplicate_count = 0
        pk_cols = [c.name for c in schema if c.name in ('symbol', 'date', 'timestamp', 'ex_date')]
        pk_present = [c for c in pk_cols if c in df.columns]
        
        if pk_present:
            sort_cols = [c for c in pk_present if c in ('date', 'timestamp', 'ex_date')]
            if sort_cols: df.sort_values(by=sort_cols, inplace=True)
            
            dup_mask = df.duplicated(subset=pk_present, keep='last')
            duplicate_count = int(dup_mask.sum())
            if duplicate_count > 0:
                warnings.append(f"Removed {duplicate_count} strict duplicate rows based on {pk_present}")
                df = df[~dup_mask]

        # Null Constraint Enforcement
        mandatory_cols = [c.name for c in schema if not c.nullable and c.name in df.columns]
        if mandatory_cols:
            missing_mask = df[mandatory_cols].isnull().any(axis=1)
            missing_count = int(missing_mask.sum())
            if missing_count > 0:
                errors.append(f"Dropped {missing_count} rows violating NOT NULL constraints on {mandatory_cols}")
                df = df[~missing_mask]
        else:
            missing_count = 0

        rows_output = len(df)
        rows_removed = rows_input - rows_output
        is_valid = rows_output > 0 and len(errors) == 0

        duration_ms = (time.perf_counter() - t0) * 1000
        stats = cls._calculate_quality_metrics(df, schema)

        SafeMetrics.increment("validation_processed_rows", amount=rows_input)
        SafeMetrics.increment("validation_dropped_rows", amount=rows_removed)
        if not is_valid:
            SafeMetrics.increment("validation_rejections")
            AuditEngine.record_event("validator.run", AuditAction.REJECT, AuditSeverity.WARNING, "Data validation failed", {"errors": errors})

        return ValidationResult(
            is_valid=is_valid, clean_dataframe=df, errors=errors, warnings=warnings,
            statistics=stats, rows_input=rows_input, rows_output=rows_output,
            rows_removed=rows_removed, duplicate_count=duplicate_count, missing_count=missing_count,
            validation_duration_ms=duration_ms
        )

    # --- Domain-Specific Public Facades ---

    @classmethod
    def validate_market_data(cls, data: Any) -> ValidationResult:
        config = ValidationConfig(check_ohlc=True, check_weekends=True, check_circuit_limits=True, check_tick_size=True)
        return cls.validate_dataframe(data, EnterpriseSchemas.OHLCV, config)

    @classmethod
    def validate_intraday(cls, data: Any) -> ValidationResult:
        config = ValidationConfig(check_ohlc=True, check_weekends=True, check_vwap=True)
        return cls.validate_dataframe(data, EnterpriseSchemas.INTRADAY, config)

    @classmethod
    def validate_live_quote(cls, data: Any) -> ValidationResult:
        return cls.validate_dataframe(data, EnterpriseSchemas.LIVE_QUOTE)

    @classmethod
    def validate_fno(cls, data: Any) -> ValidationResult:
        config = ValidationConfig(check_ohlc=True, check_weekends=True)
        return cls.validate_dataframe(data, EnterpriseSchemas.FNO_DATA, config)

    @classmethod
    def validate_etf(cls, data: Any) -> ValidationResult:
        return cls.validate_market_data(data)

    @classmethod
    def validate_reit(cls, data: Any) -> ValidationResult:
        return cls.validate_market_data(data)

    @classmethod
    def validate_invit(cls, data: Any) -> ValidationResult:
        return cls.validate_market_data(data)

    @classmethod
    def validate_bulk_deals(cls, data: Any) -> ValidationResult:
        return cls.validate_dataframe(data, EnterpriseSchemas.DEALS)

    @classmethod
    def validate_block_deals(cls, data: Any) -> ValidationResult:
        return cls.validate_dataframe(data, EnterpriseSchemas.DEALS)

    @classmethod
    def validate_surveillance(cls, data: Any) -> ValidationResult:
        return cls.validate_dataframe(data, EnterpriseSchemas.MASTER)

    @classmethod
    def validate_master(cls, data: Any) -> ValidationResult:
        return cls.validate_dataframe(data, EnterpriseSchemas.MASTER)

    @classmethod
    def validate_corporate_actions(cls, data: Any) -> ValidationResult:
        config = ValidationConfig(check_corporate_actions=True)
        return cls.validate_dataframe(data, EnterpriseSchemas.CORPORATE_ACTION, config)

    @classmethod
    def validate_symbols(cls, symbols: List[str]) -> Tuple[List[str], List[str]]:
        df = pd.DataFrame({"symbol": symbols})
        df = cls._clean_strings(df)
        df_valid, _ = cls._validate_symbols(df)
        
        valid_list = df_valid['symbol'].dropna().unique().tolist()
        valid_set = set(valid_list)
        rejected = [s for s in symbols if str(s).strip().upper() not in valid_set]
        return valid_list, rejected

    @classmethod
    def validate_ohlc(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df_clean, errs, warns = cls._validate_ohlcv_logic(df.copy(), ValidationConfig(check_ohlc=True))
        return df_clean, errs

    @classmethod
    def validate_dates(cls, df: pd.DataFrame, date_col: str) -> bool:
        if date_col not in df.columns: return False
        try:
            dates = pd.to_datetime(df[date_col], utc=True, errors='raise')
            if dates.isnull().any() or (dates > pd.Timestamp.utcnow()).any(): return False
            return bool(dates.is_monotonic_increasing and dates.is_unique)
        except Exception: return False

    @classmethod
    def validate_schema(cls, df: pd.DataFrame, expected_schema: List[ColumnSchema]) -> bool:
        actual_cols = {str(c).strip().lower() for c in df.columns}
        expected_cols = {c.name.lower() for c in expected_schema}
        return expected_cols.issubset(actual_cols)

    @classmethod
    def clean_dataframe(cls, data: Any, schema: List[ColumnSchema], config: ValidationConfig = ValidationConfig()) -> pd.DataFrame:
        df = cls._to_dataframe(data)
        df, _, _ = cls._normalize_columns(df, schema)
        df = cls._clean_strings(df)
        df, _ = cls._validate_and_clean_dates(df, schema, config)
        df, _ = cls._validate_and_clean_numerics(df, schema)
        
        if config.check_ohlc:
            df, _, _ = cls._validate_ohlcv_logic(df, config)
            
        pk_cols = [c.name for c in schema if c.name in ('symbol', 'date', 'timestamp', 'ex_date')]
        pk_present = [c for c in pk_cols if c in df.columns]
        if pk_present:
            sort_cols = [c for c in pk_present if c in ('date', 'timestamp', 'ex_date')]
            if sort_cols: df.sort_values(by=sort_cols, inplace=True)
            df.drop_duplicates(subset=pk_present, keep='last', inplace=True)
            
        mandatory_cols = [c.name for c in schema if not c.nullable and c.name in df.columns]
        if mandatory_cols:
            df.dropna(subset=mandatory_cols, inplace=True)
            
        return df

    @staticmethod
    def generate_validation_report(result: ValidationResult) -> str:
        status = "PASSED" if result.is_valid else "FAILED"
        report = [
            f"=== Enterprise Validation Report: {status} ===",
            f"Rows Processed: {result.rows_input} -> Output: {result.rows_output} (Dropped: {result.rows_removed})",
            f"Duplicates Removed: {result.duplicate_count} | Null Violations: {result.missing_count}",
            f"Processing Latency: {result.validation_duration_ms:.2f} ms",
        ]
        if result.errors:
            report.append("--- CRITICAL ERRORS ---")
            report.extend(f" * {e}" for e in result.errors)
        if result.warnings:
            report.append("--- WARNINGS ---")
            report.extend(f" * {w}" for w in result.warnings)
        if result.statistics:
            report.append("--- QUALITY METRICS ---")
            for k, v in result.statistics.items(): report.append(f" * {k}: {v}")
        return "\n".join(report)

# =========================================================================
# EXPORTS
# =========================================================================
__all__ = [
    "MarketValidator",
    "ValidationResult",
    "ValidationConfig",
    "ColumnSchema",
    "EnterpriseSchemas",
    "ValidationError",
    "SchemaError",
    "DataQualityError",
    "MarketDataError",
    "OHLCError",
    "CorporateActionError",
    "TimestampError",
    "SymbolError"
]
