"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/config/settings.py
Description: Global configuration layer. Immutable, thread-safe, singleton-based 
             configuration dataclasses governing the entire event-driven architecture.
             Contains robust runtime validation, environment variable binding, 
             and future-proof feature toggling. Production Lock v1.0.
"""

import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import Final, Tuple, Optional, Any, Type
from threading import Lock
from functools import cached_property
from datetime import time as dt_time


# -------------------------------------------------------------------------
# METADATA & CONSTANTS
# -------------------------------------------------------------------------

CONFIG_SCHEMA_VERSION: Final[str] = "1.0.0"


# -------------------------------------------------------------------------
# UTILITY HELPER FUNCTIONS (For Safe Environment Variable Parsing)
# -------------------------------------------------------------------------

def _resolve_project_root() -> Path:
    """Robustly resolves the project root directory by looking for known markers."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "backend").exists() or (current / "requirements.txt").exists():
            return current
        current = current.parent
    # Fallback to relative assumption if markers are not found
    return Path(__file__).resolve().parent.parent.parent

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def _env_list(key: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    val = os.getenv(key)
    if not val:
        return default
    return tuple(item.strip() for item in val.split(",") if item.strip())

def _env_time(key: str, default_hour: int, default_minute: int) -> dt_time:
    val = os.getenv(key)
    if val:
        try:
            parts = val.split(":")
            return dt_time(hour=int(parts[0]), minute=int(parts[1]))
        except (ValueError, IndexError):
            pass
    return dt_time(hour=default_hour, minute=default_minute)


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class Environment(Enum):
    DEVELOPMENT = "DEVELOPMENT"
    STAGING = "STAGING"
    PRODUCTION = "PRODUCTION"

class DatabaseEngine(Enum):
    SQLITE = "SQLITE"
    POSTGRESQL = "POSTGRESQL"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class JournalMode(Enum):
    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"
    PERSIST = "PERSIST"
    MEMORY = "MEMORY"
    WAL = "WAL"
    OFF = "OFF"

class SynchronousMode(Enum):
    OFF = 0
    NORMAL = 1
    FULL = 2
    EXTRA = 3

class Theme(Enum):
    LIGHT = "LIGHT"
    DARK = "DARK"
    SYSTEM = "SYSTEM"

class EventBusType(Enum):
    NONE = "NONE"
    MEMORY = "MEMORY"
    REDIS = "REDIS"
    KAFKA = "KAFKA"
    RABBITMQ = "RABBITMQ"

class AIProvider(Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GEMINI = "GEMINI"
    LOCAL_LLM = "LOCAL_LLM"


def _env_enum(enum_cls: Type[Enum], key: str, default_enum: Enum) -> Enum:
    """Safely parses Enum from environment variables, preventing startup crashes."""
    val = os.getenv(key)
    if not val:
        return default_enum
    try:
        return enum_cls(val.strip().upper())
    except ValueError:
        return default_enum


# -------------------------------------------------------------------------
# CONFIGURATION DATACLASSES (IMMUTABLE & VALIDATED)
# -------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class ApplicationConfig:
    name: str = "GREEN BULL RIDER V6"
    version: str = "6.0.0"
    environment: Environment = field(
        default_factory=lambda: _env_enum(Environment, "APP_ENV", Environment.PRODUCTION)
    )
    base_dir: Path = field(default_factory=_resolve_project_root)


@dataclass(frozen=True, kw_only=True)
class FeatureFlags:
    enable_ai: bool = field(default_factory=lambda: _env_bool("FEATURE_ENABLE_AI", True))
    enable_swing_engine: bool = field(default_factory=lambda: _env_bool("FEATURE_ENABLE_SWING_ENGINE", True))
    enable_smart_money: bool = field(default_factory=lambda: _env_bool("FEATURE_ENABLE_SMART_MONEY", True))
    enable_trace: bool = field(default_factory=lambda: _env_bool("FEATURE_ENABLE_TRACE", True))
    enable_audit: bool = field(default_factory=lambda: _env_bool("FEATURE_ENABLE_AUDIT", True))
    enable_cache: bool = field(default_factory=lambda: _env_bool("FEATURE_ENABLE_CACHE", True))


@dataclass(frozen=True, kw_only=True)
class DatabaseConfig:
    engine: DatabaseEngine = field(default_factory=lambda: _env_enum(DatabaseEngine, "DB_ENGINE", DatabaseEngine.SQLITE))
    sqlite_db_name: str = _env_str("SQLITE_DB_NAME", "universe.db")
    sqlite_db_folder: Path = field(default_factory=lambda: Path(_env_str("SQLITE_DB_FOLDER", "database")))
    pg_dsn: str = field(default_factory=lambda: _env_str("PG_DSN", "postgresql://user:pass@localhost:5432/greenbull"), repr=False)
    
    busy_timeout_ms: int = _env_int("DB_BUSY_TIMEOUT_MS", 5000)
    journal_mode: JournalMode = field(default_factory=lambda: _env_enum(JournalMode, "DB_JOURNAL", JournalMode.WAL))
    synchronous_mode: SynchronousMode = field(default_factory=lambda: _env_enum(SynchronousMode, "DB_SYNC", SynchronousMode.NORMAL))
    enable_foreign_keys: bool = True
    temp_store_memory: bool = True
    mmap_size: int = field(default_factory=lambda: _env_int("DB_MMAP_SIZE", 268435456))

    def __post_init__(self):
        if self.mmap_size < 0:
            raise ValueError(f"DatabaseConfig: mmap_size cannot be negative ({self.mmap_size})")
        if self.busy_timeout_ms < 0:
            raise ValueError("DatabaseConfig: busy_timeout_ms cannot be negative")


@dataclass(frozen=True, kw_only=True)
class LoggingConfig:
    level: LogLevel = field(default_factory=lambda: _env_enum(LogLevel, "LOG_LEVEL", LogLevel.INFO))
    log_folder: Path = field(default_factory=lambda: Path(_env_str("LOG_FOLDER", "logs")))
    rotation: dt_time = field(default_factory=lambda: _env_time("LOG_ROTATION", 0, 0))
    retention_days: int = _env_int("LOG_RETENTION_DAYS", 30)
    enable_console: bool = _env_bool("LOG_ENABLE_CONSOLE", True)
    enable_json: bool = _env_bool("LOG_ENABLE_JSON", True)
    max_file_size_bytes: int = _env_int("LOG_MAX_FILE_SIZE", 104857600)


@dataclass(frozen=True, kw_only=True)
class TraceConfig:
    trace_performance: bool = _env_bool("TRACE_PERFORMANCE", True)
    trace_execution: bool = _env_bool("TRACE_EXECUTION", True)
    trace_database: bool = _env_bool("TRACE_DATABASE", True)
    trace_provider: bool = _env_bool("TRACE_PROVIDER", True)


@dataclass(frozen=True, kw_only=True)
class AuditConfig:
    audit_log_folder: Path = field(default_factory=lambda: Path(_env_str("AUDIT_FOLDER", "audit")))
    retain_days: int = _env_int("AUDIT_RETAIN_DAYS", 365)
    strict_mode: bool = _env_bool("AUDIT_STRICT_MODE", True)


@dataclass(frozen=True, kw_only=True)
class ProviderConfig:
    name: str = _env_str("PROVIDER_NAME", "YahooFinance")
    timeout_sec: float = _env_float("PROVIDER_TIMEOUT", 15.0)
    max_retries: int = _env_int("PROVIDER_MAX_RETRIES", 5)
    backoff_factor: float = _env_float("PROVIDER_BACKOFF_FACTOR", 1.5)
    pool_connections: int = _env_int("PROVIDER_POOL_CONN", 100)
    pool_maxsize: int = _env_int("PROVIDER_POOL_MAXSIZE", 100)
    rate_limit_calls: int = _env_int("PROVIDER_RATE_LIMIT", 2000)
    rate_limit_period_sec: int = _env_int("PROVIDER_RATE_PERIOD", 3600)
    user_agent: str = _env_str(
        "PROVIDER_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __post_init__(self):
        if self.timeout_sec <= 0:
            raise ValueError("ProviderConfig: timeout_sec must be positive")


@dataclass(frozen=True, kw_only=True)
class SyncConfig:
    daily_auto_sync: bool = _env_bool("SYNC_DAILY_AUTO", True)
    incremental_sync: bool = _env_bool("SYNC_INCREMENTAL", True)
    historical_sync: bool = _env_bool("SYNC_HISTORICAL", True)
    gap_recovery: bool = _env_bool("SYNC_GAP_RECOVERY", True)
    resume_enabled: bool = _env_bool("SYNC_RESUME", True)
    crash_recovery: bool = _env_bool("SYNC_CRASH_RECOVERY", True)
    max_parallel_symbols: int = _env_int("SYNC_MAX_PARALLEL", 10)
    max_retries: int = _env_int("SYNC_MAX_RETRIES", 3)


@dataclass(frozen=True, kw_only=True)
class SchedulerConfig:
    daily_sync_time: dt_time = field(default_factory=lambda: _env_time("SCHEDULER_DAILY_TIME", 18, 30))
    weekly_universe_update_day: str = _env_str("SCHEDULER_WEEKLY_DAY", "SATURDAY")
    weekly_universe_update_time: dt_time = field(default_factory=lambda: _env_time("SCHEDULER_WEEKLY_TIME", 10, 0))
    monthly_cleanup_day: int = _env_int("SCHEDULER_MONTHLY_DAY", 1)
    retry_interval_minutes: int = _env_int("SCHEDULER_RETRY_INTERVAL", 15)
    missed_job_recovery: bool = _env_bool("SCHEDULER_MISSED_RECOVERY", True)


@dataclass(frozen=True, kw_only=True)
class EventBusConfig:
    engine: EventBusType = field(default_factory=lambda: _env_enum(EventBusType, "EVENT_BUS_TYPE", EventBusType.MEMORY))
    host: str = _env_str("EVENT_BUS_HOST", "localhost")
    port: int = _env_int("EVENT_BUS_PORT", 6379)
    retry_on_failure: bool = _env_bool("EVENT_BUS_RETRY", True)

    def __post_init__(self):
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"EventBusConfig: Invalid port number {self.port}")


@dataclass(frozen=True, kw_only=True)
class NotificationConfig:
    telegram_enabled: bool = _env_bool("NOTIFY_TELEGRAM_ENABLED", False)
    telegram_bot_token: str = field(default_factory=lambda: _env_str("NOTIFY_TELEGRAM_TOKEN", ""), repr=False)
    telegram_chat_id: str = field(default_factory=lambda: _env_str("NOTIFY_TELEGRAM_CHAT_ID", ""), repr=False)
    
    slack_enabled: bool = _env_bool("NOTIFY_SLACK_ENABLED", False)
    slack_webhook_url: str = field(default_factory=lambda: _env_str("NOTIFY_SLACK_WEBHOOK", ""), repr=False)
    
    discord_enabled: bool = _env_bool("NOTIFY_DISCORD_ENABLED", False)
    discord_webhook_url: str = field(default_factory=lambda: _env_str("NOTIFY_DISCORD_WEBHOOK", ""), repr=False)
    
    email_enabled: bool = _env_bool("NOTIFY_EMAIL_ENABLED", False)
    email_smtp_host: str = _env_str("NOTIFY_EMAIL_SMTP", "")


@dataclass(frozen=True, kw_only=True)
class MarketConfig:
    default_exchange: str = "NSE"
    default_market: str = "India"
    default_currency: str = "INR"
    default_timeframe: str = "1d"
    supported_timeframes: Tuple[str, ...] = ("1m", "5m", "15m", "60m", "1d", "1wk", "1mo")
    holiday_calendar_enabled: bool = True


@dataclass(frozen=True, kw_only=True)
class CacheConfig:
    backend: str = _env_str("CACHE_BACKEND", "MEMORY")
    redis_url: str = _env_str("CACHE_REDIS_URL", "redis://localhost:6379/1")
    max_size_mb: int = _env_int("CACHE_MAX_SIZE_MB", 2048)
    ttl_sec: int = _env_int("CACHE_TTL_SEC", 86400)


@dataclass(frozen=True, kw_only=True)
class PerformanceConfig:
    db_batch_size: int = _env_int("PERF_DB_BATCH_SIZE", 10000)
    fetch_chunk_size: int = _env_int("PERF_FETCH_CHUNK", 5000)
    memory_limit_mb: int = _env_int("PERF_MEMORY_LIMIT_MB", 8192)
    gc_threshold_gen0: int = _env_int("PERF_GC_GEN0", 700)
    gc_threshold_gen1: int = _env_int("PERF_GC_GEN1", 10)
    gc_threshold_gen2: int = _env_int("PERF_GC_GEN2", 10)
    
    def __post_init__(self):
        if self.db_batch_size < 1 or self.fetch_chunk_size < 1:
            raise ValueError("PerformanceConfig: Batch and chunk sizes must be >= 1")


@dataclass(frozen=True, kw_only=True)
class ThreadConfig:
    max_workers: int = _env_int("THREAD_MAX_WORKERS", 16)
    queue_size: int = _env_int("THREAD_QUEUE_SIZE", 5000)
    thread_timeout_sec: int = _env_int("THREAD_TIMEOUT", 300)

    def __post_init__(self):
        if self.max_workers < 1:
            raise ValueError(f"ThreadConfig: max_workers must be >= 1, got {self.max_workers}")


@dataclass(frozen=True, kw_only=True)
class RetryConfig:
    default_retries: int = _env_int("RETRY_DEFAULT", 3)
    max_delay_sec: float = _env_float("RETRY_MAX_DELAY", 60.0)
    exponential_base: float = _env_float("RETRY_EXP_BASE", 2.0)


@dataclass(frozen=True, kw_only=True)
class SecurityConfig:
    read_only_mode: bool = _env_bool("SEC_READ_ONLY", False)
    safe_mode: bool = _env_bool("SEC_SAFE_MODE", True)
    strict_validation: bool = _env_bool("SEC_STRICT_VALIDATION", True)
    checksum_enabled: bool = _env_bool("SEC_CHECKSUM", True)
    encrypt_at_rest: bool = _env_bool("SEC_ENCRYPT_REST", False)


@dataclass(frozen=True, kw_only=True)
class IndicatorConfig:
    default_lookback: int = _env_int("IND_DEFAULT_LOOKBACK", 200)
    max_lookback: int = _env_int("IND_MAX_LOOKBACK", 1000)
    pre_compute_all: bool = _env_bool("IND_PRECOMPUTE", False)
    precision_decimals: int = _env_int("IND_PRECISION", 4)


@dataclass(frozen=True, kw_only=True)
class AnalyzerConfig:
    default_depth_days: int = _env_int("ANALYZER_DEPTH_DAYS", 252)
    correlation_threshold: float = _env_float("ANALYZER_CORRELATION", 0.85)
    anomaly_threshold: float = _env_float("ANALYZER_ANOMALY", 3.0)


@dataclass(frozen=True, kw_only=True)
class AIConfig:
    provider: AIProvider = field(default_factory=lambda: _env_enum(AIProvider, "AI_PROVIDER", AIProvider.GEMINI))
    model_name: str = _env_str("AI_MODEL_NAME", "gemini-1.5-pro")
    api_key: str = field(default_factory=lambda: _env_str("AI_API_KEY", ""), repr=False)
    inference_timeout_sec: float = _env_float("AI_TIMEOUT", 60.0)
    max_tokens: int = _env_int("AI_MAX_TOKENS", 4096)
    confidence_threshold: float = _env_float("AI_CONFIDENCE", 0.80)

    def __post_init__(self):
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"AIConfig: confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")


@dataclass(frozen=True, kw_only=True)
class APIConfig:
    host: str = _env_str("API_HOST", "0.0.0.0")
    port: int = _env_int("API_PORT", 8000)
    timeout_sec: int = _env_int("API_TIMEOUT", 30)
    compression_enabled: bool = _env_bool("API_COMPRESSION", True)
    allowed_origins: Tuple[str, ...] = field(default_factory=lambda: _env_list("API_CORS_ORIGINS", ("*",)))

    def __post_init__(self):
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"APIConfig: Invalid port number {self.port}")


@dataclass(frozen=True, kw_only=True)
class PortfolioConfig:
    default_currency: str = "INR"
    broker_charge_pct: float = _env_float("PORTFOLIO_BROKER_PCT", 0.0003)
    tax_profile_stcg_pct: float = _env_float("PORTFOLIO_STCG_PCT", 0.15)
    tax_profile_ltcg_pct: float = _env_float("PORTFOLIO_LTCG_PCT", 0.10)
    risk_free_rate: float = _env_float("PORTFOLIO_RF_RATE", 0.07)


@dataclass(frozen=True, kw_only=True)
class DashboardConfig:
    refresh_interval_sec: int = _env_int("DASH_REFRESH_SEC", 15)
    default_chart_history_days: int = _env_int("DASH_CHART_DAYS", 365)
    theme: Theme = field(default_factory=lambda: _env_enum(Theme, "DASH_THEME", Theme.DARK))


# -------------------------------------------------------------------------
# GLOBAL SINGLETON SETTINGS
# -------------------------------------------------------------------------

class Settings:
    """
    Global Configuration Singleton.
    Provides immutable access to all configuration groups across the platform.
    Ensures strict thread-safety using a thread lock during instantiation.
    """
    _instance = None
    _lock = Lock()
    _locked = False

    def __new__(cls):
        # Double-checked locking pattern for Thread-Safe Singleton
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._initialize()
                    cls._instance._locked = True
        return cls._instance

    def _initialize(self) -> None:
        """Instantiates all configuration groups dynamically using object.__setattr__."""
        object.__setattr__(self, 'schema_version', CONFIG_SCHEMA_VERSION)
        object.__setattr__(self, 'app', ApplicationConfig())
        object.__setattr__(self, 'features', FeatureFlags())
        object.__setattr__(self, 'database', DatabaseConfig())
        object.__setattr__(self, 'logging', LoggingConfig())
        object.__setattr__(self, 'trace', TraceConfig())
        object.__setattr__(self, 'audit', AuditConfig())
        object.__setattr__(self, 'provider', ProviderConfig())
        object.__setattr__(self, 'sync', SyncConfig())
        object.__setattr__(self, 'scheduler', SchedulerConfig())
        object.__setattr__(self, 'event_bus', EventBusConfig())
        object.__setattr__(self, 'notification', NotificationConfig())
        object.__setattr__(self, 'market', MarketConfig())
        object.__setattr__(self, 'cache', CacheConfig())
        object.__setattr__(self, 'performance', PerformanceConfig())
        object.__setattr__(self, 'threading', ThreadConfig())
        object.__setattr__(self, 'retry', RetryConfig())
        object.__setattr__(self, 'security', SecurityConfig())
        object.__setattr__(self, 'indicator', IndicatorConfig())
        object.__setattr__(self, 'analyzer', AnalyzerConfig())
        object.__setattr__(self, 'ai', AIConfig())
        object.__setattr__(self, 'api', APIConfig())
        object.__setattr__(self, 'portfolio', PortfolioConfig())
        object.__setattr__(self, 'dashboard', DashboardConfig())

    @property
    def environment(self):
        """QA compatibility alias."""
        return self.app.environment

    @property
    def metrics(self):
        """QA compatibility alias."""
        return self.performance

    @cached_property
    def database_path(self) -> Path:
        """
        Resolves and caches the absolute path to the active SQLite database.
        mkdir is executed strictly once on the first property access.
        """
        db_dir = self.app.base_dir / self.database.sqlite_db_folder
        db_dir.mkdir(parents=True, exist_ok=True)
        return db_dir / self.database.sqlite_db_name

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Enforces immutability on the singleton container.
        Allows @cached_property to populate __dict__, but prevents manual overrides.
        """
        if getattr(self, "_locked", False):
            # @cached_property mechanisms write directly to __dict__ once initialized
            if name not in self.__dict__:
                super().__setattr__(name, value)
                return
            raise AttributeError(f"Global Settings are strictly read-only at runtime. Cannot set '{name}'.")
        super().__setattr__(name, value)


# -------------------------------------------------------------------------
# EXPORT GLOBAL INSTANCE
# -------------------------------------------------------------------------
# This export is the ONLY allowed entry point for configuration imports.
# Usage: from backend.config.settings import settings

settings: Final[Settings] = Settings()
