"""
GREEN BULL RIDER V6
Module: backend/config/settings.py

Production Configuration Layer
Python 3.13 Compatible
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from dataclasses import field
from datetime import time
from enum import Enum
from functools import cached_property
from pathlib import Path
from threading import Lock
from typing import Any
from typing import Final
from typing import Tuple
from typing import Type


# =============================================================================
# Metadata
# =============================================================================

CONFIG_SCHEMA_VERSION: Final[str] = "6.0.0"


# =============================================================================
# Helpers
# =============================================================================


def _project_root() -> Path:
    """
    Resolve project root automatically.
    """

    current = Path(__file__).resolve().parent

    while current != current.parent:

        if (current / "backend").exists():
            return current

        current = current.parent

    return Path(__file__).resolve().parent.parent.parent


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:

    try:
        return int(os.getenv(key, str(default)))

    except Exception:
        return default


def _env_float(key: str, default: float) -> float:

    try:
        return float(os.getenv(key, str(default)))

    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:

    value = os.getenv(key)

    if value is None:
        return default

    return value.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _env_list(
    key: str,
    default: Tuple[str, ...],
) -> Tuple[str, ...]:

    value = os.getenv(key)

    if not value:
        return default

    return tuple(
        x.strip()
        for x in value.split(",")
        if x.strip()
    )


def _env_time(
    key: str,
    default_hour: int,
    default_minute: int,
) -> time:

    value = os.getenv(key)

    if value:

        try:

            h, m = value.split(":")

            return time(
                hour=int(h),
                minute=int(m),
            )

        except Exception:
            pass

    return time(
        hour=default_hour,
        minute=default_minute,
    )


def _env_enum(
    enum_cls: Type[Enum],
    key: str,
    default: Enum,
) -> Enum:

    value = os.getenv(key)

    if not value:
        return default

    try:

        return enum_cls(
            value.upper(),
        )

    except Exception:

        return default


# =============================================================================
# Enums
# =============================================================================


class Environment(Enum):

    DEVELOPMENT = "DEVELOPMENT"

    STAGING = "STAGING"

    PRODUCTION = "PRODUCTION"


class DatabaseEngine(Enum):

    SQLITE = "SQLITE"

    POSTGRESQL = "POSTGRESQL"


class JournalMode(Enum):

    DELETE = "DELETE"

    WAL = "WAL"

    MEMORY = "MEMORY"


class SynchronousMode(Enum):

    OFF = 0

    NORMAL = 1

    FULL = 2


class LogLevel(Enum):

    DEBUG = "DEBUG"

    INFO = "INFO"

    WARNING = "WARNING"

    ERROR = "ERROR"

    CRITICAL = "CRITICAL"


class AIProvider(Enum):

    GEMINI = "GEMINI"

    OPENAI = "OPENAI"

    ANTHROPIC = "ANTHROPIC"

    LOCAL = "LOCAL"


# =============================================================================
# Application
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class ApplicationConfig:

    name: str = "GREEN BULL RIDER V6"

    version: str = "6.0.0"

    environment: Environment = field(
        default_factory=lambda: _env_enum(
            Environment,
            "APP_ENV",
            Environment.PRODUCTION,
        )
    )

    project_root: Path = field(
        default_factory=_project_root,
    )


# =============================================================================
# Database
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class DatabaseConfig:
    """
    Green-Bull-Data-Engine Database
    """

    engine: DatabaseEngine = field(
        default_factory=lambda: _env_enum(
            DatabaseEngine,
            "DB_ENGINE",
            DatabaseEngine.SQLITE,
        )
    )

    data_engine_root: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "DATA_ENGINE_ROOT",
                str(
                    _project_root().parent
                    / "Green-Bull-Data-Engine"
                ),
            )
        ).expanduser().resolve()
    )

    database_name: str = _env_str(
        "SQLITE_DB_NAME",
        "market.db",
    )

    busy_timeout_ms: int = _env_int(
        "DB_BUSY_TIMEOUT_MS",
        30000,
    )

    journal_mode: JournalMode = field(
        default_factory=lambda: _env_enum(
            JournalMode,
            "DB_JOURNAL",
            JournalMode.WAL,
        )
    )

    synchronous_mode: SynchronousMode = field(
        default_factory=lambda: _env_enum(
            SynchronousMode,
            "DB_SYNC",
            SynchronousMode.NORMAL,
        )
    )

    foreign_keys: bool = True

    temp_store_memory: bool = True

    mmap_size: int = _env_int(
        "DB_MMAP_SIZE",
        268435456,
    )

    @cached_property
    def database_path(self) -> Path:

        return (
            self.data_engine_root
            / "database"
            / self.database_name
        )

    def __post_init__(self):

        if self.busy_timeout_ms < 0:

            raise ValueError(
                "busy_timeout_ms cannot be negative."
            )

        if self.mmap_size < 0:

            raise ValueError(
                "mmap_size cannot be negative."
            )


# =============================================================================
# Logging
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class LoggingConfig:

    level: LogLevel = field(
        default_factory=lambda: _env_enum(
            LogLevel,
            "LOG_LEVEL",
            LogLevel.INFO,
        )
    )

    log_directory: Path = field(
        default_factory=lambda: (
            _project_root() / "logs"
        )
    )

    enable_console: bool = _env_bool(
        "LOG_CONSOLE",
        True,
    )

    enable_file: bool = _env_bool(
        "LOG_FILE",
        True,
    )

    rotation_time: time = field(
        default_factory=lambda: _env_time(
            "LOG_ROTATION",
            0,
            0,
        )
    )

    retention_days: int = _env_int(
        "LOG_RETENTION_DAYS",
        30,
    )

    max_file_size_mb: int = _env_int(
        "LOG_MAX_FILE_SIZE_MB",
        100,
    )


# =============================================================================
# Provider
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class ProviderConfig:

    provider_name: str = _env_str(
        "PROVIDER_NAME",
        "Yahoo",
    )

    timeout: float = _env_float(
        "PROVIDER_TIMEOUT",
        20.0,
    )

    max_retries: int = _env_int(
        "PROVIDER_MAX_RETRIES",
        5,
    )

    retry_backoff: float = _env_float(
        "PROVIDER_BACKOFF",
        2.0,
    )

    max_connections: int = _env_int(
        "PROVIDER_MAX_CONNECTIONS",
        50,
    )

    requests_per_second: int = _env_int(
        "PROVIDER_RPS",
        5,
    )

    user_agent: str = _env_str(
        "PROVIDER_USER_AGENT",
        "GreenBullRiderV6/6.0",
    )

    def __post_init__(self):

        if self.timeout <= 0:

            raise ValueError(
                "timeout must be positive."
            )


# =============================================================================
# Performance
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class PerformanceConfig:

    batch_size: int = _env_int(
        "PERF_BATCH_SIZE",
        1000,
    )

    chunk_size: int = _env_int(
        "PERF_CHUNK_SIZE",
        500,
    )

    max_workers: int = _env_int(
        "PERF_MAX_WORKERS",
        8,
    )

    memory_limit_mb: int = _env_int(
        "PERF_MEMORY_MB",
        4096,
    )

    sqlite_cache_mb: int = _env_int(
        "SQLITE_CACHE_MB",
        64,
    )

    def __post_init__(self):

        if self.batch_size < 1:

            raise ValueError(
                "batch_size must be >=1."
            )


# =============================================================================
# Retry
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class RetryConfig:

    retries: int = _env_int(
        "RETRY_COUNT",
        3,
    )

    base_delay: float = _env_float(
        "RETRY_BASE_DELAY",
        2.0,
    )

    max_delay: float = _env_float(
        "RETRY_MAX_DELAY",
        60.0,
    )


# =============================================================================
# Feature Flags
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class FeatureFlags:

    enable_ai: bool = _env_bool(
        "ENABLE_AI",
        True,
    )

    enable_cache: bool = _env_bool(
        "ENABLE_CACHE",
        True,
    )

    enable_trace: bool = _env_bool(
        "ENABLE_TRACE",
        True,
    )

    enable_audit: bool = _env_bool(
        "ENABLE_AUDIT",
        True,
    )

    enable_parallel: bool = _env_bool(
        "ENABLE_PARALLEL",
        True,
    )


# =============================================================================
# Security
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class SecurityConfig:

    strict_validation: bool = _env_bool(
        "STRICT_VALIDATION",
        True,
    )

    read_only_mode: bool = _env_bool(
        "READ_ONLY_MODE",
        False,
    )

    checksum_validation: bool = _env_bool(
        "CHECKSUM_VALIDATION",
        True,
    )


# =============================================================================
# Synchronization
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class SyncConfig:

    auto_daily_sync: bool = _env_bool(
        "AUTO_DAILY_SYNC",
        True,
    )

    auto_weekly_sync: bool = _env_bool(
        "AUTO_WEEKLY_SYNC",
        True,
    )

    history_days: int = _env_int(
        "SYNC_HISTORY_DAYS",
        400,
    )

    trading_candles: int = _env_int(
        "SYNC_TRADING_CANDLES",
        250,
    )

    incremental_sync: bool = _env_bool(
        "INCREMENTAL_SYNC",
        True,
    )

    resume_after_failure: bool = _env_bool(
        "RESUME_AFTER_FAILURE",
        True,
    )

    max_parallel_symbols: int = _env_int(
        "MAX_PARALLEL_SYMBOLS",
        10,
    )


# =============================================================================
# AI
# =============================================================================


@dataclass(
    frozen=True,
    kw_only=True,
)
class AIConfig:

    provider: AIProvider = field(
        default_factory=lambda: _env_enum(
            AIProvider,
            "AI_PROVIDER",
            AIProvider.GEMINI,
        )
    )

    model: str = _env_str(
        "AI_MODEL",
        "gemini-2.5-pro",
    )

    api_key: str = field(
        default_factory=lambda: _env_str(
            "AI_API_KEY",
            "",
        ),
        repr=False,
    )

    timeout: float = _env_float(
        "AI_TIMEOUT",
        60.0,
    )

    max_tokens: int = _env_int(
        "AI_MAX_TOKENS",
        4096,
    )

    temperature: float = _env_float(
        "AI_TEMPERATURE",
        0.20,
    )

    confidence_threshold: float = _env_float(
        "AI_CONFIDENCE_THRESHOLD",
        0.80,
    )

    def __post_init__(self):

        if not 0.0 <= self.confidence_threshold <= 1.0:

            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )


# =============================================================================
# Settings Singleton
# =============================================================================


class Settings:

    _instance = None

    _lock = Lock()

    _initialized = False

    def __new__(cls):

        if cls._instance is None:

            with cls._lock:

                if cls._instance is None:

                    cls._instance = super().__new__(cls)

                    cls._instance._initialize()

        return cls._instance

    def _initialize(self):

        if self._initialized:
            return

        object.__setattr__(
            self,
            "schema_version",
            CONFIG_SCHEMA_VERSION,
        )

        object.__setattr__(
            self,
            "app",
            ApplicationConfig(),
        )

        object.__setattr__(
            self,
            "database",
            DatabaseConfig(),
        )

        object.__setattr__(
            self,
            "logging",
            LoggingConfig(),
        )

        object.__setattr__(
            self,
            "provider",
            ProviderConfig(),
        )

        object.__setattr__(
            self,
            "performance",
            PerformanceConfig(),
        )

        object.__setattr__(
            self,
            "retry",
            RetryConfig(),
        )

        object.__setattr__(
            self,
            "features",
            FeatureFlags(),
        )

        object.__setattr__(
            self,
            "security",
            SecurityConfig(),
        )

        object.__setattr__(
            self,
            "sync",
            SyncConfig(),
        )

        object.__setattr__(
            self,
            "ai",
            AIConfig(),
        )

        object.__setattr__(
            self,
            "_initialized",
            True,
        )

    @property
    def project_root(self) -> Path:

        return self.app.project_root

    @property
    def database_path(self) -> Path:

        return self.database.database_path

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> None:

        raise AttributeError(
            "Settings are read-only."
        )


# =============================================================================
# Global Instance
# =============================================================================

settings: Final[Settings] = Settings()

