"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/database/migrations.py
Description: Enterprise-grade Database Migration Engine.
             Responsible for schema creation, upgrades, deterministic validation, 
             version management, dependency graph resolution, and history tracking.
             Features global execution locks, backup retention, and cyclic checks.
             Fully decoupled from business logic and strictly integrated with 
             Trace, Audit, Retry, and Logging pipelines.
"""

import json
import time
import uuid
import hashlib
import importlib.util
import inspect
import sqlite3
import threading
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Tuple, Final

from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.exceptions import MigrationError, SchemaError
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity
from backend.core.retry import retry, RetryStrategy
from backend.database.connection import (
    db_manager,
    DatabaseSession,
    IsolationLevel
)

# -------------------------------------------------------------------------
# CONSTANTS & LOGGER
# -------------------------------------------------------------------------

_logger = AppLogger("MigrationEngine")
MAX_BACKUP_RETENTION = 5


# -------------------------------------------------------------------------
# ENUMS
# -------------------------------------------------------------------------

class MigrationStatus(str, Enum):
    """Lifecycle tracking states for a distinct schema migration."""
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


class MigrationType(str, Enum):
    """Execution vector for the schema modification."""
    UPGRADE = "UPGRADE"
    DOWNGRADE = "DOWNGRADE"
    REPAIR = "REPAIR"


class SchemaObjectType(str, Enum):
    """Categorized schema structural entities."""
    TABLE = "TABLE"
    INDEX = "INDEX"
    VIEW = "VIEW"
    TRIGGER = "TRIGGER"
    COLUMN = "COLUMN"


# -------------------------------------------------------------------------
# IMMUTABLE DATACLASSES
# -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MigrationInfo:
    """Immutable metadata representing a registered migration definition."""
    version: int
    name: str
    description: str
    checksum: str
    dependencies: Tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "checksum": self.checksum,
            "dependencies": self.dependencies
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """Immutable representation of an actively applied schema version."""
    version: int
    checksum: str
    applied_at: str
    execution_time_ms: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "checksum": self.checksum,
            "applied_at": self.applied_at,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "description": self.description
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Immutable verdict of a completed or failed migration attempt."""
    version: int
    status: MigrationStatus
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True, slots=True)
class MigrationStatistics:
    """Immutable operational telemetry for a bulk migration run."""
    total_migrations: int
    successful: int
    failed: int
    total_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_migrations": self.total_migrations,
            "successful": self.successful,
            "failed": self.failed,
            "total_duration_ms": round(self.total_duration_ms, 3)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# -------------------------------------------------------------------------
# BASE MIGRATION CONTRACT
# -------------------------------------------------------------------------

class BaseMigration:
    """
    Abstract contract for all schema migrations.
    Every migration file (e.g., 0001_initial.py) must define a subclass of this.
    """
    version: int = 0
    name: str = "Unnamed Migration"
    description: str = "No description provided."
    dependencies: Tuple[int, ...] = ()

    @classmethod
    def up(cls) -> None:
        """Executes the forward schema transformation (DDL/DML)."""
        raise NotImplementedError("Migration up() must be implemented.")

    @classmethod
    def down(cls) -> None:
        """Executes the reverse schema transformation (Rollback)."""
        raise NotImplementedError("Migration down() must be implemented.")

    @classmethod
    def generate_checksum(cls) -> str:
        """
        Generates a deterministic SHA256 checksum identifying the migration's structure.
        Hashes the actual source code to prevent stealth modifications.
        """
        try:
            source = inspect.getsource(cls)
        except Exception:
            source = "Source extraction failed."
            
        raw_seed = f"{cls.version}|{cls.name}|{cls.description}|{cls.dependencies}|{source}"
        return hashlib.sha256(raw_seed.encode('utf-8')).hexdigest()

    @classmethod
    def get_info(cls) -> MigrationInfo:
        """Extracts the immutable operational metadata payload."""
        return MigrationInfo(
            version=cls.version,
            name=cls.name,
            description=cls.description,
            checksum=cls.generate_checksum(),
            dependencies=cls.dependencies
        )


# -------------------------------------------------------------------------
# MIGRATION ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class MigrationEngine:
    """
    Enterprise Migration Engine. Orchestrates schema definitions, 
    idempotent upgrades, automated backups, and structural validations.
    Operates strictly within transactional boundaries and telemetry pipelines.
    Enforces thread-safe global execution locks.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(MigrationEngine, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._registry: Dict[int, Type[BaseMigration]] = {}
        self._registry_lock = threading.Lock()
        
        # Global lock to prevent race conditions if multiple threads trigger migrations
        self._execution_lock = threading.Lock()
        
        self._backup_dir = settings.app.base_dir / "backups" / "database"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # DISCOVERY, REGISTRATION & GRAPH VALIDATION
    # -------------------------------------------------------------------------

    def register_migration(self, migration_cls: Type[BaseMigration]) -> None:
        """Registers a migration class securely into the O(1) tracking registry. Idempotent."""
        with self._registry_lock:
            if migration_cls.version in self._registry:
                existing = self._registry[migration_cls.version]
                if existing.generate_checksum() == migration_cls.generate_checksum():
                    return  
                raise MigrationError(f"Migration version collision detected for v{migration_cls.version}.")
                
            self._registry[migration_cls.version] = migration_cls
            _logger.debug(f"Registered migration v{migration_cls.version}: {migration_cls.name}")

    @trace_span(operation="migration.discover", component="migrations")
    def discover_migrations(self, directory_path: Path) -> None:
        """
        Dynamically scans and registers physical python migration files.
        Validates the dependency graph to prevent circular dependencies.
        """
        if not directory_path.exists() or not directory_path.is_dir():
            _logger.warning(f"Migration directory not found: {directory_path}")
            return

        py_files = sorted([f for f in directory_path.iterdir() if f.is_file() and f.suffix == '.py'])
        
        for file_path in py_files:
            module_name = f"migrations_{file_path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, BaseMigration) and obj is not BaseMigration:
                            self.register_migration(obj)
                            
            except Exception as e:
                TraceEngine.record_exception(e)
                raise MigrationError(f"Failed to dynamically load migration {file_path.name}: {e}") from e

        self._check_circular_dependencies()

    def _check_circular_dependencies(self) -> None:
        """DFS algorithm to detect cyclic dependencies in the migration graph."""
        visited = set()
        path = set()

        def visit(version: int) -> None:
            if version in path:
                raise SchemaError(f"Circular dependency detected involving migration v{version}.")
            if version in visited:
                return
            
            path.add(version)
            if version in self._registry:
                for dep in self._registry[version].dependencies:
                    visit(dep)
            path.remove(version)
            visited.add(version)

        with self._registry_lock:
            for v in self._registry.keys():
                visit(v)

    def _validate_dependencies_met(self, migration_cls: Type[BaseMigration], applied_versions: set) -> None:
        """Ensures all explicit dependencies have been applied prior to execution."""
        for dep in migration_cls.dependencies:
            if dep not in applied_versions:
                raise SchemaError(f"Dependency missing: v{migration_cls.version} requires v{dep} to be applied first.")

    # -------------------------------------------------------------------------
    # INFRASTRUCTURE SETUP
    # -------------------------------------------------------------------------

    @retry(operation_name="migration.initialize_database", max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_JITTER)
    @trace_span(operation="migration.initialize", component="migrations")
    def initialize_database(self) -> None:
        """Idempotently constructs the engine's internal tracking infrastructure."""
        try:
            with DatabaseSession(isolation=IsolationLevel.IMMEDIATE):
                with db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id TEXT PRIMARY KEY,
                        version INTEGER UNIQUE NOT NULL,
                        checksum TEXT NOT NULL,
                        applied_at TEXT NOT NULL,
                        execution_time_ms REAL NOT NULL,
                        description TEXT NOT NULL
                    );
                """): pass
                    
                with db_manager.execute("""
                    CREATE TABLE IF NOT EXISTS migration_history (
                        migration_id TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        duration_ms REAL,
                        checksum TEXT NOT NULL
                    );
                """): pass
                    
                with db_manager.execute("""
                    CREATE INDEX IF NOT EXISTS idx_migration_history_version 
                    ON migration_history(version);
                """): pass
                    
            _logger.info("Database migration infrastructure initialized successfully.")
            AuditEngine.record_success("migration.initialize", AuditAction.SYSTEM, "Migration tables created.")
        except Exception as e:
            TraceEngine.record_exception(e)
            raise MigrationError(f"Critical failure initializing migration tables: {e}") from e

    # -------------------------------------------------------------------------
    # STATE ENQUIRY (O(1) Operations)
    # -------------------------------------------------------------------------

    def current_version(self) -> int:
        """Retrieves the highest applied schema version from the active database."""
        try:
            result = db_manager.fetch_scalar("SELECT MAX(version) FROM schema_version;")
            return result if result is not None else 0
        except Exception:
            return 0

    def latest_version(self) -> int:
        """Retrieves the highest registered schema version from the codebase."""
        with self._registry_lock:
            if not self._registry:
                return 0
            return max(self._registry.keys())

    def _object_exists(self, obj_type: str, name: str) -> bool:
        """Generic existence validator against the SQLite master schema table."""
        sql = "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?;"
        return db_manager.fetch_scalar(sql, (obj_type, name)) is not None

    def schema_exists(self) -> bool:
        return self._object_exists("table", "schema_version")

    def table_exists(self, table_name: str) -> bool:
        return self._object_exists("table", table_name)

    def view_exists(self, view_name: str) -> bool:
        return self._object_exists("view", view_name)

    def trigger_exists(self, trigger_name: str) -> bool:
        return self._object_exists("trigger", trigger_name)

    def index_exists(self, index_name: str) -> bool:
        return self._object_exists("index", index_name)

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Validates specific column presence utilizing PRAGMA table_info."""
        if not self.table_exists(table_name):
            return False
        rows = db_manager.fetch_all(f"PRAGMA table_info({table_name});")
        return any(row['name'] == column_name for row in rows)

    def _get_schema_metrics(self) -> Tuple[int, int]:
        """Helper to extract raw schema counts for telemetry."""
        tables = db_manager.fetch_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='table';") or 0
        indexes = db_manager.fetch_scalar("SELECT COUNT(*) FROM sqlite_master WHERE type='index';") or 0
        return tables, indexes

    # -------------------------------------------------------------------------
    # STRUCTURAL VALIDATION
    # -------------------------------------------------------------------------

    @trace_span(operation="migration.validate_schema", component="migrations")
    def validate_schema(self) -> bool:
        """Executes strict native SQLite integrity and foreign key constraint checks."""
        try:
            is_valid = True
            
            integrity_result = db_manager.fetch_all("PRAGMA integrity_check;")
            if not integrity_result or integrity_result[0][0].lower() != "ok":
                errors = [r[0] for r in integrity_result]
                _logger.error("Schema Integrity Validation FAILED.", metadata={"errors": errors})
                AuditEngine.record_failure("migration.validate", AuditAction.SYSTEM, "PRAGMA integrity_check failed.")
                is_valid = False

            fk_result = db_manager.fetch_all("PRAGMA foreign_key_check;")
            if fk_result:
                violations = [{"table": r[0], "rowid": r[1], "parent": r[2], "fkid": r[3]} for r in fk_result]
                _logger.error("Foreign Key Validation FAILED.", metadata={"violations": violations})
                AuditEngine.record_failure("migration.validate", AuditAction.SYSTEM, "PRAGMA foreign_key_check failed.")
                is_valid = False

            if is_valid:
                _logger.info("Schema Validation PASSED successfully.")
                AuditEngine.record_success("migration.validate", AuditAction.SYSTEM, "Schema integrity verified.")
            
            return is_valid
            
        except Exception as e:
            TraceEngine.record_exception(e)
            raise SchemaError(f"Validation engine encountered a critical error: {e}") from e

    @trace_span(operation="migration.verify_checksums", component="migrations")
    def verify_checksums(self) -> None:
        """Cryptographically verifies historical migrations against codebase hashes."""
        applied = db_manager.fetch_all("SELECT version, checksum FROM schema_version;")
        
        with self._registry_lock:
            for row in applied:
                version = row['version']
                applied_hash = row['checksum']
                
                if version in self._registry:
                    codebase_hash = self._registry[version].generate_checksum()
                    if codebase_hash != applied_hash:
                        error_msg = f"Checksum mismatch for Migration v{version}. Database compromised or codebase altered."
                        _logger.critical(error_msg)
                        AuditEngine.record_failure("migration.checksum", AuditAction.SYSTEM, error_msg, AuditSeverity.CRITICAL)
                        raise SchemaError(error_msg)
                        
        _logger.debug("Cryptographic migration checksum verification passed.")

    # -------------------------------------------------------------------------
    # DATA SAFEGUARDS (BACKUP & RETENTION)
    # -------------------------------------------------------------------------

    @trace_span(operation="migration.backup", component="migrations")
    def backup_before_upgrade(self) -> Tuple[Path, str]:
        """
        Automates native SQLite physical backup, applies retention limits,
        and generates a verifiable SHA256 checksum of the backup file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = self._backup_dir / f"pre_upgrade_backup_{timestamp}.sqlite3"
        
        try:
            source_conn = db_manager.get_connection()
            with closing(sqlite3.connect(backup_path.as_posix())) as backup_conn:
                source_conn.backup(backup_conn)
                
            # Compute File Checksum
            sha256_hash = hashlib.sha256()
            with open(backup_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            backup_checksum = sha256_hash.hexdigest()

            TraceEngine.attach_metadata("backup_destination", backup_path.name)
            TraceEngine.attach_metadata("backup_checksum", backup_checksum)
            
            # Retention Policy Enforcement
            all_backups = sorted([f for f in self._backup_dir.iterdir() if f.is_file()], key=os.path.getmtime)
            if len(all_backups) > MAX_BACKUP_RETENTION:
                for old_backup in all_backups[:-MAX_BACKUP_RETENTION]:
                    old_backup.unlink(missing_ok=True)
                TraceEngine.attach_metadata("backups_pruned", len(all_backups) - MAX_BACKUP_RETENTION)

            _logger.info(f"Pre-upgrade database backup generated securely at {backup_path.name}")
            AuditEngine.record_success("migration.backup", AuditAction.SYSTEM, "Backup and retention policy completed.")
            return backup_path, backup_checksum
            
        except Exception as e:
            TraceEngine.record_exception(e)
            raise MigrationError(f"Failed to generate safety backup: {e}") from e

    # -------------------------------------------------------------------------
    # CORE EXECUTION MECHANICS
    # -------------------------------------------------------------------------

    def _execute_migration(self, migration_cls: Type[BaseMigration], direction: MigrationType) -> MigrationResult:
        """
        Isolated execution bounded securely within a manual SQLite SAVEPOINT transaction.
        Handles cursor lifecycle properly using the Context Manager.
        """
        info = migration_cls.get_info()
        migration_id = uuid.uuid4().hex
        start_time_perf = time.perf_counter()
        start_time_utc = datetime.now(timezone.utc).isoformat()
        
        TraceEngine.attach_tags({"migration.version": str(info.version), "migration.direction": direction.value})

        with db_manager.execute(
            """INSERT INTO migration_history 
               (migration_id, version, status, started_at, checksum) 
               VALUES (?, ?, ?, ?, ?)""",
            (migration_id, info.version, MigrationStatus.RUNNING.value, start_time_utc, info.checksum)
        ):
            pass

        try:
            with DatabaseSession(isolation=IsolationLevel.IMMEDIATE):
                if direction == MigrationType.UPGRADE:
                    migration_cls.up()
                elif direction == MigrationType.DOWNGRADE:
                    migration_cls.down()
                else:
                    raise MigrationError(f"Unsupported migration direction: {direction.value}")

                duration_ms = (time.perf_counter() - start_time_perf) * 1000.0
                end_time_utc = datetime.now(timezone.utc).isoformat()

                if direction == MigrationType.UPGRADE:
                    with db_manager.execute(
                        """INSERT INTO schema_version (id, version, checksum, applied_at, execution_time_ms, description)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (uuid.uuid4().hex, info.version, info.checksum, end_time_utc, duration_ms, info.description)
                    ):
                        pass
                elif direction == MigrationType.DOWNGRADE:
                    with db_manager.execute("DELETE FROM schema_version WHERE version = ?;", (info.version,)):
                        pass

                with db_manager.execute(
                    """UPDATE migration_history 
                       SET status = ?, finished_at = ?, duration_ms = ? 
                       WHERE migration_id = ?""",
                    (MigrationStatus.SUCCESS.value, end_time_utc, duration_ms, migration_id)
                ):
                    pass

            tables, indexes = self._get_schema_metrics()
            TraceEngine.attach_metadata("table_count", tables)
            TraceEngine.attach_metadata("index_count", indexes)

            _logger.info(f"Migration v{info.version} [{direction.value}] completed in {duration_ms:.2f}ms.")
            AuditEngine.record_success(f"migration.{direction.value.lower()}", AuditAction.UPDATE, f"v{info.version} applied.")
            
            return MigrationResult(version=info.version, status=MigrationStatus.SUCCESS, duration_ms=duration_ms)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time_perf) * 1000.0
            end_time_utc = datetime.now(timezone.utc).isoformat()
            
            with db_manager.execute(
                """UPDATE migration_history 
                   SET status = ?, finished_at = ?, duration_ms = ? 
                   WHERE migration_id = ?""",
                (MigrationStatus.FAILED.value, end_time_utc, duration_ms, migration_id)
            ):
                pass
            
            TraceEngine.record_exception(e)
            _logger.error(f"Migration v{info.version} [{direction.value}] FAILED.", exc_info=e)
            AuditEngine.record_failure(f"migration.{direction.value.lower()}", AuditAction.UPDATE, f"v{info.version} failed.", AuditSeverity.CRITICAL)
            
            return MigrationResult(version=info.version, status=MigrationStatus.FAILED, duration_ms=duration_ms, error=str(e))

    @retry(operation_name="migration.upgrade_schema", max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_JITTER)
    @trace_span(operation="migration.upgrade", component="migrations")
    def upgrade_schema(self, target_version: Optional[int] = None) -> MigrationStatistics:
        """
        Advances the database schema safely to the target version (or latest).
        Enforces Thread-Safe global locking to prevent simultaneous execution.
        """
        with self._execution_lock:
            if not self.schema_exists():
                self.initialize_database()

            self.verify_checksums()
            
            current = self.current_version()
            target = target_version if target_version is not None else self.latest_version()

            TraceEngine.attach_metadata("current_version", current)
            TraceEngine.attach_metadata("target_version", target)

            if current >= target:
                _logger.info("Database schema is already up to date.")
                return MigrationStatistics(0, 0, 0, 0.0)

            with self._registry_lock:
                pending = [cls for ver, cls in self._registry.items() if current < ver <= target]
            
            pending.sort(key=lambda x: x.version)

            if not pending:
                return MigrationStatistics(0, 0, 0, 0.0)

            # Retrieve all applied versions for dependency validation
            applied_rows = db_manager.fetch_all("SELECT version FROM schema_version;")
            applied_versions = {r['version'] for r in applied_rows}

            backup_path, backup_chk = self.backup_before_upgrade()

            successful, failed = 0, 0
            start_time_perf = time.perf_counter()

            for migration_cls in pending:
                self._validate_dependencies_met(migration_cls, applied_versions)
                
                result = self._execute_migration(migration_cls, MigrationType.UPGRADE)
                if result.status == MigrationStatus.SUCCESS:
                    successful += 1
                    applied_versions.add(migration_cls.version)
                else:
                    failed += 1
                    break  

            total_duration = (time.perf_counter() - start_time_perf) * 1000.0
            
            stats = MigrationStatistics(
                total_migrations=len(pending),
                successful=successful,
                failed=failed,
                total_duration_ms=total_duration
            )
            
            TraceEngine.attach_metadata("successful_migrations", successful)
            TraceEngine.attach_metadata("failed_migrations", failed)
            TraceEngine.attach_metadata("total_duration_ms", round(total_duration, 3))
            
            _logger.info("Schema upgrade sequence finalized.", metadata=stats.to_dict())
            return stats

    @retry(operation_name="migration.downgrade_schema", max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_JITTER)
    @trace_span(operation="migration.downgrade", component="migrations")
    def downgrade_schema(self, target_version: int) -> MigrationStatistics:
        """Reverses the database schema deterministically to a lower target version."""
        with self._execution_lock:
            if not self.schema_exists():
                raise MigrationError("Migration infrastructure missing. Cannot downgrade.")

            current = self.current_version()
            TraceEngine.attach_metadata("current_version", current)
            TraceEngine.attach_metadata("target_version", target_version)

            if target_version >= current:
                _logger.warning("Target version must be strictly less than current version for downgrade.")
                return MigrationStatistics(0, 0, 0, 0.0)

            with self._registry_lock:
                revertable = [cls for ver, cls in self._registry.items() if target_version < ver <= current]
            
            revertable.sort(key=lambda x: x.version, reverse=True)

            if not revertable:
                return MigrationStatistics(0, 0, 0, 0.0)

            self.backup_before_upgrade()

            successful, failed = 0, 0
            start_time_perf = time.perf_counter()

            for migration_cls in revertable:
                result = self._execute_migration(migration_cls, MigrationType.DOWNGRADE)
                if result.status == MigrationStatus.SUCCESS:
                    successful += 1
                else:
                    failed += 1
                    break 

            total_duration = (time.perf_counter() - start_time_perf) * 1000.0
            
            stats = MigrationStatistics(
                total_migrations=len(revertable),
                successful=successful,
                failed=failed,
                total_duration_ms=total_duration
            )
            
            TraceEngine.attach_metadata("successful_migrations", successful)
            TraceEngine.attach_metadata("failed_migrations", failed)
            TraceEngine.attach_metadata("total_duration_ms", round(total_duration, 3))
            
            _logger.warning("Schema downgrade sequence finalized.", metadata=stats.to_dict())
            return stats

    def create_schema(self) -> MigrationStatistics:
        return self.upgrade_schema()

    def run_migrations(self) -> MigrationStatistics:
        return self.upgrade_schema()

    @trace_span(operation="migration.repair", component="migrations")
    def repair_schema(self) -> None:
        """Purges orphaned locks resulting from fatal crashes during schema alterations."""
        with self._execution_lock:
            if not self.schema_exists():
                return
                
            try:
                with DatabaseSession():
                    with db_manager.execute(
                        """UPDATE migration_history 
                           SET status = 'FAILED' 
                           WHERE status = 'RUNNING';"""
                    ):
                        pass
                
                self.validate_schema()
                self.verify_checksums()
                _logger.info("Schema repair maintenance routine completed securely.")
                
            except Exception as e:
                TraceEngine.record_exception(e)
                raise MigrationError(f"Failed to execute schema repair routines: {e}") from e

    def rollback_failed_migration(self) -> None:
        self.repair_schema()

    def database_statistics(self) -> Dict[str, Any]:
        """Provides operational metrics surrounding the active schema framework."""
        if not self.schema_exists():
            return {"status": "UNINITIALIZED"}
            
        total_applied = db_manager.fetch_scalar("SELECT COUNT(*) FROM schema_version;") or 0
        history_count = db_manager.fetch_scalar("SELECT COUNT(*) FROM migration_history;") or 0
        failures = db_manager.fetch_scalar("SELECT COUNT(*) FROM migration_history WHERE status = 'FAILED';") or 0
        
        return {
            "status": "INITIALIZED",
            "current_version": self.current_version(),
            "latest_available_version": self.latest_version(),
            "total_applied": total_applied,
            "total_execution_history": history_count,
            "recorded_failures": failures
        }


# -------------------------------------------------------------------------
# EXPORT GLOBAL INSTANCE
# -------------------------------------------------------------------------

migration_engine: Final[MigrationEngine] = MigrationEngine()

__all__ = [
    "MigrationStatus",
    "MigrationType",
    "SchemaObjectType",
    "MigrationInfo",
    "SchemaVersion",
    "MigrationResult",
    "MigrationStatistics",
    "BaseMigration",
    "MigrationEngine",
    "migration_engine"
]
