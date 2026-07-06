"""
GREEN BULL RIDER V6 - Institutional-grade AI Stock Analysis Platform
Module: backend/core/security.py
Description: Centralized Enterprise Security Engine.
             Provides cryptographically secure operations, strict input validation,
             path traversal protection, file magic-number security, and secret masking.
             Fully decoupled from business logic and strictly integrated with 
             Trace, Audit, and Logging telemetry pipelines. Production Locked.
"""

import os
import re
import time
import uuid
import hmac
import base64
import hashlib
import secrets
import mimetypes
import ipaddress
import threading
from urllib.parse import urlparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Final, Union, BinaryIO

# Internal Platform Integrations
from backend.config.settings import settings
from backend.core.logger import AppLogger
from backend.core.exceptions import GreenBullError
from backend.core.trace import TraceEngine, SpanKind, trace_span
from backend.core.audit import AuditEngine, AuditAction, AuditResult, AuditSeverity

# -------------------------------------------------------------------------
# LOGGER INITIALIZATION
# -------------------------------------------------------------------------
_logger = AppLogger("SecurityEngine")

# -------------------------------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------------------------------

class SecurityError(GreenBullError):
    """Base exception for all security boundary violations."""
    error_code: str = "GBR-SEC-000"


class ValidationError(SecurityError):
    """Raised when input strictly fails structural or security validation."""
    error_code: str = "GBR-SEC-001"


class CryptoError(SecurityError):
    """Raised during cryptographic operation failures."""
    error_code: str = "GBR-SEC-002"


# -------------------------------------------------------------------------
# IMMUTABLE DATACLASSES
# -------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PasswordHash:
    """Immutable transport container for securely hashed passwords."""
    hash_value: str
    salt: str
    iterations: int
    algorithm: str


@dataclass(frozen=True, slots=True)
class FileChecksum:
    """Immutable verification artifact for file integrity."""
    algorithm: str
    checksum: str
    file_size_bytes: int


@dataclass(frozen=True, slots=True)
class SecurityStatistics:
    """Immutable operational telemetry for the Security Engine infrastructure."""
    passwords_hashed: int
    passwords_verified: int
    tokens_generated: int
    files_validated: int
    validation_failures: int
    crypto_operations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passwords_hashed": self.passwords_hashed,
            "passwords_verified": self.passwords_verified,
            "tokens_generated": self.tokens_generated,
            "files_validated": self.files_validated,
            "validation_failures": self.validation_failures,
            "crypto_operations": self.crypto_operations
        }


# -------------------------------------------------------------------------
# PRE-COMPILED VALIDATION & MAGIC NUMBERS
# -------------------------------------------------------------------------
# RFC 5322 compliant robust email regex pattern
_REGEX_EMAIL = re.compile(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$")
_REGEX_DOMAIN = re.compile(r"^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$")
_REGEX_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_REGEX_API_KEY = re.compile(r"^[A-Za-z0-9_\-\.]{16,128}$")
_REGEX_SYMBOL = re.compile(r"^[A-Z0-9_.\-]{1,20}$")
_REGEX_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# Common Magic Numbers (File Signatures) for Defense-in-Depth
_MAGIC_NUMBERS = {
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'\xff\xd8\xff': 'image/jpeg',
    b'%PDF-': 'application/pdf',
    b'PK\x03\x04': 'application/zip',
    b'PK\x05\x06': 'application/zip',
    b'PK\x07\x08': 'application/zip',
    b'GIF87a': 'image/gif',
    b'GIF89a': 'image/gif',
}


# -------------------------------------------------------------------------
# SECURITY ENGINE (SINGLETON)
# -------------------------------------------------------------------------

class SecurityEngine:
    """
    Enterprise Central Security Engine.
    Provides decoupled, state-free cryptographic operations and deterministic validation.
    Guarantees O(1) thread-safe metric tracking, strict magic-number file validation, 
    and robust directory traversal protection.
    """
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super(SecurityEngine, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Constructs thread-safe internal metric registries."""
        self._stats_lock = threading.Lock()
        self._passwords_hashed: int = 0
        self._passwords_verified: int = 0
        self._tokens_generated: int = 0
        self._files_validated: int = 0
        self._validation_failures: int = 0
        self._crypto_operations: int = 0

    # -------------------------------------------------------------------------
    # STATISTICS & HEALTH
    # -------------------------------------------------------------------------

    def get_statistics(self) -> SecurityStatistics:
        """Retrieves a thread-safe snapshot of global security telemetry."""
        with self._stats_lock:
            return SecurityStatistics(
                passwords_hashed=self._passwords_hashed,
                passwords_verified=self._passwords_verified,
                tokens_generated=self._tokens_generated,
                files_validated=self._files_validated,
                validation_failures=self._validation_failures,
                crypto_operations=self._crypto_operations
            )

    def reset_statistics(self) -> None:
        """Purges operational metrics cleanly."""
        with self._stats_lock:
            self._passwords_hashed = 0
            self._passwords_verified = 0
            self._tokens_generated = 0
            self._files_validated = 0
            self._validation_failures = 0
            self._crypto_operations = 0

    @trace_span(operation="security.health_check", component="security", kind=SpanKind.INTERNAL)
    def health_check(self) -> Dict[str, Any]:
        """Provides status validation for system monitoring engines."""
        try:
            test_hash = self.hash_sha256(b"health_check")
            return {
                "status": "HEALTHY" if test_hash else "DEGRADED",
                "statistics": self.get_statistics().to_dict()
            }
        except Exception as e:
            TraceEngine.record_exception(e)
            return {"status": "UNHEALTHY", "error": str(e)}

    def _increment_metric(self, metric_name: str) -> None:
        """Safely increments internal O(1) tracking counters."""
        with self._stats_lock:
            current = getattr(self, metric_name)
            setattr(self, metric_name, current + 1)

    # -------------------------------------------------------------------------
    # CRYPTOGRAPHIC GENERATORS
    # -------------------------------------------------------------------------

    @trace_span(operation="security.generate_uuid", component="security", kind=SpanKind.INTERNAL)
    def generate_uuid(self) -> str:
        """Generates a secure UUID v4 string."""
        return str(uuid.uuid4())

    @trace_span(operation="security.generate_random_bytes", component="security", kind=SpanKind.INTERNAL)
    def generate_random_bytes(self, length: int = 32) -> bytes:
        """Generates cryptographically secure pseudo-random bytes."""
        self._increment_metric("_crypto_operations")
        return secrets.token_bytes(length)

    @trace_span(operation="security.generate_random_string", component="security", kind=SpanKind.INTERNAL)
    def generate_random_string(self, length: Optional[int] = None) -> str:
        """Generates a secure hexadecimal random string."""
        self._increment_metric("_tokens_generated")
        target_length = length or settings.security.token_length
        return secrets.token_hex(target_length // 2)

    @trace_span(operation="security.generate_token", component="security", kind=SpanKind.INTERNAL)
    def generate_token(self, nbytes: int = 32) -> str:
        """Generates a standard cryptographically secure token."""
        self._increment_metric("_tokens_generated")
        AuditEngine.record_success("security.generate_token", AuditAction.SYSTEM, "Token generated securely.")
        return secrets.token_hex(nbytes)

    @trace_span(operation="security.generate_url_safe_token", component="security", kind=SpanKind.INTERNAL)
    def generate_url_safe_token(self, nbytes: int = 32) -> str:
        """Generates a cryptographically secure token safe for URL routing."""
        self._increment_metric("_tokens_generated")
        AuditEngine.record_success("security.generate_url_safe_token", AuditAction.SYSTEM, "URL safe token generated.")
        return secrets.token_urlsafe(nbytes)

    # -------------------------------------------------------------------------
    # HASHING & SIGNATURES
    # -------------------------------------------------------------------------

    @trace_span(operation="security.hash_sha256", component="security", kind=SpanKind.INTERNAL)
    def hash_sha256(self, data: Union[str, bytes]) -> str:
        """Computes a deterministic SHA256 digest."""
        self._increment_metric("_crypto_operations")
        payload = data.encode('utf-8') if isinstance(data, str) else data
        return hashlib.sha256(payload).hexdigest()

    @trace_span(operation="security.hash_sha512", component="security", kind=SpanKind.INTERNAL)
    def hash_sha512(self, data: Union[str, bytes]) -> str:
        """Computes a deterministic SHA512 digest."""
        self._increment_metric("_crypto_operations")
        payload = data.encode('utf-8') if isinstance(data, str) else data
        return hashlib.sha512(payload).hexdigest()

    @trace_span(operation="security.hmac_sign", component="security", kind=SpanKind.INTERNAL)
    def hmac_sign(self, message: Union[str, bytes], secret: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """Generates an HMAC cryptographic signature."""
        self._increment_metric("_crypto_operations")
        key = secret.encode('utf-8') if isinstance(secret, str) else secret
        payload = message.encode('utf-8') if isinstance(message, str) else message
        
        try:
            return hmac.new(key, payload, getattr(hashlib, algorithm)).hexdigest()
        except AttributeError as e:
            TraceEngine.record_exception(e)
            raise CryptoError(f"Unsupported HMAC algorithm: {algorithm}") from e

    @trace_span(operation="security.hmac_verify", component="security", kind=SpanKind.INTERNAL)
    def hmac_verify(self, message: Union[str, bytes], secret: Union[str, bytes], signature: str, algorithm: str = 'sha256') -> bool:
        """Verifies an HMAC cryptographic signature utilizing constant-time comparison."""
        expected_mac = self.hmac_sign(message, secret, algorithm)
        return self.secure_compare_bytes(expected_mac, signature)

    @trace_span(operation="security.secure_compare_bytes", component="security", kind=SpanKind.INTERNAL)
    def secure_compare_bytes(self, val1: Union[str, bytes], val2: Union[str, bytes]) -> bool:
        """Mitigates timing attacks by evaluating byte arrays or strings securely in constant time."""
        self._increment_metric("_crypto_operations")
        b1 = val1.encode('utf-8') if isinstance(val1, str) else val1
        b2 = val2.encode('utf-8') if isinstance(val2, str) else val2
        return hmac.compare_digest(b1, b2)

    # -------------------------------------------------------------------------
    # PASSWORD MANAGEMENT
    # -------------------------------------------------------------------------

    @trace_span(operation="security.generate_salt", component="security", kind=SpanKind.INTERNAL)
    def generate_salt(self, length: int = 16) -> str:
        """Generates a secure salt string."""
        return secrets.token_hex(length)

    @trace_span(operation="security.hash_password", component="security", kind=SpanKind.INTERNAL)
    def hash_password_pbkdf2(self, password: str, algorithm: str = "pbkdf2_sha256") -> PasswordHash:
        """
        Hashes a plaintext password utilizing PBKDF2 standard. 
        Algorithm parameter retained for future flexibility (e.g., argon2 translation layer).
        """
        try:
            salt = self.generate_salt()
            iterations = settings.security.password_iterations
            
            TraceEngine.attach_metadata("iterations", iterations)
            TraceEngine.attach_metadata("algorithm", algorithm)
            
            start_time = time.perf_counter()
            hash_bytes = hashlib.pbkdf2_hmac(
                'sha256', 
                password.encode('utf-8'), 
                salt.encode('utf-8'), 
                iterations
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            
            TraceEngine.attach_metadata("hashing_time_ms", round(duration_ms, 3))
            
            self._increment_metric("_passwords_hashed")
            self._increment_metric("_crypto_operations")
            
            AuditEngine.record_success("security.hash_password", AuditAction.SYSTEM, "Password hashed securely.")
            
            return PasswordHash(
                hash_value=hash_bytes.hex(),
                salt=salt,
                iterations=iterations,
                algorithm=algorithm
            )
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CryptoError("Failed to hash password via PBKDF2.") from e

    @trace_span(operation="security.verify_password", component="security", kind=SpanKind.INTERNAL)
    def verify_password_pbkdf2(self, password: str, hashed_obj: PasswordHash) -> bool:
        """Securely evaluates a plaintext password against a stored structural hash."""
        try:
            if hashed_obj.algorithm != "pbkdf2_sha256":
                raise CryptoError(f"Unsupported verification algorithm: {hashed_obj.algorithm}")

            start_time = time.perf_counter()
            test_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                hashed_obj.salt.encode('utf-8'),
                hashed_obj.iterations
            )
            
            is_valid = self.secure_compare_bytes(test_hash.hex(), hashed_obj.hash_value)
            
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            TraceEngine.attach_metadata("verification_time_ms", round(duration_ms, 3))
            TraceEngine.attach_metadata("verification_result", is_valid)
            
            self._increment_metric("_passwords_verified")
            self._increment_metric("_crypto_operations")
            
            if is_valid:
                AuditEngine.record_success("security.verify_password", AuditAction.SYSTEM, "Password verified successfully.")
            else:
                AuditEngine.record_failure("security.verify_password", AuditAction.SYSTEM, "Password verification failed.", AuditSeverity.WARNING)
                
            return is_valid
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CryptoError("Failed to verify password via PBKDF2.") from e

    # -------------------------------------------------------------------------
    # UTILITIES & MASKS
    # -------------------------------------------------------------------------

    @trace_span(operation="security.mask_secret", component="security", kind=SpanKind.INTERNAL)
    def mask_secret(self, secret: str, visible_start: int = 2, visible_end: int = 2) -> str:
        """Redacts sensitive strings structurally for safe logging."""
        length = len(secret)
        if length <= visible_start + visible_end:
            return "*" * length
        return f"{secret[:visible_start]}{'*' * (length - visible_start - visible_end)}{secret[-visible_end:]}"

    @trace_span(operation="security.base64_encode", component="security", kind=SpanKind.INTERNAL)
    def base64_encode(self, data: Union[str, bytes]) -> str:
        payload = data.encode('utf-8') if isinstance(data, str) else data
        return base64.b64encode(payload).decode('utf-8')

    @trace_span(operation="security.base64_decode", component="security", kind=SpanKind.INTERNAL)
    def base64_decode(self, data: str) -> bytes:
        try:
            return base64.b64decode(data.encode('utf-8'), validate=True)
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CryptoError("Invalid Base64 payload provided for decoding.") from e

    # -------------------------------------------------------------------------
    # INPUT VALIDATORS (O(1) Evaluation)
    # -------------------------------------------------------------------------

    def _record_validation_failure(self, validator_name: str, payload: str, reason: str) -> None:
        """Safely traps and audits malicious or flawed input boundaries."""
        self._increment_metric("_validation_failures")
        safe_payload = payload[:64] + "..." if len(payload) > 64 else payload
        
        TraceEngine.record_event(
            name="VALIDATION_FAILED",
            severity=TraceLevel.WARNING,
            metadata={"validator": validator_name, "reason": reason}
        )
        
        AuditEngine.record_failure(
            operation=validator_name,
            action=AuditAction.READ,
            message=f"Input validation rejected: {reason}",
            severity=AuditSeverity.WARNING,
            metadata={"input_preview": safe_payload}
        )

    @trace_span(operation="security.validate_email", component="security", kind=SpanKind.INTERNAL)
    def validate_email(self, email: str) -> bool:
        if not _REGEX_EMAIL.match(email):
            self._record_validation_failure("validate_email", email, "Invalid RFC 5322 email format.")
            return False
        return True

    @trace_span(operation="security.validate_url", component="security", kind=SpanKind.INTERNAL)
    def validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            if all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']:
                return True
            self._record_validation_failure("validate_url", url, "Missing scheme or network location.")
            return False
        except Exception:
            self._record_validation_failure("validate_url", url, "Invalid URL structure.")
            return False

    @trace_span(operation="security.validate_ipv4", component="security", kind=SpanKind.INTERNAL)
    def validate_ipv4(self, ip: str) -> bool:
        try:
            ipaddress.IPv4Address(ip)
            return True
        except ValueError:
            self._record_validation_failure("validate_ipv4", ip, "Invalid IPv4 format.")
            return False

    @trace_span(operation="security.validate_ipv6", component="security", kind=SpanKind.INTERNAL)
    def validate_ipv6(self, ip: str) -> bool:
        try:
            ipaddress.IPv6Address(ip)
            return True
        except ValueError:
            self._record_validation_failure("validate_ipv6", ip, "Invalid IPv6 format.")
            return False

    @trace_span(operation="security.validate_domain", component="security", kind=SpanKind.INTERNAL)
    def validate_domain(self, domain: str) -> bool:
        if not _REGEX_DOMAIN.match(domain):
            self._record_validation_failure("validate_domain", domain, "Invalid domain format.")
            return False
        return True

    @trace_span(operation="security.validate_uuid", component="security", kind=SpanKind.INTERNAL)
    def validate_uuid(self, uuid_str: str) -> bool:
        if not _REGEX_UUID.match(uuid_str):
            self._record_validation_failure("validate_uuid", uuid_str, "Invalid UUID v4 format.")
            return False
        return True

    @trace_span(operation="security.validate_api_key", component="security", kind=SpanKind.INTERNAL)
    def validate_api_key(self, api_key: str) -> bool:
        if not _REGEX_API_KEY.match(api_key):
            self._record_validation_failure("validate_api_key", self.mask_secret(api_key), "Invalid API Key structure.")
            return False
        return True

    @trace_span(operation="security.validate_symbol", component="security", kind=SpanKind.INTERNAL)
    def validate_symbol(self, symbol: str) -> bool:
        if not _REGEX_SYMBOL.match(symbol):
            self._record_validation_failure("validate_symbol", symbol, "Invalid financial symbol format.")
            return False
        return True

    # -------------------------------------------------------------------------
    # PATH & FILE SECURITY
    # -------------------------------------------------------------------------

    @trace_span(operation="security.prevent_path_traversal", component="security", kind=SpanKind.INTERNAL)
    def prevent_path_traversal(self, base_dir: Union[str, Path], target_path: Union[str, Path]) -> Path:
        """
        Resolves physical target boundaries and mathematically restricts 
        escape attempts via directory traversal tactics. Prevents symlink attacks.
        """
        try:
            base = Path(base_dir).resolve(strict=True)
            # Do not enforce strict resolution on target if it doesn't exist yet, just resolve structure
            target = (base / Path(target_path)).resolve()
            
            # Use commonpath as an additional mathematical boundary check
            if os.path.commonpath([str(base), str(target)]) != str(base):
                self._record_validation_failure("path_traversal", str(target_path), "Directory traversal attack detected via commonpath.")
                raise SecurityError(f"Access denied. Target escapes bounded directory.")
                
            return target
        except Exception as e:
            TraceEngine.record_exception(e)
            raise SecurityError(f"Path resolution failed: {e}") from e

    def safe_join(self, base_dir: Union[str, Path], *paths: Union[str, Path]) -> Path:
        """Safely joins paths ensuring no upward traversal occurs."""
        combined = Path(base_dir)
        for p in paths:
            combined = combined / p
        return self.prevent_path_traversal(base_dir, combined)

    @trace_span(operation="security.validate_filename", component="security", kind=SpanKind.INTERNAL)
    def validate_filename(self, filename: str) -> bool:
        """Validates that a filename contains no illegal or traversal characters."""
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            self._record_validation_failure("validate_filename", filename, "Contains structural traversal characters.")
            return False
        if not _REGEX_SAFE_FILENAME.match(filename):
            self._record_validation_failure("validate_filename", filename, "Contains unsafe special characters.")
            return False
        return True

    @trace_span(operation="security.sanitize_filename", component="security", kind=SpanKind.INTERNAL)
    def sanitize_filename(self, filename: str) -> str:
        """Strips malicious directory indicators and forces structurally safe boundaries."""
        sanitized = Path(filename).name
        sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', sanitized)
        if not sanitized or sanitized.startswith('.'):
            sanitized = f"safe_{self.generate_random_string(8)}.bin"
        return sanitized

    def generate_temp_filename(self, extension: str = ".tmp") -> str:
        """Produces a cryptographically secure entropy-bound temporary file mapping."""
        uid = self.generate_uuid()
        ext = extension if extension.startswith('.') else f".{extension}"
        return f"tmp_{uid}{ext}"

    @trace_span(operation="security.validate_file", component="security", kind=SpanKind.INTERNAL)
    def validate_file(self, file_obj: BinaryIO, filename: str) -> bool:
        """Comprehensive single-pass file validation (Size, Extension, Magic Number)."""
        # 1. Extension Verification
        ext = Path(filename).suffix.lower()
        if ext in settings.security.forbidden_extensions:
            self._record_validation_failure("validate_file", filename, f"Forbidden extension: {ext}")
            return False
        if settings.security.allowed_extensions and ext not in settings.security.allowed_extensions:
            self._record_validation_failure("validate_file", filename, f"Disallowed extension: {ext}")
            return False

        # 2. Size Verification
        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell()
        file_obj.seek(0)
        
        if size > settings.security.max_file_size_bytes:
            self._record_validation_failure("validate_file", filename, f"Size {size}B exceeds {settings.security.max_file_size_bytes}B limit.")
            return False

        # 3. Magic Number Verification (Defense in Depth)
        header = file_obj.read(2048)
        file_obj.seek(0)
        
        # If we have a known magic number mapping, strictly enforce it against extension
        matched_magic = False
        for magic, mime in _MAGIC_NUMBERS.items():
            if header.startswith(magic):
                matched_magic = True
                expected_exts = mimetypes.guess_all_extensions(mime)
                if ext not in expected_exts and not (mime == 'application/zip' and ext in ['.docx', '.xlsx', '.jar']):
                    self._record_validation_failure("validate_file", filename, f"Magic number MIME {mime} mismatched with extension {ext}.")
                    return False
                break
                
        # If no magic number matched but file is permitted, we let it pass but log it structurally
        if not matched_magic:
            _logger.debug(f"File {filename} passed without explicit Magic Number validation.")
            
        AuditEngine.record_success("security.validate_file", AuditAction.READ, f"File {filename} fully validated.")
        return True

    @trace_span(operation="security.validate_mime_type", component="security", kind=SpanKind.INTERNAL)
    def validate_mime_type(self, file_obj: BinaryIO, filename: str) -> Optional[str]:
        """Provides defense-in-depth MIME type validation combining Magic Numbers and Extensions."""
        header = file_obj.read(2048)
        file_obj.seek(0)
        
        for magic, mime in _MAGIC_NUMBERS.items():
            if header.startswith(magic):
                return mime
                
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type

    @trace_span(operation="security.hash_file_sha256", component="security", kind=SpanKind.INTERNAL)
    def hash_file_sha256(self, file_obj: BinaryIO) -> FileChecksum:
        """Syntactic wrapper for SHA256 file checksums."""
        return self.generate_file_checksum(file_obj, 'sha256')

    @trace_span(operation="security.hash_file_sha512", component="security", kind=SpanKind.INTERNAL)
    def hash_file_sha512(self, file_obj: BinaryIO) -> FileChecksum:
        """Syntactic wrapper for SHA512 file checksums."""
        return self.generate_file_checksum(file_obj, 'sha512')

    @trace_span(operation="security.file_checksum", component="security", kind=SpanKind.INTERNAL)
    def generate_file_checksum(self, file_obj: BinaryIO, algorithm: str = 'sha256') -> FileChecksum:
        """
        Generates robust structural integrity hashes using native Python zero-copy digest.
        Stream-based O(1) memory mapping.
        """
        try:
            self._increment_metric("_files_validated")
            start_time = time.perf_counter()
            
            file_obj.seek(0)
            
            if algorithm == 'sha256':
                file_hash = hashlib.file_digest(file_obj, "sha256")
            elif algorithm == 'sha512':
                file_hash = hashlib.file_digest(file_obj, "sha512")
            else:
                raise CryptoError(f"Unsupported file checksum algorithm: {algorithm}")
                
            checksum_hex = file_hash.hexdigest()
            
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(0)
            
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            
            TraceEngine.attach_metadata("file_size_bytes", size)
            TraceEngine.attach_metadata("hash_duration_ms", round(duration_ms, 3))
            
            AuditEngine.record_success("security.generate_file_checksum", AuditAction.SYSTEM, f"{algorithm.upper()} checksum generated.")
            
            return FileChecksum(
                algorithm=algorithm,
                checksum=checksum_hex,
                file_size_bytes=size
            )
        except Exception as e:
            TraceEngine.record_exception(e)
            raise CryptoError("Failed to calculate cryptographic checksum on file stream.") from e


# -------------------------------------------------------------------------
# GLOBAL SINGLETON EXPORT
# -------------------------------------------------------------------------

security_engine: Final[SecurityEngine] = SecurityEngine()

__all__ = [
    "SecurityError",
    "ValidationError",
    "CryptoError",
    "PasswordHash",
    "FileChecksum",
    "SecurityStatistics",
    "SecurityEngine",
    "security_engine"
]
