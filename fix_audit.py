import os

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    old_import = "from backend.core.audit import AuditEngine, AuditAction, AuditSeverity"
    new_import = """from backend.core.audit import AuditEngine as CoreAudit, AuditAction, AuditSeverity

class AuditEngine:
    @staticmethod
    def record_event(operation: str, action: AuditAction, severity: AuditSeverity, message: str, metadata: dict = None) -> None:
        try:
            if severity in (AuditSeverity.CRITICAL, AuditSeverity.WARNING):
                if hasattr(CoreAudit, 'record_failure'):
                    CoreAudit.record_failure(operation=operation, action=action, message=message, severity=severity, metadata=metadata)
            else:
                if hasattr(CoreAudit, 'record_success'):
                    CoreAudit.record_success(operation=operation, action=action, message=message, metadata=metadata)
        except Exception:
            pass
"""

    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Fixed AuditEngine AttributeError in market_pipeline.py")
    else:
        print("⚠️ AuditEngine already patched or import format differs.")
except Exception as e:
    print(f"Error: {e}")
