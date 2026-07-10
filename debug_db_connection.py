from backend.database.connection import db_manager

print(f"DEBUG: Current DB URL: {db_manager.db_url}")

try:
    tables = db_manager.fetch_all("SELECT name FROM sqlite_master WHERE type='table';")
    print("DEBUG: Tables found in THIS connection:")
    for row in tables:
        print(f" - {row['name']}")
except Exception as e:
    print(f"DEBUG Error: {e}")
