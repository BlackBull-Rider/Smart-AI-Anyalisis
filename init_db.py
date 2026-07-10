from backend.database.schema import schema_engine
try:
    print("🚀 Initializing Schema...")
    schema_engine.create_schema()
    print("✅ Success: All tables created in universe.db!")
except Exception as e:
    print(f"❌ Error: {e}")
