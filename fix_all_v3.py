import sqlite3
import os
import sys

# ফাইল পাথ ঠিক করা
db_path = "backend/database/universe.db"
schema_file = "backend/database/schema.py"

print("Step 1: Updating schema.py...")
with open(schema_file, "r", encoding="utf-8") as f:
    content = f.read()

# is_fno নিশ্চিত করা
new_ddl = 'is_active BOOLEAN NOT NULL DEFAULT 1,\n        is_fno BOOLEAN DEFAULT 0,'
if 'is_fno BOOLEAN DEFAULT 0,' not in content:
    content = content.replace('is_active BOOLEAN NOT NULL DEFAULT 1,', new_ddl)
    with open(schema_file, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ schema.py updated.")

print("Step 2: Initializing DB at backend/database/universe.db...")
from backend.database.schema import schema_engine
try:
    # জোর করে টেবিল ড্রপ এবং নতুন করে তৈরি
    schema_engine.create_schema()
    print("🎉 SUCCESS: Clean schema deployed to backend/database/universe.db")
except Exception as e:
    print(f"❌ ERROR: {e}")
