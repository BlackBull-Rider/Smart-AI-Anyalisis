import re

filepath = "backend/database/schema.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # stock_master টেবিল ডেফিনিশন আপডেট করা
    target = 'is_active BOOLEAN NOT NULL DEFAULT 1,'
    replacement = 'is_active BOOLEAN NOT NULL DEFAULT 1,\n        is_fno BOOLEAN DEFAULT 0,'
    
    if target in content and 'is_fno' not in content:
        content = content.replace(target, replacement)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Successfully patched schema.py with 'is_fno' column.")
    else:
        print("ℹ️ Column might already be in schema.py.")

except Exception as e:
    print(f"❌ Error patching schema: {e}")
