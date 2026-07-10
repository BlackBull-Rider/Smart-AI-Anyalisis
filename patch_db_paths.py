import os

target_db = "universe.db"
wrong_dbs = ["gbr_market_master.db", "gbr_master.db"]
directory = "backend"

patched_files = []

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content
                for wrong_db in wrong_dbs:
                    if wrong_db in new_content:
                        new_content = new_content.replace(wrong_db, target_db)
                
                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    patched_files.append(filepath)
            except Exception as e:
                print(f"Skipped {filepath} due to error: {e}")

if patched_files:
    print("✅ Successfully patched the following files to use 'universe.db':")
    for p in patched_files:
        print(f"  - {p}")
else:
    print("⚠️ No incorrect database paths found to patch.")
