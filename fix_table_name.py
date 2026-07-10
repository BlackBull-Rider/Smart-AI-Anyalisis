import os

directory = "backend"
patched_files = []

for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Replace the wrong table name with the original one
                if "master_stock" in content:
                    new_content = content.replace("master_stock", "stock_master")
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    patched_files.append(filepath)
            except Exception as e:
                pass

if patched_files:
    print("✅ Successfully fixed table name (master_stock -> stock_master) in:")
    for p in patched_files:
        print(f"  - {p}")
else:
    print("⚠️ No fixes needed.")
