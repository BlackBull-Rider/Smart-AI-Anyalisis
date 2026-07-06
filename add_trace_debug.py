from pathlib import Path

p = Path("tests/test_core_pipeline.py")
text = p.read_text(encoding="utf-8")

text = text.replace(
    "ctx_start = ResourceTracker.get_contextvars_count()",
    "ctx_start = ResourceTracker.get_contextvars_count()\n            print(f'DEBUG ctx_start = {ctx_start}')",
    1
)

text = text.replace(
    "ctx_end = ResourceTracker.get_contextvars_count()",
    "ctx_end = ResourceTracker.get_contextvars_count()\n            print(f'DEBUG ctx_end = {ctx_end}')",
    1
)

p.write_text(text, encoding="utf-8")
print("Done.")
