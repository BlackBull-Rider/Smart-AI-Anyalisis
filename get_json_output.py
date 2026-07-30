import json
import numpy as np
import pandas as pd
from backend.analyzers.analyzer_engine import analyzer_engine

def make_serializable(o):
    # Numpy ডেটাকে Python native টাইপে কনভার্ট করার ফাংশন
    if isinstance(o, (np.int64, np.int32)): return int(o)
    if isinstance(o, (np.float64, np.float32)): return float(o)
    if isinstance(o, (np.bool_)): return bool(o)
    if isinstance(o, dict): return {k: make_serializable(v) for k, v in o.items()}
    if isinstance(o, list): return [make_serializable(i) for i in o]
    
    # NaN চেক করার সময় যেন এরর না দেয়
    try:
        if pd.isna(o): return None
    except:
        pass
    return o

# ইঞ্জিন রান করা
result = analyzer_engine.run({"symbol": "RELIANCE"})

# সিরিয়ালাইজ করে প্রিন্ট করা
print(json.dumps(make_serializable(result), indent=4))
