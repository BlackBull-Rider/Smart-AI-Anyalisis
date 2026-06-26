import os
from pathlib import Path
import logging

# AI প্রজেক্টের রুট পাথ 
BASE_DIR = Path(__file__).resolve().parent

# Data Engine এর ডেটাবেস পাথ 
DB_PATH = BASE_DIR.parent / "Green-Bull-Data-Engine" / "database" / "market.db"

# System Constants
DEFAULT_LOOKBACK = 200  # টেকনিক্যাল অ্যানালাইসিসের জন্য ডিফল্ট বাফার
LOG_LEVEL = logging.INFO

# Logging Setup
logging.basicConfig(
    level=LOG_LEVEL, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
