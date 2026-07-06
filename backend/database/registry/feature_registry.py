"""
Green Bull Rider V6
Global Feature Registry

Single Source of Truth
"""

from dataclasses import dataclass

# ==========================================================
# FEATURE MODEL
# ==========================================================

@dataclass(slots=True)
class Feature:

    name: str
    module: str
    function: str

    history: int

    db_column: str

    dtype: str = "REAL"


FEATURES = {}
