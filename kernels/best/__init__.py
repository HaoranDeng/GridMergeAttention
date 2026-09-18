"""Best measured BF16 snapshots. FP8 has a separate namespace.

The C8 r2 package exposes only its screened numerics implementation. Historical
alternatives remain documented in c8_r2/evidence.json, but are not runtime
choices.
"""
from .c4 import create as create_c4
from .c8 import create as create_c8
from .c8_r2 import create as create_c8_r2
from .c10 import create as create_c10

__all__ = ["create_c4", "create_c8", "create_c8_r2", "create_c10"]
