"""Explicit pre-r2 C8 reference; the default C8 factory uses r2."""

from ..c8 import create_slow_reference as create

__all__ = ["create"]
