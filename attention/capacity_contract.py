"""Repository-wide exact-capacity contract for C4 and C8.

C4 and C8 name the complete number of merged K/V representatives.  Neither
operator may append or stream a raw/singleton/residual K/V token.
"""

EXACT_MERGED_CAPACITY = {4: 4, 8: 8}
FORBIDDEN_SINGLETON_CAPACITIES = frozenset(EXACT_MERGED_CAPACITY)


def require_exact_merged_capacity(capacity: int, singleton_count: int = 0) -> int:
    """Validate and return the complete representative count."""
    capacity = int(capacity)
    singleton_count = int(singleton_count)
    if capacity in FORBIDDEN_SINGLETON_CAPACITIES and singleton_count:
        raise ValueError(
            f"C{capacity} is exactly {capacity} merged K/V representatives; "
            "singleton/SQ1/C+1 routing is forbidden"
        )
    if capacity in EXACT_MERGED_CAPACITY:
        return EXACT_MERGED_CAPACITY[capacity]
    return capacity + singleton_count


__all__ = [
    "EXACT_MERGED_CAPACITY",
    "FORBIDDEN_SINGLETON_CAPACITIES",
    "require_exact_merged_capacity",
]
