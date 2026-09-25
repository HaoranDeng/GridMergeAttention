"""Regenerate the C8-Hunyuan candidate table.

Mirrors attention/adaptive_seed_pytorch.py:multiscale_candidate_table exactly
(copied inline below so this script has no repo-package import dependency),
for the Hunyuan Video 13B shape: query_rows=30 (H), candidate_count=16.

geometry_count choice: the packaged Wan table used geometry_count == query_rows
(45 == 45), i.e. full-reach multiscale geometry. For rows=30/candidates=16 that
exact analogue (geometry_count=30) raises multiscale_candidate_table's own
"more mandatory rows than target pool" ValueError (radius=max(1,30//4)=7 and
uniform_count=7 together produce >16 mandatory rows for interior key_rows).
geometry_count in [16,20] is feasible; this script uses 20, the largest
feasible value, to keep as much multiscale spread as the C16 pool allows.
"""
import json
from pathlib import Path


def multiscale_candidate_table(query_rows, candidate_count, *, geometry_count=None):
    query_rows = int(query_rows)
    candidate_count = int(candidate_count)
    geometry_count = int(geometry_count or candidate_count)
    if not 4 <= candidate_count <= query_rows:
        raise ValueError(f"candidate_count must be in [4, {query_rows}], got {candidate_count}")
    if not candidate_count <= geometry_count <= query_rows:
        raise ValueError("geometry_count must be between candidate_count and query_rows")

    table = []
    anchors = []
    radius = max(1, geometry_count // 4)
    uniform_count = max(4, geometry_count // 4)
    for key_row in range(query_rows):
        chosen = {key_row}
        chosen.update(range(max(0, key_row - radius), min(query_rows, key_row + radius + 1)))
        chosen.update(
            round(index * (query_rows - 1) / (uniform_count - 1))
            for index in range(uniform_count)
        )
        if len(chosen) > candidate_count:
            raise ValueError(
                f"candidate geometry creates more mandatory rows ({len(chosen)}) "
                f"than target pool ({candidate_count}) at key_row={key_row}"
            )
        while len(chosen) < candidate_count:
            remaining = [row for row in range(query_rows) if row not in chosen]
            row = max(
                remaining,
                key=lambda item: (min(abs(item - seed) for seed in chosen), -item),
            )
            chosen.add(row)
        selected = sorted(chosen)[:candidate_count]
        if key_row not in selected:
            raise RuntimeError("multiscale candidate truncation removed the anchor row")
        table.append(selected)
        anchors.append(selected.index(key_row))
    return table, anchors


def main():
    rows = 30
    candidate_count = 16
    geometry_count = 20
    table, anchors = multiscale_candidate_table(rows, candidate_count, geometry_count=geometry_count)
    for row in table:
        assert len(row) == candidate_count
        assert row == sorted(row)
    payload = {
        "rows": rows,
        "candidate_count": candidate_count,
        "geometry_count": geometry_count,
        "candidates": table,
        "anchors": anchors,
    }
    out = Path(__file__).resolve().parent / "gma_c8h_candidate_table.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({rows} rows, C{candidate_count}, geometry {geometry_count})")


if __name__ == "__main__":
    main()
