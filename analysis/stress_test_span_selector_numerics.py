"""Stress-test C4 span selectors on increasingly ill-conditioned profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _reference_mgs2(profile: np.ndarray, weight: np.ndarray) -> tuple[int, ...]:
    residual = profile.astype(np.float64, copy=True)
    chosen = [0]
    used = np.zeros(profile.shape[0], dtype=bool)
    used[0] = True
    basis_source = residual[0]
    for _ in range(3):
        basis = basis_source / max(np.linalg.norm(basis_source), 1e-30)
        # Two passes make the float64 reference insensitive to residual
        # components along previously selected directions.
        for _ in range(2):
            residual -= (residual @ basis)[:, None] * basis[None, :]
        objective = np.sum(residual * residual, axis=1) * weight
        objective[used] = -np.inf
        pick = int(np.argmax(objective))
        chosen.append(pick)
        used[pick] = True
        basis_source = residual[pick].copy()
    return tuple(chosen)


def _mgs(profile: np.ndarray, weight: np.ndarray) -> tuple[int, ...]:
    residual = profile.astype(np.float32, copy=True)
    chosen = [0]
    used = np.zeros(profile.shape[0], dtype=bool)
    used[0] = True
    basis_source = residual[0]
    for _ in range(3):
        norm = np.sqrt(np.sum(basis_source * basis_source, dtype=np.float32))
        basis = basis_source / np.maximum(norm, np.float32(1e-12))
        projection = np.sum(residual * basis[None, :], axis=1, dtype=np.float32)
        residual -= projection[:, None] * basis[None, :]
        objective = np.sum(residual * residual, axis=1, dtype=np.float32) * weight
        objective[used] = -np.inf
        pick = int(np.argmax(objective))
        chosen.append(pick)
        used[pick] = True
        basis_source = residual[pick].copy()
    return tuple(chosen)


def _pivoted_cholesky(
    profile: np.ndarray,
    weight: np.ndarray,
) -> tuple[tuple[int, ...], int]:
    x = profile.astype(np.float32, copy=False)
    gram = x @ x.T
    diagonal = np.diag(gram).copy()
    columns: list[np.ndarray] = []
    chosen = [0]
    used = np.zeros(profile.shape[0], dtype=bool)
    used[0] = True
    clipped = 0
    pivot = 0
    for _ in range(3):
        correction = np.zeros_like(diagonal)
        for column in columns:
            correction += column * column[pivot]
        denominator = np.sqrt(max(float(diagonal[pivot]), 1e-12))
        column = (gram[:, pivot] - correction) / np.float32(denominator)
        columns.append(column)
        updated = diagonal - column * column
        clipped += int(np.count_nonzero(updated < 0.0))
        diagonal = np.maximum(updated, np.float32(0.0))
        objective = diagonal * weight
        objective[used] = -np.inf
        pivot = int(np.argmax(objective))
        chosen.append(pivot)
        used[pivot] = True
    return tuple(chosen), clipped


def _two_stage(profile: np.ndarray, weight: np.ndarray) -> tuple[int, ...]:
    residual = profile.astype(np.float32, copy=True)
    used = np.zeros(profile.shape[0], dtype=bool)
    used[0] = True
    chosen = [0]
    basis_source = residual[0]
    for _ in range(2):
        norm = np.sqrt(np.sum(basis_source * basis_source, dtype=np.float32))
        basis = basis_source / np.maximum(norm, np.float32(1e-12))
        projection = np.sum(residual * basis[None, :], axis=1, dtype=np.float32)
        residual -= projection[:, None] * basis[None, :]
        objective = np.sum(residual * residual, axis=1, dtype=np.float32) * weight
        objective[used] = -np.inf
        pick = int(np.argmax(objective))
        chosen.append(pick)
        used[pick] = True
        basis_source = residual[pick].copy()
    objective[used] = -np.inf
    final = int(np.argmax(objective))
    return tuple((*chosen, final))


def _profile(rng: np.random.Generator, condition: float) -> np.ndarray:
    candidates, dimension = 32, 16
    left, _ = np.linalg.qr(rng.normal(size=(candidates, dimension)))
    right, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    singular = np.geomspace(1.0, 1.0 / condition, dimension)
    profile = (left * singular[None, :]) @ right.T
    # Hellinger profiles are centered and nonnegative before centering; a
    # common offset does not affect the centered routing features.
    profile -= profile.mean(axis=1, keepdims=True)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--conditions",
        default="10,100,1000,10000,100000,1000000,10000000",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    conditions = [float(value) for value in args.conditions.split(",")]
    result = {"trials": args.trials, "seed": args.seed, "conditions": {}}

    for condition in conditions:
        exact = {"mgs": 0, "pivoted_cholesky": 0, "two_stage": 0}
        set_overlap = {name: 0.0 for name in exact}
        clipped = 0
        for _ in range(args.trials):
            profile = _profile(rng, condition)
            weight = np.exp(rng.normal(scale=0.2, size=profile.shape[0])).astype(
                np.float32
            )
            reference = _reference_mgs2(profile, weight)
            selections = {
                "mgs": _mgs(profile, weight),
                "two_stage": _two_stage(profile, weight),
            }
            selections["pivoted_cholesky"], trial_clipped = _pivoted_cholesky(
                profile,
                weight,
            )
            clipped += trial_clipped
            reference_set = set(reference)
            for name, selection in selections.items():
                exact[name] += int(selection == reference)
                set_overlap[name] += len(reference_set.intersection(selection))
        row = {
            name: {
                "exact_sequence_fraction": exact[name] / args.trials,
                "mean_set_overlap": set_overlap[name] / args.trials,
            }
            for name in exact
        }
        row["pivoted_cholesky"]["mean_clipped_diagonals"] = clipped / args.trials
        result["conditions"][str(condition)] = row
        print(condition, json.dumps(row), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
