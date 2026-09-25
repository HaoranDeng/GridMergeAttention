"""Run the canonical Wan Diffusers benchmark with repo-native MonarchRT blocks."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


REPO = Path(
    os.environ.get("GMA_WORKTREE", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(REPO))

import attention.token_merging as token_merging


F_TIED = int(os.environ.get("MONARCHRT_F_TIED", "1"))
H_REDUCE = int(os.environ.get("MONARCHRT_H_REDUCE", "1"))
W_REDUCE = int(os.environ.get("MONARCHRT_W_REDUCE", "1"))
METHOD_LABEL = os.environ.get(
    "MONARCHRT_METHOD_LABEL",
    f"MonarchRT f{F_TIED} h{H_REDUCE} w{W_REDUCE}",
)

for name, value in (
    ("MONARCHRT_F_TIED", F_TIED),
    ("MONARCHRT_H_REDUCE", H_REDUCE),
    ("MONARCHRT_W_REDUCE", W_REDUCE),
):
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")

# The official 720p layout is F=21, H=45, W=80. Restrict this runner to exact
# factorizations so its curve never mixes padded custom variants.
for name, extent, reduction in (
    ("F", 21, F_TIED),
    ("H", 45, H_REDUCE),
    ("W", 80, W_REDUCE),
):
    if extent % reduction:
        raise ValueError(
            f"Repo-native 720p MonarchRT requires {name}={extent} to be "
            f"divisible by its reduction, got {reduction}."
        )


def _repo_native_monarchrt(query, key, value, **kwargs):
    return token_merging._causal_one_token_attention(
        query,
        key,
        value,
        f_tied=F_TIED,
        h_reduce=H_REDUCE,
        w_reduce=W_REDUCE,
        h=kwargs["h"],
        w=kwargs["w"],
        sm_scale=kwargs["sm_scale"],
        q_init="ith",
        query_outer_chunk=kwargs["query_outer_chunk"],
        backend="monarch_rt",
        head_chunk=kwargs["head_chunk"],
    )


def _argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Required benchmark argument missing: {name}") from exc


token_merging._hard_c4_attention = _repo_native_monarchrt
runpy.run_path(
    str(REPO / "analysis" / "benchmark_wan14b_720p_methods.py"),
    run_name="__main__",
)

runtime_path = Path(_argument_value("--output_dir")) / "runtime.json"
payload = json.loads(runtime_path.read_text(encoding="utf-8"))
protocol = payload.setdefault("protocol", {})
density = H_REDUCE / (F_TIED * 45) + W_REDUCE / 80
protocol.update(
    {
        "method": METHOD_LABEL,
        "method_classification": "training-free baseline",
        "attention_schedule": {
            "dense_first_steps": 0,
            "dense_first_layers": 0,
            "transformer_layers": 40,
            "remaining_attention": "pure all-layer repo-native MonarchRT",
        },
        "pure_all_layer": True,
        "pure_all_layer_gma": False,
        "monarch_rt": {
            "f_tied": F_TIED,
            "h_reduce": H_REDUCE,
            "w_reduce": W_REDUCE,
            "q_init": "ith",
            "num_merge_tokens": 1,
            "head_chunk": 1,
            "attention_density": density,
            "attention_sparsity": 1.0 - density,
            "uses_padding": False,
        },
        "prompt_manifest": str(runtime_path.parent / "prompt_manifest.json"),
    }
)
runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
