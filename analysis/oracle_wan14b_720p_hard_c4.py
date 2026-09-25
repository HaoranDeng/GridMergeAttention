"""Measure the representation ceiling of four convex KV summaries per block.

This is a diagnostic, not a production attention implementation.  On one real
Wan2.1-14B 720p self-attention call it freezes Q/K/V, then directly optimizes
four merge distributions for every 80-key block.  Each compressed key and
value must use the same convex weights over the original block, so the oracle
cannot invent arbitrary K/V vectors.

Two upper bounds are reported:

* ``entropy`` keeps the existing entropy-derived routing bias.
* ``free_bias`` additionally optimizes one scalar log-mass per compressed token.
* ``masked_entropy`` also preserves the original hard key-to-seed assignment.

If neither bound approaches the target error, changing seed selection cannot
make ordinary four-token GridMerge reach that target.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


FRAMES = 21
ROWS = 45
COLS = 80
SEEDS = 4

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
    "limbs, fused fingers, still picture, messy background, three legs, many "
    "people in the background, walking backwards"
)


class StopOracle(Exception):
    """Stop generation after the requested real-QKV call has been measured."""


def parse_columns(value: str) -> list[int]:
    if value.lower() == "all":
        return list(range(COLS))
    return [int(item) for item in value.replace(",", " ").split()]


def error_metrics(output: torch.Tensor, dense: torch.Tensor) -> dict[str, float]:
    difference = output.float() - dense.float()
    squared_error = float(difference.square().sum())
    reference_squared = float(dense.float().square().sum())
    return {
        "rel_l2_vs_dense": math.sqrt(squared_error / max(reference_squared, 1e-20)),
        "mse_vs_dense": squared_error / dense.numel(),
        "max_abs_vs_dense": float(difference.abs().max()),
        "cosine_error_vs_dense": 1.0
        - float(F.cosine_similarity(output.float(), dense.float(), dim=-1).mean()),
    }


def current_c4_weights(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    key_frames: int,
    key_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current hard-C4 weights and active-token mask.

    ``query`` is ``[J, I, D]`` and ``key_blocks`` is ``[F*K, L, D]``.
    The weights and assignment support are ``[J, F*K, C, L]``.
    """
    query_rows = query.size(1)
    starts = (torch.arange(key_rows, device=query.device) - 1).clamp(
        0, query_rows - SEEDS
    )
    seed_rows = starts[:, None] + torch.arange(SEEDS, device=query.device)
    seeds = query[:, seed_rows]
    keys = key_blocks.reshape(key_frames, key_rows, key_blocks.size(1), -1)
    logits = torch.einsum("jkcd,fkld->jfkcl", seeds, keys).float()
    logits *= query.size(-1) ** -0.5

    assignment_score = torch.log_softmax(logits, dim=-1)
    assignment = F.one_hot(
        assignment_score.argmax(dim=-2), num_classes=SEEDS
    ).permute(0, 1, 2, 4, 3).bool()
    selected_logits = logits.masked_fill(~assignment, float("-inf"))
    log_weights = torch.log_softmax(selected_logits, dim=-1)
    log_weights = torch.where(assignment, log_weights, 0.0)
    weights = torch.where(assignment, log_weights.exp(), 0.0)
    return (
        weights.flatten(1, 2),
        assignment.any(dim=-1).flatten(1, 2),
        assignment.flatten(1, 2),
    )


def compressed_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    weights: torch.Tensor,
    active: torch.Tensor | None = None,
    free_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate full second-stage attention over convex C4 block summaries."""
    merged_key = torch.einsum("jbcl,bld->jbcd", weights, key_blocks)
    merged_value = torch.einsum("jbcl,bld->jbcd", weights, value_blocks)
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(dim=-1)
    logits = torch.einsum("jid,jbcd->jibc", query, merged_key)
    logits = logits * (query.size(-1) ** -0.5) + entropy[:, None]
    if free_bias is not None:
        logits = logits + free_bias[:, None]
    if active is not None:
        logits = logits.masked_fill(~active[:, None], float("-inf"))
    probabilities = torch.softmax(logits.flatten(-2), dim=-1)
    values = merged_value.flatten(1, 2)
    return torch.einsum("jit,jtd->jid", probabilities, values)


def optimize_oracle(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    dense: torch.Tensor,
    initial_weights: torch.Tensor,
    mode: str,
    steps: int,
    learning_rate: float,
    report_every: int,
    support: torch.Tensor | None = None,
    return_weights: bool = False,
) -> dict | tuple[dict, torch.Tensor]:
    """Optimize convex merge weights, optionally with a free routing bias."""
    active_support = None
    safe_support = support
    smoothed = initial_weights.float() + 1e-5
    if support is not None:
        active_support = support.any(dim=-1)
        safe_support = support.clone()
        safe_support[..., 0] |= ~active_support
        smoothed = smoothed.masked_fill(~safe_support, 0.0)
    smoothed /= smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    merge_logits = torch.nn.Parameter(smoothed.clamp_min(1e-30).log())
    free_bias = None
    parameters = [merge_logits]
    parameter_groups = [{"params": [merge_logits], "lr": learning_rate}]
    if mode == "free_bias":
        free_bias = torch.nn.Parameter(torch.zeros_like(merge_logits[..., 0]))
        parameters.append(free_bias)
        parameter_groups.append(
            {"params": [free_bias], "lr": learning_rate * 0.1}
        )

    optimizer = torch.optim.Adam(parameter_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.05
        + 0.95 * (1.0 + math.cos(math.pi * step / max(steps, 1))) / 2.0,
    )
    reference_squared = dense.float().square().sum().clamp_min(1e-20)
    trajectory = []
    best_rel_l2 = float("inf")
    best_step = 0

    for step in range(steps + 1):
        weight_logits = merge_logits
        if safe_support is not None:
            weight_logits = weight_logits.masked_fill(~safe_support, float("-inf"))
        weights = torch.softmax(weight_logits, dim=-1)
        active = None
        if active_support is not None:
            active = active_support
            weights = torch.where(active.unsqueeze(-1), weights, 0.0)
        output = compressed_attention(
            query,
            key_blocks,
            value_blocks,
            weights,
            active=active,
            free_bias=free_bias,
        )
        loss = (output - dense).square().sum() / reference_squared
        rel_l2 = math.sqrt(float(loss.detach()))
        if rel_l2 < best_rel_l2:
            best_rel_l2 = rel_l2
            best_step = step
        if step == 0 or step == steps or step % report_every == 0:
            metrics = error_metrics(output.detach(), dense)
            metrics.update({"step": step, "objective": float(loss.detach())})
            trajectory.append(metrics)
            print(
                f"[{mode}] step={step:04d} rel_l2={metrics['rel_l2_vs_dense']:.6f} "
                f"mse={metrics['mse_vs_dense']:.8f}",
                flush=True,
            )
        if step == steps:
            final_output = output.detach()
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    result = error_metrics(final_output, dense)
    result.update(
        {
            "best_rel_l2_seen": best_rel_l2,
            "best_step_seen": best_step,
            "trajectory": trajectory,
            "optimized_parameters": sum(parameter.numel() for parameter in parameters),
        }
    )
    if return_weights:
        return result, weights.detach()
    return result


def run_oracle_on_call(query, key, value, args) -> dict:
    head = args.head
    frame = args.frame
    columns = torch.tensor(args.query_columns, device=query.device)
    dim = query.size(-1)

    query_grid = query[0, head].reshape(FRAMES, ROWS, COLS, dim)
    key_grid = key[0, head].reshape(FRAMES, ROWS, COLS, dim)
    value_grid = value[0, head].reshape(FRAMES, ROWS, COLS, dim)
    query_frame = query_grid[frame]

    with torch.no_grad():
        dense = F.scaled_dot_product_attention(
            query_frame.flatten(0, 1)[None, None],
            key_grid.flatten(0, 2)[None, None],
            value_grid.flatten(0, 2)[None, None],
            dropout_p=0.0,
            is_causal=False,
        )[0, 0].reshape(ROWS, COLS, dim)

    query_columns = query_frame.index_select(1, columns).permute(1, 0, 2).float()
    dense_columns = dense.index_select(1, columns).permute(1, 0, 2).float()
    key_blocks = key_grid.reshape(FRAMES * ROWS, COLS, dim).float()
    value_blocks = value_grid.reshape(FRAMES * ROWS, COLS, dim).float()
    initial_weights, active, assignment = current_c4_weights(
        query_columns, key_blocks, FRAMES, ROWS
    )
    with torch.no_grad():
        baseline = compressed_attention(
            query_columns,
            key_blocks,
            value_blocks,
            initial_weights,
            active=active,
        )

    result = {
        "current_hard_c4": error_metrics(baseline, dense_columns),
        "current_empty_cluster_fraction": float((~active).float().mean()),
        "query_columns": args.query_columns,
        "query_elements": dense_columns.numel(),
        "oracles": {},
    }
    print(
        "[current_hard_c4] "
        f"rel_l2={result['current_hard_c4']['rel_l2_vs_dense']:.6f} "
        f"mse={result['current_hard_c4']['mse_vs_dense']:.8f}",
        flush=True,
    )

    tensor_payload = None
    if args.tensor_output:
        tensor_payload = {
            "query": query_columns.to(torch.bfloat16).cpu(),
            "key": key_blocks.to(torch.bfloat16).cpu(),
            "value": value_blocks.to(torch.bfloat16).cpu(),
            "dense": dense_columns.to(torch.bfloat16).cpu(),
            # Q/K/V originate in BF16, but merge probabilities are computed
            # in FP32. Preserve that precision for formula fitting.
            "initial_weights": initial_weights.float().cpu(),
            "assignment": assignment.cpu(),
        }
    for mode in args.modes:
        oracle = optimize_oracle(
            query_columns,
            key_blocks,
            value_blocks,
            dense_columns,
            initial_weights,
            mode,
            args.oracle_steps,
            args.learning_rate,
            args.report_every,
            support=assignment if mode == "masked_entropy" else None,
            return_weights=tensor_payload is not None,
        )
        if tensor_payload is None:
            result["oracles"][mode] = oracle
        else:
            result["oracles"][mode], oracle_weights = oracle
            tensor_payload[f"{mode}_weights"] = oracle_weights.float().cpu()
        torch.cuda.empty_cache()
    if tensor_payload is not None:
        result["_tensor_payload"] = tensor_payload
    return result


def install_probe(pipe, args, state):
    layer_count = len(pipe.transformer.blocks)

    class Processor:
        def __init__(self, layer):
            self.layer = layer

        def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            rotary_emb=None,
            timestep=None,
        ):
            del timestep
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
            if rotary_emb is not None:
                q_complex = torch.view_as_complex(
                    query.to(torch.float64).unflatten(3, (-1, 2))
                )
                k_complex = torch.view_as_complex(
                    key.to(torch.float64).unflatten(3, (-1, 2))
                )
                query = torch.view_as_real(q_complex * rotary_emb).flatten(3, 4).type_as(query)
                key = torch.view_as_real(k_complex * rotary_emb).flatten(3, 4).type_as(key)

            step = state["call"] // layer_count
            if step == args.target_step and self.layer == args.target_layer:
                with torch.inference_mode(False), torch.enable_grad():
                    q = query.detach().clone()
                    k = key.detach().clone()
                    v = value.detach().clone()
                    state["result"] = run_oracle_on_call(q, k, v, args)
                state["captured_call"] = state["call"]
                raise StopOracle

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            state["call"] += 1
            output = output.transpose(1, 2).flatten(2, 3).type_as(query)
            return attn.to_out[1](attn.to_out[0](output))

    for layer, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(Processor(layer))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor_output")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--target_step", type=int, default=2)
    parser.add_argument("--target_layer", type=int, default=20)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--frame", type=int, default=10)
    parser.add_argument("--query_columns", type=parse_columns, default=list(range(COLS)))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["entropy", "free_bias", "masked_entropy"],
        default=["entropy", "free_bias"],
    )
    parser.add_argument("--oracle_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.08)
    parser.add_argument("--report_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipe = WanPipeline.from_pretrained(
        args.model,
        vae=vae,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=5.0,
    )
    pipe.to("cuda")

    state = {"call": 0, "result": None}
    install_probe(pipe, args, state)
    prompts = [
        line.strip()
        for line in Path(args.prompt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    try:
        with torch.inference_mode():
            pipe(
                prompt=prompts[args.prompt_index],
                negative_prompt=NEGATIVE_PROMPT,
                height=720,
                width=1280,
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=args.steps,
                generator=generator,
            )
    except StopOracle:
        pass
    torch.cuda.synchronize()

    tensor_payload = state["result"].pop("_tensor_payload", None)
    payload = {
        "device": torch.cuda.get_device_name(),
        "model": str(Path(args.model).resolve()),
        "prompt_index": args.prompt_index,
        "steps": args.steps,
        "target_step": args.target_step,
        "target_layer": args.target_layer,
        "head": args.head,
        "frame": args.frame,
        "captured_call": state.get("captured_call"),
        "oracle_definition": {
            "compressed_tokens_per_80_key_block": SEEDS,
            "same_convex_weights_for_key_and_value": True,
            "full_second_stage_attention": True,
            "entropy_mode_uses_only_entropy_routing_bias": True,
            "free_bias_mode_adds_one_scalar_per_compressed_token": True,
            "diagnostic_only": True,
        },
        "optimization": {
            "steps": args.oracle_steps,
            "learning_rate": args.learning_rate,
            "modes": args.modes,
        },
        "result": state["result"],
        "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if tensor_payload is not None:
        tensor_output = Path(args.tensor_output)
        tensor_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_payload, tensor_output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
