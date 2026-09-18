"""Matched Wan2.2-A14B T2V comparison on at most two exclusively clean H200s.

The two experts share one global 50-step schedule. Both requested methods use
10 dense steps and one dense layer, as explicitly requested on 2026-09-08.
The published SVG-EAR operator and the existing standard Hard C4 are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
MODEL_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
SVG_REVISION = "f89aedaf169ac2ae5b186bda674e53c3dc08c476"
SVG_ROOT = Path("/local2/denghaoran/GridMergeAttention_worktrees/svg_official_20260906ch")
MODEL = Path("/local2/denghaoran/GridMergeAttention/models/Wan2.2-T2V-A14B-Diffusers")
METHODS = ("dense", "ear", "gma_c4_mix")
STEPS, LAYERS, CFG_PASSES = 50, 40, 2
FIRST_STEPS, FIRST_LAYERS = 10, 1
CALLS = STEPS * LAYERS * CFG_PASSES
DENSE_CALLS = CFG_PASSES * (FIRST_STEPS * LAYERS + (STEPS - FIRST_STEPS) * FIRST_LAYERS)
SPARSE_CALLS = CALLS - DENSE_CALLS
DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness",
              "dynamic_degree", "aesthetic_quality", "imaging_quality")
STOP_REQUESTED = threading.Event()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def density_summary(values, count):
    require(len(values) == count * SPARSE_CALLS, "incomplete sparse density records")
    require(all(0 <= value <= 1 for value in values), "invalid density")
    sparse = sum(values) / len(values)
    effective = (sum(values) + count * DENSE_CALLS) / (count * CALLS)
    return dict(sparse_call_density=sparse, all_call_density=effective,
                sparse_call_sparsity=1 - sparse, all_call_sparsity=1 - effective,
                sparse_calls=len(values), dense_calls=count * DENSE_CALLS,
                definition="Exact retained token-pair area weighted by Q/K cluster populations; dense calls count as 1. Compensation and routing overhead are separate.")


class ExpertProxy:
    """Use one expert with the live pipeline timestep and shared scheduler."""
    def __init__(self, pipeline, expert):
        self.pipeline, self.transformer = pipeline, expert

    def __getattr__(self, key):
        return getattr(self.pipeline, key)


def legacy_rope_pair(rotary):
    """Convert Diffusers 0.35's duplicated real RoPE to SVG's complex layout."""
    import torch
    cosine, sine = rotary
    require(cosine.ndim == sine.ndim == 4 and cosine.shape == sine.shape,
            "unexpected Wan2.2 rotary layout")
    real = cosine[0, :, 0, 0::2].contiguous().float()
    imag = sine[0, :, 0, 1::2].contiguous().float()
    return torch.complex(real, imag)[None, None]


class NativeProcessorBridge:
    """Bridge native Wan2.2 RoPE/timesteps into the unchanged attention cores."""
    def __init__(self, inner, proxy):
        self.inner, self.proxy, self.rotary = inner, proxy, None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, timestep=None):
        import torch
        if rotary_emb is not None and self.rotary is None:
            frequencies = legacy_rope_pair(rotary_emb)[0, 0]
            self.rotary = (frequencies.real.contiguous(), frequencies.imag.contiguous())
        if self.rotary is not None:
            require(hidden_states.shape[1] == self.rotary[0].shape[0], "cached RoPE sequence length changed")
        current = self.proxy._current_timestep if timestep is None else timestep
        current = current.reshape(1) if isinstance(current, torch.Tensor) else torch.tensor([current], device=hidden_states.device)
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask,
                          self.rotary if rotary_emb is not None else None, current)


def configure_native_processors(proxy, backend, timer, full_timer, **kwargs):
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanTransformerBlock
    from analysis import benchmark_wan14b_720p_methods as bench
    model_forward, block_forward = WanTransformer3DModel.forward, WanTransformerBlock.forward
    try:
        audit = bench.configure_processors(proxy, backend, timer, full_timer,
            bench.AttentionErrorStats(0), SVG_ROOT, STEPS, 720, 1280, 81,
            wan_forward_backend="svg", qk_norm_backend="triton", rope_backend="svg",
            gma_preprocess_backend="separate", gma_output_layout="token", gma_head_chunk=8,
            svg_top_p=0.85, svg_min_kc_ratio=0.05, **kwargs)
    finally:
        WanTransformer3DModel.forward, WanTransformerBlock.forward = model_forward, block_forward
    for block in proxy.transformer.blocks:
        block.attn1.set_processor(NativeProcessorBridge(block.attn1.processor, proxy))
    return audit


def configure_vae_math_decode(vae):
    """Keep FP32 VAE decoding compatible without changing transformer SDPA."""
    from functools import wraps
    from torch.nn.attention import SDPBackend, sdpa_kernel
    original_decode = vae.decode

    @wraps(original_decode)
    def decode(*args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return original_decode(*args, **kwargs)

    vae.decode = decode


def validate_vae_decode(args):
    import torch
    from diffusers import AutoencoderKLWan
    from torch.nn.attention import SDPBackend, sdpa_kernel
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32,
                                          local_files_only=True).eval().to("cuda")
    vae.enable_tiling()
    configure_vae_math_decode(vae)
    torch.manual_seed(0)
    latent = torch.randn(1, 16, 21, 90, 160, device="cuda", dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = vae.decode(latent, return_dict=False)[0]
        require(not torch.backends.cuda.math_sdp_enabled(), "VAE leaked math backend into transformer")
        require(output.shape == (1, 3, 81, 720, 1280), "unexpected VAE output shape")
        require(torch.isfinite(output).all().item(), "VAE output is nonfinite")
    torch.cuda.synchronize()
    write_json(args.output / "setup/vae_compatibility.json", dict(valid=True,
        scope="Actual checkpoint VAE, FP32, tiled full 81-frame 720p decode under outer flash-only SDPA",
        output_shape=list(output.shape), elapsed_seconds=time.perf_counter() - started,
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        transformer_backend_restored=True, completed_utc=now()))


def validate_dispatch(audits, expert_forwards, method):
    expected_branch = "sparse" if method == "ear" else "gca"
    layers = [{branch: sum(a.layers[layer][branch] for a in audits)
               for branch in ("dense", "sparse", "gca")} for layer in range(LAYERS)]
    if method != "dense":
        for layer, counts in enumerate(layers):
            expected_dense = 100 if layer < FIRST_LAYERS else 20
            require(counts["dense"] == expected_dense and counts[expected_branch] == 100 - expected_dense,
                    f"incorrect warm-up dispatch at layer {layer}")
    forward_counts = Counter(row["step"] for row in expert_forwards)
    require(forward_counts == Counter({i: CFG_PASSES for i in range(STEPS)}), "missing denoising/CFG passes")
    require(all(len({r["expert"] for r in expert_forwards if r["step"] == i}) == 1 for i in range(STEPS)),
            "CFG passes used different experts")
    require({r["expert"] for r in expert_forwards} == {"high_noise", "low_noise"}, "both experts must execute")
    require(all(sum(row.values()) == 100 for row in layers), "missing layer calls")
    result = {branch: sum(row[branch] for row in layers) for branch in ("dense", "sparse", "gca")}
    result.update(per_layer=layers, expert_forwards=expert_forwards,
                  expert_forward_counts=dict(Counter(r["expert"] for r in expert_forwards)))
    return result


def validate_compatibility(args):
    """Exercise the native and SVG forwards on both tiny BF16 expert models."""
    import types
    import torch
    from diffusers import WanTransformer3DModel, UniPCMultistepScheduler
    from analysis import benchmark_wan14b_720p_methods as bench
    torch.manual_seed(0)
    models = [WanTransformer3DModel(num_attention_heads=2, attention_head_dim=128,
        in_channels=16, out_channels=16, text_dim=64, freq_dim=32, ffn_dim=512,
        num_layers=2).eval().to(device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    hidden = torch.randn(1, 16, 3, 8, 8, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)
    times = [torch.tensor([950], device="cuda"), torch.tensor([500], device="cuda")]
    with torch.no_grad():
        references = [m(hidden, t, text).sample for m, t in zip(models, times)]
    for index, reference in enumerate(references):
        print(json.dumps(dict(event="native_reference", expert=index,
            finite=bool(torch.isfinite(reference).all()),
            abs_max=reference.float().abs().max().item())), flush=True)
        require(torch.isfinite(reference).all().item(), "native tiny-model reference is nonfinite")
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler", local_files_only=True)
    pipeline = types.SimpleNamespace(scheduler=scheduler, _current_timestep=950)
    timer, full_timer = bench.AttentionTimer(), bench.AttentionTimer()
    rows = []
    for index, (model, timestep, reference) in enumerate(zip(models, times, references)):
        proxy = ExpertProxy(pipeline, model)
        configure_native_processors(proxy, "dense", timer, full_timer)
        pipeline._current_timestep = int(timestep.item())
        require(proxy._current_timestep == pipeline._current_timestep, "proxy cached the wrong expert timestep")
        with torch.no_grad():
            actual = model(hidden, timestep, text).sample
        print(json.dumps(dict(event="adapted_reference", expert=index,
            finite=bool(torch.isfinite(actual).all()),
            abs_max=actual.float().abs().max().item())), flush=True)
        relative_rmse = ((actual.float() - reference.float()).square().mean()
                         / reference.float().square().mean()).sqrt().item()
        require(torch.isfinite(actual).all().item() and relative_rmse < 0.02,
                f"Wan2.2/SVG forward mismatch: {relative_rmse}")
        rows.append(dict(expert=index, relative_rmse=relative_rmse,
                         max_absolute_error=(actual.float() - reference.float()).abs().max().item()))
    require(timer.finish()[1] == full_timer.finish()[1] == 4, "both experts were not instrumented")
    write_json(args.output / "setup/compatibility.json", dict(valid=True, experts=rows, completed_utc=now()))


def validate_ear_core(args):
    import torch
    from analysis import benchmark_wan14b_720p_methods as bench
    sys.path[:0] = [str(SVG_ROOT), str(SVG_ROOT / "svg/kernels/build")]
    bench.install_rocm_import_stubs()
    bench.install_flashinfer_workspace_compat()
    from svg.models.wan.attention import WanAttn_EARAttn_Processor
    processor = WanAttn_EARAttn_Processor(layer_idx=1)
    for name, value in dict(context_length=0, num_frame=1, frame_size=1024,
            first_layers_fp=1, first_times_fp=0, num_q_centroids=32, num_k_centroids=64,
            top_p_kmeans=0.85, min_kc_ratio=0.05, kmeans_iter_init=5, kmeans_iter_step=2,
            zero_step_kmeans_init=False, num_layers=40,
            logging_file=str(args.output / "setup/ear_smoke_density.jsonl")).items():
        setattr(processor, name, value)
    require(not Path(processor.logging_file).exists(), "fresh EAR validation log required")
    torch.manual_seed(0)
    tensors = [torch.randn(1, 2, 1024, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with torch.no_grad():
        for _ in range(2):
            output = processor.attention_core_logic(*tensors, torch.tensor([0], device="cuda"))
            require(output.shape == tensors[0].shape and torch.isfinite(output).all().item(), "EAR core returned invalid output")
    densities = [json.loads(line)["avg_density"] for line in Path(processor.logging_file).read_text().splitlines()]
    require(len(densities) == 2 and all(0 < value <= 1 for value in densities), "EAR density logging failed")
    write_json(args.output / "setup/ear_compatibility.json", dict(valid=True, densities=densities,
        scope="Small synthetic tensors; initial and continued clustering, routing, compensation and realized density", completed_utc=now()))


def worker(args):
    from utils.prompt_contract import (build_prompt_manifest, ensure_output_prompt_manifest,
                                       load_nonempty_prompts, validate_canonical_prompt_contract)
    prompt_path = ROOT / "prompts/MovieGenVideoBench_extended.txt"
    prompts = load_nonempty_prompts(prompt_path)
    validate_canonical_prompt_contract(prompt_path, prompts)
    video_dir = args.output / args.method / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    require(not list(video_dir.glob("*.mp4")) and not (video_dir / "runtime.json").exists(), "fresh outputs required")
    ensure_output_prompt_manifest(video_dir, build_prompt_manifest(prompt_path, prompts,
        selected_count=args.count, seed=0, num_samples=1, vbench_filenames=False, allow_noncanonical=False))
    subprocess.run([sys.executable, str(ROOT / "analysis/verify_prompt_contract.py"),
                    "--prompt_path", str(prompt_path), "--num_videos", str(args.count),
                    "--seed", "0", "--manifest_out", str(video_dir.parent / "prompt_contract.json")], check=True)
    import random
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    from analysis import benchmark_wan14b_720p_methods as bench
    require(torch.cuda.device_count() == 1, "worker must see one GPU")
    require(git(SVG_ROOT, "rev-parse", "HEAD") == SVG_REVISION, "SVG revision changed")
    require(json.loads((MODEL / "model_index.json").read_text())["boundary_ratio"] == 0.875,
            "wrong Wan2.2 checkpoint")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    write_json(video_dir.parent / "progress.json", dict(stage="loading_weights", updated_utc=now()))
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    require(pipe.transformer_2 is not None and float(pipe.config.boundary_ratio) == 0.875, "missing second expert")
    # Keep both BF16 experts resident. This fits one 141-GiB H200 without model offload.
    pipe.to("cuda")
    write_json(video_dir.parent / "progress.json", dict(stage="configuring_attention", updated_utc=now()))
    pipe.vae.enable_tiling()
    configure_vae_math_decode(pipe.vae)
    pipe.scheduler.set_timesteps(STEPS)
    timesteps = [float(t) for t in pipe.scheduler.timesteps]
    timer, full_timer = bench.AttentionTimer(), bench.AttentionTimer()
    audits, proxies, hooks = [], [], []
    stage = dict(step=0, forwards=[])
    os.environ["SVG_DENSITY_LOG"] = str(video_dir / "density.jsonl")
    for expert_name, expert in (("high_noise", pipe.transformer), ("low_noise", pipe.transformer_2)):
        require(len(expert.blocks) == LAYERS and expert.config.num_attention_heads == 40
                and expert.config.attention_head_dim == 128, "unexpected Wan2.2 expert architecture")
        proxy = ExpertProxy(pipe, expert)
        proxies.append(proxy)
        audit = configure_native_processors(proxy, args.method, timer, full_timer, output_dir=video_dir)
        if audit is not None:
            audits.append(audit)
        else:
            audit = bench.DiffusersDispatchAudit("dense", STEPS, LAYERS, STEPS, 0)
            audits.append(audit)
            for layer, block in enumerate(expert.blocks):
                def dense_call(module, positional, kwargs, which=layer, collector=audit):
                    collector.record(which, "dense")
                hooks.append(block.attn1.register_forward_pre_hook(dense_call, with_kwargs=True))
        def before_forward(module, positional, kwargs, name=expert_name):
            expected_expert = "high_noise" if timesteps[stage["step"]] >= 875 else "low_noise"
            require(name == expected_expert, "expert switch disagrees with checkpoint boundary")
            stage["forwards"].append(dict(step=stage["step"], expert=name))
        hooks.append(expert.register_forward_pre_hook(before_forward, with_kwargs=True))
    def step_finished(pipeline, index, timestep, callback_kwargs):
        require(index == stage["step"], "denoising step sequence changed")
        stage["step"] += 1
        write_json(video_dir.parent / "progress.json", dict(prompt_index=stage["index"],
            steps_completed=stage["step"], total_steps=STEPS, updated_utc=now()))
        return callback_kwargs
    protocol = dict(model=MODEL_REPO, model_revision=MODEL_REVISION, model_directory=str(MODEL),
        num_steps=STEPS, num_frames=81, resolution="1280x720", seed=0, videos=args.count,
        guidance_scale=4.0, guidance_scale_2=3.0, boundary_ratio=0.875,
        scheduler=dict(pipe.scheduler.config), timesteps=timesteps,
        dense_first_steps=FIRST_STEPS if args.method != "dense" else STEPS,
        dense_first_layers=FIRST_LAYERS, calls_per_video=CALLS,
        dense_calls_per_video=DENSE_CALLS if args.method != "dense" else CALLS,
        replacement_calls_per_video=SPARSE_CALLS if args.method != "dense" else 0,
        svg_top_p=0.85, svg_query_clusters=300, svg_key_clusters=1000,
        svg_min_kc_ratio=0.05, svg_kmeans_iterations=[50, 2], svg_commit=SVG_REVISION,
        gma="standard Hard C4, 4 fixed seeds, entropy bias, full compressed attention, head chunk 8",
        common_preprocessing="Native Diffusers Wan2.2 forward, Triton RMSNorm, SVG RoPE; separate QKV preprocessing for all methods",
        attention_timing="CUDA events, complete core and self-attention module, dense warm-up included",
        sdpa_backend="Transformer: PyTorch SDPA flash only; FP32 VAE decode: scoped math SDPA",
        source_commit=git(ROOT, "rev-parse", "HEAD"))
    write_json(video_dir.parent / "protocol.json", protocol)
    records = {}
    for index in range(args.count):
        stage.update(step=0, forwards=[], index=index)
        if args.method == "ear":
            for proxy in proxies:
                for block in proxy.transformer.blocks:
                    inner = block.attn1.processor.inner
                    inner.centroids_init, inner.q_centroids, inner.k_centroids = False, None, None
        for audit in audits:
            audit.begin_video(index)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        generator = torch.Generator(device="cuda").manual_seed(0)
        timer.reset()
        full_timer.reset()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        frames = pipe(prompt=prompts[index], negative_prompt=bench.NEGATIVE_PROMPT,
            height=720, width=1280, num_frames=81, num_inference_steps=STEPS,
            guidance_scale=4.0, guidance_scale_2=3.0, generator=generator,
            callback_on_step_end=step_finished).frames[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        core_ms, core_calls = timer.finish()
        full_ms, full_calls = full_timer.finish()
        require(core_calls == full_calls == CALLS and stage["step"] == STEPS, "incomplete attention timing")
        dispatch = validate_dispatch(audits, stage["forwards"], args.method)
        for audit in audits:
            audit.active_index = None
        video_path = video_dir / f"{index:03d}-00.mp4"
        export_started = time.perf_counter()
        export_to_video(frames, str(video_path), fps=16)
        row = dict(index=index, prompt=prompts[index], seed=0, video=video_path.name,
            attention_calls=core_calls, attention_total_ms=core_ms, mean_attention_ms=core_ms / core_calls,
            full_attention_calls=full_calls, full_attention_total_ms=full_ms, mean_full_attention_ms=full_ms / full_calls,
            inference_seconds=elapsed, export_seconds=time.perf_counter() - export_started,
            max_memory_gib=torch.cuda.max_memory_allocated() / 2**30, attention_dispatch=dispatch)
        records[index] = row
        payload = dict(backend=args.method, protocol=protocol, videos=list(records.values()), summary=bench.summarize(records))
        summary = payload["summary"]
        summary["attention_share"] = summary["steady_state_mean_attention_ms"] * CALLS / (1000 * summary["steady_state_mean_inference_seconds"])
        if args.method == "ear":
            values = [json.loads(line)["avg_density"] for line in (video_dir / "density.jsonl").read_text().splitlines()]
            payload["density"] = density_summary(values, index + 1)
            write_json(video_dir.parent / "density_summary.json", payload["density"])
        write_json(video_dir / "runtime.json", payload)
        print(json.dumps(dict(event="video_complete", method=args.method, index=index,
                             inference_seconds=elapsed, mean_attention_ms=core_ms / core_calls)), flush=True)


def guarded(command, output, gpu, label, environment=None):
    from analysis import launch_quality_first_videos as guard
    require(not STOP_REQUESTED.is_set(), "comparison interrupted")
    state = guard.require_clean(gpu)
    require(len(guard.account_devices()) < 2, "no free account GPU slot")
    output.mkdir(parents=True, exist_ok=True)
    env = guard.child_environment(output, state["gpu"]["uuid"])
    # Reuse installed bytecode; compiling a second cache on NFS delays imports.
    env.pop("PYTHONPYCACHEPREFIX", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(PYTHONPATH=os.pathsep.join((str(args_root(output) / "setup/python"), str(ROOT), str(SVG_ROOT))),
               LD_LIBRARY_PATH="/home/denghaoran/miniconda3/envs/monarch_rt/lib:/usr/local/cuda-12.9/lib64:" + env.get("LD_LIBRARY_PATH", ""))
    env.update(CUDA_HOME="/usr/local/cuda-12.9", PATH="/usr/local/cuda-12.9/bin:" + env["PATH"])
    if environment:
        env.update(environment)
    child, failure, code = None, None, None
    started = now()
    try:
        guard.require_clean(gpu, expected_uuid=state["gpu"]["uuid"])
        with (output / f"{label}.log").open("x") as log, (output / f"{label}_gpu.jsonl").open("x") as monitor:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / f"{label}_process.json", dict(pid=child.pid, parent_pid=os.getpid(), gpu=gpu, command=command))
            while True:
                require(not STOP_REQUESTED.is_set(), "comparison interrupted; stopping own worker")
                processes = guard.gpu_processes(state["gpu"]["uuid"])
                monitor.write(json.dumps(dict(utc=now(), processes=processes)) + "\n")
                monitor.flush()
                require(all(row["pid"] == child.pid for row in processes), "foreign GPU process; timing invalid")
                require(len(guard.account_devices()) <= 2, "account exceeded two GPUs")
                if child.poll() is not None:
                    break
                time.sleep(5)
            code = child.wait()
            require(code == 0, f"{label} exited {code}")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        guard.stop_own_child(child)
    receipt = dict(valid=failure is None, failure=failure, exit_code=code, command=command,
                   started_utc=started, completed_utc=now(), gpu=state["gpu"])
    write_json(output / f"{label}_completion.json", receipt)
    require(receipt["valid"], failure)


def args_root(output):
    output = Path(output)
    for directory in (output, *output.parents):
        if (directory / "setup").is_dir():
            return directory
    raise ValueError("comparison setup directory missing")


def generate(args, method, gpu):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--method", method,
               "--output", str(args.output), "--count", str(args.count)]
    guarded(command, args.output / method, gpu, "generation")


def evaluate(args, method, gpu):
    from utils.prompt_contract import assert_matching_prompt_manifests
    output = args.output / method / "results"
    assert_matching_prompt_manifests(args.output / "dense/videos", args.output / method / "videos")
    quality = [sys.executable, "-u", str(ROOT / "analysis/evaluate_video_quality.py"),
        "--reference_dir", str(args.output / "dense/videos"), "--method_dir", str(args.output / method / "videos"),
        "--method_name", method, "--output", str(output / "quality.json"),
        "--expected_videos", str(args.count), "--expected_frames", "81", "--lpips_batch_size", "8"]
    failures = {}
    try:
        guarded(quality, output, gpu, "quality")
    except Exception as error:
        failures["quality"] = str(error)
    vbench = output / "vbench"
    vbench.mkdir()
    subprocess.run([sys.executable, str(ROOT / "analysis/prepare_vbench_prompt_map.py"),
        "--prompt_path", str(ROOT / "prompts/MovieGenVideoBench_extended.txt"),
        "--video_dir", str(args.output / method / "videos"), "--output", str(vbench / "prompts.json")], check=True)
    cache = "/local2/denghaoran/GridMergeAttention/vbench_home/.cache"
    try:
        guarded([sys.executable, "-u", "/home/denghaoran/workspace/VBench/evaluate.py",
            "--videos_path", str(args.output / method / "videos"), "--dimension", *DIMENSIONS,
            "--mode", "custom_input", "--load_ckpt_from_local", "True", "--prompt_file", str(vbench / "prompts.json"),
            "--output_path", str(vbench)], output, gpu, "vbench", dict(VBENCH_CACHE_DIR=cache + "/vbench",
                XDG_CACHE_HOME=cache, TORCH_HOME=cache + "/torch", HF_HOME=cache + "/huggingface"))
    except Exception as error:
        failures["vbench"] = str(error)
    write_json(output / "evaluation_status.json", dict(valid=not failures, failures=failures,
        vbench_dimensions=DIMENSIONS, vbench_official_total_score=False))
    return failures


def coordinator(args):
    from concurrent.futures import ThreadPoolExecutor
    from analysis import launch_quality_first_videos as guard
    require(len(set(args.gpus)) == 2, "two distinct GPUs required")
    require(len(guard.account_devices()) == 0, "two account GPU slots required")
    initial = [guard.require_clean(gpu) for gpu in args.gpus]
    require((args.output / "setup/model_verified.json").is_file(), "model verification required")
    compatibility = json.loads((args.output / "setup/compatibility.json").read_text())
    require(compatibility.get("valid") is True, "Wan2.2 interface compatibility must pass")
    require(json.loads((args.output / "setup/ear_compatibility.json").read_text()).get("valid") is True,
            "EAR sparse-kernel compatibility must pass")
    require(json.loads((args.output / "setup/vae_compatibility.json").read_text()).get("valid") is True,
            "full-size FP32 VAE decode must pass")
    require(not git(ROOT, "diff", "HEAD", "--name-only"), "experiment source must be committed")
    require(not (args.output / "comparison.json").exists(), "fresh comparison required")
    write_json(args.output / "comparison.json", dict(created_utc=now(), initial_gpus=initial,
        model=MODEL_REPO, model_revision=MODEL_REVISION, source_commit=git(ROOT, "rev-parse", "HEAD"),
        svg_commit=SVG_REVISION, steps=STEPS, videos_per_method=args.count, seed=0,
        user_override="scai6, up to two H200s, identical 10-step and first-layer dense warm-up for EAR and standard C4",
        schedule=f"GPU {args.gpus[0]} generates Dense SDPA; GPU {args.gpus[1]} generates GMA-C4 then SVG-EAR; paired evaluation follows on both GPUs",
        vbench_dimensions=DIMENSIONS, vbench_official_total_score=False))
    failure = None
    try:
        def methods_lane():
            generate(args, "gma_c4_mix", args.gpus[1])
            generate(args, "ear", args.gpus[1])
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(generate, args, "dense", args.gpus[0]), pool.submit(methods_lane)]
            for task in tasks:
                task.result()
            tasks = [pool.submit(evaluate, args, method, gpu)
                     for method, gpu in zip(("gma_c4_mix", "ear"), args.gpus)]
            evaluation_failures = [task.result() for task in tasks]
        reference = json.loads((args.output / "dense/videos/runtime.json").read_text())
        methods = {}
        for method in ("ear", "gma_c4_mix"):
            result = json.loads((args.output / method / "videos/runtime.json").read_text())
            dense_summary, method_summary = reference["summary"], result["summary"]
            methods[method] = dict(summary=method_summary, density=result.get("density"),
                speedup_vs_sdpa=dense_summary["steady_state_mean_attention_ms"] / method_summary["steady_state_mean_attention_ms"],
                full_attention_speedup_vs_sdpa=dense_summary["steady_state_mean_full_attention_ms"] / method_summary["steady_state_mean_full_attention_ms"],
                end_to_end_speedup_vs_sdpa=dense_summary["steady_state_mean_inference_seconds"] / method_summary["steady_state_mean_inference_seconds"],
                timing_comparability="Same checkpoint, prompts, seed and measurement; separate generation trajectories, not an identical-QKV replay. Different H200s on scai6; cross-device speedup is approximate.")
        write_json(args.output / "summary.json", dict(dense=reference["summary"], methods=methods))
        require(not any(evaluation_failures), f"evaluation failures: {evaluation_failures}")
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    write_json(args.output / "comparison_completion.json", dict(valid=failure is None,
        failure=failure, completed_utc=now()))
    require(failure is None, failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gpus", nargs=2, type=int, default=[4, 5])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-ear", action="store_true")
    parser.add_argument("--validate-vae", action="store_true")
    args = parser.parse_args()
    if not args.worker and not args.validate:
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda *_: STOP_REQUESTED.set())
    require(args.count in (1, 20), "one-video integration test or full twenty-video comparison required")
    if args.validate_vae:
        validate_vae_decode(args)
    elif args.validate_ear:
        validate_ear_core(args)
    elif args.validate:
        validate_compatibility(args)
    elif args.worker:
        worker(args)
    else:
        coordinator(args)


if __name__ == "__main__":
    main()
