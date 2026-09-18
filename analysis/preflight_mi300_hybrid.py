"""Exercise the actual hybrid core on a saved full-shape real QKV call."""
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch


def run(benchmark, method, sparse_root, output):
    from diffusers import UniPCMultistepScheduler
    class Attention:
        processor = None
        def set_processor(self, value):
            self.processor = value
    pipe = SimpleNamespace(
        scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
            num_train_timesteps=1000, flow_shift=5.0),
        transformer=SimpleNamespace(blocks=[SimpleNamespace(attn1=Attention()) for _ in range(40)],
            config=SimpleNamespace(patch_size=(1,2,2))),
        vae_scale_factor_temporal=4, vae_scale_factor_spatial=8, device=torch.device('cuda'))
    timer, full = benchmark.AttentionTimer(), benchmark.AttentionTimer()
    previous_log = os.environ.pop('SVG_DENSITY_LOG', None)
    try:
        audit = benchmark.configure_processors(pipe, method, timer, full,
            benchmark.AttentionErrorStats(0), sparse_root, 50, 720, 1280, 81,
            output_dir=output)
        schedule = pipe.scheduler
        schedule.set_timesteps(50)
        observed = {'dense': 0, 'method': 0}
        for step in schedule.timesteps:
            for block in pipe.transformer.blocks:
                proc = block.attn1.processor
                dense = proc.layer_idx < proc.first_layers_fp or step > proc.first_times_fp
                observed['dense' if dense else 'method'] += 2
        assert observed == {'dense':880, 'method':3120}, observed
        capture = torch.load('/work1/jasoncong/denghaoran/GridMergeAttention/hip_20260914/real_full40_step0_layer19.pt',
                             map_location='cpu', weights_only=True)
        qkv = [capture[k].reshape(1,75600,40,128).transpose(1,2).contiguous().cuda()
               for k in ('q','k','v')]
        del capture
        audit.begin_video(0)
        # Full-sized non-dense call compiles and validates the actual ROCm path
        # before model loading or the long dense warmup schedule.
        y = pipe.transformer.blocks[1].attn1.processor.attention_core_logic(
            *qkv, torch.tensor([0], device='cuda'))
        torch.cuda.synchronize()
        assert tuple(y.shape) == (1,40,75600,128)
        assert bool(torch.isfinite(y).all()), 'non-finite hybrid output'
        (output / 'native_preflight.json').write_text(json.dumps(dict(
            valid=True, method=method, observed_schedule=observed,
            shape=list(y.shape), finite=True, dispatch=audit.layers[1]), indent=2))
        del qkv, y, pipe, audit
    finally:
        if previous_log is not None:
            os.environ['SVG_DENSITY_LOG'] = previous_log
    gc.collect()
    torch.cuda.empty_cache()
