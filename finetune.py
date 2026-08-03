import argparse
import bisect
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper

# Per-channel statistics of the Wan2.1 VAE latent space, mirroring
# utils/wan_wrapper.py:WanVAECore. The FastVideo Wan-Syn parquets store raw VAE latents,
# while this repo's models operate on (raw - mean) / std. Shaped to broadcast over a
# [channel, frame, height, width] row latent.
_LATENT_MEAN = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]).view(-1, 1, 1, 1)
_LATENT_STD = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]).view(-1, 1, 1, 1)


class FastVideoLatentDataset(Dataset):
    """Reads FastVideo Wan-Syn parquet rows containing precomputed Wan latents."""

    def __init__(self, data_dir: str, max_files: int | None = None):
        self.files = sorted(Path(data_dir).rglob("*.parquet"))
        if max_files is not None:
            self.files = self.files[:max_files]
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")

        self.lengths = [pq.ParquetFile(path).metadata.num_rows for path in self.files]
        self.offsets = []
        total = 0
        for length in self.lengths:
            self.offsets.append(total)
            total += length
        self.total_rows = total
        self._cache_path = None
        self._cache_rows = None

    def __len__(self):
        return self.total_rows

    @staticmethod
    def _array_from_bytes(row: dict, prefix: str) -> np.ndarray:
        raw = row[f"{prefix}_bytes"]
        shape = row[f"{prefix}_shape"]
        dtype = np.dtype(row[f"{prefix}_dtype"])
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    def _read_file_rows(self, path: Path):
        if self._cache_path != path:
            self._cache_rows = pq.read_table(path).to_pylist()
            self._cache_path = path
        return self._cache_rows

    def __getitem__(self, index: int):
        if index < 0 or index >= self.total_rows:
            raise IndexError(index)
        file_index = bisect.bisect_right(self.offsets, index) - 1
        row_index = index - self.offsets[file_index]
        row = self._read_file_rows(self.files[file_index])[row_index]

        latent = torch.from_numpy(self._array_from_bytes(row, "vae_latent"))
        latent = (latent - _LATENT_MEAN) / _LATENT_STD
        text_embedding = torch.from_numpy(self._array_from_bytes(row, "text_embedding"))
        return {
            "latent": latent,
            "text_embedding": text_embedding,
            "caption": row.get("caption", ""),
        }


def collate_latent_batch(batch, text_len: int):
    latents = torch.stack([item["latent"] for item in batch])
    # Dataset latent is [channel, frame, height, width]; Wan wrapper expects [batch, frame, channel, height, width].
    latents = latents.permute(0, 2, 1, 3, 4).contiguous()

    text_embeddings = []
    for item in batch:
        embedding = item["text_embedding"]
        if embedding.shape[0] > text_len:
            embedding = embedding[:text_len]
        text_embeddings.append(embedding.contiguous())

    return {
        "latents": latents,
        "text_embeddings": text_embeddings,
        "captions": [item["caption"] for item in batch],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Dense Wan2.1 finetuning on FastVideo Wan-Syn parquet latents.")
    parser.add_argument("--config", default="configs/dense_wan_448p_train.yaml")
    parser.add_argument("--data_dir", default="data/wan_syn_smoke")
    parser.add_argument("--output_dir", default="checkpoints/dense_wan_syn_smoke")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--wandb_project", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max_steps must be at least 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be at least 1")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be greater than 0")
    return args


def init_wandb(args, config, attention_mode: str, dataset: FastVideoLatentDataset):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. "
            "Install the dependencies from requirements.txt."
        ) from exc

    run_config = vars(args).copy()
    run_config.update(
        {
            "attention_mode": attention_mode,
            "dataset_rows": len(dataset),
            "dataset_files": len(dataset.files),
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "model_config": OmegaConf.to_container(config, resolve=True),
        }
    )
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        group=args.wandb_group or None,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        config=run_config,
    )


def load_generator_checkpoint(generator: WanDiffusionWrapper, checkpoint_path: str, use_ema: bool):
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and ("generator" in state or "generator_ema" in state):
        key = "generator_ema" if use_ema and "generator_ema" in state else "generator"
        state = state[key]
    generator.load_state_dict(state)
    print(f"Loaded generator checkpoint from {checkpoint_path}")


def save_generator_checkpoint(generator: WanDiffusionWrapper, output_dir: Path, step: int, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint-step-{step:06d}.pt"
    payload = {
        "step": step,
        "generator": generator.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(payload, path)
    print(f"Saved {path}")


def choose_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def sample_timesteps(scheduler, batch: int, frames: int, device):
    timesteps = scheduler.timesteps.to(device=device)
    weights = getattr(scheduler, "linear_timesteps_weights", None)
    if weights is None:
        ids = torch.randint(0, timesteps.numel(), (batch,), device=device)
    else:
        ids = torch.multinomial(weights.to(device=device).float(), batch, replacement=True)
    values = timesteps[ids]
    return values[:, None].expand(batch, frames)


def main():
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm GPU is required for finetuning.")

    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = choose_dtype(args.dtype)
    config = OmegaConf.load(args.config)

    dataset = FastVideoLatentDataset(args.data_dir, max_files=args.max_files)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_latent_batch(batch, args.text_len),
        pin_memory=True,
    )

    model_kwargs = dict(config.get("model_kwargs", {}))
    generator = WanDiffusionWrapper(
        **model_kwargs,
        image_or_video_shape=config.image_or_video_shape,
    )
    generator.model.token_merging_args = dict(config.get("token_merging_args", {}))
    load_generator_checkpoint(generator, args.checkpoint, args.use_ema)
    generator = generator.to(device=device, dtype=dtype)
    generator.train()
    generator.model.train()
    generator.model.gradient_checkpointing = bool(args.gradient_checkpointing)

    optimizer = torch.optim.AdamW(
        (p for p in generator.model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    attention_mode = "token-merging" if config.get("token_merging_args", {}).get("enable", False) else "dense"
    wandb_run = init_wandb(args, config, attention_mode, dataset)
    print(
        f"Training {attention_mode} Wan on {len(dataset)} rows from {len(dataset.files)} parquet files, "
        f"dtype={dtype}, batch_size={args.batch_size}, "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
        f"max_steps={args.max_steps} optimizer updates."
    )

    output_dir = Path(args.output_dir)
    data_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="Finetuning", unit="update")
    micro_step = 0

    try:
        for global_step in range(1, args.max_steps + 1):
            accumulated_loss = 0.0
            for _ in range(args.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                micro_step += 1
                clean_latents = batch["latents"].to(device=device, dtype=dtype, non_blocking=True)
                text_embeddings = [
                    embedding.to(device=device, dtype=dtype, non_blocking=True)
                    for embedding in batch["text_embeddings"]
                ]
                noise = torch.randn_like(clean_latents)
                timesteps = sample_timesteps(
                    generator.scheduler, clean_latents.size(0), clean_latents.size(1), device
                )
                noisy_latents = generator.scheduler.add_noise(
                    clean_latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).unflatten(0, clean_latents.shape[:2])
                target = generator.scheduler.training_target(
                    clean_latents.flatten(0, 1),
                    noise.flatten(0, 1),
                    timesteps.flatten(0, 1),
                ).unflatten(0, clean_latents.shape[:2])

                flow_pred, _ = generator(
                    noisy_image_or_video=noisy_latents,
                    conditional_dict={"prompt_embeds": text_embeddings},
                    timestep=timesteps,
                )
                loss = F.mse_loss(flow_pred.float(), target.float())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at micro_step={micro_step}, global_step={global_step}: "
                        f"{loss.item()}"
                    )
                accumulated_loss += loss.item()
                (loss / args.gradient_accumulation_steps).backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                generator.model.parameters(), args.max_grad_norm, error_if_nonfinite=True
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at micro_step={micro_step}, "
                    f"global_step={global_step}: {grad_norm.item()}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            mean_loss = accumulated_loss / args.gradient_accumulation_steps
            progress.update(1)
            progress.set_postfix(loss=f"{mean_loss:.5f}", grad_norm=f"{grad_norm.item():.3f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": mean_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/micro_step": micro_step,
                    },
                    step=global_step,
                )
            if args.save_steps and global_step % args.save_steps == 0:
                save_generator_checkpoint(generator, output_dir, global_step, config)

        save_generator_checkpoint(generator, output_dir, args.max_steps, config)
    finally:
        progress.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("TOKEN_MERGING_FORCE_TORCH_RMSNORM", "1")
    main()
