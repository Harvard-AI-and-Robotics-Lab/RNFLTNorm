"""Train residual healthy-baseline v5 (mean map plus a bounded polar residual)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from rnfltnorm.config import V5Config, load_stats_into_v5_config
from rnfltnorm.data.dataset import FundusRnfltDataset, compute_mean_rnflt_map
from rnfltnorm.data.healthy_baseline import (
    apply_healthy_baseline_filter,
    apply_patient_split_from_table,
    load_or_build_glaucoma_cache,
)
from rnfltnorm.data.index import (
    apply_index_filters,
    dedupe_rnflt_rows,
    load_index_table,
    make_patient_holdout_split,
    patient_split_table,
)
from rnfltnorm.losses import ResidualLoss, batch_metrics
from rnfltnorm.models.predictor import ResidualPolarRnfltPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=V5Config.output_dir)
    p.add_argument("--data-dir", type=Path, default=V5Config.data_dir)
    p.add_argument("--index-csv", type=Path, default=V5Config.index_csv)
    p.add_argument("--stats-json", type=Path, default=V5Config.stats_json)
    p.add_argument("--rnflt-key", type=str, default=V5Config.rnflt_key)
    p.add_argument("--batch-size", type=int, default=V5Config.batch_size)
    p.add_argument("--num-workers", type=int, default=V5Config.num_workers)
    p.add_argument("--max-epochs", type=int, default=V5Config.max_epochs)
    p.add_argument("--backbone-lr", type=float, default=V5Config.backbone_lr)
    p.add_argument("--decoder-lr", type=float, default=V5Config.decoder_lr)
    p.add_argument("--warmup-epochs", type=int, default=V5Config.warmup_epochs)
    p.add_argument("--device", type=str, default=V5Config.device)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-crop-retina", action="store_true")
    p.add_argument("--no-canonicalize-eye", action="store_true")
    p.add_argument("--no-dedupe-rnflt", action="store_true")
    p.add_argument("--no-vessel-channel", action="store_true")
    p.add_argument("--healthy-label", type=int, default=V5Config.healthy_label)
    p.add_argument("--glaucoma-cache-csv", type=Path, default=V5Config.glaucoma_cache_csv)
    p.add_argument("--no-reuse-patient-split", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--limit-val-batches", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(unwrap(model)).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = unwrap(model)
        source_params = dict(source.named_parameters())
        for name, parameter in self.module.named_parameters():
            parameter.mul_(self.decay).add_(source_params[name], alpha=1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, buffer in self.module.named_buffers():
            buffer.copy_(source_buffers[name])


def make_dataset(cfg: V5Config, rows, augment: bool) -> FundusRnfltDataset:
    return FundusRnfltDataset(
        rows,
        cfg.data_dir,
        cfg.fundus_size,
        cfg.rnflt_min,
        cfg.rnflt_max,
        rnflt_key=cfg.rnflt_key,
        crop_retina=cfg.crop_retina,
        canonicalize_eye=cfg.canonicalize_eye,
        use_vessel_channel=cfg.use_vessel_channel,
        augment=augment,
    )


def lr_multiplier(epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    if epoch < warmup_epochs:
        return float(epoch + 1) / max(float(warmup_epochs), 1.0)
    progress = (epoch - warmup_epochs) / max(float(max_epochs - warmup_epochs), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def reduce_sums(values: Iterable[float], device: torch.device, distributed: bool) -> torch.Tensor:
    tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def load_or_build_mean_map(cfg: V5Config, train_df: pd.DataFrame, rank: int, distributed: bool) -> np.ndarray:
    mean_path = cfg.output_dir / cfg.mean_map_name
    if rank == 0:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        if not mean_path.is_file():
            print("computing training-set mean RNFLT map")
            mean, support = compute_mean_rnflt_map(
                train_df,
                cfg.data_dir,
                rnflt_key=cfg.rnflt_key,
                rnflt_min=cfg.rnflt_min,
                rnflt_max=cfg.rnflt_max,
                canonicalize_eye=cfg.canonicalize_eye,
            )
            np.savez_compressed(mean_path, mean=mean, support=support)
            print(f"wrote {mean_path}")
    if distributed:
        dist.barrier()
    payload = np.load(mean_path)
    return payload["mean"].astype(np.float32)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ResidualLoss,
    device: torch.device,
    mean_thickness: torch.Tensor,
    rnflt_scale: float,
    distributed: bool,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    names = ("loss", "mae_um", "correlation", "ssim", "mask_dice")
    sums = {name: 0.0 for name in names}
    count = 0
    for batch_idx, (fundus, target, valid_mask, _) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        fundus = fundus.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        output = model(fundus, mean_thickness)
        losses = criterion(
            output["thickness"],
            output["residual"],
            output["polar"],
            output["mask_logits"],
            target,
            valid_mask,
            mean_thickness,
        )
        metrics = batch_metrics(
            output["thickness"],
            output["mask_logits"],
            target,
            valid_mask,
            rnflt_scale,
        )
        batch_size = fundus.shape[0]
        sums["loss"] += float(losses["total"].item()) * batch_size
        for name in names[1:]:
            sums[name] += float(metrics[name].item()) * batch_size
        count += batch_size

    reduced = reduce_sums([sums[name] for name in names] + [count], device, distributed)
    total_count = max(float(reduced[-1].item()), 1.0)
    return {name: float(reduced[idx].item()) / total_count for idx, name in enumerate(names)}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cfg: V5Config,
    metrics: Dict[str, float],
    score: float,
    mean_rnflt: Optional[np.ndarray] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        payload = {
            "version": "v5-residual-polar",
            "epoch": epoch,
            "model": unwrap(model).state_dict(),
            "ema_model": ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg.to_dict(),
            "metrics": metrics,
            "score": score,
        }
        if mean_rnflt is not None:
            payload["mean_rnflt"] = np.asarray(mean_rnflt, dtype=np.float32)
        torch.save(payload, tmp)
        tmp.replace(path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"warning: failed to save {path}: {exc}")


@torch.no_grad()
def save_preview(
    model: torch.nn.Module,
    dataset: FundusRnfltDataset,
    output_path: Path,
    cfg: V5Config,
    mean_thickness: torch.Tensor,
) -> None:
    model.eval()
    n = min(cfg.preview_cases, len(dataset))
    if n == 0:
        return
    indices = np.linspace(0, len(dataset) - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 5, figsize=(16.5, 3.2 * n), squeeze=False)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("lightgray")
    device = next(model.parameters()).device
    mean_np = mean_thickness[0, 0].detach().cpu().numpy()
    for row_idx, dataset_idx in enumerate(indices):
        fundus, target, valid, meta = dataset[int(dataset_idx)]
        output = model(fundus[None].to(device), mean_thickness)
        pred = output["thickness"][0, 0].cpu().numpy()
        pred_mask = torch.sigmoid(output["mask_logits"])[0, 0].cpu().numpy() >= 0.5
        gt = target[0].numpy()
        gt_mask = valid[0].numpy().astype(bool)
        pred_um = pred * (cfg.rnflt_max - cfg.rnflt_min) + cfg.rnflt_min
        gt_um = gt * (cfg.rnflt_max - cfg.rnflt_min) + cfg.rnflt_min
        mean_um = mean_np * (cfg.rnflt_max - cfg.rnflt_min) + cfg.rnflt_min
        pred_plot = np.where(pred_mask, pred_um, np.nan)
        gt_plot = np.where(gt_mask, gt_um, np.nan)
        residual = np.where(gt_mask, pred_um - mean_um, np.nan)
        error = np.where(gt_mask, np.abs(pred_um - gt_um), np.nan)
        rgb = fundus[:3].permute(1, 2, 0).numpy()
        axes[row_idx, 0].imshow(np.clip(rgb, 0, 1))
        axes[row_idx, 0].set_title(str(meta["combined_npz"]))
        axes[row_idx, 1].imshow(pred_plot, cmap=cmap, vmin=0, vmax=350)
        axes[row_idx, 1].set_title("Predicted RNFLT")
        axes[row_idx, 2].imshow(gt_plot, cmap=cmap, vmin=0, vmax=350)
        axes[row_idx, 2].set_title("Corrected RNFLT")
        axes[row_idx, 3].imshow(residual, cmap="coolwarm", vmin=-40, vmax=40)
        axes[row_idx, 3].set_title("Pred − mean (µm)")
        axes[row_idx, 4].imshow(error, cmap="magma", vmin=0, vmax=80)
        axes[row_idx, 4].set_title("Absolute error (µm)")
        for axis in axes[row_idx]:
            axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank = init_distributed()
    cfg = load_stats_into_v5_config(V5Config())
    cfg.output_dir = args.output_dir
    cfg.data_dir = args.data_dir
    cfg.index_csv = args.index_csv
    cfg.stats_json = args.stats_json
    cfg.rnflt_key = args.rnflt_key
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.max_epochs = args.max_epochs
    cfg.backbone_lr = args.backbone_lr
    cfg.decoder_lr = args.decoder_lr
    cfg.warmup_epochs = args.warmup_epochs
    cfg.amp = not args.no_amp
    cfg.pretrained = not args.no_pretrained
    cfg.crop_retina = not args.no_crop_retina
    cfg.canonicalize_eye = not args.no_canonicalize_eye
    cfg.dedupe_rnflt = not args.no_dedupe_rnflt
    cfg.use_vessel_channel = not args.no_vessel_channel
    cfg.healthy_label = int(args.healthy_label)
    cfg.glaucoma_cache_csv = args.glaucoma_cache_csv
    cfg.reuse_patient_split = not args.no_reuse_patient_split

    seed = cfg.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = load_index_table(cfg.index_csv, dedupe_rnflt=False)
    df = apply_index_filters(
        df,
        imagetypes=cfg.imagetype_filter,
        viewingangle_max=cfg.viewingangle_max,
        delta_days_max=cfg.delta_days_max,
    )
    if cfg.dedupe_rnflt:
        df = dedupe_rnflt_rows(df)
    if cfg.healthy_baseline:
        if rank == 0:
            print(f"v5 healthy-baseline: residual_scale={cfg.residual_scale}")
            load_or_build_glaucoma_cache(df, cfg.data_dir, cfg.glaucoma_cache_csv)
        if distributed:
            dist.barrier()
        cache = pd.read_csv(cfg.glaucoma_cache_csv)
        n_before = len(df)
        df = apply_healthy_baseline_filter(df, cache, healthy_label=cfg.healthy_label)
        if rank == 0:
            n_eyes = int(df["eye_id"].nunique()) if len(df) else 0
            print(f"healthy-baseline pairs: {n_before} -> {len(df)} (eyes={n_eyes})")
    if cfg.reuse_patient_split and Path(cfg.patient_split_csv).is_file():
        train_df, val_df, holdout_df = apply_patient_split_from_table(
            df, cfg.patient_split_csv
        )
    else:
        train_df, val_df, holdout_df = make_patient_holdout_split(
            df,
            holdout_ratio=cfg.holdout_ratio,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
    if rank == 0:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        used_split = pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val"),
                holdout_df.assign(split="holdout"),
            ],
            ignore_index=True,
        )
        if len(used_split):
            used_split["patient_id"] = used_split["full_pid_rnflt"].map(
                lambda x: str(x).split("_")[0]
            )
            used_split[["patient_id", "split"]].drop_duplicates().to_csv(
                cfg.output_dir / "patient_split.csv", index=False
            )
        if not (cfg.reuse_patient_split and Path(cfg.patient_split_csv).is_file()):
            split_df = patient_split_table(
                df,
                holdout_ratio=cfg.holdout_ratio,
                val_ratio=cfg.val_ratio,
                seed=cfg.seed,
            )
            cfg.patient_split_csv.parent.mkdir(parents=True, exist_ok=True)
            split_df.to_csv(cfg.patient_split_csv, index=False)
        print(
            f"RNFLT exams — train: {len(train_df)}, val: {len(val_df)}, "
            f"holdout: {len(holdout_df)}"
        )

    mean_np = load_or_build_mean_map(cfg, train_df, rank, distributed)
    mean_thickness = torch.from_numpy(mean_np)[None, None].to(device)

    train_ds = make_dataset(cfg, train_df, augment=True)
    val_ds = make_dataset(cfg, val_df, augment=False)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=device.type == "cuda",
    )

    in_channels = 4 if cfg.use_vessel_channel else 3
    model = ResidualPolarRnfltPredictor(
        output_size=cfg.image_size,
        decoder_channels=cfg.decoder_channels,
        pretrained=cfg.pretrained,
        in_channels=in_channels,
        residual_scale=cfg.residual_scale,
        polar_bins=cfg.polar_bins,
        polar_inner_radius=cfg.polar_inner_radius,
        polar_outer_radius=cfg.polar_outer_radius,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": cfg.backbone_lr},
            {"params": model.decoder_parameters(), "lr": cfg.decoder_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: lr_multiplier(epoch, cfg.warmup_epochs, cfg.max_epochs),
    )
    ema = ModelEMA(model, cfg.ema_decay)
    criterion = ResidualLoss(
        cfg.image_size,
        l1_weight=cfg.l1_weight,
        peak_weight=cfg.peak_weight,
        residual_l1_weight=cfg.residual_l1_weight,
        ms_ssim_weight=cfg.ms_ssim_weight,
        gradient_weight=cfg.gradient_weight,
        polar_weight=cfg.polar_weight,
        polar_consistency_weight=cfg.polar_consistency_weight,
        mask_weight=cfg.mask_weight,
        polar_bins=cfg.polar_bins,
        polar_inner_radius=cfg.polar_inner_radius,
        polar_outer_radius=cfg.polar_outer_radius,
        residual_scale=cfg.residual_scale,
    ).to(device)
    scaler = GradScaler(enabled=cfg.amp and device.type == "cuda")

    start_epoch = 1
    best_score = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.module.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("score", float("inf")))
        if rank == 0:
            print(f"resumed {args.resume} at epoch {start_epoch}")

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        print(
            f"v5 model={cfg.backbone}-FPN residual_scale={cfg.residual_scale} "
            f"polar={cfg.polar_weight} "
            f"in_channels={in_channels} train_steps={len(train_loader)}"
        )

    scale_um = cfg.rnflt_max - cfg.rnflt_min
    global_step = 0
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}", disable=rank != 0)
        for batch_idx, (fundus, target, valid_mask, _) in enumerate(progress):
            if args.limit_train_batches is not None and batch_idx >= args.limit_train_batches:
                break
            fundus = fundus.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp and device.type == "cuda"):
                output = model(fundus, mean_thickness)
                losses = criterion(
                    output["thickness"],
                    output["residual"],
                    output["polar"],
                    output["mask_logits"],
                    target,
                    valid_mask,
                    mean_thickness,
                )
                loss = losses["total"]
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            global_step += 1
            if rank == 0 and global_step % cfg.log_every_steps == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l1=f"{losses['l1'].item():.4f}",
                    polar=f"{losses['polar'].item():.4f}",
                )

        metrics = validate(
            ema.module,
            val_loader,
            criterion,
            device,
            mean_thickness,
            scale_um,
            distributed,
            limit_batches=args.limit_val_batches,
        )
        scheduler.step()
        score = (
            metrics["mae_um"] / 20.0
            + (1.0 - metrics["correlation"])
            + (1.0 - metrics["ssim"])
            + 0.25 * (1.0 - metrics["mask_dice"])
        )
        if rank == 0:
            print(
                f"epoch {epoch} val_loss={metrics['loss']:.4f} "
                f"mae={metrics['mae_um']:.2f}um corr={metrics['correlation']:.4f} "
                f"ssim={metrics['ssim']:.4f} mask_dice={metrics['mask_dice']:.4f} "
                f"score={score:.4f}"
            )
            save_checkpoint(
                cfg.output_dir / "checkpoint_latest.pt",
                epoch,
                model,
                ema,
                optimizer,
                scheduler,
                cfg,
                metrics,
                score,
                mean_rnflt=mean_np,
            )
            if score < best_score:
                best_score = score
                save_checkpoint(
                    cfg.output_dir / "checkpoint_best.pt",
                    epoch,
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    cfg,
                    metrics,
                    score,
                    mean_rnflt=mean_np,
                )
                print(f"new best -> {cfg.output_dir / 'checkpoint_best.pt'}")
            if epoch == 1 or epoch % cfg.preview_every_epochs == 0:
                try:
                    save_preview(
                        ema.module,
                        val_ds,
                        cfg.output_dir / "samples" / f"epoch_{epoch:03d}.png",
                        cfg,
                        mean_thickness,
                    )
                except Exception as exc:
                    print(f"warning: failed to save preview: {exc}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
