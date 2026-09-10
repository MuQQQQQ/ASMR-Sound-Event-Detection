from __future__ import annotations

import argparse
import math
import os
import platform
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RandomFileSubsetBatchSampler, SEDWindowDataset, split_files
from model import EMA, ResNetConformerSED, UNetConformerSED
from utils import (
    build_label_map,
    color_map,
    load_annotations,
    save_checkpoint,
    save_json,
)

warnings.filterwarnings("ignore")


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,          # 🔥 控制周期数
    use_restarts: bool = False,   # 🔥 开关
    min_lr_ratio: float = 0.0,    # 最低 lr 比例
):
    """Create warmup + cosine (optionally restarted) LR scheduler."""
    """
    Args:
        optimizer
        num_warmup_steps
        num_training_steps
        num_cycles: cosine 周期数（只有 use_restarts=True 时生效）
        use_restarts: 是否使用多周期 cosine
        min_lr_ratio: 最低 lr = base_lr * min_lr_ratio
    """

    def lr_lambda(current_step: int):

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        if not use_restarts:
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        else:
            progress_in_cycle = (progress * num_cycles) % 1.0
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress_in_cycle))

        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, num_classes: int, device: str, checkpoint: str = None):
    """Build model instance from command-line arguments."""
    audio_channels = 2 if args.audio_mode == "stereo" else 1
    if args.model_type == "resnet":
        model = ResNetConformerSED(
            num_classes=num_classes,
            sample_rate=args.sample_rate,
            use_specaug=args.use_specaug,
            freq_mask_param=args.freq_mask_param,
            time_mask_param=args.time_mask_param,
            feature_extractor=args.feature_extractor,
            device=device,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            audio_channels=audio_channels,
            conformer_layers=4,
            conformer_heads=4,
            conformer_dim=256
        ).to(device)

    else:
        model = UNetConformerSED(
            num_classes=num_classes,
            sample_rate=args.sample_rate,
            use_specaug=args.use_specaug,
            freq_mask_param=args.freq_mask_param,
            time_mask_param=args.time_mask_param,
            feature_extractor=args.feature_extractor,
            audio_channels=audio_channels,
        ).to(device)
    if checkpoint is not None and os.path.exists(checkpoint):
        print(f"Loading checkpoint from {checkpoint}")
        payload = torch.load(checkpoint, map_location=device)
        payload_state_dict = payload.get("model_state_dict", payload)
        model.load_state_dict(payload_state_dict, strict=False)
    return model


def build_criterion(loss_type: str):
    """Build loss function from name."""
    pos_weight = torch.tensor([
        2.0,   # Speech
        8,   # Chewing
        5,   # Mouth_Sounds
        4,   # Breathing
        1.0,  # Tapping
        5.0,  # Water_Bottle
        10.0,  # Slime
        8.0,   # Rub
        5.0,  # Scrub
        3.0,  # Rasp
    ], device='cuda')
    gap = torch.ones(50).cuda()
    pos_weight = torch.cat([pos_weight, gap])
    if loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == "weighted":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = alpha * (1 - p_t) ** gamma * bce_loss
        return loss.mean()

    return focal_loss


def build_scheduler(args, optimizer, train_loader_len: int):
    """Build learning-rate scheduler from command-line arguments."""
    if args.lr_scheduler == "cosine":
        warmup_epochs = 5
        base_lr = args.lr
        min_lr = 1e-6

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs

            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr / base_lr + (1 - min_lr / base_lr) * cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda
        )
        return scheduler
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5000000, gamma=0.1)
    if args.lr_scheduler == "custom":
        total_steps = train_loader_len * args.epochs
        warmup_steps = int(0.1 * total_steps)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            use_restarts=True,
            num_cycles=3,
        )
    return None


def train_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer,
    lr_scheduler=None,
    amp_enabled=False,
    amp_dtype=torch.float16,
    scaler=None,
    ema=None,
    dataset_type='group',
    virtual_classes=0
):
    model.train()
    total_loss = 0.0
    n = 0

    autocast_ctx = (
        lambda: torch.amp.autocast(
            device_type=device,
            dtype=amp_dtype,
            enabled=amp_enabled,
        )
        if amp_enabled else nullcontext()
    )

    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        if dataset_type == 'slice':
            x = batch[0].to(device, non_blocking=True)
            x = x.mean(dim=1, keepdim=True)
            y = batch[1].to(device, non_blocking=True)
        elif dataset_type == 'group':
            x = batch['waveform'].to(device, non_blocking=True)
            y = batch['target'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits = model(x)
            t = min(logits.size(1), y.size(1))
            logits = logits[:, :t]
            y = y[:, :t]
            loss = criterion(logits, y)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if ema is not None:
            ema.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        del x, y, logits, loss
    if lr_scheduler is not None:
        lr_scheduler.step()
    return total_loss / max(1, n)


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    amp_enabled=False,
    amp_dtype=torch.float16,
    ema=None,
    virtual_classes=0
):
    model.eval()

    if ema is not None:
        ema.apply_shadow()

    total_loss = 0.0
    n = 0

    all_probs = []
    all_targets = []

    autocast_ctx = (
        lambda: torch.amp.autocast(
            device_type=device,
            dtype=amp_dtype,
            enabled=amp_enabled,
        )
        if amp_enabled else nullcontext()
    )

    pbar = tqdm(loader, desc="val", leave=False)

    for batch in pbar:
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with autocast_ctx():
            logits = model(x)
            t = min(logits.size(1), y.size(1))
            logits = logits[:, :t]
            y = y[:, :t]
            loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        if virtual_classes != 0:
            probs = probs[:, :, :-virtual_classes]
            y = y[:, :, :-virtual_classes]

        all_probs.append(probs.float().cpu())
        all_targets.append(y.float().cpu())

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        del x, y, logits, probs, loss

    if ema is not None:
        ema.restore()

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # [BS, T, C] -> [BS*T, C]
    probs = probs.reshape(-1, probs.shape[-1])
    targets = targets.reshape(-1, targets.shape[-1])

    ap = average_precision_score(
        targets,
        probs,
        average=None,
    )

    map_score = np.nanmean(ap)

    thresholds = np.linspace(0.05, 1, 30)
    f1s = np.zeros(len(thresholds))
    ps = np.zeros(len(thresholds))
    rs = np.zeros(len(thresholds))
    best_f1 = 0.0
    best_threshold = 0.5
    for i, threshold in enumerate(thresholds):
        preds = probs >= threshold

        targets_bool = targets.astype(bool)

        tp = np.logical_and(preds, targets_bool).sum(axis=0)
        fp = np.logical_and(preds, ~targets_bool).sum(axis=0)
        fn = np.logical_and(~preds, targets_bool).sum(axis=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        macro_precision = np.nanmean(precision)
        macro_recall = np.nanmean(recall)
        macro_f1 = np.nanmean(f1)

        f1s[i] = macro_f1
        ps[i] = macro_precision
        rs[i] = macro_recall
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = threshold

    ziped = (thresholds, f1s, ps, rs)
    return (
        total_loss / max(1, n),
        float(map_score),
        float(macro_precision),
        float(macro_recall),
        float(macro_f1),
        ap,
        ziped,
        best_threshold,
    )


def plot_loss(
    train_loss_history,
    val_loss_history,
    lr_history,
    val_map_history,
    val_ap_history,
    idx_to_label,
    log_dir,
    f1s
):
    plt.figure(figsize=(10, 8))

    # =========================
    # 1. Loss
    # =========================
    plt.subplot(2, 2, 1)

    plt.plot(train_loss_history, label="Train Loss", color="#2953dcff")
    plt.plot(val_loss_history, label="Val Loss", color="#ff820eff")

    plt.xlabel("Epoch")
    plt.ylim(0, 0.005)
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid()

    # =========================
    # 2. mAP
    # =========================
    plt.subplot(2, 2, 2)

    plt.plot(val_map_history, label="Val mAP", color="#d8907eff")

    if val_map_history:
        best_epoch, best_map = max(
            enumerate(val_map_history),
            key=lambda x: x[1],
        )

        plt.scatter(
            best_epoch,
            best_map,
            color="red",
            s=50,
            zorder=5,
        )

        plt.annotate(
            f"{best_map:.4f}",
            (best_epoch, best_map),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.ylim(0, 1)

    plt.title("Validation mAP")
    plt.grid()

    # =========================
    # 3. Per-class AP
    # =========================
    plt.subplot(2, 2, 3)

    ap_history = np.asarray(val_ap_history)

    for idx in range(ap_history.shape[1]):
        label = idx_to_label[idx]
        color = color_map[label]
        plt.plot(
            ap_history[:, idx],
            label=label,
            color=color
        )

    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.ylim(0, 1)
    plt.title("Per-class AP")
    plt.legend(fontsize=8, loc='upper left')
    plt.grid()

    # plot ziped f1s
    plt.subplot(2, 2, 4)
    plt.plot(f1s[0], f1s[1], label='F1')
    plt.plot(f1s[0], f1s[2], label='Precision')
    plt.plot(f1s[0], f1s[3], label='Recall')

    plt.xlabel("Threshold")
    plt.legend()
    plt.tight_layout()

    try:
        plt.savefig(
            os.path.join(log_dir, "training_curves.png"),
            dpi=150,
        )
    except:
        pass
    plt.close()
# 5, 00:05:35


def main():
    """Training script entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="mono_data_all")
    ap.add_argument("--annotations", type=str,
                    default="mono_data_all/1.csv,mono_data_all/2.csv,mono_data_all/3.csv")
    # "data/anno.csv,data/annotations.csv,data/anno3.csv,data/anno4.csv,data/anno5.csv,data/anno6.csv,data/anno7.csv"
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--frame_hop_sec", type=float, default=0.02)
    ap.add_argument("--windows_per_file", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--use_specaug", action="store_true", default=False)
    ap.add_argument("--use_mixup", action="store_true", default=False)
    ap.add_argument("--freq_mask_param", type=int, default=20)
    ap.add_argument("--time_mask_param", type=int, default=40)
    ap.add_argument("--log_dir", type=str, default="mono_logs")
    ap.add_argument("--feature_extractor", type=str,
                    default="melspec", choices=["wav2vec2", "melspec", 'panns', 'fbank'])
    ap.add_argument("--amp", dest="amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bfloat16",
                    choices=["float16", "bfloat16"])
    ap.add_argument("--use_ema", action="store_true", default=False)

    ap.add_argument("--model_type", type=str, default="resnet",
                    choices=["resnet", "unet"])
    ap.add_argument("--lr_scheduler", type=str, default="custom",
                    choices=["cosine", "step", "custom", "none"])

    ap.add_argument("--loss_type", type=str, default="bce",
                    choices=["bce", "focal", "weighted"])
    ap.add_argument("--n_fft", type=int, default=1024)

    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--hop_length", type=int, default=320)
    ap.add_argument("--audio_mode", type=str, default="mono",
                    choices=["mono", "stereo"])

    ap.add_argument("--use_lazy_loading", action="store_true", default=False,
                    help="Enable lazy loading (load windows on demand)")
    ap.add_argument("--files_per_batch", type=int, default=60,
                    help="If >0, each train batch is sampled from this many random audio files")
    ap.add_argument("--batches_per_group", type=int, default=4,
                    help="When using --files_per_batch, number of batches to sample before reshuffling files")
    ap.add_argument("--virtual_classes", type=int, default=1)
    ap.add_argument("--dataset_type", type=str,
                    default='group', choices=["slice", "group"])
    # early stop 296 4min
    ap.add_argument("--early_stop_patience", type=int, default=30)
    args = ap.parse_args()

    args.batches_per_group = int(0.5*args.files_per_batch *
                                 args.windows_per_file // args.batch_size)
    print(f"Calculated batches_per_group: {args.batches_per_group}")
    current_time = time.localtime()
    # log_dir = os.path.join(
    #     args.log_dir, time.strftime("%Y%m%d-%H%M%S", current_time))
    os.makedirs(args.log_dir, exist_ok=True)
    num_logs = len([d for d in os.listdir(args.log_dir)
                   if os.path.isdir(os.path.join(args.log_dir, d))])
    log_dir = os.path.join(args.log_dir, f"run_{num_logs+1:03d}")
    os.makedirs(log_dir, exist_ok=True)

    # save config
    with open(os.path.join(log_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = bool(args.amp and device == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    print(f"Using device: {device}")
    print(f"AMP enabled: {amp_enabled} (dtype={args.amp_dtype})")

    annotation_files = args.annotations.split(",")
    total_df = []
    for af in annotation_files:
        if not os.path.exists(af):
            raise FileNotFoundError(f"Annotation file not found: {af}")
        df = load_annotations(af)
        total_df.append(df)

    df = pd.concat(total_df, ignore_index=True)
    label_to_idx, idx_to_label = build_label_map(df, args.virtual_classes)
    files = sorted(df["filename"].unique().tolist())
    train_files, val_files = split_files(
        files, val_ratio=args.val_ratio, seed=args.seed)
    # train_files = train_files[:3]
    # train_files = ['out000.mp3', 'out001.mp3', 'out003.mp3',
    #                'out004.mp3', 'out005.mp3', 'out006.mp3']
    # val_files = ['out002.mp3', 'out007.mp3']
    print(f"Classes ({len(label_to_idx)}): {list(label_to_idx.keys())}")
    print(f"Train files: {train_files}, {len(train_files)}")
    print(f"Val files: {val_files}, {len(val_files)}")
    executor = ThreadPoolExecutor(max_workers=4)
    futures = []
    if args.dataset_type == 'group':
        train_ds = SEDWindowDataset(
            data_dir=args.data_dir,
            annotations=df,
            files=train_files,
            label_to_idx=label_to_idx,
            sample_rate=args.sample_rate,
            window_sec=args.window_sec,
            frame_hop_sec=args.frame_hop_sec,
            windows_per_file=args.windows_per_file,
            training=True,
            seed=args.seed,
            audio_mode=args.audio_mode,
            use_lazy_loading=args.use_lazy_loading,
        )
    elif args.dataset_type == 'slice':
        # 不必在意
        train_ds = AudioDataset(
            root=r'mono_data_all\slices',
            files=train_files,
        )
    val_ds = SEDWindowDataset(
        data_dir=args.data_dir,
        annotations=df,
        files=val_files,
        label_to_idx=label_to_idx,
        sample_rate=args.sample_rate,
        window_sec=args.window_sec,
        frame_hop_sec=args.frame_hop_sec,
        windows_per_file=max(20, args.windows_per_file // 4),
        training=False,
        seed=args.seed,
        full_audio_eval=True,
        eval_hop_sec=10,
        audio_mode=args.audio_mode,
        use_lazy_loading=False,
    )

    if args.files_per_batch > 0 and args.dataset_type == 'group':
        train_batch_sampler = RandomFileSubsetBatchSampler(
            dataset=train_ds,
            batch_size=args.batch_size,
            files_per_batch=args.files_per_batch,
            batches_per_group=args.batches_per_group,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=0
        )
    elif args.dataset_type == 'slice':
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )  # 4, 4min39s; 6, 4min49; 2, 4min28; 3, 4min30
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args, num_classes=len(label_to_idx),
                        device=device, checkpoint=args.checkpoint)
    model.use_specaug = args.use_specaug
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=0.999)

    criterion = build_criterion(args.loss_type)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr)
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    scaler = None
    if amp_enabled:
        try:
            scaler = torch.amp.GradScaler(device=device, enabled=amp_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)

    best_val = 0
    history = []
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    val_map_history = []
    val_f1_history = []
    val_ap_history = []
    train_loader.dataset.use_mixup = args.use_mixup
    timer = time.time()
    for ep in range(1, args.epochs + 1):
        if ep > 0.7*args.epochs:
            model.use_specaug = False
            train_loader.dataset.use_mixup = False

        tr = train_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            ema=ema if args.use_ema else None,
            dataset_type=args.dataset_type,
            virtual_classes=args.virtual_classes
        )
        va, val_map, val_precision, val_recall, val_f1, class_ap, f1s, best_threshold = validate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            ema=ema if args.use_ema else None,
            virtual_classes=args.virtual_classes
        )

        history.append({
            "epoch": ep,
            "train_loss": tr,
            "val_loss": va,
            "val_map": val_map,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "class_ap": class_ap.tolist(),
            "best_threshold": best_threshold,
        })

        train_loss_history.append(tr)
        val_loss_history.append(va)
        val_map_history.append(val_map)
        val_ap_history.append(class_ap)
        lr_history.append(lr_scheduler.get_last_lr()[
                          0] if lr_scheduler else None)
        # clear memory
        torch.cuda.empty_cache()

        time_used = time.time() - timer
        # format
        used_hours, rem = divmod(time_used, 3600)
        used_minutes, used_seconds = divmod(rem, 60)
        avg_epoch_time = time_used / ep
        eta = (args.epochs - ep) * avg_epoch_time
        eta_hours, rem = divmod(eta, 3600)
        eta_minutes, eta_seconds = divmod(rem, 60)

        print(
            f"E {ep:03d}: t_loss={tr:.4f} v_loss={va:.4f}, mAP={val_map:.4f}, best_threshold={best_threshold:.2f}, lr={lr_scheduler.get_last_lr()[0] if lr_scheduler else 'N/A':.6f}. Time elapsed: {int(used_hours):02d}:{int(used_minutes):02d}:{int(used_seconds):02d}. ETA: {int(eta_hours):02d}:{int(eta_minutes):02d}:{int(eta_seconds):02d}")
        # plot_loss(train_loss_history,val_loss_history,lr_history,val_map_history,val_ap_history,idx_to_label,log_dir,f1s )
        executor.submit(
            plot_loss,
            train_loss_history,
            val_loss_history,
            lr_history,
            val_map_history,
            val_ap_history,
            idx_to_label,
            log_dir,
            f1s if f1s is not None else None,
        )
        if val_map > best_val:
            best_val = val_map
            if args.use_ema:
                state_dict = model.state_dict()

                for name, value in ema.shadow.items():
                    state_dict[name] = value.detach().cpu().clone()
            else:
                state_dict = model.state_dict()

            payload = {
                "model_state_dict": state_dict,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "sample_rate": args.sample_rate,
                "frame_hop_sec": args.frame_hop_sec,
                "window_sec": args.window_sec,
                "audio_mode": args.audio_mode,
                "audio_channels": (2 if args.audio_mode == "stereo" else 1),
                "history": history,
            }

            save_checkpoint(os.path.join(
                log_dir, "best_checkpoint.pth"), payload)
            print(
                f"Saved best checkpoint to {os.path.join(log_dir, 'best_checkpoint.pth')}")

    save_json(os.path.join(log_dir, "history.json"), {"history": history})
    if args.use_ema:
        state_dict = model.state_dict()

        for name, value in ema.shadow.items():
            state_dict[name] = value.detach().cpu().clone()
    else:
        state_dict = model.state_dict()
    payload = {
        "model_state_dict": state_dict,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "sample_rate": args.sample_rate,
        "frame_hop_sec": args.frame_hop_sec,
        "window_sec": args.window_sec,
        "audio_mode": args.audio_mode,
        "audio_channels": (2 if args.audio_mode == "stereo" else 1),
        "history": history,
    }

    save_checkpoint(os.path.join(
        log_dir, "last_checkpoint.pth"), payload)
    print("Training done.")
    executor.shutdown(wait=True)
    return best_val


if __name__ == "__main__":
    best_val = main()

# python src/train.py --amp --virtual_classes 50 --loss_type focal --lr_scheduler cosine --lr 1e-4 --use_ema --batch_size 30 --dataset_type group --epochs 60;python src/train.py --amp --virtual_classes 50 --sample_rate 48000 --n_fft 3072 --hop_length 960 --win_length 3072 --n_mels 256 --loss_type focal --lr_scheduler cosine --lr 1e-4 --use_ema --batch_size 30 --dataset_type group --epochs 60;python src/train.py --amp --virtual_classes 50 --sample_rate 48000 --n_fft 3072 --hop_length 960 --win_length 3072 --n_mels 256 --loss_type focal --lr_scheduler cosine --lr 1e-4 --use_ema --batch_size 30 --dataset_type slice
