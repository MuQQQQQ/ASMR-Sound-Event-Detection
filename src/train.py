from __future__ import annotations

import argparse
import math
import os
import platform
import random
import time
from contextlib import nullcontext

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SEDWindowDataset, split_files
from model import EMA, ResNetConformerSED, UNetConformerSED
from utils import build_label_map, load_annotations, save_checkpoint, save_json


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

        # =========================
        # 1️⃣ Warmup
        # =========================
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # =========================
        # 2️⃣ 进入 cosine 区间
        # =========================
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        # =========================
        # 🥇 单周期 cosine
        # =========================
        if not use_restarts:
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # =========================
        # 🥈 多周期 cosine（restart）
        # =========================
        else:
            progress_in_cycle = (progress * num_cycles) % 1.0
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress_in_cycle))

        # =========================
        # 3️⃣ 最小 lr 控制
        # =========================
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# 针对 Windows 环境下 VS Code 调试器找不到 CUDA DLL 的修复
if platform.system() == "Windows":
    # 填入你刚才确认的路径
    conda_bin_custom = r"C:\Users\11703\AppData\Local\miniconda3\envs\py39\bin"

    if os.path.exists(conda_bin_custom):
        # 核心：告诉 Python 解释器去哪里找 DLL
        os.add_dll_directory(conda_bin_custom)
        # 兼容性补充：同时更新环境变量
        os.environ['PATH'] = conda_bin_custom + os.pathsep + os.environ['PATH']
        print(f"成功挂载 DLL 路径: {conda_bin_custom}")
    else:
        print("警告：指定的 Conda bin 路径不存在，请检查路径是否正确。")
matplotlib.use('Agg')  # 使用 Agg 后端，只负责生成图片文件，不弹出窗口


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, num_classes: int, device: str):
    """Build model instance from command-line arguments."""
    if args.model_type == "resnet":
        return ResNetConformerSED(
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
        ).to(device)

    return UNetConformerSED(
        num_classes=num_classes,
        sample_rate=args.sample_rate,
        use_specaug=args.use_specaug,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
        feature_extractor=args.feature_extractor,
    ).to(device)


def build_criterion(loss_type: str):
    """Build loss function from name."""
    if loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()

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
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
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


def run_epoch(model, loader, criterion, device, optimizer=None, lr_scheduler=None, amp_enabled=False, amp_dtype=torch.float16, scaler=None, ema=None):
    """Run one train/validation epoch and return mean loss/accuracy."""
    training = optimizer is not None
    model.train(training)
    if not training and ema is not None:
        ema.apply_shadow()
    total_loss = 0.0
    n = 0
    total_acc = 0
    pbar = tqdm(loader, desc="train" if training else "val", leave=False)
    for batch in pbar:

        x = batch["waveform"].to(device)  # [B, S]
        y = batch["target"].to(device)  # [B, T, C]

        autocast_ctx = (
            torch.amp.autocast(device_type=device,
                               dtype=amp_dtype, enabled=amp_enabled)
            if amp_enabled
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(x)  # [B, Tm, C]
            t = min(logits.size(1), y.size(1))
            logits = logits[:, :t, :]
            y = y[:, :t, :]

            loss = criterion(logits, y)
            cur_acc = ((logits.sigmoid() > 0.5) ==
                       y.bool()).float().mean().item()
        if training:
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=5.0)
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
        if training and ema is not None:
            ema.update()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += cur_acc * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{total_acc / max(1, n):.4f}")

    if not training and ema is not None:
        ema.restore()  # 恢复原始参数
    return total_loss / max(1, n), total_acc / max(1, n)


def plot_loss(train_loss_history, val_loss_history, lr_history, val_acc_history, log_dir):
    """Save training curves (loss/accuracy/LR) to disk."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label="Train Loss", color="#2953dcff")
    plt.plot(val_loss_history, label="Val Loss", color="#ff820eff")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves {max(val_acc_history):.4f}")
    # plt.legend()
    plt.grid()
    # set another y-axis for accuracy
    ax2 = plt.gca().twinx()
    ax2.plot(val_acc_history, label="Val Acc", color="#d8907eff")
    ax2.set_ylabel("Val Accuracy")
    # ax2.legend(loc="right")

    plt.subplot(1, 2, 2)
    plt.plot(lr_history, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curves.png"))
    plt.close()


def main():
    """Training script entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--annotations", type=str,
                    default="data/anno.csv,data/annotations.csv,data/anno3.csv,data/anno4.csv,data/anno5.csv,data/anno6.csv")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--window_sec", type=float, default=10.0)
    ap.add_argument("--frame_hop_sec", type=float, default=0.02)
    ap.add_argument("--windows_per_file", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    ap.add_argument("--use_specaug", action="store_true", default=True)
    ap.add_argument("--freq_mask_param", type=int, default=20)
    ap.add_argument("--time_mask_param", type=int, default=40)
    ap.add_argument("--log_dir", type=str, default="logs")
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
                    choices=["bce", "focal"])
    ap.add_argument("--n_fft", type=int, default=1024)

    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--hop_length", type=int, default=320)
    args = ap.parse_args()

    current_time = time.localtime()
    log_dir = os.path.join(
        args.log_dir, time.strftime("%Y%m%d-%H%M%S", current_time))
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
    label_to_idx, idx_to_label = build_label_map(df)
    files = sorted(df["filename"].unique().tolist())
    train_files, val_files = split_files(
        files, val_ratio=args.val_ratio, seed=args.seed)
    train_files = ['out000.mp3', 'out001.mp3', 'out003.mp3',
                   'out004.mp3', 'out005.mp3', 'out006.mp3']
    val_files = ['out002.mp3', 'out007.mp3']
    print(f"Classes ({len(label_to_idx)}): {list(label_to_idx.keys())}")
    print(f"Train files: {train_files}")
    print(f"Val files: {val_files}")

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
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args, num_classes=len(label_to_idx), device=device)
    # ckpt = torch.load(
    #     r'logs\20260401-000923\best_checkpoint.pth', map_location='cpu')
    # model.load_state_dict(ckpt["model_state_dict"])
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=0.99)

    criterion = build_criterion(args.loss_type)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = build_scheduler(args, optimizer, len(train_loader))
    scaler = None
    if amp_enabled:
        try:
            scaler = torch.amp.GradScaler(device=device, enabled=amp_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)

    best_val = 1e9
    history = []
    train_loss_history = []
    val_loss_history = []
    lr_history = []
    val_acc_history = []
    for ep in range(1, args.epochs + 1):
        if ep > 0.7*args.epochs:
            model.use_specaug = False
        tr, train_acc = run_epoch(model, train_loader, criterion,
                                  device, optimizer=optimizer, lr_scheduler=lr_scheduler, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler, ema=ema if args.use_ema else None)

        va, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None, lr_scheduler=lr_scheduler,
                                amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=None, ema=ema if args.use_ema else None)
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va,
                       "train_acc": train_acc, "val_acc": val_acc})
        train_loss_history.append(tr)
        val_loss_history.append(va)
        val_acc_history.append(val_acc)
        lr_history.append(lr_scheduler.get_last_lr()[
                          0] if lr_scheduler else None)
        # clear memory
        torch.cuda.empty_cache()

        print(
            f"E {ep:03d}: t_loss={tr:.4f} v_loss={va:.4f}, t_acc={train_acc:.4f}, v_acc={val_acc:.4f}, lr={lr_scheduler.get_last_lr()[0] if lr_scheduler else 'N/A':.6f}")
        plot_loss(train_loss_history, val_loss_history,
                  lr_history, val_acc_history, log_dir)
        if va < best_val:
            best_val = va
            payload = {
                "model_state_dict": model.state_dict(),
                "shadow_state_dict": ema.shadow if args.use_ema else None,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "sample_rate": args.sample_rate,
                "window_sec": args.window_sec,
                "history": history,
            }

            save_checkpoint(os.path.join(
                log_dir, "best_checkpoint.pth"), payload)
            print(
                f"Saved best checkpoint to {os.path.join(log_dir, 'best_checkpoint.pth')}")

    save_json(os.path.join(log_dir, "history.json"), {"history": history})
    print("Training done.")
    return best_val


if __name__ == "__main__":
    best_val = main()
