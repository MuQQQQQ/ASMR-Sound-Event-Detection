from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib.ticker import FuncFormatter, MultipleLocator

matplotlib.use("Agg")
CLASSES = [
    "Speech",
    "Chewing",
    "Mouth_Sounds",
    "Breathing",
    "Water_Bottle",
    "Slime",
    "Rub",
    "Scrub",
    "Rasp"
]
color_map_255 = {
    "Speech":       (239, 83, 80),     # 红
    "Chewing":      (255, 145, 55),    # 橙
    "Mouth_Sounds": (250, 200, 55),    # 黄
    "Breathing":    (90, 195, 105),    # 绿
    "Water_Bottle": (50, 190, 195),    # 青
    "Slime":        (60, 135, 230),    # 蓝
    "Rub":          (105, 90, 220),     # 蓝紫
    "Scrub":        (175, 80, 210),    # 紫
    "Rasp":         (225, 75, 155),    # 粉紫
}
# to hex
color_map = {k: '#%02x%02x%02x' % v for k, v in color_map_255.items()}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# Labels that are excluded during training/evaluation data loading.
DEFAULT_DISABLED_LABELS = {"Scraping", "Tapping",
                           "Breathing", "Water_Bottle", "Slime"}
DEFAULT_DISABLED_LABELS = {}


def name_to_color(name: str) -> str:
    """Generate a deterministic hex color from a string label."""
    hash_code = abs(hash(name)) % (256**3)  # Get a hash and limit to RGB range
    r = (hash_code >> 16) & 0xFF
    g = (hash_code >> 8) & 0xFF
    b = hash_code & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Load and validate annotation CSV for SED training/inference."""
    df = pd.read_csv(csv_path)
    required = {"filename", "start_time", "end_time", "event_label"}
    disabled_labels = DEFAULT_DISABLED_LABELS
    df = df[~df['event_label'].isin(disabled_labels)].reset_index(drop=True)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {missing}")
    return df


def build_label_map(df: pd.DataFrame, virtual_classes=0) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build bidirectional mappings between class labels and indices."""
    labels = CLASSES + \
        [f"virtual_{i}" for i in range(virtual_classes)]
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    idx_to_label = {i: lb for lb, i in label_to_idx.items()}

    return label_to_idx, idx_to_label


def format_audio_channels(wav: torch.Tensor, audio_mode: str = "mono") -> torch.Tensor:
    """Normalize raw waveform channels according to target audio mode.

    Args:
        wav: Tensor with shape [channels, samples].
        audio_mode: "mono" or "stereo".

    Returns:
        mono   -> [samples]
        stereo -> [2, samples]
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    if audio_mode == "mono":
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0)

    if audio_mode == "stereo":
        if wav.size(0) == 1:
            wav = wav.repeat(2, 1)
        elif wav.size(0) > 2:
            wav = wav[:2]
        return wav

    raise ValueError(f"Unsupported audio_mode: {audio_mode}")


def load_audio(path: str, target_sr: int, audio_mode: str = "mono") -> torch.Tensor:
    """Load full audio with configurable mono/stereo output and resampling."""
    wav, sr = torchaudio.load(path)
    wav = format_audio_channels(wav, audio_mode=audio_mode)

    if sr != target_sr:
        if wav.dim() == 1:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, target_sr).squeeze(0)
        else:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    """Backward-compatible mono loader wrapper."""
    return load_audio(path=path, target_sr=target_sr, audio_mode="mono")


def spans_to_frame_targets(
    spans: List[Tuple[float, float, str]],
    label_to_idx: Dict[str, int],
    clip_start_sec: float,
    clip_end_sec: float,
    frame_hop_sec: float,
    num_frames: int,
) -> np.ndarray:
    """Convert labeled temporal spans to frame-wise multi-hot targets."""
    y = np.zeros((num_frames, len(label_to_idx)), dtype=np.float32)

    for st, ed, label in spans:
        if label not in label_to_idx:
            continue
        overlap_st = max(st, clip_start_sec)
        overlap_ed = min(ed, clip_end_sec)
        if overlap_ed <= overlap_st:
            continue

        s = int(np.floor((overlap_st - clip_start_sec) / frame_hop_sec))
        e = int(np.ceil((overlap_ed - clip_start_sec) / frame_hop_sec))
        s = max(0, min(s, num_frames))
        e = max(0, min(e, num_frames))
        if e > s:
            y[s:e, label_to_idx[label]] = 1.0
    return y


def frame_probs_to_spans_np(
    probs: np.ndarray,
    idx_to_label: Dict[int, str],
    frame_hop_sec: float,
    threshold: float = 0.5,
    min_duration_sec: float = 0.0,
) -> List[Dict]:
    """Convert frame-level probabilities to merged event spans by class."""
    t, c = probs.shape
    spans = []

    min_frames = int(np.ceil(min_duration_sec / frame_hop_sec))

    for cls in range(c):
        active = probs[:, cls] >= threshold
        if not active.any():
            continue
        changes = np.diff(active.astype(np.int8))
        starts = np.flatnonzero(changes == 1) + 1
        ends = np.flatnonzero(changes == -1) + 1

        if active[0]:
            starts = np.r_[0, starts]

        if active[-1]:
            ends = np.r_[ends, t]

        lengths = ends - starts
        mask = lengths >= min_frames

        starts = starts[mask]
        ends = ends[mask]

        if len(starts) == 0:
            continue

        label = idx_to_label[cls]

        for s, e in zip(starts, ends):
            spans.append({
                "event_label": label,
                "start_time": float(s * frame_hop_sec),
                "end_time": float(e * frame_hop_sec),
                "score": float(probs[s:e, cls].mean()),
            })

    spans.sort(key=lambda x: (x["start_time"], x["event_label"]))
    return spans


def frame_probs_to_spans(
    probs: np.ndarray,
    idx_to_label: Dict[int, str],
    frame_hop_sec: float,
    threshold: float = 0.5,
    min_duration_sec: float = 0.0,
) -> List[Dict]:
    """Convert frame-level probabilities to merged event spans by class."""
    spans: List[Dict] = []
    t, c = probs.shape
    min_frames = int(np.ceil(min_duration_sec / frame_hop_sec))

    for cls in range(c):
        active = probs[:, cls] >= threshold
        i = 0
        while i < t:
            if not active[i]:
                i += 1
                continue
            s = i
            while i < t and active[i]:
                i += 1
            e = i
            st_sec = s * frame_hop_sec
            ed_sec = e * frame_hop_sec
            if (e - s) >= min_frames:
                spans.append(
                    {
                        "event_label": idx_to_label[cls],
                        "start_time": float(st_sec),
                        "end_time": float(ed_sec),
                        "score": float(probs[s:e, cls].mean()),
                    }
                )
    spans.sort(key=lambda x: (x["start_time"], x["event_label"]))
    return spans


def save_checkpoint(path: str, payload: Dict):
    """Save model checkpoint dictionary to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def save_json(path: str, obj: Dict):
    """Save dictionary object as UTF-8 JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_timeline(
    waveform: np.ndarray,
    sr: int,
    gt_spans: List[Dict],
    pred_spans: List[Dict],
    output_path: str,
    title: str = "SED Timeline",
    plot_waveform: bool = True,
):
    """Plot waveform + ground-truth spans + predicted spans as a timeline image."""

    duration = len(waveform) / sr
    t = np.arange(len(waveform)) / sr

    fig, axes = plt.subplots(
        3, 1,
        figsize=(24, 10),
        sharex=True,
    )

    if plot_waveform:
        axes[0].plot(
            t,
            waveform,
            color="steelblue",
            linewidth=0.7,
        )
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title(title)
        axes[0].grid(alpha=0.2)

    def draw_spans(ax, spans, span_title, labels=None):
        """Draw horizontal labeled time spans on a given axis."""

        y_map = {lb: i for i, lb in enumerate(labels)}

        for s in spans:
            label = s["event_label"]

            if label not in y_map:
                continue

            y = y_map[label]

            ax.broken_barh(
                [(s["start_time"], s["end_time"] - s["start_time"])],
                (y - 0.4, 0.8),
                facecolors=color_map.get(label, "#888888"),

                linewidth=0.3,
            )

        ax.set_yticks(list(y_map.values()))
        ax.set_yticklabels(labels)

        # Y轴：第一个类别在最上面
        ax.set_ylim(-1, len(labels) - 0.5)
        ax.invert_yaxis()

        # Y轴文字使用对应类别的颜色
        for text, label in zip(ax.get_yticklabels(), labels):
            text.set_color(color_map.get(label, "#888888"))
            text.set_fontweight("bold")

        ax.set_xlim(0, duration)
        ax.set_title(span_title)
        ax.grid(alpha=0.2, axis="x")

    draw_spans(
        axes[1],
        gt_spans,
        "Ground Truth Spans",
        labels=CLASSES,
    )

    draw_spans(
        axes[2],
        pred_spans,
        "Predicted Spans",
        labels=CLASSES,
    )

    # =========================
    # X-axis: seconds -> M'SS"
    # =========================

    def format_time(seconds, pos):
        seconds = int(round(seconds))
        minutes = seconds // 60
        seconds = seconds % 60

        if minutes == 0:
            return f'{seconds}"'

        return f"{minutes}'{seconds:02d}\""

    time_formatter = FuncFormatter(format_time)

    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_major_formatter(time_formatter)
        ax.tick_params(axis='x', labelrotation=30)  # 倾斜30度

    axes[2].set_xlabel("Time (min:s)")

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)
