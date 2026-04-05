from __future__ import annotations

import math
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import load_audio_mono, spans_to_frame_targets


class SEDWindowDataset(Dataset):
    """Window-based dataset for sound event detection (SED)."""

    def __init__(
        self,
        data_dir: str,
        annotations,
        files: List[str],
        label_to_idx: Dict[str, int],
        sample_rate: int = 16000,
        window_sec: float = 10.0,
        frame_hop_sec: float = 0.02,
        windows_per_file: int = 100,
        training: bool = True,
        seed: int = 42,
        full_audio_eval: bool = False,
        eval_hop_sec: float = 2.0,

        # ===== mixup 参数 =====
        use_mixup: bool = True,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
    ):
        """Initialize dataset and pre-cache audio + span metadata."""
        self.data_dir = data_dir
        self.annotations = annotations
        self.files = files
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.frame_hop_sec = frame_hop_sec
        self.windows_per_file = windows_per_file
        self.training = training
        self.seed = seed
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        self.window_samples = int(round(window_sec * sample_rate))
        self.num_frames = int(math.ceil(window_sec / frame_hop_sec))

        self.file_to_spans = {}
        self.file_to_wav = {}
        self.file_durations = {}

        for fn in self.files:
            wav = load_audio_mono(os.path.join(
                self.data_dir, fn), self.sample_rate)
            self.file_to_wav[fn] = wav
            self.file_durations[fn] = wav.numel() / self.sample_rate

            sdf = self.annotations[self.annotations["filename"] == fn]
            spans = [
                (float(r.start_time), float(r.end_time), str(r.event_label))
                for r in sdf.itertuples(index=False)
            ]
            self.file_to_spans[fn] = spans

        self.index = []

        if self.training or not full_audio_eval:
            for fn in self.files:
                n = self.windows_per_file if self.training else 20
                for i in range(n):
                    self.index.append((fn, i))
        else:
            for fn in self.files:
                dur = self.file_durations[fn]
                stride = eval_hop_sec
                max_start = max(0.0, dur - self.window_sec)

                starts = []
                cur = 0.0
                while cur < max_start:
                    starts.append(cur)
                    cur += stride

                if len(starts) == 0 or starts[-1] < max_start:
                    starts.append(max_start)

                for st in starts:
                    self.index.append((fn, st))

    def __len__(self):
        """Return number of indexed windows."""
        return len(self.index)

    def _sample_start(self, fn: str, pseudo_idx: int) -> float:
        """Sample (or deterministically generate) a window start time."""
        dur = self.file_durations[fn]
        max_start = max(0.0, dur - self.window_sec)
        if max_start == 0.0:
            return 0.0

        if self.training:
            return random.random() * max_start

        rnd = random.Random(self.seed + pseudo_idx + hash(fn) % 10000)
        return rnd.random() * max_start

    def _get_clip(self, fn: str, st: float):
        """Extract a fixed-length clip and build frame-wise labels."""
        wav = self.file_to_wav[fn]
        s = int(round(st * self.sample_rate))
        e = s + self.window_samples

        clip = wav[s:e]
        if clip.numel() < self.window_samples:
            pad = self.window_samples - clip.numel()
            clip = F.pad(clip, (0, pad))

        ed = st + self.window_sec

        y = spans_to_frame_targets(
            spans=self.file_to_spans[fn],
            label_to_idx=self.label_to_idx,
            clip_start_sec=st,
            clip_end_sec=ed,
            frame_hop_sec=self.frame_hop_sec,
            num_frames=self.num_frames,
        )

        return clip.float(), torch.from_numpy(y)

    @staticmethod
    def _is_float_like(value) -> bool:
        """Check whether an index metadata item behaves like a float start time."""
        return hasattr(value, "__float__")

    def __getitem__(self, idx):
        """Fetch one sample and optionally apply mixup augmentation."""

        fn, meta = self.index[idx]

        if self.training or not self._is_float_like(meta):
            st = self._sample_start(fn, int(meta))
        else:
            st = float(meta)

        clip, target = self._get_clip(fn, st)

        # =========================
        # Mixup
        # =========================
        if (
            self.training
            and self.use_mixup
            and random.random() < self.mixup_prob
        ):
            idx2 = random.randint(0, len(self.index) - 1)
            fn2, meta2 = self.index[idx2]

            if self.training or not self._is_float_like(meta2):
                st2 = self._sample_start(fn2, int(meta2))
            else:
                st2 = float(meta2)

            clip2, target2 = self._get_clip(fn2, st2)

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            clip = lam * clip + (1 - lam) * clip2
            target = lam * target + (1 - lam) * target2

        return {
            "waveform": clip,
            "target": target,
            "filename": fn,
            "clip_start": float(st),
        }


def split_files(all_files: List[str], val_ratio: float = 0.2, seed: int = 42):
    """Split file list into train/validation subsets with a fixed seed."""
    files = sorted(all_files)
    rnd = random.Random(seed)
    rnd.shuffle(files)
    n_val = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 else 0
    val_files = files[:n_val]
    train_files = files[n_val:] if n_val > 0 else files
    if not train_files:
        train_files = files
        val_files = files
    return train_files, val_files
