from __future__ import annotations

import math
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchcodec
from torch.utils.data import Dataset, Sampler

from utils import format_audio_channels, spans_to_frame_targets


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
        audio_mode: str = "mono",
        use_lazy_loading: bool = True,
    ):
        """Initialize dataset with selectable lazy/eager loading."""
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
        self.audio_mode = audio_mode
        self.audio_channels = 2 if audio_mode == "stereo" else 1
        self.use_lazy_loading = bool(use_lazy_loading)

        self.window_samples = int(round(window_sec * sample_rate))
        self.num_frames = int(math.ceil(window_sec / frame_hop_sec))

        self.file_to_spans = {}
        self.file_durations = {}
        self.file_meta = {}
        self.file_to_wav = {}
        self._active_file_to_wav: Dict[str, torch.Tensor] = {}

        for fn in self.files:
            path = os.path.join(self.data_dir, fn)

            metadata = torchcodec.decoders.AudioDecoder(path).metadata
            orig_sr = int(metadata.sample_rate)
            num_frames = int(metadata.duration_seconds * orig_sr)
            if num_frames <= 0:
                wav_full, sr_full = torchaudio.load(path)
                orig_sr = int(sr_full)
                num_frames = int(wav_full.shape[-1])

            duration = float(num_frames / max(1, orig_sr))
            self.file_meta[fn] = {
                "path": path,
                "orig_sr": orig_sr,
                "num_frames": num_frames,
            }
            self.file_durations[fn] = duration

            sdf = self.annotations[self.annotations["filename"] == fn]
            spans = [
                (float(r.start_time), float(r.end_time), str(r.event_label))
                for r in sdf.itertuples(index=False)
            ]
            self.file_to_spans[fn] = spans

        if not self.use_lazy_loading:
            for fn in self.files:
                self.file_to_wav[fn] = self._load_full_processed(fn)

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

    def _load_full_processed(self, fn: str) -> torch.Tensor:
        """Load full audio, normalize channel mode and sample-rate once."""
        meta = self.file_meta[fn]
        wav, sr = torchaudio.load(meta["path"])
        wav = format_audio_channels(wav, audio_mode=self.audio_mode)

        if int(sr) != self.sample_rate:
            if wav.dim() == 1:
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), int(sr), self.sample_rate
                ).squeeze(0)
            else:
                wav = torchaudio.functional.resample(
                    wav, int(sr), self.sample_rate)
        return wav.float()

    def set_active_files(self, active_files: List[str]):
        """Preload only selected files for current batch into memory.

        This is designed to be called by a custom batch sampler before yielding
        batch indices.
        """
        if not self.use_lazy_loading:
            return

        selected = {fn for fn in active_files if fn in self.file_meta}
        if not selected:
            self._active_file_to_wav = {}
            return

        new_cache: Dict[str, torch.Tensor] = {}
        for fn in selected:
            if fn in self._active_file_to_wav:
                new_cache[fn] = self._active_file_to_wav[fn]
            else:
                new_cache[fn] = self._load_full_processed(fn)
        self._active_file_to_wav = new_cache

    def _load_clip_from_full(self, wav: torch.Tensor, st: float) -> torch.Tensor:
        """Slice and pad one clip from a full processed waveform."""
        s = int(round(st * self.sample_rate))
        e = s + self.window_samples

        if wav.dim() == 1:
            clip = wav[s:e]
            if clip.numel() < self.window_samples:
                clip = F.pad(clip, (0, self.window_samples - clip.numel()))
            return clip[:self.window_samples].float()

        clip = wav[:, s:e]
        cur_len = clip.shape[-1]
        if cur_len < self.window_samples:
            clip = F.pad(clip, (0, self.window_samples - cur_len))
        return clip[:, :self.window_samples].float()

    def _load_clip_lazy(self, fn: str, st: float) -> torch.Tensor:
        """Load one clip on demand from active batch file cache or file IO."""
        cached = self._active_file_to_wav.get(fn)
        if cached is not None:
            return self._load_clip_from_full(cached, st)

        meta = self.file_meta[fn]
        path = meta["path"]
        orig_sr = meta["orig_sr"]
        start_frame = int(round(st * orig_sr))
        num_frames = max(1, int(math.ceil(self.window_sec * orig_sr)))

        try:
            wav, sr = torchaudio.load(
                path,
                frame_offset=max(0, start_frame),
                num_frames=num_frames,
            )
        except RuntimeError:
            wav, sr = torchaudio.load(path)
            wav = wav[:, max(0, start_frame): max(0, start_frame) + num_frames]

        wav = format_audio_channels(wav, audio_mode=self.audio_mode)

        if int(sr) != self.sample_rate:
            if wav.dim() == 1:
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), int(sr), self.sample_rate
                ).squeeze(0)
            else:
                wav = torchaudio.functional.resample(
                    wav, int(sr), self.sample_rate)

        if wav.dim() == 1:
            if wav.numel() < self.window_samples:
                wav = F.pad(wav, (0, self.window_samples - wav.numel()))
            wav = wav[:self.window_samples]
        else:
            cur_len = wav.shape[-1]
            if cur_len < self.window_samples:
                wav = F.pad(wav, (0, self.window_samples - cur_len))
            wav = wav[:, :self.window_samples]

        return wav.float()

    def _get_clip(self, fn: str, st: float):
        """Extract a fixed-length clip and build frame-wise labels."""
        if self.use_lazy_loading:
            clip = self._load_clip_lazy(fn, st)
        else:
            clip = self._load_clip_from_full(self.file_to_wav[fn], st)

        ed = st + self.window_sec

        y = spans_to_frame_targets(
            spans=self.file_to_spans[fn],
            label_to_idx=self.label_to_idx,
            clip_start_sec=st,
            clip_end_sec=ed,
            frame_hop_sec=self.frame_hop_sec,
            num_frames=self.num_frames,
        )

        return clip, torch.from_numpy(y)

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


class RandomFileSubsetBatchSampler(Sampler):
    """Sample each batch from a random subset of files.

    For each batch, pick `files_per_batch` files, ask dataset to preload only
    those files into memory, then sample indices from the chosen files.
    """


class RandomFileSubsetBatchSampler(Sampler):
    """
    Improved version:
    Each selected file subset persists for multiple batches.

    Args:
        dataset
        batch_size
        files_per_batch
        batches_per_group: number of batches to use same file subset
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        files_per_batch: int,
        batches_per_group: int = 2,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.files_per_batch = max(1, int(files_per_batch))
        self.batches_per_group = max(1, int(batches_per_group))

        # build mapping
        self.file_to_indices = {}
        for i, (fn, _meta) in enumerate(dataset.index):
            self.file_to_indices.setdefault(fn, []).append(i)

        self.files = list(self.file_to_indices.keys())

        self.num_batches = max(1, len(dataset) // max(1, self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        batch_count = 0

        while batch_count < self.num_batches:

            k = min(self.files_per_batch, len(self.files))
            chosen_files = random.sample(self.files, k=k)

            self.dataset.set_active_files(chosen_files)

            pool = []
            for fn in chosen_files:
                pool.extend(self.file_to_indices[fn])

            if not pool:
                continue

            for _ in range(self.batches_per_group):

                if batch_count >= self.num_batches:
                    break

                batch = [random.choice(pool) for _ in range(self.batch_size)]
                yield batch

                batch_count += 1
