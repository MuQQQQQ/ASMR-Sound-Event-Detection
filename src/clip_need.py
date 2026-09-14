import argparse
import os
import random
import re
import shutil
import subprocess
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from infer_and_visualize import collect_audio_files, infer_full_audio
from model import ResNetConformerSED
from to_frame_gpu import WORD_MAP, process_one_item_v2
from utils import (
    CLASSES,
    frame_probs_to_spans,
    frame_probs_to_spans_np,
    load_audio,
    save_json,
)

CHUNK_SEC = 2 * 3600
LAST_CHUNK_MERGE_MIN_SEC = 30 * 60

MERGE_GAP_SEC = 3.0
MIN_KEEP_SEC = 5.0
KEEP_PADDING_SEC = 1.0
FADE_SEC = 1.0

continuous_labels = {
    'speech',
    'breathing'
}

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a",
    ".aac", ".ogg", ".opus", ".wma"
}


def normalize_label(label: str) -> str:
    x = label.strip().lower()
    x = re.sub(r"[\s\-]+", "_", x)

    if x == "mouth_sound":
        x = "mouth_sounds"

    if x in {"waterbottle", "water_bottle"}:
        x = "water_bottle"

    return x


def load_model(args, device: str):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = (
        {int(k): v for k, v in ckpt["idx_to_label"].items()}
        if isinstance(list(ckpt["idx_to_label"].keys())[0], str)
        else ckpt["idx_to_label"]
    )
    sample_rate = int(ckpt.get("sample_rate", args.sample_rate))
    frame_hop_sec = float(ckpt.get("frame_hop_sec", 0.02))
    window_sec = (
        float(args.window_sec)
        if args.window_sec is not None
        else float(ckpt.get("window_sec", 10.0))
    )
    ckpt_audio_mode = str(ckpt.get("audio_mode", "mono"))
    print(f"Loaded checkpoint: {args.checkpoint}, mode = {ckpt_audio_mode}")
    audio_mode = ckpt_audio_mode if args.audio_mode == "auto" else args.audio_mode
    audio_channels = 2 if audio_mode == "stereo" else 1

    model = ResNetConformerSED(
        num_classes=len(label_to_idx),
        sample_rate=sample_rate,
        feature_extractor="melspec",
        use_specaug=False,
        audio_channels=audio_channels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
        conformer_dim=256,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, idx_to_label, sample_rate, frame_hop_sec, window_sec, audio_mode


def merge_spans(
    spans: List[Dict],
    gap_sec: float = 3.0,
    merge_all: bool = False,
) -> List[Dict]:
    """
    Merge spans.

    If merge_all=True, event labels are ignored and all spans are treated
    as belonging to one class.
    """
    if not spans:
        return []

    items = []

    for s in spans:
        label = normalize_label(s["event_label"])
        items.append({
            "start_time": float(s["start_time"]),
            "end_time": float(s["end_time"]),
            "event_label": label,
            "score": float(s.get("score", 0.0)),
        })

    if merge_all:
        items.sort(key=lambda x: (x["start_time"], x["end_time"]))
    else:
        items.sort(
            key=lambda x: (
                x["event_label"],
                x["start_time"],
                x["end_time"],
            )
        )

    merged = []

    if merge_all:
        groups = [items]
    else:
        groups = []
        current = []
        current_label = None

        for item in items:
            if current_label is None or item["event_label"] == current_label:
                current.append(item)
                current_label = item["event_label"]
            else:
                groups.append(current)
                current = [item]
                current_label = item["event_label"]

        if current:
            groups.append(current)

    for group in groups:
        if not group:
            continue

        cur_start = group[0]["start_time"]
        cur_end = group[0]["end_time"]
        scores = [group[0]["score"]]

        for item in group[1:]:
            if item["start_time"] - cur_end < gap_sec:
                cur_end = max(cur_end, item["end_time"])
                scores.append(item["score"])
            else:
                merged.append({
                    "start_time": cur_start,
                    "end_time": cur_end,
                    "event_label": group[0]["event_label"],
                    "score": float(np.mean(scores)),
                })

                cur_start = item["start_time"]
                cur_end = item["end_time"]
                scores = [item["score"]]

        merged.append({
            "start_time": cur_start,
            "end_time": cur_end,
            "event_label": group[0]["event_label"],
            "score": float(np.mean(scores)),
        })

    merged.sort(key=lambda x: (x["start_time"], x["end_time"]))
    return merged


def get_audio_duration(
    audio_path: str,
    ffprobe_bin: str = "ffprobe",
) -> float:
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def get_chunk_ranges(
    duration: float,
    chunk_sec: float = CHUNK_SEC,
    last_merge_min_sec: float = LAST_CHUNK_MERGE_MIN_SEC,
) -> List[Tuple[float, float]]:
    """
    Generate logical chunks.

    If the last chunk is shorter than 30 minutes, merge it into
    the previous chunk.
    """
    if duration <= chunk_sec:
        return [(0.0, duration)]

    ranges = []

    start = 0.0

    while start < duration:
        end = min(start + chunk_sec, duration)
        ranges.append((start, end))
        start = end

    if len(ranges) >= 2:
        last_duration = ranges[-1][1] - ranges[-1][0]

        if last_duration < last_merge_min_sec:
            ranges[-2] = (
                ranges[-2][0],
                ranges[-1][1],
            )
            ranges.pop()

    return ranges


def load_audio_chunk_16k(
    audio_path: str,
    start_sec: float,
    duration_sec: float,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Read one chunk from the original audio and convert it to 16 kHz stereo.

    Returns:
        Tensor [2, T]
    """
    samples = int(round(duration_sec * sample_rate))

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_sec:.6f}",
        "-t", f"{duration_sec:.6f}",
        "-i", str(audio_path),
        "-vn",
        "-ac", "2",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "pipe:1",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.int16,
    )

    expected = samples * 2

    if audio.size < expected:
        audio = np.pad(
            audio,
            (0, expected - audio.size),
        )
    elif audio.size > expected:
        audio = audio[:expected]

    audio = audio.reshape(-1, 2).T

    waveform = torch.from_numpy(
        audio.astype(np.float32) / 32768.0
    )

    return waveform


def infer_one_channel(
    model,
    channel_waveform: torch.Tensor,
    sample_rate: int,
    frame_hop_sec: float,
    window_sec: float,
    args,
    device,
    idx_to_label,
):
    t0 = time.time()
    probs = infer_full_audio(
        model=model,
        waveform=channel_waveform,
        sample_rate=sample_rate,
        frame_hop_sec=frame_hop_sec,
        window_sec=window_sec,
        overlap=args.overlap,
        device=device,
    )
    t1 = time.time()
    # print(f' > infer_full_audio: {t1-t0:.2f}s')
    t0 = time.time()
    res = frame_probs_to_spans_np(
        probs=probs,
        idx_to_label=idx_to_label,
        frame_hop_sec=frame_hop_sec,
        threshold=args.threshold,
        min_duration_sec=args.min_duration,
    )
    t1 = time.time()
    # print(f' > frame_probs_to_spans_np: {t1-t0:.2f}s')

    return res


def infer_audio_in_chunks(
    audio_path: str,
    model,
    args,
    device,
    idx_to_label,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
):
    """
    First pass:
    48 kHz original -> logical chunks -> 16 kHz -> SED.

    FFmpeg decoding of the next chunk is performed in a background
    thread while the current chunk is being processed by PyTorch.

    Returned spans use global timestamps.
    """

    duration = get_audio_duration(
        audio_path,
        ffprobe_bin,
    )

    chunks = get_chunk_ranges(duration)

    all_left_spans = []
    all_right_spans = []

    print(
        f"\nAudio: {audio_path}"
        f"\nDuration: {duration / 3600:.2f} h"
        f"\nChunks: {len(chunks)}"
    )

    def load_chunk(chunk):
        chunk_idx, (chunk_start, chunk_end) = chunk
        chunk_duration = chunk_end - chunk_start

        waveform = load_audio_chunk_16k(
            audio_path=audio_path,
            start_sec=chunk_start,
            duration_sec=chunk_duration,
            ffmpeg_bin=ffmpeg_bin,
            sample_rate=16000,
        )

        return waveform

    # Only one FFmpeg job at a time.
    # The main thread handles GPU inference.
    with ThreadPoolExecutor(max_workers=1) as executor:

        # Start loading the first chunk immediately.
        future = executor.submit(
            load_chunk,
            (0, chunks[0]),
        )

        for chunk_idx in range(len(chunks)):

            chunk_start, chunk_end = chunks[chunk_idx]
            chunk_duration = chunk_end - chunk_start

            print(
                f"\n[{chunk_idx + 1}/{len(chunks)}] "
                f"{chunk_start / 3600:.2f}h -> "
                f"{chunk_end / 3600:.2f}h"
            )

            # ---------------------------------------------------------
            # Wait for current chunk's FFmpeg decoding.
            # While this waits, FFmpeg is doing the work.
            # ---------------------------------------------------------

            waveform = future.result()

            # ---------------------------------------------------------
            # Immediately start decoding the NEXT chunk.
            #
            # This runs in the background while GPU inference below
            # is running.
            # ---------------------------------------------------------
            if chunk_idx + 1 < len(chunks):
                future = executor.submit(
                    load_chunk,
                    (
                        chunk_idx + 1,
                        chunks[chunk_idx + 1],
                    ),
                )

            left = waveform[0]
            right = waveform[1]

            # ---------------------------------------------------------
            # GPU inference: left + right
            # ---------------------------------------------------------

            left_spans = infer_one_channel(
                model=model,
                channel_waveform=left,
                sample_rate=16000,
                frame_hop_sec=args.frame_hop_sec,
                window_sec=args.window_sec,
                args=args,
                device=device,
                idx_to_label=idx_to_label,
            )

            t0 = time.time()

            right_spans = infer_one_channel(
                model=model,
                channel_waveform=right,
                sample_rate=16000,
                frame_hop_sec=args.frame_hop_sec,
                window_sec=args.window_sec,
                args=args,
                device=device,
                idx_to_label=idx_to_label,
            )

            # ---------------------------------------------------------
            # Convert local timestamps to global timestamps
            # ---------------------------------------------------------

            for span in left_spans:
                span["start_time"] += chunk_start
                span["end_time"] += chunk_start

            for span in right_spans:
                span["start_time"] += chunk_start
                span["end_time"] += chunk_start

            all_left_spans.extend(left_spans)
            all_right_spans.extend(right_spans)

            del waveform, left, right

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return duration, all_left_spans, all_right_spans


def extend_keep_to_continuous(
    start: float,
    end: float,
    continuous_spans: List[Dict],
    continuous_starts: List[float],
    duration: float,
) -> Tuple[float, float]:

    if continuous_spans:

        # Left boundary
        i = bisect_right(
            continuous_starts,
            start,
        ) - 1

        if i >= 0:
            c = continuous_spans[i]

            if c["start_time"] <= start < c["end_time"]:
                start = c["start_time"]

        # Right boundary
        i = bisect_right(
            continuous_starts,
            end,
        ) - 1

        if i >= 0:
            c = continuous_spans[i]

            if c["start_time"] < end <= c["end_time"]:
                end = c["end_time"]

    return (
        max(0.0, start),
        min(duration, end),
    )


def build_keep_intervals(
    left_spans: List[Dict],
    right_spans: List[Dict],
    duration: float,
    need_events,
    condiment_events={}
):
    """
    Build final keep intervals.

    All NEED_EVENTS are treated as ONE class.
    """
    left_spans = merge_spans(
        left_spans,
        gap_sec=MERGE_GAP_SEC,
        merge_all=False,
    )

    right_spans = merge_spans(
        right_spans,
        gap_sec=MERGE_GAP_SEC,
        merge_all=False,
    )

    strict_left_need = [
        s for s in left_spans
        if normalize_label(s["event_label"]) in need_events
    ]

    strict_right_need = [
        s for s in right_spans
        if normalize_label(s["event_label"]) in need_events
    ]

    strict_need = strict_left_need + strict_right_need

    strict_need = merge_spans(
        strict_need,
        gap_sec=MERGE_GAP_SEC,
        merge_all=True,
    )
    strict_need = [
        s for s in strict_need
        if s["end_time"] - s["start_time"] >= MIN_KEEP_SEC
    ]
    n_need = len(strict_need)

    condiment_candidates = [
        s for s in left_spans + right_spans
        if normalize_label(s["event_label"]) in condiment_events
    ]
    condiment_candidates = merge_spans(
        condiment_candidates,
        gap_sec=MERGE_GAP_SEC,
        merge_all=True,
    )
    condiment_candidates = [
        s for s in condiment_candidates
        if s["end_time"] - s["start_time"] >= MIN_KEEP_SEC
    ]
    condiment_spans = []

    for label, ratio in condiment_events.items():
        candidates = [
            s for s in condiment_candidates
            if normalize_label(s["event_label"]) == label
        ]

        n_target = round(n_need * ratio)

        if n_target >= len(candidates):
            selected = candidates
        else:
            selected = random.sample(candidates, n_target)

        condiment_spans.extend(selected)
    need = strict_need+condiment_spans
    need = merge_spans(
        need,
        gap_sec=MERGE_GAP_SEC,
        merge_all=True
    )
    continuous_spans = left_spans + right_spans

    continuous_spans = [
        s for s in continuous_spans
        if normalize_label(s["event_label"]) in continuous_labels
    ]

    continuous_spans = merge_spans(
        continuous_spans,
        gap_sec=MERGE_GAP_SEC,
        merge_all=False,
    )

    continuous_starts = [
        s["start_time"]
        for s in continuous_spans
    ]

    # Add context, then extend to continuous-event boundaries.
    keep = []

    for s in need:
        original_start = s["start_time"]
        original_end = s["end_time"]

        # Normal context padding.
        start = max(
            0.0,
            original_start - KEEP_PADDING_SEC,
        )

        end = min(
            duration,
            original_end + KEEP_PADDING_SEC,
        )

        start, end = extend_keep_to_continuous(
            start,
            end,
            continuous_spans,
            continuous_starts,
            duration,
        )

        start = max(0.0, start)
        end = min(duration, end)

        if end > start:
            keep.append((start, end))
    # Padding may cause different intervals to overlap.
    keep.sort()

    merged_keep = []

    for start, end in keep:
        if not merged_keep:
            merged_keep.append([start, end])
            continue

        if start <= merged_keep[-1][1]:
            merged_keep[-1][1] = max(
                merged_keep[-1][1],
                end,
            )
        else:
            merged_keep.append([start, end])

    return merged_keep, left_spans, right_spans


def get_attenuation_intervals(
    spans: List[Dict],
    attenuation_db
):
    """
    Return only events requiring attenuation.
    """
    result = []

    for s in spans:
        label = normalize_label(s["event_label"])

        if label in attenuation_db:
            result.append({
                "start": float(s["start_time"]),
                "end": float(s["end_time"]),
                "db": attenuation_db[label],
                "label": label,
            })

    return result


def make_ffmpeg_channel_volume_filter(
    spans: List[Dict],
    clip_start: float,
    clip_end: float,
    attenuation_db,
    fade_sec: float = 2.0,
):
    """
    Build a time-dependent volume expression for one channel.

    Gain: 
        before fade      -> 1.0
        fade-in          -> 1.0 -> target_gain
        event            -> target_gain
        fade-out         -> target_gain -> 1.0

    If an attenuation event overlaps another one, the strongest
    attenuation (smallest gain) wins.
    """
    events = get_attenuation_intervals(spans, attenuation_db)

    expressions = []

    for e in events:
        event_start_abs = e["start"]
        event_end_abs = e["end"]

        # Ignore events that do not overlap the clip.
        if event_end_abs <= clip_start or event_start_abs >= clip_end:
            continue

        gain = 10 ** (e["db"] / 20.0)

        # Original attenuation + fade region.
        fade_start_abs = event_start_abs - fade_sec
        fade_end_abs = event_end_abs + fade_sec

        # Clamp everything to the actual clip.
        fade_start = max(
            fade_start_abs,
            clip_start,
        ) - clip_start

        event_start = max(
            event_start_abs,
            clip_start,
        ) - clip_start

        event_end = min(
            event_end_abs,
            clip_end,
        ) - clip_start

        fade_end = min(
            fade_end_abs,
            clip_end,
        ) - clip_start

        # Safety.
        event_start = max(0.0, min(event_start, clip_end - clip_start))
        event_end = max(event_start, min(event_end, clip_end - clip_start))
        fade_start = max(0.0, min(fade_start, event_start))
        fade_end = max(event_end, min(fade_end, clip_end - clip_start))

        fade_in_duration = event_start - fade_start
        fade_out_duration = fade_end - event_end

        # Build expression from the four possible regions.
        if fade_in_duration > 1e-6 and fade_out_duration > 1e-6:
            expr = (
                f"if(lt(t,{fade_start:.6f}),1,"
                f"if(lt(t,{event_start:.6f}),"
                f"1-({1.0-gain:.8f})*(t-{fade_start:.6f})/"
                f"{fade_in_duration:.6f},"
                f"if(lt(t,{event_end:.6f}),"
                f"{gain:.8f},"
                f"if(lt(t,{fade_end:.6f}),"
                f"{gain:.8f}+({1.0-gain:.8f})*(t-{event_end:.6f})/"
                f"{fade_out_duration:.6f},"
                f"1))))"
            )

        elif fade_in_duration > 1e-6:
            expr = (
                f"if(lt(t,{fade_start:.6f}),1,"
                f"if(lt(t,{event_start:.6f}),"
                f"1-({1.0-gain:.8f})*(t-{fade_start:.6f})/"
                f"{fade_in_duration:.6f},"
                f"{gain:.8f}))"
            )

        elif fade_out_duration > 1e-6:
            expr = (
                f"if(lt(t,{event_end:.6f}),"
                f"{gain:.8f},"
                f"if(lt(t,{fade_end:.6f}),"
                f"{gain:.8f}+({1.0-gain:.8f})*(t-{event_end:.6f})/"
                f"{fade_out_duration:.6f},"
                f"1))"
            )

        else:
            expr = f"{gain:.8f}"

        expressions.append(expr)

    if not expressions:
        return "1"

    # Strongest attenuation wins.
    volume_expr = expressions[0]

    for expr in expressions[1:]:
        volume_expr = f"min({volume_expr},{expr})"

    return volume_expr


def extract_processed_clip(
    audio_path: str,
    output_path: str,
    clip_start: float,
    clip_end: float,
    left_spans: List[Dict],
    right_spans: List[Dict],
    attenuation_db,
    ffmpeg_bin: str = "ffmpeg",
):
    """
    Extract one final stereo MP3 directly from the original 48 kHz audio.

    Left/right attenuation are independent.
    Final clip fades are applied after attenuation.
    """
    duration = clip_end - clip_start

    left_volume = make_ffmpeg_channel_volume_filter(
        left_spans,
        clip_start,
        clip_end,
        attenuation_db,
        fade_sec=FADE_SEC,
    )

    right_volume = make_ffmpeg_channel_volume_filter(
        right_spans,
        clip_start,
        clip_end,
        attenuation_db,
        fade_sec=FADE_SEC,
    )

    fade = min(
        FADE_SEC,
        duration / 2.0,
    )

    filter_complex = (
        "[0:a]channelsplit=channel_layout=stereo[left][right];"
        f"[left]volume=eval=frame:volume='{left_volume}',"
        f"afade=t=in:st=0:d={fade:.6f},"
        f"afade=t=out:st={duration-fade:.6f}:d={fade:.6f}[l];"
        f"[right]volume=eval=frame:volume='{right_volume}',"
        f"afade=t=in:st=0:d={fade:.6f},"
        f"afade=t=out:st={duration-fade:.6f}:d={fade:.6f}[r];"
        "[l][r]amerge=inputs=2[a]"
    )

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        "-ss", f"{clip_start:.6f}",
        "-t", f"{duration:.6f}",
        "-i", str(audio_path),

        "-filter_complex", filter_complex,
        "-map", "[a]",

        "-ar", "48000",
        "-ac", "2",

        "-c:a", "libmp3lame",
        "-q:a", "2",

        str(output_path),
    ]

    subprocess.run(
        cmd,
        check=True,
    )


def process_one_audio(
    audio_path: str,
    output_root: str,
    model,
    args,
    device,
    idx_to_label,
    need_events,
    attenuation_db,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    condiment_events={},
    max_workers: int = 8,
):
    audio_path = Path(audio_path)

    output_dir = Path(output_root) / audio_path.stem

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"\n{'=' * 70}")
    print(f"Processing: {audio_path}")
    print(f"{'=' * 70}")

    duration, left_spans, right_spans = infer_audio_in_chunks(
        audio_path=audio_path,
        model=model,
        args=args,
        device=device,
        idx_to_label=idx_to_label,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
    )

    keep_intervals, left_spans, right_spans = build_keep_intervals(
        left_spans,
        right_spans,
        duration,
        need_events,
        condiment_events=condiment_events,
    )

    print(f"\nDetected keep intervals: {len(keep_intervals)}")

    prefix = get_date_key(output_dir)
    if prefix is not None:
        prefix = prefix.strftime("%Y年%m月%d日")
    else:
        prefix = ""

    def extract_one(idx, start, end):
        output_path = (
            output_dir
            / f"{idx:04d}_{start:.2f}-{end:.2f}.mp3"
        )

        extract_processed_clip(
            audio_path=audio_path,
            output_path=output_path,
            clip_start=start,
            clip_end=end,
            left_spans=left_spans,
            right_spans=right_spans,
            attenuation_db=attenuation_db,
            ffmpeg_bin=ffmpeg_bin,
        )

        return idx, start, end, output_path

    total = len(keep_intervals)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_one, idx, start, end)
            for idx, (start, end) in enumerate(keep_intervals, 1)
        ]

        for future in as_completed(futures):
            idx, start, end, output_path = future.result()

            print(
                f"[{idx:04d}/{total:04d}] "
                f"{start:.2f}s -> {end:.2f}s "
                f"({end - start:.2f}s)"
            )

    print(f"\nFinished: {audio_path.name}")


def process_folder(
    input_folder: str,
    output_root: str,
    model,
    args,
    device,
    idx_to_label,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
):
    input_folder = Path(input_folder)

    files = sorted(
        p for p in input_folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTS
    )

    print(
        f"Found {len(files)} audio files."
    )

    for i, audio_path in enumerate(files, 1):
        print(
            f"\n######## "
            f"{i}/{len(files)} "
            f"########"
        )

        try:
            process_one_audio(
                audio_path=audio_path,
                output_root=output_root,
                model=model,
                args=args,
                device=device,
                idx_to_label=idx_to_label,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
            )
        except Exception as e:
            print(
                f"ERROR: {audio_path}\n"
                f"{type(e).__name__}: {e}"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stereo ASMR event based audio cleaner. "
            "Each channel is inferred independently, "
            "but required events are combined across channels."
        )
    )
    parser.add_argument("--mode", type=str, default='speech',
                        choices=['speech', 'no_speech'])
    parser.add_argument("--checkpoint", type=str,
                        default=r"mono_logs\run_029\last_checkpoint.pth")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r'D:\ASMR\1\gzy_cut',
        help="Input folder containing audio files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder.",
    )

    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="FFmpeg executable.",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min_duration", type=float, default=0.06)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--audio_mode", type=str, default="auto",
                        choices=["auto", "mono", "stereo"])
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=320)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--window_sec", type=float, default=10.0)
    parser.add_argument("--frame_hop_sec", type=float, default=0.02)

    parser.add_argument(
        "--bitrate",
        type=str,
        default="192k",
    )

    return parser.parse_args()


date_pattern = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")


def get_date_key(item):

    if hasattr(item, "name"):
        s = item.name
    else:
        s = str(item)

    match = date_pattern.search(s)
    if not match:
        return None

    y, m, d = match.groups()
    return datetime(int(y), int(m), int(d))

# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)

    if args.output_dir is None:
        output_dir = os.path.join(
            input_dir,
            "_cleaned",
        )
    else:
        output_dir = os.path.abspath(
            args.output_dir
        )

    os.makedirs(output_dir, exist_ok=True)

    audio_files = collect_audio_files(
        None,
        input_dir
    )

    # Don't accidentally process our own output directory.
    output_dir_abs = os.path.abspath(output_dir)

    audio_files = [
        p for p in audio_files
        if not os.path.abspath(p).startswith(
            output_dir_abs + os.sep
        )
    ]
    audio_files.sort(key=get_date_key)
    audio_files = audio_files[::-1]

    if not audio_files:
        raise ValueError(
            f"No audio files found in {input_dir}"
        )
    print(audio_files)
    print(
        f"Found {len(audio_files)} audio files."
    )

    # --------------------------------------------------------
    # Your existing model loader
    # --------------------------------------------------------

    device = "cuda"

    model, idx_to_label, sample_rate, frame_hop_sec, window_sec, audio_mode = load_model(
        args,
        device,
    )
    args.frame_hop_sec = frame_hop_sec
    args.window_sec = window_sec
    print()
    print(f"Device:       {device}")
    print(f"Sample rate:  {sample_rate}")
    print(f"Frame hop:    {frame_hop_sec}s")
    print(f"Window:       {window_sec}s")
    print(f"Output dir:   {output_dir}")

    if args.mode == 'speech':
        need_events = {
            "chewing",
            "mouth_sounds",
        }
        condiment_events = {
            "breathing": 0.3,
            "speech": 0.
        }

        attenuation_db = {
            "rasp": -25.0,          # 90% reduction -> gain 0.10
            "scrub": -10.4576,      # 70% reduction -> gain 0.30
            "rub": -10.0206,         # 50% reduction -> gain 0.50
            "water_bottle": -10.0206,
            "slime": -10.4576,
        }

    elif args.mode == 'no_speech':
        need_events = {
            "chewing",
            "mouth_sounds",
            "breathing",
            # "speech",
            "rasp",
            "scrub",
            "rub",
            "water_bottle"
        }

        attenuation_db = {
            "speech": -30.0,
        }
        condiment_events = {
        }
    for i, audio_path in enumerate(audio_files):
        print()
        print(
            f"[{i + 1}/{len(audio_files)}]"
        )

        process_one_audio(
            audio_path=audio_path,
            output_root=output_dir,
            model=model,

            args=args,
            device=device,
            idx_to_label=idx_to_label,
            need_events=need_events,
            attenuation_db=attenuation_db,
            ffmpeg_bin=args.ffmpeg,
            condiment_events=condiment_events
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
