from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import torch

from infer_and_visualize import collect_audio_files, infer_full_audio
from model import ResNetConformerSED
from to_frame_gpu import WORD_MAP, process_one_item_v2
from utils import CLASSES, frame_probs_to_spans, load_audio, save_json

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import test


DEFAULT_FFMPEG = r"D:\ffmpeg-6.1.1-essentials_build\bin\ffmpeg"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def clean_audio_name(org_audio_name: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F000-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "\U00002B00-\U00002BFF"
        "\U0001F1E0-\U0001F1FF"
        "\U0001F900-\U0001F9FF"
        "\u200D"
        "\uFE0F"
        "\u20E3"
        "]+",
        flags=re.UNICODE,
    )

    name = emoji_pattern.sub("", org_audio_name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def format_cn_time(seconds: float) -> str:
    """Format seconds as hh:mm:ss text used in clip names."""
    total_seconds = int(max(0.0, seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}：{minutes:02d}：{secs:02d}"


def clip_filename(org_stem: str, start_sec: float, end_sec: float, ext: str) -> str:
    start_txt = format_cn_time(start_sec)
    end_txt = format_cn_time(end_sec)
    return f"{start_txt} ~ {end_txt}{ext}"


def resolve_ffmpeg_bins(ffmpeg_bin: str) -> Tuple[str, str]:
    ffmpeg_bin = os.path.abspath(ffmpeg_bin)
    ffprobe_bin = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe.exe")
    if not os.path.isfile(ffprobe_bin):
        ffprobe_bin = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe")
    if not os.path.isfile(ffprobe_bin):
        ffprobe_bin = "ffprobe"
    return ffmpeg_bin, ffprobe_bin


def get_audio_duration(path: str, ffprobe_bin: str) -> float:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    text = result.stdout.decode("utf-8", errors="replace").strip()
    if result.returncode != 0 or not text:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffprobe failed for {path}: {err or text}")
    return float(text)


def run_ffmpeg(cmd: List[str]) -> None:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{err}")


def split_audio(
    audio_path: str,
    chunk_min: float,
    temp_folder: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    max_workers: int = 8,
) -> List[str]:
    os.makedirs(temp_folder, exist_ok=True)

    org_name = os.path.basename(audio_path)
    org_stem, ext = os.path.splitext(org_name)
    duration = get_audio_duration(audio_path, ffprobe_bin)
    chunk_sec = float(chunk_min) * 60.0

    if chunk_sec <= 0:
        raise ValueError("chunk_min must be positive")

    segments = []
    start = 0.0
    while start < duration - 1e-3:
        end = min(start + chunk_sec, duration)
        segments.append((start, end))
        start = end

    if len(segments) >= 2:
        last_start, last_end = segments[-1]
        if last_end - last_start < 10 * 60:
            prev_start, _ = segments[-2]
            segments[-2] = (prev_start, last_end)
            segments.pop()

    def process_segment(index_start_end):
        index, start, end = index_start_end

        out_name = clip_filename(org_stem, start, end, ext)
        out_path = os.path.join(temp_folder, out_name)
        seg_dur = max(0.01, end - start)

        copy_cmd = [
            ffmpeg_bin, "-y",
            "-ss", f"{start:.3f}",
            "-i", audio_path,
            "-t", f"{seg_dur:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]

        try:
            run_ffmpeg(copy_cmd)
        except RuntimeError:
            reencode_cmd = [
                ffmpeg_bin, "-y",
                "-ss", f"{start:.3f}",
                "-i", audio_path,
                "-t", f"{seg_dur:.3f}",
                "-c:a", "copy",
                out_path,
            ]

            if ext.lower() != ".mp3":
                reencode_cmd = [
                    ffmpeg_bin, "-y",
                    "-ss", f"{start:.3f}",
                    "-i", audio_path,
                    "-t", f"{seg_dur:.3f}",
                    out_path,
                ]

            run_ffmpeg(reencode_cmd)

        print(
            f"Split: {org_name} -> {out_name} "
            f"({start:.1f}s-{end:.1f}s)"
        )

        return index, os.path.abspath(out_path)

    # 并行执行 FFmpeg
    tasks = [
        (i, start, end)
        for i, (start, end) in enumerate(segments)
    ]

    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_segment, task)
            for task in tasks
        ]

        for future in as_completed(futures):
            index, path = future.result()
            results[index] = path

    return results


def temp_dir_for_audio(audio_dir: str, temp_dir: Optional[str]) -> str:
    if temp_dir:
        return temp_dir
    return os.path.join(audio_dir, "temp")


def collect_inputs(input_path: Optional[str], audio: Optional[str], audio_dir: Optional[str]) -> List[str]:
    if input_path:
        if os.path.isdir(input_path):
            audio_dir = input_path
        elif os.path.isfile(input_path):
            audio = input_path
        else:
            raise ValueError(f"Input path does not exist: {input_path}")

    files = collect_audio_files(audio, audio_dir)
    filtered = []
    for path in files:
        parent = os.path.basename(os.path.dirname(path)).lower()
        if parent == "temp":
            continue
        filtered.append(path)
    return filtered


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


def infer_clip(model, audio_path, sample_rate, audio_mode, frame_hop_sec, window_sec, args, device, idx_to_label):
    waveform = load_audio(audio_path, sample_rate, audio_mode=audio_mode)
    probs = infer_full_audio(
        model=model,
        waveform=waveform,
        sample_rate=sample_rate,
        frame_hop_sec=frame_hop_sec,
        window_sec=window_sec,
        overlap=args.overlap,
        device=device,
    )
    pred_spans = frame_probs_to_spans(
        probs=probs,
        idx_to_label=idx_to_label,
        frame_hop_sec=frame_hop_sec,
        threshold=args.threshold,
        min_duration_sec=args.min_duration,
    )
    return pred_spans


def parse_args():
    ap = argparse.ArgumentParser(
        description="Split audio, run SED inference, and render timeline videos."
    )
    ap.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Audio file or directory (alternative to --audio / --audio_dir)",
    )
    ap.add_argument("--audio", type=str, default=None,
                    help="Single audio file")
    ap.add_argument("--audio_dir", type=str,
                    default=r'D:\ASMR\1\gzy_processed', help="Directory of audio files")
    ap.add_argument("--background", type=str, default="1.png")
    ap.add_argument("--author", type=str, default="顾子韵_w")
    ap.add_argument("--ffmpeg", type=str, default=DEFAULT_FFMPEG)
    ap.add_argument("--chunk_min", type=float, default=30.0,
                    help="Split length in minutes")

    ap.add_argument("--temp_dir", type=str, default=None,
                    help="Temp folder for split clips")
    ap.add_argument("--checkpoint", type=str,
                    default=r"mono_logs\run_017\last_checkpoint.pth")
    ap.add_argument("--video_dir", type=str, default=None,
                    help="Directory for rendered videos")

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min_duration", type=float, default=0.06)
    ap.add_argument("--window_sec", type=float, default=None)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--audio_mode", type=str, default="auto",
                    choices=["auto", "mono", "stereo"])
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=320)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--skip_video", action="store_true",
                    help="Only split + infer, do not render")
    ap.add_argument("--skip_split", action="store_true",
                    )
    return ap.parse_args()


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


def main():
    args = parse_args()
    audio_files = collect_inputs(args.input_path, args.audio, args.audio_dir)
    audio_files.sort(key=get_date_key)
    audio_files = audio_files[-5:]
    if not audio_files:
        raise ValueError(
            "Provide a file or folder via positional path, --audio, or --audio_dir.")

    ffmpeg_bin, ffprobe_bin = resolve_ffmpeg_bins(args.ffmpeg)

    if args.video_dir is None:
        args.video_dir = os.path.join(args.audio_dir, 'video')

    os.makedirs(args.video_dir, exist_ok=True)

    clip_paths: List[str] = []
    clip_folders = []
    temp_folder = temp_dir_for_audio(args.audio_dir, args.temp_dir)
    for audio_path in audio_files:
        audio_name = audio_path.split('\\')[-1].split('.')[0]

        current_temp_folder = os.path.join(temp_folder, audio_name)
        clip_folders.append(current_temp_folder)
        clip_paths.extend(
            split_audio(
                audio_path=audio_path,
                chunk_min=args.chunk_min,
                temp_folder=current_temp_folder,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
            )
        )

    if not clip_paths:
        raise ValueError("No clips were produced.")

    device = "cuda"
    model, idx_to_label, sample_rate, frame_hop_sec, window_sec, audio_mode = load_model(
        args, device
    )

    simple_preds = []

    for i, clip_folder in enumerate(clip_folders):
        duration_per_class = {k: 0 for k in CLASSES}
        total_duration = 0
        org_audio_name = os.path.basename(clip_folder)
        cleaned_org_audio_name = clean_audio_name(org_audio_name)
        clips = os.listdir(clip_folder)
        current_video_dir = os.path.join(args.video_dir, org_audio_name)
        print(f"Infer Audios [{i + 1}/{len(clip_folders)}]: {org_audio_name}")
        for j, clip_path in enumerate(clips):
            print(f"\tInfer Clips [{j + 1}/{len(clips)}]: {clip_path}")
            clip_path = os.path.join(clip_folder, clip_path)
            pred_spans = infer_clip(
                model=model,
                audio_path=clip_path,
                sample_rate=sample_rate,
                audio_mode=audio_mode,
                frame_hop_sec=frame_hop_sec,
                window_sec=window_sec,
                args=args,
                device=device,
                idx_to_label=idx_to_label,
            )
            item = {
                "audio": os.path.basename(clip_path),
                "audio_path": clip_path,
                "pred_spans": pred_spans,
            }
            simple_preds.append(item)
            for span in pred_spans:
                duration_per_class[span['event_label']
                                   ] += span['end_time']-span['start_time']
            if args.skip_video:
                continue
            save_img = j == 0
            cover_path = os.path.join(current_video_dir, 'cover.png')
            duration = process_one_item_v2(
                item,
                i,
                output_dir=current_video_dir,
                background=args.background,
                ffmpeg_bin=ffmpeg_bin,
                author=args.author,
                org_audio_name=cleaned_org_audio_name,
                saved_name=os.path.basename(item["audio_path"]).split('.')[0],
                save_img=save_img, cover_path=cover_path
            )
            total_duration += duration

        ratio_per_class = {
            k: duration_per_class[k]/total_duration for k in duration_per_class}
        information_path = os.path.join(current_video_dir, 'inf.txt')
        with open(information_path, 'w') as f:
            f.write(f'各类别占比：\n')
            for k in ratio_per_class:
                if k in {'Rub', 'Scrub', 'Rasp'}:
                    f.write(
                        f'底噪{WORD_MAP[k]}:\t{ratio_per_class[k]*100:.2f}%\n')
                else:
                    f.write(f'{WORD_MAP[k]}:\t{ratio_per_class[k]*100:.2f}%\n')

    print(f"Clips in temp folders; videos in: {args.video_dir}")


if __name__ == "__main__":
    main()
