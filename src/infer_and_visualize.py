from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import string
from typing import Dict, List
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
import torch
import torchaudio

from model import ResNetConformerSED
from utils import frame_probs_to_spans, load_audio, plot_timeline, save_json


@torch.no_grad()
def infer_full_audio(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    sample_rate: int,
    frame_hop_sec: float,
    window_sec: float = 10.0,
    overlap: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """Run sliding-window inference on a full audio waveform."""
    model.eval()
    w = waveform.to(device)
    total_samples = w.shape[-1]

    window_samples = int(round(window_sec * sample_rate))
    stride_samples = max(1, int(round(window_samples * (1.0 - overlap))))

    in_channels = int(getattr(model, "audio_channels", 1))
    if in_channels == 1:
        dummy = torch.zeros(1, window_samples, device=device)
    else:
        dummy = torch.zeros(1, in_channels, window_samples, device=device)
    out_t = model(dummy).size(1)

    total_sec = total_samples / sample_rate
    total_frames = int(np.ceil(total_sec / frame_hop_sec))
    num_classes = model.classifier.out_features

    prob_sum = np.zeros((total_frames, num_classes), dtype=np.float32)
    prob_cnt = np.zeros((total_frames, 1), dtype=np.float32)

    starts = list(
        range(0, max(1, total_samples - window_samples + 1), stride_samples)
    )
    if not starts or starts[-1] + window_samples < total_samples:
        starts.append(max(0, total_samples - window_samples))

    for s in starts:
        if w.dim() == 1:
            clip = w[s:s + window_samples]
        else:
            clip = w[:, s:s + window_samples]

        if clip.shape[-1] < window_samples:
            clip = torch.nn.functional.pad(
                clip, (0, window_samples - clip.shape[-1]))
        logits = model(clip.unsqueeze(0))
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        start_sec = s / sample_rate
        g0 = int(round(start_sec / frame_hop_sec))
        t = min(out_t, probs.shape[0])
        g1 = min(total_frames, g0 + t)
        if g1 <= g0:
            continue
        used = g1 - g0

        prob_sum[g0:g1] += probs[:used]
        prob_cnt[g0:g1] += 1.0

    prob_cnt = np.maximum(prob_cnt, 1.0)
    return prob_sum / prob_cnt


def load_gt_spans(annotations_csv: str, audio_filename: str) -> List[Dict]:
    """Load ground-truth spans for one audio file from CSV."""
    if not annotations_csv or not os.path.exists(annotations_csv):
        return []
    df = pd.read_csv(annotations_csv)
    if "filename" not in df.columns:
        return []
    sdf = df[df["filename"] == audio_filename]
    spans = []
    for r in sdf.itertuples(index=False):
        spans.append(
            {
                "event_label": str(r.event_label),
                "start_time": float(r.start_time),
                "end_time": float(r.end_time),
            }
        )
    return spans


def _safe_name(s: str) -> str:
    """Convert string to a filesystem-safe name."""
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    return s.strip().replace(" ", "_")


def save_predicted_audio_clips(
    waveform: torch.Tensor,
    sample_rate: int,
    pred_spans: List[Dict],
    clip_root: str,
    min_samples: int = 1,
) -> int:
    """Export predicted event spans as per-class audio clips."""
    saved = 0
    total_samples = waveform.shape[-1]
    waveform_2d = waveform if waveform.dim() == 2 else waveform.unsqueeze(0)

    for span in pred_spans:
        label = str(span["event_label"])
        st = float(span["start_time"])
        ed = float(span["end_time"])
        if ed <= st:
            continue

        s = max(0, int(round(st * sample_rate)))
        e = min(total_samples, int(round(ed * sample_rate)))
        if e - s < min_samples:
            continue

        class_dir = os.path.join(clip_root, _safe_name(label))
        os.makedirs(class_dir, exist_ok=True)

        fn = f"{st:.3f}_{ed:.3f}.wav"
        out_path = os.path.join(class_dir, fn)
        clip = waveform_2d[:, s:e].cpu()
        torchaudio.save(out_path, clip, sample_rate)
        saved += 1
    return saved


def clear_directory_contents(dir_path: str):
    """Ensure a directory exists and clear all its existing files/subdirs."""
    os.makedirs(dir_path, exist_ok=True)
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass


def normalize_slashes(path: str) -> str:
    """Normalize backslashes to forward slashes."""
    return path.replace("\\", "/")


def to_labelstudio_audio_uri(original_path: str) -> str:
    """Convert local audio path into a Label Studio-compatible URI."""
    # normalize mixed slashes first
    p = normalize_slashes(original_path)

    # replace D:/ prefix with /data/local-files/?d=
    if re.match(r"^[dD]:/", p):
        rel = p[3:]
        encoded = quote(rel, safe="/")
        return f"/data/local-files/?d={encoded}"

    # fallback: keep path but URL-encode unicode and unsafe chars
    return quote(p, safe="/:")


def build_labelstudio_task(
    audio_uri: str,
    pred_spans: List[Dict],
    original_length: float,
    from_name: str,
    to_name: str,
    prediction_id: int,
    completed_by: int,
    origin: str,
) -> Dict:
    """Build one Label Studio task object from predicted spans."""
    disabled_labels = set(['Scraping', 'Tapping'])
    result_items = []
    for s in pred_spans:
        st = float(s["start_time"])
        ed = float(s["end_time"])
        lb = str(s["event_label"])
        score = float(s.get("score", 0.0))
        if ed-st < 0.4:
            continue
        if lb in disabled_labels:
            continue
        if ed <= st:
            continue
        region_id = "".join(random.choices(
            string.ascii_letters + string.digits, k=5))
        result_items.append(
            {
                "original_length": float(original_length),
                "value": {
                    "start": st,
                    "end": ed,
                    "channel": 0,
                    "labels": [lb],
                },
                "id": region_id,
                "from_name": from_name,
                "to_name": to_name,
                "type": "labels",
                "origin": origin,
                "score": score,
            }
        )
    filename = unquote(os.path.basename(audio_uri))
    return {
        "predictions": [
            {
                "id": int(prediction_id),
                "completed_by": int(completed_by),
                "result": result_items,
            }
        ],
        "data": {"audio": audio_uri, "filename": filename},
    }


def collect_audio_files(audio: str | None, audio_dir: str | None) -> List[str]:
    """Collect and deduplicate audio files from file and/or directory input."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    items: List[str] = []

    if audio:
        items.append(audio)

    if audio_dir:
        for filename in os.listdir(audio_dir):
            if os.path.splitext(filename)[1].lower() in exts:
                items.append(os.path.join(audio_dir, filename))

    dedup = []
    seen = set()
    for p in items:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.exists(ap):
            seen.add(ap)
            dedup.append(ap)
    return sorted(dedup)


def main():
    """Inference script entry point for timeline and Label Studio output."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=str, default=None,
                    help="Single audio file path")
    ap.add_argument("--audio_dir", type=str, default=None,
                    help="Directory for batch inference")
    ap.add_argument("--annotations", type=str, default="data/anno3.csv")
    ap.add_argument("--checkpoint", type=str,
                    default=r"logs\20260405-001204\best_checkpoint.pth")
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--pred_json", type=str, default="pred_spans.json")
    ap.add_argument("--pred_labelstudio_json", type=str,
                    default="labelstudio.json")

    ap.add_argument("--labelstudio_from_name", type=str, default="labels")
    ap.add_argument("--labelstudio_to_name", type=str, default="audio")
    ap.add_argument("--labelstudio_completed_by", type=int, default=1)
    ap.add_argument("--labelstudio_prediction_id_start", type=int, default=1)
    ap.add_argument("--labelstudio_origin", type=str, default="manual")

    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min_duration", type=float, default=0.06)
    ap.add_argument("--window_sec", type=float, default=None)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--save_pred_clips", action="store_true")
    ap.add_argument("--audio_mode", type=str, default="auto",
                    choices=["auto", "mono", "stereo"])
    args = ap.parse_args()

    if not args.audio and not args.audio_dir:
        raise ValueError("Provide either --audio or --audio_dir (or both).")

    clear_directory_contents(args.output_dir)

    audio_files = collect_audio_files(args.audio, args.audio_dir)
    if not audio_files:
        raise ValueError("No valid audio files found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = (
        {int(k): v for k, v in ckpt["idx_to_label"].items()}
        if isinstance(list(ckpt["idx_to_label"].keys())[0], str)
        else ckpt["idx_to_label"]
    )
    sample_rate = int(ckpt.get("sample_rate", 16000))
    frame_hop_sec = float(ckpt.get("frame_hop_sec", 0.02))
    window_sec = float(args.window_sec) if args.window_sec is not None else float(
        ckpt.get("window_sec", 10.0)
    )
    ckpt_audio_mode = str(ckpt.get("audio_mode", "mono"))
    audio_mode = ckpt_audio_mode if args.audio_mode == "auto" else args.audio_mode
    audio_channels = 2 if audio_mode == "stereo" else 1

    model = ResNetConformerSED(
        num_classes=len(label_to_idx),
        sample_rate=sample_rate,
        feature_extractor='melspec',
        use_specaug=False,
        audio_channels=audio_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    simple_preds = []
    labelstudio_tasks = []

    pred_id = int(args.labelstudio_prediction_id_start)
    for audio_path in audio_files:
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

        audio_filename = os.path.basename(audio_path)
        audio_stem = os.path.splitext(audio_filename)[0]
        gt_spans = load_gt_spans(args.annotations, audio_filename)

        timeline_out = os.path.join(
            args.output_dir, f"{audio_stem}_timeline.png")
        waveform_plot = waveform[0] if waveform.dim() == 2 else waveform
        plot_timeline(
            waveform=waveform_plot.cpu().numpy(),
            sr=sample_rate,
            gt_spans=gt_spans,
            pred_spans=pred_spans,
            output_path=timeline_out,
            title=f"SED Timeline: {audio_filename}",
        )

        if args.save_pred_clips:
            clip_dir = os.path.join(args.output_dir, "pred_clips", audio_stem)
            save_predicted_audio_clips(
                waveform=waveform,
                sample_rate=sample_rate,
                pred_spans=pred_spans,
                clip_root=clip_dir,
            )

        simple_preds.append(
            {
                "audio": audio_filename,
                "audio_path": audio_path,
                "pred_spans": pred_spans,
            }
        )

        audio_uri = to_labelstudio_audio_uri(audio_path)
        labelstudio_tasks.append(
            build_labelstudio_task(
                audio_uri=audio_uri,
                pred_spans=pred_spans,
                original_length=float(waveform.shape[-1] / sample_rate),
                from_name=args.labelstudio_from_name,
                to_name=args.labelstudio_to_name,
                prediction_id=pred_id,
                completed_by=args.labelstudio_completed_by,
                origin=args.labelstudio_origin,
            )
        )
        pred_id += 1
        print(f"Processed: {audio_path} -> {timeline_out}")

    save_json(os.path.join(args.output_dir, args.pred_json),
              {"items": simple_preds})
    pred_labelstudio_json = os.path.join(
        args.output_dir, args.pred_labelstudio_json)
    os.makedirs(os.path.dirname(pred_labelstudio_json)
                or ".", exist_ok=True)
    with open(pred_labelstudio_json, "w", encoding="utf-8") as f:
        json.dump(labelstudio_tasks, f, ensure_ascii=False, indent=2)

    print(f"Saved aggregated simple predictions: {args.pred_json}")
    print(f"Saved aggregated Label Studio JSON: {pred_labelstudio_json}")


if __name__ == "__main__":
    main()
