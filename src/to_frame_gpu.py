import json
import math
import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm

JSON_PATH = r"outputs_0909\pred_spans.json"
BACKGROUND = "1.png"
OUTPUT_DIR = "vds/0909"
FPS = 10
WIDTH = 1920
HEIGHT = 1080


TRACK_HEIGHT = 40       # 轨道高度

CLASSES = [
    "Speech",
    "Chewing",
    "Mouth_Sounds",
    "Breathing",
    # "Tapping",
    "Water_Bottle",
    "Slime",
    "Rub",
    "Scrub",
    "Rasp"
]
COLOR_MAP = {
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
WORD_MAP = {
    "Speech": "人声",
    "Chewing": "食音",
    "Mouth_Sounds": "口腔",
    "Slime": "凝胶",
    "Breathing": "吹气",
    "Water_Bottle": "水瓶",
    "Rub": "一般",
    "Scrub": "较强",
    "Rasp": "猛烈"
}

FONT_PATH = r"C:\Users\11703\AppData\Local\Microsoft\Windows\Fonts/SarasaTermSCNerd.ttc"  # 微软雅黑
FONT_SIZE = 30


def get_audio_duration(path, ffprobe_bin="ffprobe"):
    result = subprocess.run(
        [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)


def draw_class_ratios(img, class_ratios, x=50, y=80, line_gap=5):
    font = ImageFont.truetype(
        FONT_PATH,
        25,
        index=7
    )
    draw = ImageDraw.Draw(img)

    lines = ["当前片段各类别占比："]
    for k in class_ratios:
        if k in {'Rub', 'Scrub', 'Rasp'}:
            lines.append(
                f'底噪{WORD_MAP[k]}:  {class_ratios[k]*100:5.2f} %')
        else:
            lines.append(f'{WORD_MAP[k]}:  {class_ratios[k]*100:5.2f} %')
    bbox = draw.multiline_textbbox(
        (x, y),
        "\n".join(lines),
        font=font,
        spacing=line_gap
    )

    padding = 12
    bg_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg_layer)
    bg_draw.rounded_rectangle(
        (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding
        ),
        radius=8,
        fill=(0, 0, 0, 125)
    )

    img = Image.alpha_composite(img.convert("RGBA"), bg_layer)
    draw = ImageDraw.Draw(img)

    text = "\n".join(lines)

    draw.multiline_text(
        (x + 1, y + 1),
        text,
        font=font,
        fill=(0, 0, 0, 180),
        spacing=line_gap, align="right"
    )

    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(245, 245, 245, 225),
        spacing=line_gap, align="right"
    )

    return img


def draw_cover_title(
    img,
    title="ASMR",
    subtitle="时间轴可视化",
    title_size=130,
    subtitle_size=130,
):
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_PATH, title_size)
    subtitle_font = ImageFont.truetype(FONT_PATH, subtitle_size)

    def draw_centered(text, font, y, fill=(242, 123, 31, 245), shadow=(0, 0, 0, 180)):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        x = (img.width - w) // 2

        draw.text((x+5, y+6), text, font=font, fill=shadow)
        draw.text((x, y), text, font=font, fill=fill)
        return y + bbox[3] - bbox[1]

    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_h = title_bbox[3] - title_bbox[1]
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_h = subtitle_bbox[3] - subtitle_bbox[1]

    total_h = title_h + 50 + subtitle_h
    y = (img.height - total_h) // 2 + 200

    draw_centered(title, title_font, y)
    draw_centered(subtitle, subtitle_font, y + title_h + 50)

    return img


def draw_static_background(bg_img, spans, duration, title="默认标题", author="顾子韵_w", ratio_per_class=None, save_img=False, cover_path=None):

    OUT_W, OUT_H = 1920, 1080

    AVATAR_SIZE = 300
    AVATAR_TOP = 50

    TRACK_START_Y = 480
    TRACK_HEIGHT = 55
    TRACK_GAP = 7
    NOISE_GAP = 7
    TRACK_MARGIN_X = 35
    LABEL_WIDTH = 70
    TIMELINE_GAP = 2
    TITLE_TOP = AVATAR_TOP + AVATAR_SIZE + 70
    AUTHOR_TOP = AVATAR_TOP + AVATAR_SIZE - 40

    timeline_x1 = TRACK_MARGIN_X + LABEL_WIDTH + TIMELINE_GAP
    timeline_x2 = OUT_W - TRACK_MARGIN_X
    timeline_width = timeline_x2 - timeline_x1
    # TIMELINE_DURATION = 30 * 60
    TIMELINE_STEP = 5 * 60
    timeline_duration = math.ceil(duration / TIMELINE_STEP) * TIMELINE_STEP
    timeline_duration = duration

    KAI_FONT_PATH = r"C:\Windows\Fonts\simkai.ttf"

    bg = ImageOps.fit(
        bg_img.convert("RGB"),
        (OUT_W, OUT_H),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    ).convert("RGBA")

    bg = bg.filter(ImageFilter.GaussianBlur(14))

    dark_layer = Image.new(
        "RGBA",
        (OUT_W, OUT_H),
        (0, 0, 0, 155)
    )
    img = Image.alpha_composite(bg, dark_layer)

    glass_layer = Image.new(
        "RGBA",
        (OUT_W, OUT_H),
        (255, 255, 255, 10)
    )
    img = Image.alpha_composite(img, glass_layer)

    src_w, src_h = bg_img.size
    crop_size = min(src_w, src_h)
    crop_x = (src_w - crop_size) // 2
    crop_y = (src_h - crop_size) // 2

    avatar = bg_img.convert("RGB").crop((
        crop_x,
        crop_y,
        crop_x + crop_size,
        crop_y + crop_size
    ))

    avatar = avatar.resize(
        (AVATAR_SIZE, AVATAR_SIZE),
        Image.Resampling.LANCZOS
    ).convert("RGBA")

    avatar_x = (OUT_W - AVATAR_SIZE) // 2

    avatar_mask = Image.new(
        "L",
        (AVATAR_SIZE, AVATAR_SIZE),
        0
    )

    ImageDraw.Draw(avatar_mask).ellipse(
        (0, 0, AVATAR_SIZE - 1, AVATAR_SIZE - 1),
        fill=255
    )
    SHADOW_PADDING = 30

    shadow_mask = Image.new(
        "L",
        (
            AVATAR_SIZE + SHADOW_PADDING * 2,
            AVATAR_SIZE + SHADOW_PADDING * 2
        ),
        0
    )

    shadow_draw = ImageDraw.Draw(shadow_mask)

    shadow_draw.ellipse(
        (
            SHADOW_PADDING,
            SHADOW_PADDING,
            SHADOW_PADDING + AVATAR_SIZE - 1,
            SHADOW_PADDING + AVATAR_SIZE - 1
        ),
        fill=190
    )

    shadow_mask = shadow_mask.filter(
        ImageFilter.GaussianBlur(14)
    )

    shadow = Image.new(
        "RGBA",
        shadow_mask.size,
        (0, 0, 0, 0)
    )
    shadow.putalpha(shadow_mask)

    img.alpha_composite(
        shadow,
        (
            avatar_x - SHADOW_PADDING,
            AVATAR_TOP - SHADOW_PADDING
        )
    )

    img.paste(
        avatar,
        (avatar_x, AVATAR_TOP),
        avatar_mask
    )

    draw = ImageDraw.Draw(img)

    # Avatar 外圈
    draw.ellipse(
        (
            avatar_x,
            AVATAR_TOP,
            avatar_x + AVATAR_SIZE - 1,
            AVATAR_TOP + AVATAR_SIZE - 1
        ),
        outline=(255, 255, 255, 210),
        width=5
    )

    try:
        label_font = ImageFont.truetype(
            FONT_PATH,
            max(FONT_SIZE - 6, 12)
        )
    except Exception:
        label_font = ImageFont.load_default()

    # Title 字体
    try:
        title_font = ImageFont.truetype(
            FONT_PATH,
            40
        )
    except Exception:
        title_font = ImageFont.load_default()

    # Author：楷体 + 斜体
    try:
        author_font = ImageFont.truetype(
            KAI_FONT_PATH,
            50
        )
    except Exception:
        author_font = title_font

    title = str(title).strip()

    if title:
        bbox = draw.textbbox(
            (0, 0),
            title,
            font=title_font
        )

        title_w = bbox[2] - bbox[0]
        title_h = bbox[3] - bbox[1]

        title_x = (OUT_W - title_w) // 2
        title_y = TITLE_TOP - bbox[1]

        # 阴影
        draw.text(
            (title_x + 1, title_y + 1),
            title,
            font=title_font,
            fill=(0, 0, 0, 180)
        )

        draw.text(
            (title_x, title_y),
            title,
            font=title_font,
            fill=(255, 255, 255, 225)
        )

    author = str(author).strip()

    if author:
        author_text = "@" + author

        bbox = draw.textbbox(
            (0, 0),
            author_text,
            font=author_font
        )

        author_w = bbox[2] - bbox[0]
        author_h = bbox[3] - bbox[1]

        # 位于 Avatar 右下侧
        author_x = (
            avatar_x
            + AVATAR_SIZE
            - author_w
            + 200
        )

        author_y = AUTHOR_TOP - bbox[1]

        # 阴影
        draw.text(
            (author_x + 1, author_y + 1),
            author_text,
            font=author_font,
            fill=(0, 0, 0, 180)
        )

        draw.text(
            (author_x, author_y),
            author_text,
            font=author_font,
            fill=(235, 235, 235, 220)
        )

    labels = list(CLASSES)
    label_y_map = {}

    track_layer = Image.new(
        "RGBA",
        (OUT_W, OUT_H),
        (0, 0, 0, 0)
    )

    track_draw = ImageDraw.Draw(track_layer)
    if len(labels) >= 3:
        noise_top = (
            TRACK_START_Y
            + (len(labels) - 3) * (TRACK_HEIGHT + TRACK_GAP)
        )
        noise_bottom = (
            TRACK_START_Y
            + (len(labels) - 1) * (TRACK_HEIGHT + TRACK_GAP)
            + TRACK_HEIGHT
        )

        noise_text = "底噪强度"

        try:
            noise_font = ImageFont.truetype(
                FONT_PATH,
                25
            )
        except Exception:
            noise_font = label_font

        bbox = track_draw.textbbox(
            (0, 0),
            "底",
            font=noise_font
        )
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]

        total_h = len(noise_text) * char_h + (len(noise_text) - 1) * 2

        noise_x = max(
            2,
            (TRACK_MARGIN_X - char_w) // 2
        )
        noise_y = (
            noise_top
            + (noise_bottom - noise_top - total_h) // 2
        )

        for j, char in enumerate(noise_text):
            cy = noise_y + j * (char_h + 2)

            track_draw.text(
                (noise_x + 1, cy + 1),
                char,
                font=noise_font,
                fill=(0, 0, 0)
            )

            track_draw.text(
                (noise_x, cy),
                char,
                font=noise_font,
                fill=(255, 255, 255, 185)
            )
    for i, label in enumerate(labels):

        y = TRACK_START_Y + i * (TRACK_HEIGHT + TRACK_GAP)
        if i >= max(0, len(labels) - 3):
            y += NOISE_GAP
        if i == max(0, len(labels) - 3):
            sep_y = y - NOISE_GAP // 2
            track_draw.line(
                (TRACK_MARGIN_X-30, sep_y-5, OUT_W - TRACK_MARGIN_X+30, sep_y-5),
                fill=(255, 255, 255, 255),
                width=2
            )

        label_y_map[label] = y

        color = COLOR_MAP.get(
            label,
            (170, 170, 170)
        )

        track_draw.rectangle(
            (
                TRACK_MARGIN_X,
                y,
                OUT_W - TRACK_MARGIN_X,
                y + TRACK_HEIGHT
            ),
            fill=(0, 0, 0, 105)
        )

        divider_x = TRACK_MARGIN_X + LABEL_WIDTH

        track_draw.rectangle(
            (
                timeline_x1,
                y + 8,
                timeline_x2,
                y + TRACK_HEIGHT - 8
            ),
            fill=(0, 0, 0, 90)
        )

        # Label
        text = WORD_MAP.get(label, label)

        bbox = track_draw.textbbox(
            (0, 0),
            text,
            font=label_font
        )

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x = (
            TRACK_MARGIN_X
            + (LABEL_WIDTH - text_w) // 2
        )

        text_y = (
            y
            + (TRACK_HEIGHT - text_h) // 2
            - bbox[1]
        )

        track_draw.text(
            (text_x + 1, text_y + 1),
            text,
            font=label_font,
            fill=(0, 0, 0, 180)
        )

        track_draw.text(
            (text_x, text_y),
            text,
            font=label_font,
            fill=color
        )

    img = Image.alpha_composite(
        img,
        track_layer
    )

    span_layer = Image.new(
        "RGBA",
        (OUT_W, OUT_H),
        (0, 0, 0, 0)
    )

    span_draw = ImageDraw.Draw(span_layer)

    for span in spans:

        label = span["event_label"]

        if label not in label_y_map:
            continue

        y = label_y_map[label]

        start_time = max(
            0,
            span["start_time"]
        )
        end_time = min(
            timeline_duration,
            span["end_time"]
        )

        start_x = int(
            timeline_x1
            + start_time / timeline_duration
            * timeline_width
        )

        end_x = int(
            timeline_x1
            + end_time / timeline_duration
            * timeline_width
        )

        if end_x <= start_x:
            end_x = start_x + 2

        start_x = max(
            timeline_x1,
            min(start_x, timeline_x2)
        )

        end_x = max(
            timeline_x1,
            min(end_x, timeline_x2)
        )

        color = COLOR_MAP.get(
            label,
            (170, 170, 170)
        )

        span_draw.rectangle(
            (
                start_x,
                y + 10,
                end_x,
                y + TRACK_HEIGHT - 10
            ),
            fill=(*color, 235)
        )

    img = Image.alpha_composite(
        img,
        span_layer
    )

    grid_layer = Image.new(
        "RGBA",
        (OUT_W, OUT_H),
        (0, 0, 0, 0)
    )

    grid_draw = ImageDraw.Draw(grid_layer)

    if labels:
        tracks_bottom = (
            TRACK_START_Y
            + (len(labels) - 1) * (TRACK_HEIGHT + TRACK_GAP)
            + TRACK_HEIGHT
        )

    grid_layer = Image.new(
        "RGBA",
        (OUT_W, OUT_H),
        (0, 0, 0, 0)
    )

    grid_draw = ImageDraw.Draw(grid_layer)

    if labels:
        tracks_bottom = (
            TRACK_START_Y
            + (len(labels) - 1) * (TRACK_HEIGHT + TRACK_GAP)
            + TRACK_HEIGHT
        )

        try:
            time_font = ImageFont.truetype(
                FONT_PATH,
                20
            )
        except Exception:
            time_font = label_font

        end_text_x = timeline_x1 + timeline_width

        end_text = f'{int(duration)//60:02d}:{int(duration)%60:02d}'
        ticks = (0, 5, 10, 15, 20, 25, 30, 35, 40)
        need_end = False
        for i, minute in enumerate(ticks):

            x = int(
                timeline_x1
                + minute * 60 / timeline_duration
                * timeline_width
            )
            if x > end_text_x:
                need_end = True
                break

            grid_draw.line(
                (
                    x,
                    TRACK_START_Y,
                    x,
                    tracks_bottom
                ),
                fill=(255, 255, 255, 45),
                width=1
            )

            time_text = f"{minute:02d}:00"

            bbox = grid_draw.textbbox(
                (0, 0),
                time_text,
                font=time_font
            )

            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = x - text_w // 2

            text_y = tracks_bottom + 8

            grid_draw.text(
                (text_x + 1, text_y + 1),
                time_text,
                font=time_font,
                fill=(0, 0, 0, 150)
            )

            grid_draw.text(
                (text_x, text_y),
                time_text,
                font=time_font,
                fill=(255, 255, 255, 180)
            )

        last_text_x = int(
            timeline_x1
            + ticks[i-1] * 60 / timeline_duration
            * timeline_width
        )
        if last_text_x < end_text_x-100 and need_end:
            bbox = grid_draw.textbbox(
                (0, 0),
                end_text,
                font=time_font
            )

            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = end_text_x - text_w // 2

            text_y = tracks_bottom + 8

            grid_draw.text(
                (text_x + 1, text_y + 1),
                end_text,
                font=time_font,
                fill=(0, 0, 0, 150)
            )

            grid_draw.text(
                (text_x, text_y),
                end_text,
                font=time_font,
                fill=(255, 255, 255, 180)
            )

    img = Image.alpha_composite(
        img,
        grid_layer
    )
    if ratio_per_class is not None:
        img = draw_class_ratios(img, ratio_per_class)
    # img = draw_cover_title(img)
    result = img.convert("RGB")
    if save_img:
        cover = img.copy()
        cover = draw_cover_title(cover)
        cover.convert("RGB").save(cover_path)
    result.save("debug_static_bg.png")

    return result, label_y_map, timeline_x1, timeline_x2, timeline_width


def draw_frame_fast_v2(static_bg_np, t, duration, label_y_map):

    x = int((t / duration) * WIDTH)
    x = max(2, min(x, WIDTH - 3))

    y_values = label_y_map.values()
    y_start = min(y_values) - 30
    y_end = max(y_values) + TRACK_HEIGHT + 10

    original_slice = static_bg_np[y_start:y_end, x-1:x+2].copy()

    static_bg_np[y_start:y_end, x-1:x+2] = [255, 255, 255]

    yield static_bg_np

    static_bg_np[y_start:y_end, x-1:x+2] = original_slice


def merge_spans(spans, gap_threshold=1):
    grouped = {}
    for span in spans:
        grouped.setdefault(span["event_label"], []).append(span)

    merged = []
    for label, items in grouped.items():
        items.sort(key=lambda x: x["start_time"])
        cur = items[0].copy()

        for span in items[1:]:
            if span["start_time"] - cur["end_time"] < gap_threshold:
                cur["end_time"] = max(cur["end_time"], span["end_time"])
            else:
                merged.append(cur)
                cur = span.copy()

        merged.append(cur)

    return sorted(merged, key=lambda x: x["start_time"])


def process_one_item_v2(
    item,
    index,
    output_dir=None,
    background=None,
    ffmpeg_bin=None,
    author="顾子韵_w",
    org_audio_name=None,
    saved_name=None,
    save_img=False, cover_path=None
):
    ffmpeg_bin = (
        r"D:\ffmpeg-6.1.1-essentials_build\bin\ffmpeg"
        if ffmpeg_bin is None
        else ffmpeg_bin
    )
    ffprobe_bin = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe.exe")
    if not os.path.isfile(ffprobe_bin):
        ffprobe_bin = os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe")
    if not os.path.isfile(ffprobe_bin):
        ffprobe_bin = "ffprobe"
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    background = BACKGROUND if background is None else background
    audio_path = item["audio_path"]
    duration_per_class = {k: 0 for k in CLASSES}
    duration = get_audio_duration(audio_path, ffprobe_bin)
    total_frames = int(duration * FPS)
    os.makedirs(output_dir, exist_ok=True)

    for span in item["pred_spans"]:
        duration_per_class[span['event_label']
                           ] += span['end_time']-span['start_time']
    ratio_per_class = {
        k: duration_per_class[k]/duration for k in duration_per_class}
    spans = merge_spans(item["pred_spans"], gap_threshold=5)

    title = os.path.splitext(os.path.basename(audio_path))[0]
    if saved_name is None:
        saved_name = title
    if org_audio_name is not None:
        title = f'{org_audio_name} {title}'
    print(f"渲染启动: {os.path.basename(audio_path)}")

    bg = Image.open(background).convert('RGB')
    title_cleaned = title.replace('：', ':')
    static_bg, label_y_map, timeline_x1, timeline_x2, timeline_width = draw_static_background(
        bg, spans, duration, title_cleaned, author=author, ratio_per_class=ratio_per_class, save_img=save_img, cover_path=cover_path)
    frame_array = np.array(static_bg, dtype=np.uint8)

    y_values = list(label_y_map.values())
    y_start = min(y_values) - 20
    y_end = max(y_values) + TRACK_HEIGHT + 30

    video_path = os.path.join(output_dir, f"{saved_name}.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-af",
        f"afade=t=in:st=0:d=1,"
        f"afade=t=out:st={max(0, duration - 1):.3f}:d=1",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        video_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10**7)
    pointer_slice = np.full((y_end - y_start, 3, 3), 255, dtype=np.uint8)
    write = proc.stdin.write
    get_bytes = frame_array.tobytes
    for i in range(total_frames):
        t = i / FPS
        x = int((t / duration) * timeline_width) + timeline_x1
        dirty_rect_backup = frame_array[y_start:y_end, x-1:x+2].copy()
        frame_array[y_start:y_end, x-1:x+2] = pointer_slice
        write(get_bytes())
        frame_array[y_start:y_end, x-1:x+2] = dirty_rect_backup
    proc.stdin.close()
    proc.wait()
    print(f"视频已保存至: {video_path}")
    return duration


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data["items"][:]):

        process_one_item_v2(item, i)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"总耗时: {t2 - t1:.2f} 秒")
