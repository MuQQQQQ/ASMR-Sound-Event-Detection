# 合并clip_need.py生成的片段，一个文件夹合并出一个文件
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(r"D:\ASMR\1\gzy_cut\_cleaned")
OUTPUT_DIR = Path(r"D:\ASMR\1\gzy_cut\_merged_run029")
FFMPEG = "ffmpeg"
MAX_WORKERS = 8

AUDIO_EXTS = {".mp3", ".aac", ".m4a", ".wav", ".flac", ".ogg", ".opus"}


def natural_key(path: Path):
    import re
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


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


def merge_folder(folder: Path):
    files = sorted(
        [p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=natural_key
    )

    if not files:
        return folder.name, False, "没有音频文件"

    prefix = get_date_key(folder.name)
    if prefix is not None:
        prefix = prefix.strftime("%Y年%m月%d日")
    else:
        prefix = ''
    # 使用第一个文件的扩展名作为输出格式
    output = OUTPUT_DIR / f"{prefix}_{folder.name}{files[0].suffix}"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8"
    ) as f:
        concat_file = Path(f.name)

        for audio in files:
            path = audio.resolve().as_posix().replace("'", r"'\''")
            f.write(f"file '{path}'\n")

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),

        # 不重新编码
        "-c:a", "libmp3lame",


        str(output),
    ]

    try:
        subprocess.run(cmd, check=True)
        return folder.name, True, f"{len(files)} 个文件 -> {output.name}"
    except subprocess.CalledProcessError as e:
        return folder.name, False, f"FFmpeg 失败，返回码 {e.returncode}"
    finally:
        concat_file.unlink(missing_ok=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        [p for p in INPUT_DIR.iterdir()
         if p.is_dir() and p.name != OUTPUT_DIR.name],
        key=lambda p: p.name
    )

    print(f"输入目录 : {INPUT_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"文件夹数 : {len(folders)}")
    print(f"线程数   : {MAX_WORKERS}")
    print("-" * 70)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(merge_folder, folder): folder
            for folder in folders
        }

        for future in as_completed(futures):
            name, success, message = future.result()

            if success:
                print(f"[完成] {name}: {message}")
            else:
                print(f"[失败] {name}: {message}")

    print("-" * 70)
    print("全部处理完成")


if __name__ == "__main__":
    main()
