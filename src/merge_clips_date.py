# 按照日期进行合并
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(r"D:\ASMR\1\gzy_cut\_cleaned")


# 开始日期、结束日期（包含首尾）
START_DATE = datetime(2026, 7, 1)
END_DATE = datetime(2026, 8, 31)

# 最终输出目录
OUTPUT_DIR = Path(r"D:\ASMR\1\gzy_cut\_merged_date")

# 中间文件目录
TEMP_DIR = OUTPUT_DIR / "_temp"

FFMPEG = "ffmpeg"

# 并行处理文件夹数量
MAX_WORKERS = min(4, os.cpu_count() or 1)

AUDIO_EXTS = {
    ".mp3",
    ".aac",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".opus",
}

# 文件夹日期：
# 顶级轻语！ 2026年07月25日
date_pattern = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")

# 文件序号：
# 0002_1024.00-1057.62.mp3
number_pattern = re.compile(r"^(\d{4})_")


# =========================
# 日期解析
# =========================

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


# =========================
# 文件序号解析
# =========================

def get_sequence_number(path):
    match = number_pattern.match(path.name)

    if match:
        return int(match.group(1))

    return None


# =========================
# 创建 concat 文件
# =========================

def create_concat_file(files):
    f = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    )

    concat_path = Path(f.name)

    try:
        for audio in files:
            path = audio.resolve().as_posix().replace("'", r"'\''")
            f.write(f"file '{path}'\n")
    finally:
        f.close()

    return concat_path


# =========================
# FFmpeg 合并
# =========================

def concat_audio(files, output, code='libmp3lame'):
    if not files:
        raise ValueError("没有输入文件")

    if len(files) == 1:
        shutil.copy2(files[0], output)
        return

    concat_file = create_concat_file(files)

    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),

        # 不重新编码
        "-c", code,


        str(output),
    ]

    try:
        subprocess.run(cmd, check=True)
    finally:
        concat_file.unlink(missing_ok=True)


# =========================
# 第一阶段：
# 合并单个文件夹
# =========================

def merge_one_folder(folder, index, total):
    date = get_date_key(folder)

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]

    # 只保留能解析出四位序号的文件
    numbered_files = []

    for file in files:
        number = get_sequence_number(file)

        if number is not None:
            numbered_files.append((number, file))

    numbered_files.sort(key=lambda x: x[0])

    if not numbered_files:
        return {
            "success": False,
            "folder": folder,
            "date": date,
            "output": None,
            "message": "没有找到符合 XXXX_ 开头格式的音频",
        }

    ordered_files = [file for _, file in numbered_files]

    # 使用日期 + 文件夹序号生成唯一中间文件
    # 不使用文件夹名，避免特殊字符造成 FFmpeg 路径问题
    output = TEMP_DIR / f"{index:06d}_{folder.name}.mp3"

    try:
        concat_audio(ordered_files, output)

        return {
            "success": True,
            "folder": folder,
            "date": date,
            "output": output,
            "message": f"{len(ordered_files)} 个文件",
        }

    except Exception as e:
        return {
            "success": False,
            "folder": folder,
            "date": date,
            "output": None,
            "message": str(e),
        }


# =========================
# 第二阶段：
# 按日期合并文件夹
# =========================

def merge_by_date(results):
    # 按日期排序
    results = sorted(results, key=lambda x: x["date"])

    files = [item["output"] for item in results]

    if not files:
        print("没有可以合并的文件。")
        return

    start_str = START_DATE.strftime("%Y%m%d")
    end_str = END_DATE.strftime("%Y%m%d")

    output = OUTPUT_DIR / f"{start_str}-{end_str}.mp3"

    print()
    print("=" * 70)
    print("开始按照日期合并")
    print("=" * 70)

    for item in results:
        print(
            f"{item['date'].strftime('%Y-%m-%d')}  "
            f"{item['folder'].name}"
        )

    print("-" * 70)

    try:
        concat_audio(files, output, 'copy')

        print(f"[完成] 最终文件：{output}")
        print(f"[完成] 共合并 {len(files)} 个文件夹")

    except subprocess.CalledProcessError as e:
        print(f"[失败] 最终合并失败，FFmpeg 返回码：{e.returncode}")

    except Exception as e:
        print(f"[失败] 最终合并失败：{e}")


# =========================
# 主程序
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("按日期合并音频")
    print("=" * 70)

    print(f"输入目录 : {INPUT_DIR}")
    print(f"开始日期 : {START_DATE.strftime('%Y-%m-%d')}")
    print(f"结束日期 : {END_DATE.strftime('%Y-%m-%d')}")
    print(f"线程数   : {MAX_WORKERS}")
    print()

    # 找到所有一级子文件夹
    folders = [
        p for p in INPUT_DIR.iterdir()
        if p.is_dir()
        and p.name not in {"_merged", "_date_merged"}
    ]

    # 解析日期并筛选
    selected = []

    for folder in folders:
        date = get_date_key(folder)

        if date is None:
            print(f"[跳过] 无法解析日期：{folder.name}")
            continue

        if START_DATE <= date <= END_DATE:
            selected.append(folder)

    # 首先按照日期排序
    selected.sort(key=get_date_key)

    print(f"找到符合日期范围的文件夹：{len(selected)}")
    print("-" * 70)

    if not selected:
        print("没有符合条件的文件夹。")
        return

    for i, folder in enumerate(selected, 1):
        print(
            f"[{i:03d}] "
            f"{get_date_key(folder).strftime('%Y-%m-%d')}  "
            f"{folder.name}"
        )

    print()
    print("=" * 70)
    print("第一阶段：并行合并每个文件夹")
    print("=" * 70)

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                merge_one_folder,
                folder,
                index,
                len(selected),
            ): folder
            for index, folder in enumerate(selected, 1)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            folder = result["folder"]

            if result["success"]:
                print(
                    f"[完成] {result['date'].strftime('%Y-%m-%d')} "
                    f"{folder.name}: {result['message']}"
                )
            else:
                print(
                    f"[失败] {folder.name}: "
                    f"{result['message']}"
                )

    # 只保留成功的文件夹
    success_results = [
        r for r in results
        if r["success"]
    ]

    # 如果某个日期的文件夹失败，则不会进入最终合并
    if not success_results:
        print("没有成功生成任何中间文件。")
        return

    # 第二阶段：按照日期合并
    merge_by_date(success_results)

    print()
    print("=" * 70)
    print("全部处理完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
