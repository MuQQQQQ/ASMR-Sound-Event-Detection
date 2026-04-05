<a id="top"></a>

# ASMR Sound Event Detection

<p>
  <a href="#english"><button>English</button></a>
  <a href="#chinese"><button>中文</button></a>
</p>

---

<a id="english"></a>

## English

This project provides a full workflow for **training** and **inference** of a frame-level Sound Event Detection model focused on ASMR audio.

Labels include:
- Speech
- Chewing
- Mouth sounds
- Scraping (not currently included in training data)
- Tapping (not currently included in training data)

Core code is under `src/`:

- `src/train.py`: model training entry point
- `src/infer_and_visualize.py`: inference + timeline visualization + Label Studio export
- `src/data.py`: dataset and split logic
- `src/model.py`: model definitions (ResNet/UNet + Conformer) and EMA
- `src/utils.py`: shared helpers
- `src/convert_labels.py`: convert Label Studio JSON annotations to CSV

---

### 1) Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Any Python version should work.

---

### 2) Data and Annotation Format

Audio files are typically placed in `data/`.

Training annotation CSV must contain these columns:

- `filename`
- `start_time` (in seconds)
- `end_time` (in seconds)
- `event_label`

Example:

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
```

---

### 3) Training

Basic training command:

```bash
python src/train.py \
  --data_dir data \
  --annotations data/anno.csv \
  --epochs 200 \
  --batch_size 32 \
  --lr 1e-4
```

You can pass several annotation files via `--annotations data/anno1.csv,data/anno2.csv,data/anno3.csv` (comma-separated). All audio files must be in the same `data_dir`.

Useful options:

- `--use_ema` to enable EMA
- `--loss_type {bce,focal}`
- `--lr_scheduler {cosine,step,custom,none}`
- `--amp` mixed precision control

Training outputs are written to timestamped folders under `logs/`, including:

- `best_checkpoint.pth`
- `history.json`
- `loss_curves.png`
- `config.txt`

---

### 4) Inference and Visualization

#### Single audio file

```bash
python src/infer_and_visualize.py \
  --audio data/out000.mp3 \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

#### Batch directory inference

```bash
python src/infer_and_visualize.py \
  --audio_dir data \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

Optional:

- `--save_pred_clips` to export predicted event clips by class
- `--pred_json` to customize plain prediction JSON name
- `--pred_labelstudio_json` to customize Label Studio prediction JSON name

Inference outputs in `outputs/` include:

- `*_timeline.png` (waveform + GT + prediction timeline)
- `pred_spans.json`
- `labelstudio.json`
- optional `pred_clips/`

---

### 5) Convert Label Studio JSON to CSV

If your annotations come from Label Studio:

```bash
python src/convert_labels.py --input data/<label_studio_file>.json --output data/<output_file>.csv
```

This generates a training-ready CSV with columns:

- `filename`
- `start_time`
- `end_time`
- `event_label`

<p><a href="#top"><button>Back to Top</button></a></p>

---

<a id="chinese"></a>

## 中文

本项目提供了一个完整流程，用于对 ASMR 音频进行帧级声音事件检测模型的**训练**与**推理**。

标签类别包括：
- Speech（轻语）
- Chewing（咀嚼声）
- Mouth sounds（口腔音）
- Scraping（刮擦，当前训练数据中暂未包含）
- Tapping（敲击，当前训练数据中暂未包含）

核心代码位于 `src/`：

- `src/train.py`：模型训练入口
- `src/infer_and_visualize.py`：推理 + 时间轴可视化 + Label Studio 格式导出
- `src/data.py`：数据集与数据划分逻辑
- `src/model.py`：模型定义（ResNet/UNet + Conformer）与 EMA
- `src/utils.py`：通用辅助函数
- `src/convert_labels.py`：将 Label Studio 的 JSON 标注转换为 CSV

---

### 1）环境安装

安装依赖：

```bash
pip install -r requirements.txt
```

任意 Python 版本通常都可以运行。

---

### 2）数据与标注格式

音频文件放在 `data/` 目录下。

训练标注 CSV 必须包含以下列：

- `filename`
- `start_time`（单位：秒）
- `end_time`（单位：秒）
- `event_label`

示例：

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
```

---

### 3）训练

基础训练命令：

```bash
python src/train.py \
  --data_dir data \
  --annotations data/anno.csv \
  --epochs 200 \
  --batch_size 32 \
  --lr 1e-4
```

你也可以通过 `--annotations data/anno1.csv,data/anno2.csv,data/anno3.csv` 传入多个标注文件（用逗号分隔）。但所有音频文件必须位于同一个 `data_dir` 中。

常用可选参数：

- `--use_ema`：启用 EMA
- `--loss_type {bce,focal}`
- `--lr_scheduler {cosine,step,custom,none}`
- `--amp`：混合精度控制

训练输出会写入 `logs/` 下按时间戳命名的目录，通常包括：

- `best_checkpoint.pth`
- `history.json`
- `loss_curves.png`
- `config.txt`

---

### 4）推理与可视化

#### 单个音频文件推理

```bash
python src/infer_and_visualize.py \
  --audio data/out000.mp3 \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

#### 批量目录推理

```bash
python src/infer_and_visualize.py \
  --audio_dir data \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

可选参数：

- `--save_pred_clips`：按类别导出预测到的事件片段
- `--pred_json`：自定义普通预测 JSON 文件名
- `--pred_labelstudio_json`：自定义 Label Studio 预测 JSON 文件名

`outputs/` 中的推理结果包括：

- `*_timeline.png`（波形 + GT + 预测时间轴）
- `pred_spans.json`
- `labelstudio.json`
- 可选的 `pred_clips/`

---

### 5）将 Label Studio JSON 转换为 CSV

如果标注来自 Label Studio：

```bash
python src/convert_labels.py --input data/<label_studio_file>.json --output data/<output_file>.csv
```

该命令会生成可直接用于训练的 CSV，包含以下列：

- `filename`
- `start_time`
- `end_time`
- `event_label`

<p><a href="#top"><button>返回顶部</button></a></p>

test