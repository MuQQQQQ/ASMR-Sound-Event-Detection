
<a id="top"></a>

# ASMR Sound Event Detection

<p>
  <a href="#english"><button>English</button></a>
  <a href="#chinese"><button>简体中文</button></a>
</p>

---

<a id="english"></a>

## English

This repository provides an end-to-end pipeline for **training** and **inference** of a **frame-level Sound Event Detection (SED)** model tailored for ASMR audio.

## 1. Supported Event Labels

* Speech
* Chewing
* Mouth sounds
* Scraping *(currently not included in training data)*
* Tapping *(currently not included in training data)*

---

## 2. Project Structure

Core implementation is located in `src/`:

* `train.py` — training entry point
* `infer_and_visualize.py` — inference, timeline visualization, and Label Studio export
* `data.py` — dataset handling and split logic
* `model.py` — model definitions (ResNet + Conformer) and EMA
* `utils.py` — shared utilities
* `convert_labels.py` — convert Label Studio JSON annotations to CSV


## 3. Model Architecture

The model follows a lightweight **ResNet + Conformer** design for frame-wise prediction.

### 3.1 Input

* `waveform`: **[B, S]**
  *(batch size, number of audio samples)*


### 3.2Feature Extraction

Using `torchaudio.transforms.MelSpectrogram`:

* Key parameters: `n_fft`, `win_length`, `hop_length`, `n_mels`, `sample_rate`
* Output: **[B, M, T]**
  *(batch size, mel bins, time frames)*


### 3.3 ResNet Backbone

Input: `x ∈ [B, M, T]`

(1). Expand channel dimension
   → `[B, 1, M, T]`

(2). Stem:

   * Conv2d(1 → 32, kernel=3, stride=1, padding=1)
   * BatchNorm + ReLU

(3). Residual blocks:

   * `ResBlock2D(32, 64, stride=(2,1))`
   * `ResBlock2D(64, 128, stride=(2,1))`
   * `ResBlock2D(128, 128, stride=(1,1))`

Output: **[B, 128, M/4, T]**

#### 3.3.1 Frequency Pooling

* Mean over frequency → `[B, 128, T]`
* Transpose → `[B, T, 128]`

This converts 2D time-frequency features into a temporal sequence.


### 3.4 Projection Layer

* `Linear(128 → conformer_dim) + Dropout`
* `conformer_dim = 256`

Output: **[B, T, 256]**


### 3.5 Conformer Encoder

Stack of **4 Conformer blocks**, each including:

(1). Feed-forward module (half residual)
(2). Multi-head self-attention
(3). Convolution module:

   * LayerNorm → pointwise Conv1d → GLU
   * Depthwise Conv1d
   * BatchNorm + SiLU
   * Pointwise Conv1d + Dropout
(4). Second feed-forward module
(5). Final LayerNorm

Output shape remains: **[B, T, 256]**


### 3.6 Frame-wise Classifier

* `Linear(256 → num_classes)`

Output:

* Logits: **[B, T, C]**

---

## 4. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Most modern Python versions should work.

---

## 5. Data Format

Audio files should be placed in `data/`.

Annotation CSV format:

Required columns:

* `filename`
* `start_time` (seconds)
* `end_time` (seconds)
* `event_label`

Example:

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
```

---

## 6. Training

Basic command:

```bash
python src/train.py \
  --data_dir data \
  --annotations data/anno.csv \
  --epochs 200 \
  --batch_size 32 \
  --lr 1e-4
```

### 6.1 Multiple annotation files

```bash
--annotations data/a.csv,data/b.csv,data/c.csv
```

> All referenced audio files must reside in the same `data_dir`.


### 6.2 Useful options

* `--use_ema` — enable EMA
* `--loss_type {bce,focal}`
* `--lr_scheduler {cosine,step,custom,none}`
* `--amp` — mixed precision
* `--audio_mode {mono,stereo}` — audio channel mode (stereo loads each clip as 2 channels)
* `--use_lazy_loading` — use lazy loading 
* `--files_per_batch N` — each batch samples windows only from `N` random audio files

### 6.3 Training Outputs

Saved under `logs/<timestamp>/`:

* `best_checkpoint.pth`
* `history.json`
* `loss_curves.png`
* `config.txt`

---

## 7. Inference & Visualization

### 7.1 Single file

```bash
python src/infer_and_visualize.py \
  --audio data/out000.mp3 \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

### 7.2 Batch inference

```bash
python src/infer_and_visualize.py \
  --audio_dir data \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```


### 7.3 Optional arguments

* `--save_pred_clips` — export predicted clips by class
* `--pred_json` — custom prediction JSON filename
* `--pred_labelstudio_json` — custom Label Studio JSON filename
* `--audio_mode {auto,mono,stereo}` — inference audio mode
  * `auto`: follow checkpoint training setting
  * `mono`/`stereo`: manually override

### 7.4 Outputs

Saved in `outputs/`:

* `*_timeline.png` — waveform + GT + predictions
* `pred_spans.json`
* `labelstudio.json`
* `pred_clips/` *(optional)*

---

## 8. Convert Label Studio JSON to CSV

```bash
python src/convert_labels.py \
  --input data/<label_studio_file>.json \
  --output data/<output_file>.csv
```

Generated CSV contains:

* `filename`
* `start_time`
* `end_time`
* `event_label`

---

<p><a href="#top"><button>Back to Top</button></a></p>

---

<a id="chinese"></a>

## 简体中文

本项目提供了一套完整的流程，用于对 **ASMR 音频进行帧级声音事件检测（Sound Event Detection, SED）模型的训练与推理**。


## 1. 支持的事件类别

* 语音（Speech）
* 咀嚼声（Chewing）
* 口腔音（Mouth sounds）
* 刮擦声（Scraping，当前未用于训练）
* 敲击声（Tapping，当前未用于训练）

---

## 2. 项目结构

核心代码位于 `src/` 目录：

* `train.py` —— 训练入口
* `infer_and_visualize.py` —— 推理、时间轴可视化及 Label Studio 导出
* `data.py` —— 数据加载与划分逻辑
* `model.py` —— 模型定义（ResNet + Conformer）及 EMA
* `utils.py` —— 通用工具函数
* `convert_labels.py` —— 将 Label Studio JSON 转换为 CSV

---

## 3. 模型结构

模型采用轻量级 **ResNet + Conformer** 架构，实现帧级预测。

### 3.1 输入

* `waveform`：形状 **[B, S]**
  （批大小，音频采样点数）


### 3.2 特征提取

使用 `MelSpectrogram`：

* 关键参数：`n_fft`、`win_length`、`hop_length`、`n_mels`、`sample_rate`
* 输出：**[B, M, T]**（批大小 × 梅尔频带 × 时间帧）


### 3.3 ResNet 主干

输入：`[B, M, T]`

处理流程：

(1). 扩展通道 → `[B, 1, M, T]`
(2). 卷积 Stem（Conv + BN + ReLU）
(3). 三个残差块（逐步压缩频率维）

输出： **[B, 128, M/4, T]**

#### 3.3.1 频率维池化

* 对频率维取均值 → `[B, 128, T]`
* 转置 → `[B, T, 128]`

将时频特征转换为时间序列特征


### 3.4 投影层

* `Linear(128 → 256) + Dropout`

输出： **[B, T, 256]**


### 3.5 Conformer 编码器

包含 4 个 Conformer Block，每个 Block 包括：

(1). 前馈网络（半残差）
(2). 多头自注意力
(3). 卷积模块（GLU + 深度可分离卷积等）
(4). 第二个前馈网络
(5). LayerNorm

输出维度保持不变：**[B, T, 256]**


### 3.6 帧级分类器

* `Linear(256 → 类别数)`

输出： **[B, T, C]**

---

## 4. 环境安装

```bash
pip install -r requirements.txt
```

支持大多数 Python 版本。

---

## 5. 数据格式

音频文件放置于 `data/` 目录。

标注 CSV 格式:

必须包含以下字段：

* `filename`
* `start_time`（秒）
* `end_time`（秒）
* `event_label`

示例：

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
```

---

## 6. 模型训练

```bash
python src/train.py \
  --data_dir data \
  --annotations data/anno.csv \
  --epochs 200 \
  --batch_size 32 \
  --lr 1e-4
```

### 6.1 多标注文件

```bash
--annotations data/a.csv,data/b.csv,data/c.csv
```

> 所有音频必须位于同一个 `data_dir` 下


### 6.2 常用参数

* `--use_ema`：启用 EMA
* `--loss_type {bce,focal}`
* `--lr_scheduler {cosine,step,custom,none}`
* `--amp`：混合精度
* `--audio_mode {mono,stereo}`：音频通道模式（stereo 会按双通道读取）
* `--use_lazy_loading`：懒加载
* `--files_per_batch N`：每个 batch 仅从 `N` 个随机音频文件中采样窗口

### 6.3 输出结果

保存在 `logs/<时间戳>/`：

* `best_checkpoint.pth`
* `history.json`
* `loss_curves.png`
* `config.txt`

---

## 7. 推理与可视化

### 7.1 单文件

```bash
python src/infer_and_visualize.py \
  --audio data/out000.mp3 \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

### 7.2 批量推理

```bash
python src/infer_and_visualize.py \
  --audio_dir data \
  --checkpoint logs/<run_id>/best_checkpoint.pth \
  --output_dir outputs
```

### 7.3 可选参数

* `--save_pred_clips`：导出预测片段
* `--pred_json`：自定义预测 JSON 名称
* `--pred_labelstudio_json`：自定义 Label Studio JSON
* `--audio_mode {auto,mono,stereo}`：推理音频模式
  * `auto`：跟随 checkpoint 中训练配置
  * `mono`/`stereo`：手动覆盖

### 7.4 输出内容

位于 `outputs/`：

* `*_timeline.png`（波形 + 标注 + 预测）
* `pred_spans.json`
* `labelstudio.json`
* `pred_clips/`（可选）

---

## 8. Label Studio 转 CSV

```bash
python src/convert_labels.py \
  --input data/<label_studio_file>.json \
  --output data/<output_file>.csv
```

生成训练用 CSV，包含：

* `filename`
* `start_time`
* `end_time`
* `event_label`

---

<p><a href="#top"><button>返回顶部</button></a></p>
