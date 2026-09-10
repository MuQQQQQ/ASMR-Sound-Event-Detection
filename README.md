<a id="top"></a>

# ASMR Sound Event Detection

<div align="center">


[![Language](https://img.shields.io/badge/Language-Python-blue)](#)

</div>

<p align="center">
  <a href="#简体中文"><button>简体中文</button></a>
  &nbsp;&nbsp;
  <a href="#english"><button>English</button></a>
</p>

---

<a id="简体中文"></a>

# 简体中文

## 项目简介

本项目提供了一套完整的ASMR音频声音事件检测（Sound Event Detection, SED）流程，涵盖数据标注转换、模型训练、音频推理、预测结果导出、Label Studio 辅助标注以及最终的 ASMR 音频时间轴可视化。

模型以音频波形为输入，通过Mel Spectrogram + ResNet + Conformer提取时频特征，并对音频进行逐帧（frame-level）声音事件预测。项目主要面向 ASMR 音频中的人声、食音、口腔音、呼吸声以及不同类型的摩擦/刮擦等声音事件。

> 如果你只想处理自己的ASMR音频并生成最终的时间轴可视化视频，无需进行模型训练，可以直接使用 `all_in_one.py`。

---

## 1. 支持的声音事件

当前模型主要包含以下事件类别：

| 类别      | 标签名称           | 说明             |
| ------- | -------------- | -------------- |
| 人声      | `Speech`       | 说话、讲话等人声       |
| 咀嚼声     | `Chewing`      | 咀嚼、咬食等声音       |
| 口腔音     | `Mouth_Sounds` | 吸吮、舔舐、口腔动作等声音  |
| 呼吸声     | `Breathing`    | 呼气、吸气等呼吸声音     |
| 水瓶声     | `Water_Bottle` | 水瓶、塑料瓶等产生的声音   |
| 一般噪声/摩擦 | `Rub`          | 较弱的摩擦、摩挲等声音    |
| 较强噪声/摩擦 | `Scrub`        | 较明显的摩擦、刮擦等声音   |
| 强烈噪声/摩擦 | `Rasp`         | 强烈、粗糙或尖锐的刮擦类声音 |

其中 `Rub`、`Scrub` 和 `Rasp` 并不严格对应传统意义上的“底噪（background noise）”。它们主要用于归类 ASMR 中的**非目标声音以及不同强度的摩擦、刮擦、敲击等音效**。

由于这类声音边界较模糊，且人工标注时很难对每一种具体音效进行稳定、细粒度的分类，因此项目将其统一归入三个不同强度的类别。



---

## 2. 快速开始

### 2.1 安装环境

安装项目依赖：

```bash
pip install -r requirements.txt
```

建议使用 Python 3.10 或兼容的较新 Python 版本，并根据实际 CUDA 环境安装对应版本的 PyTorch。

### 2.2 直接处理自己的 ASMR 音频

如果你不需要训练自己的 SED 模型，而只是希望对已有 ASMR 音频进行处理并生成最终的时间轴可视化视频，可以直接运行：

```bash
python src/all_in_one.py --audio_dir <音频目录> --background <头像图片> --author <作者>
```

例如：

```bash
python src/all_in_one.py --audio_dir D:\ASMR\input --background avatar.jpg --author "Name"
```

该脚本负责将音频分析结果、事件时间轴以及背景信息整合为最终的视频可视化结果。

---

## 3. 项目结构

核心代码位于 `src/` 目录：

```text
src/
├── train.py
├── infer_and_visualize.py
├── data.py
├── model.py
├── utils.py
├── convert_labels.py
└── all_in_one.py
```

各文件主要功能如下：

| 文件                       | 功能                                                                       |
| ------------------------ | ------------------------------------------------------------------------ |
| `train.py`               | SED 模型训练入口，包括训练、验证、指标计算和 checkpoint 保存                                   |
| `infer_and_visualize.py` | 使用训练好的模型对完整音频进行推理，并生成预测事件结果；同时导出可直接用于 Label Studio 人工检查、修正和重新标注的 JSON 文件 |
| `data.py`                | 数据集读取、音频切片、标签构建以及训练/验证数据划分                                               |
| `model.py`               | ResNet + Conformer SED 模型，以及 EMA（Exponential Moving Average）相关实现         |
| `utils.py`               | 通用工具函数，包括训练、评估、日志及其他辅助功能                                                 |
| `convert_labels.py`      | 将 Label Studio 导出的标注 JSON 转换为模型训练所需的 CSV 格式                              |
| `all_in_one.py`          | 将 ASMR 音频处理结果整合为最终的时间轴可视化视频                                              |

---

# 4. 模型结构

模型采用轻量级 **ResNet + Conformer** 架构，实现帧级预测。

### 4.1 输入

* `waveform`：形状 **[B, S]**
  （批大小，音频采样点数）


### 4.2 特征提取

使用 `MelSpectrogram`：

* 关键参数：`n_fft`、`win_length`、`hop_length`、`n_mels`、`sample_rate`
* 输出：**[B, M, T]**（批大小 × 梅尔频带 × 时间帧）


### 4.3 ResNet 主干

输入：`[B, M, T]`

处理流程：

(1). 扩展通道 → `[B, 1, M, T]`
(2). 卷积 Stem（Conv + BN + ReLU）
(3). 三个残差块（逐步压缩频率维）

输出： **[B, 128, M/4, T]**

#### 4.3.1 频率维池化

* 对频率维取均值 → `[B, 128, T]`
* 转置 → `[B, T, 128]`

将时频特征转换为时间序列特征


### 4.4 投影层

* `Linear(128 → 256) + Dropout`

输出： **[B, T, 256]**


### 4.5 Conformer 编码器

包含 4 个 Conformer Block，每个 Block 包括：

(1). 前馈网络（半残差）
(2). 多头自注意力
(3). 卷积模块（GLU + 深度可分离卷积等）
(4). 第二个前馈网络
(5). LayerNorm

输出维度保持不变：**[B, T, 256]**


### 4.6 帧级分类器

* `Linear(256 → 类别数)`

输出： **[B, T, C]**


---

# 5. 数据格式

## 5.1 音频文件

训练音频可以放置在指定的数据目录中，例如：

```text
data/
├── audio_001.mp3
├── audio_002.mp3
├── audio_003.wav
└── ...
```

训练时通过 `--data_dir` 指定音频目录。

---

## 5.2 CSV 标注格式

训练标注使用 CSV 文件，每条记录表示一个声音事件区间。

CSV 必须包含以下字段：

```text
filename
start_time
end_time
event_label
```

例如：

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
out001.mp3,10.50,12.20,Breathing
```

其中：

* `filename`：音频文件名
* `start_time`：事件开始时间，单位为秒
* `end_time`：事件结束时间，单位为秒
* `event_label`：事件类别

同一时间段可以存在多个标签，以支持多标签声音事件检测。

---

# 6. 模型训练

一个典型的训练命令如下：

```bash
python src/train.py --data_dir data --annotations data/anno.csv --amp --virtual_classes 50 --loss_type focal --lr_scheduler cosine --lr 1e-4 --batch_size 30 --dataset_type group --sample_rate 16000 --n_fft 1024 --hop_length 320 --win_length 1024 --n_mels 64 --use_ema --epochs 100
```

这里使用的主要配置包括：

```text
Sample rate : 16 kHz
n_fft       : 1024
hop length  : 320
window      : 1024
Mel bands   : 64
Loss        : Focal Loss
Scheduler   : Cosine
EMA         : Enabled
AMP         : Enabled
Epochs      : 100
```

### Virtual Classes

`--virtual_classes` 用于增加额外的**虚拟类别（virtual classes）**。

这些类别并不是实际需要识别的 ASMR 声音事件，而是训练过程中的辅助类别，可用于增加训练任务的约束，并在当前数据设置下帮助模型优化和收敛。具体数量应根据数据规模和实验结果进行调整。

---

## 6.1 使用多个标注文件

可以同时指定多个 CSV 标注文件：

```bash
--annotations data/a.csv,data/b.csv,data/c.csv
```

例如：

```bash
python src/train.py \
  --data_dir data \
  --annotations data/a.csv,data/b.csv,data/c.csv
```

所有 CSV 中引用的音频文件需要位于同一个 `data_dir` 下。

---

## 6.2 常用训练参数

### EMA

```bash
--use_ema
```

启用 Exponential Moving Average，对模型参数进行指数移动平均，通常用于获得更加稳定的验证和推理模型。

### Loss

```bash
--loss_type {bce,focal}
```

支持：

* `bce`：Binary Cross Entropy
* `focal`：Focal Loss

对于类别分布不均衡的 SED 数据，Focal Loss 可以降低大量容易分类样本对训练的影响。

### Learning Rate Scheduler

```bash
--lr_scheduler {cosine,step,custom,none}
```

支持多种学习率策略：

```text
cosine
step
custom
none
```

### AMP

```bash
--amp
```

启用 Automatic Mixed Precision，以降低显存占用并提高 GPU 训练效率。

### Audio Mode

```bash
--audio_mode {mono,stereo}
```

控制音频读取模式：

```text
mono   → 单声道处理
stereo → 双声道处理
```

### Lazy Loading

```bash
--use_lazy_loading
```

启用懒加载，减少数据集初始化时的内存占用，适合音频文件较多或单个音频较大的情况。

### Files per Batch

```bash
--files_per_batch N
```

限制每个 batch 从多少个随机音频文件中采样训练窗口。

例如：

```bash
--files_per_batch 10
```

每个 batch 仅从随机选择的 10 个音频文件中采样窗口，可用于控制数据读取方式以及减少大规模音频数据带来的 I/O 和内存压力。

---

# 7. 训练输出

训练结果默认保存到：

```text
logs/<timestamp>/
```

典型结构如下：

```text
logs/
└── 20260910_123456/
    ├── best_checkpoint.pth
    ├── history.json
    ├── loss_curves.png
    └── config.txt
```

主要文件：

| 文件                    | 说明                    |
| --------------------- | --------------------- |
| `best_checkpoint.pth` | 验证集表现最好的模型 checkpoint |
| `history.json`        | 保存训练过程中的 loss、指标等历史记录 |
| `loss_curves.png`     | 训练过程曲线                |
| `config.txt`          | 当前训练所使用的配置参数          |

---

# 8. 模型推理与 Label Studio

训练完成后，可以使用 `infer_and_visualize.py` 对新的 ASMR 音频进行推理：

```bash
python src/infer_and_visualize.py \
  --audio_dir data \
  --checkpoint last_checkpoint.pth \
  --output_dir outputs
```

该脚本不仅用于查看模型预测结果，还可以作为**模型预测 → 人工检查 → 修正标注 → 重新训练**的数据闭环的一部分。


---

## 8.1 常用参数

### Prediction JSON

```bash
--pred_json <filename>
```

自定义模型预测结果 JSON 文件名。

### Label Studio JSON

```bash
--pred_labelstudio_json <filename>
```

自定义 Label Studio 格式 JSON 文件名。

### Audio Mode

```bash
--audio_mode {auto,mono,stereo}
```

控制推理时的音频通道模式。

其中：

```text
auto
```

表示自动读取 checkpoint 中保存的训练配置。

```text
mono
```

强制使用单声道模式。

```text
stereo
```

强制使用双声道模式。

---

# 9. 推理输出

输出结果位于指定的 `output_dir`：

```text
outputs/
├── *_timeline.png
├── pred_spans.json
└── labelstudio.json
```

主要文件用途如下：

| 文件                 | 用途                         |
| ------------------ | -------------------------- |
| `*_timeline.png`   | 音频波形、人工标注及模型预测结果的时间轴可视化    |
| `pred_spans.json`  | 模型预测得到的声音事件时间区间            |
| `labelstudio.json` | 可导入 Label Studio 的模型预测标注结果 |

其中 `labelstudio.json` 的主要用途是**辅助人工重新检查和修正模型预测结果**，而不是直接作为最终人工标注结果使用。

---

# 10. Label Studio 标注转换

如果使用 Label Studio 对模型预测结果进行检查和修正，可以将 Label Studio 导出的 JSON 转换为训练所需的 CSV：

```bash
python src/convert_labels.py \
  --input data/<label_studio_file>.json \
  --output data/<output_file>.csv
```

生成的 CSV 包含：

```text
filename
start_time
end_time
event_label
```

例如：

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
```

之后即可将该 CSV 用于下一轮模型训练。

---

# 11. ASMR 时间轴可视化

如果你的目标不是训练 SED 模型，而是直接制作 ASMR 音频的可视化视频，可以使用：

```bash
python src/all_in_one.py \
  --audio_dir <音频目录> \
  --background <头像图片> \
  --author <作者>
```

该脚本用于将 ASMR 音频处理结果整合为最终的**声音事件时间轴可视化视频**。

可视化内容包括音频时间轴、声音事件类别、事件持续时间以及相应的视觉元素，可用于直观展示一段 ASMR 音频中不同声音事件随时间的分布。

---

# 12. 注意事项

训练和推理时必须保证音频采样率、Mel Spectrogram 参数以及 checkpoint 中保存的模型配置一致。特别是 `sample_rate`、`n_fft`、`hop_length`、`win_length` 和 `n_mels` 会直接影响输入特征的时间和频率分辨率。

对于 SED 任务，模型输出的是时间帧级预测，同一个时间段可能同时包含多个声音事件。

此外，ASMR 中不同声音事件之间存在较强的主观性和重叠。例如口腔音、呼吸声和咀嚼声在部分片段中可能同时出现；摩擦、刮擦、敲击等非目标音效也很难严格划分。因此，训练数据的一致性通常会直接影响模型最终的检测效果。

---

<p align="center">
  <a href="#top"><button>返回顶部</button></a>
</p>

---

<a id="english"></a>

# English

## Project Overview

This project provides a complete pipeline for **Sound Event Detection (SED) on ASMR audio**, covering annotation conversion, model training, audio inference, prediction export, Label Studio-assisted annotation, and final ASMR timeline visualization.

The model takes raw audio waveforms as input and uses a **Mel Spectrogram + ResNet + Conformer** architecture to extract time-frequency features and perform frame-level sound event prediction.

The project is primarily designed for ASMR audio containing speech, chewing, mouth sounds, breathing, water bottle sounds, and different types of rubbing, scraping, and other non-target sound effects.

> If you **only want to process your own ASMR audio and generate the final timeline visualization video**, you do not need to train the SED model. You can directly use `all_in_one.py`.

---

## 1. Supported Sound Events

The current model uses the following sound event categories:

| Category       | Description                                          |
| -------------- | ---------------------------------------------------- |
| `Speech`       | Speech and other human vocal sounds                  |
| `Chewing`      | Chewing, biting, and eating-related sounds           |
| `Mouth_Sounds` | Sucking, licking, and other mouth-related sounds     |
| `Breathing`    | Inhalation, exhalation, and breathing sounds         |
| `Water_Bottle` | Sounds produced by water bottles and plastic bottles |
| `Rub`          | Relatively weak rubbing and friction sounds          |
| `Scrub`        | More noticeable rubbing and scraping sounds          |
| `Rasp`         | Strong, rough, or sharp scraping sounds              |

`Rub`, `Scrub`, and `Rasp` do not strictly represent conventional **background noise**. They are mainly used to group **non-target sounds and various rubbing, scraping, tapping, and similar sound effects** commonly found in ASMR recordings.

Because these sounds are difficult to define consistently at a fine-grained level, they are grouped into three categories according to their approximate intensity and acoustic characteristics.


---

## 2. Quick Start

### 2.1 Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Python 3.10 or a compatible recent Python version is recommended. Install a PyTorch version compatible with your CUDA environment when GPU acceleration is required.

### 2.2 Process Your Own ASMR Audio

If you do not need to train your own SED model and only want to process ASMR audio and generate the final timeline visualization video, run:

```bash
python src/all_in_one.py --audio_dir <audio_directory> --background <avatar_image> --author <author>
```

For example:

```bash
python src/all_in_one.py --audio_dir D:\ASMR\input --background avatar.jpg --author "Name"
```

This script integrates the audio analysis results, event timeline, and visual elements into the final visualization video.

---

## 3. Project Structure

The core source code is located in `src/`:

```text
src/
├── train.py
├── infer_and_visualize.py
├── data.py
├── model.py
├── utils.py
├── convert_labels.py
└── all_in_one.py
```

| File                     | Description                                                                                                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train.py`               | Training entry point, including training, validation, metric calculation, and checkpoint saving                                                                   |
| `infer_and_visualize.py` | Runs inference on complete audio files and exports model predictions, including Label Studio-compatible JSON for manual inspection, correction, and re-annotation |
| `data.py`                | Dataset loading, audio segmentation, label construction, and train/validation data handling                                                                       |
| `model.py`               | ResNet + Conformer SED model and EMA implementation                                                                                                               |
| `utils.py`               | General utility functions used throughout the project                                                                                                             |
| `convert_labels.py`      | Converts Label Studio JSON annotations into CSV files used for model training                                                                                     |
| `all_in_one.py`          | Generates the final ASMR timeline visualization video                                                                                                             |

---

# 4. Model Architecture

The model follows a lightweight **ResNet + Conformer** design for frame-wise prediction.

### 4.1 Input

* `waveform`: **[B, S]**
  *(batch size, number of audio samples)*


### 4.2Feature Extraction

Using `torchaudio.transforms.MelSpectrogram`:

* Key parameters: `n_fft`, `win_length`, `hop_length`, `n_mels`, `sample_rate`
* Output: **[B, M, T]**
  *(batch size, mel bins, time frames)*


### 4.3 ResNet Backbone

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

#### 4.3.1 Frequency Pooling

* Mean over frequency → `[B, 128, T]`
* Transpose → `[B, T, 128]`

This converts 2D time-frequency features into a temporal sequence.


### 4.4 Projection Layer

* `Linear(128 → conformer_dim) + Dropout`
* `conformer_dim = 256`

Output: **[B, T, 256]**


### 4.5 Conformer Encoder

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


### 4.6 Frame-wise Classifier

* `Linear(256 → num_classes)`

Output:

* Logits: **[B, T, C]**


---

# 5. Data Format

## 5.1 Audio Files

Training audio files should be placed in the specified data directory:

```text
data/
├── audio_001.mp3
├── audio_002.mp3
├── audio_003.wav
└── ...
```

The directory is specified using `--data_dir`.

---

## 5.2 CSV Annotations

Training annotations are stored in CSV format, where each row represents one sound event segment.

The CSV file must contain:

```text
filename
start_time
end_time
event_label
```

Example:

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
out001.mp3,10.50,12.20,Breathing
```

Fields:

* `filename`: audio file name
* `start_time`: event start time in seconds
* `end_time`: event end time in seconds
* `event_label`: sound event category

Multiple labels may overlap in time, allowing multi-label sound event detection.

---

# 6. Model Training

A typical training command is:

```bash
python src/train.py --data_dir data --annotations data/anno.csv --amp --virtual_classes 50 --loss_type focal --lr_scheduler cosine --lr 1e-4 --batch_size 30 --dataset_type group --sample_rate 16000 --n_fft 1024 --hop_length 320 --win_length 1024 --n_mels 64 --use_ema --epochs 100
```

Example configuration:

```text
Sample rate : 16 kHz
n_fft       : 1024
hop length  : 320
window      : 1024
Mel bands   : 64
Loss        : Focal Loss
Scheduler   : Cosine
EMA         : Enabled
AMP         : Enabled
Epochs      : 100
```

### Virtual Classes

`--virtual_classes` adds additional **virtual classes** during training.

These classes are not actual ASMR sound events that need to be recognized. Instead, they are auxiliary training classes that can provide additional constraints during optimization and, under the current dataset setup, help the model converge.

The appropriate number should be determined empirically according to the dataset size and training results.

---

## 6.1 Multiple Annotation Files

Multiple CSV annotation files can be provided:

```bash
--annotations data/a.csv,data/b.csv,data/c.csv
```

For example:

```bash
python src/train.py \
  --data_dir data \
  --annotations data/a.csv,data/b.csv,data/c.csv
```

All referenced audio files must be located under the same `data_dir`.

---

## 6.2 Common Training Parameters

### EMA

```bash
--use_ema
```

Enables Exponential Moving Average of model parameters, which can provide a more stable model for evaluation and inference.

### Loss Function

```bash
--loss_type {bce,focal}
```

Supported losses:

* `bce`: Binary Cross Entropy
* `focal`: Focal Loss

Focal Loss can be useful for imbalanced SED datasets because it reduces the contribution of a large number of easy negative examples.

### Learning Rate Scheduler

```bash
--lr_scheduler {cosine,step,custom,none}
```

Supported options:

```text
cosine
step
custom
none
```

### AMP

```bash
--amp
```

Enables Automatic Mixed Precision to reduce GPU memory usage and improve training efficiency.

### Audio Mode

```bash
--audio_mode {mono,stereo}
```

Controls how audio channels are loaded:

```text
mono   → mono processing
stereo → stereo processing
```

### Lazy Loading

```bash
--use_lazy_loading
```

Enables lazy loading to reduce memory consumption during dataset initialization. This is useful when working with a large number of long audio files.

### Files per Batch

```bash
--files_per_batch N
```

Limits each batch to sampling training windows from `N` randomly selected audio files.

For example:

```bash
--files_per_batch 10
```

restricts each batch to 10 randomly selected audio files, which can help control I/O and memory usage for large audio datasets.

---

# 7. Training Outputs

Training results are saved under:

```text
logs/<timestamp>/
```

A typical output directory is:

```text
logs/
└── 20260910_123456/
    ├── best_checkpoint.pth
    ├── history.json
    ├── loss_curves.png
    └── config.txt
```

| File                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `best_checkpoint.pth` | Best-performing model checkpoint on the validation set        |
| `history.json`        | Training and validation history, including losses and metrics |
| `loss_curves.png`     | Training curves                                               |
| `config.txt`          | Configuration used for the experiment                         |

---

# 8. Inference and Label Studio

After training, use `infer_and_visualize.py` to run inference on new ASMR audio:

```bash
python src/infer_and_visualize.py \
  --audio_dir data \
  --checkpoint last_checkpoint.pth \
  --output_dir outputs
```

The script is intended not only for inspecting model predictions, but also as part of an iterative **prediction → human inspection → correction → retraining** workflow.

The model predictions are converted into temporal event spans and exported in a Label Studio-compatible JSON format. These predictions can then be loaded into Label Studio for manual inspection, correction, deletion, or additional annotation, making it easier to generate high-quality annotations for subsequent training iterations.

---

## 8.1 Common Options

### Prediction JSON

```bash
--pred_json <filename>
```

Specifies a custom filename for the model prediction JSON.

### Label Studio JSON

```bash
--pred_labelstudio_json <filename>
```

Specifies a custom filename for the Label Studio-compatible prediction file.

### Audio Mode

```bash
--audio_mode {auto,mono,stereo}
```

Controls the audio channel mode during inference.

```text
auto
```

Uses the audio configuration stored in the checkpoint.

```text
mono
```

Forces mono processing.

```text
stereo
```

Forces stereo processing.

---

# 9. Inference Outputs

The output directory contains:

```text
outputs/
├── *_timeline.png
├── pred_spans.json
└── labelstudio.json
```

| File               | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `*_timeline.png`   | Timeline visualization of the waveform, annotations, and model predictions |
| `pred_spans.json`  | Predicted sound event time spans                                           |
| `labelstudio.json` | Model predictions formatted for import into Label Studio                   |

The main purpose of `labelstudio.json` is to **assist human inspection and correction**. It should not be considered a replacement for manually verified annotations.

---

# 10. Converting Label Studio Annotations to CSV

After inspecting and correcting model predictions in Label Studio, export the annotations and convert them into the CSV format used for training:

```bash
python src/convert_labels.py \
  --input data/<label_studio_file>.json \
  --output data/<output_file>.csv
```

The generated CSV contains:

```text
filename
start_time
end_time
event_label
```

Example:

```csv
filename,start_time,end_time,event_label
out000.mp3,1.20,2.80,Chewing
out000.mp3,4.00,4.60,Speech
```

The resulting CSV can then be used for another round of model training.


---

# 11. ASMR Timeline Visualization

If your goal is not to train an SED model but simply to generate a visual timeline for ASMR audio, use:

```bash
python src/all_in_one.py \
  --audio_dir <audio_directory> \
  --background <avatar_image> \
  --author <author>
```

This script integrates the ASMR audio analysis results into a final **sound event timeline visualization video**.

The visualization can show the temporal distribution of different sound events together with the audio timeline and other visual elements, providing an intuitive overview of the sound composition of an ASMR recording.

---

# 12. Notes

For both training and inference, the audio sampling rate and Mel Spectrogram parameters should be consistent with the configuration stored in the checkpoint. In particular, `sample_rate`, `n_fft`, `hop_length`, `win_length`, and `n_mels` directly affect the time and frequency resolution of the model input.

This is a **frame-level multi-label SED system**, rather than a conventional single-label audio classification model. Multiple sound events may therefore be active within the same time interval.

ASMR sound events can also overlap substantially. For example, mouth sounds, breathing, and chewing may occur simultaneously, while rubbing, scraping, tapping, and similar non-target sounds can be difficult to distinguish consistently. Therefore, the consistency and quality of the training annotations have a significant impact on the final detection performance.

---

<p align="center">
  <a href="#top"><button>Back to Top</button></a>
</p>
