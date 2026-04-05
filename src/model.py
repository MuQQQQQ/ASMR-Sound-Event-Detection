from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi


class ResBlock2D(nn.Module):
    """Basic 2D residual block used in spectrogram CNN encoder."""

    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1)):
        """Initialize residual block layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if in_ch != out_ch or stride != (1, 1):
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        """Forward pass for residual block."""
        identity = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.down is not None:
            identity = self.down(identity)
        y = self.act(y + identity)
        return y


class ConformerBlock(nn.Module):
    """Lightweight Conformer-inspired block for temporal modeling."""

    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        """Initialize attention, convolution, and feed-forward sublayers."""
        super().__init__()
        hidden = d_model * ff_mult
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        self.mha_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.mha_drop = nn.Dropout(dropout)

        self.conv_ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                            padding=conv_kernel // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass of Conformer block."""
        x = x + 0.5 * self.ffn1(x)

        y = self.mha_ln(x)
        y, _ = self.mha(y, y, y, need_weights=False)
        x = x + self.mha_drop(y)

        y = self.conv_ln(x).transpose(1, 2)
        y = self.pw1(y)
        y = self.glu(y)
        y = self.dw(y)
        y = self.bn(y)
        y = torch.nn.functional.silu(y)
        y = self.pw2(y)
        y = self.conv_drop(y).transpose(1, 2)
        x = x + y

        x = x + 0.5 * self.ffn2(x)
        return self.final_ln(x)


class UNetBlock(nn.Module):
    """Two-layer convolutional block used by U-Net encoder/decoder."""

    def __init__(self, in_ch, out_ch):
        """Initialize U-Net convolutional block."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass for U-Net block."""
        return self.conv(x)


# =========================
# 主模型
# =========================
class UNetConformerSED(nn.Module):
    """U-Net + Conformer SED model with optional wav2vec2 frontend."""

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 320,
        n_mels: int = 64,
        conformer_dim: int = 256,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        dropout: float = 0.1,
        use_specaug: bool = True,
        freq_mask_param: int = 10,
        time_mask_param: int = 30,
        feature_extractor: str = "melspec",  # "melspec" or "wav2vec2"
    ):
        """Initialize UNetConformerSED model components."""
        super().__init__()

        self.feature_extractor = feature_extractor
        self.use_specaug = use_specaug

        # =========================
        # Mel Spectrogram
        # =========================
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        if self.use_specaug:
            self.specaug = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param),
                torchaudio.transforms.TimeMasking(time_mask_param),
            )

        # =========================
        # U-Net（只用于 melspec）
        # =========================
        self.enc1 = UNetBlock(1, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)

        self.pool = nn.MaxPool2d((2, 1))

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.dec1 = UNetBlock(64, 64)

        proj_dim = 64  # U-Net 输出

        # =========================
        # Projection
        # =========================
        self.proj = nn.Sequential(
            nn.Linear(proj_dim, conformer_dim),
            nn.Dropout(dropout),
        )

        # =========================
        # Conformer
        # =========================
        self.conformer = nn.ModuleList([
            ConformerBlock(
                d_model=conformer_dim,
                nhead=conformer_heads,
                ff_mult=4,
                conv_kernel=15,
                dropout=dropout,
            )
            for _ in range(conformer_layers)
        ])

        # =========================
        # 分类头
        # =========================
        self.classifier = nn.Linear(conformer_dim, num_classes)

    # =========================
    # 特征提取
    # =========================
    def extract_features(self, waveform):
        """Extract frame-level features from waveform."""
        if self.feature_extractor == "melspec":
            x = self.melspec(waveform) + 1e-10
            x = self.db(x)

            if self.use_specaug and self.training:
                x = self.specaug(x)

            x = (x - x.mean(dim=(-2, -1), keepdim=True)) / (
                x.std(dim=(-2, -1), keepdim=True) + 1e-5
            )

            return x  # (B, M, T)

        elif self.feature_extractor == "wav2vec2":
            with torch.no_grad():
                features = self.extractor(waveform).last_hidden_state
            return features  # (B, T, 768)

        raise ValueError(
            f"Unsupported feature extractor: {self.feature_extractor}")

    # =========================
    # forward
    # =========================
    def forward(self, waveform):
        """Forward pass producing frame-wise event logits."""

        if self.feature_extractor == "melspec":
            x = self.extract_features(waveform)  # (B, M, T)

            x = x.unsqueeze(1)  # (B,1,M,T)

            # ===== Encoder =====
            e1 = self.enc1(x)               # (B,32,M,T)
            e2 = self.enc2(self.pool(e1))   # (B,64,M/2,T)
            e3 = self.enc3(self.pool(e2))   # (B,128,M/4,T)

            # ===== Decoder =====
            d2 = self.up2(e3)               # (B,64,M/2,T)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)              # (B,64,M/2,T)

            d1 = self.up1(d2)               # (B,32,M,T)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)              # (B,64,M,T)

            # ===== 频率聚合 =====
            x = d1.mean(dim=2)              # (B,64,T)
            x = x.transpose(1, 2)           # (B,T,64)

        else:
            x = self.extract_features(waveform)  # (B,T,768)

        # ===== Projection =====
        x = self.proj(x)

        # ===== Conformer =====
        for blk in self.conformer:
            x = blk(x)

        # ===== 分类 =====
        logits = self.classifier(x)  # (B,T,C)

        return logits


class ResNetConformerSED(nn.Module):
    """ResNet-style CNN + Conformer SED model."""

    def __init__(
        self,
        num_classes: int,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 320,
        win_length: int = 1024,
        n_mels: int = 64,
        conformer_dim: int = 256,
        conformer_layers: int = 4,
        conformer_heads: int = 4,
        dropout: float = 0.1,
        use_specaug: bool = True,
        freq_mask_param: int = 10,
        time_mask_param: int = 30,
        feature_extractor: str = "wav2vec2",
        device: torch.device = torch.device("cuda"),
        use_wenet=False,
        use_wenet_ckpt=True
    ):
        """Initialize ResNetConformerSED model components."""
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_hop_sec = hop_length / sample_rate
        self.use_specaug = use_specaug
        self.feature_extractor = feature_extractor
        self.use_wenet = use_wenet
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=2.0,
            norm='slaney',
            mel_scale='slaney'
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.cnn = nn.Sequential(
            ResBlock2D(32, 64, stride=(2, 1)),
            ResBlock2D(64, 128, stride=(2, 1)),
            ResBlock2D(128, 128, stride=(1, 1)),
        )

        if feature_extractor == "melspec":
            proj_dim = 128
        elif feature_extractor == "wav2vec2":
            proj_dim = 768
        elif feature_extractor == "panns":
            proj_dim = 256
        elif feature_extractor == "fbank":
            proj_dim = 128
        self.proj = nn.Sequential(
            nn.Linear(proj_dim, conformer_dim),
            nn.Dropout(dropout),
        )

        self.conformer = nn.ModuleList([
            ConformerBlock(
                d_model=conformer_dim,
                nhead=conformer_heads,
                ff_mult=4,
                conv_kernel=15,
                dropout=dropout,
            )
            for _ in range(conformer_layers)
        ])

        self.classifier = nn.Linear(conformer_dim, num_classes)

        if self.use_specaug:
            self.specaug = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param),
                torchaudio.transforms.TimeMasking(time_mask_param)
            )

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from raw waveform."""
        if self.feature_extractor == "melspec":
            x = self.melspec(waveform) + 1e-10  # [B, M, T]
            x = self.db(x)
            if self.use_specaug and self.training:
                x = self.specaug(x)
            x = (x - x.mean(dim=(-2, -1), keepdim=True)) / \
                (x.std(dim=(-2, -1), keepdim=True) + 1e-5)
            return x

        elif self.feature_extractor == "fbank":
            bs = waveform.size(0)
            features = []
            for i in range(bs):
                feat = kaldi.fbank(waveform[i].unsqueeze(0), num_mel_bins=128,
                                   frame_length=25, frame_shift=20, sample_frequency=self.sample_rate)
                features.append(feat)
            features = torch.stack(features).to(waveform.device)  # [B, T', D]

            x = (features - features.mean(dim=1, keepdim=True)) / \
                (features.std(dim=1, keepdim=True) + 1e-5)
            return x
        else:
            raise ValueError(
                f"Unsupported feature extractor: {self.feature_extractor}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass producing frame-wise event logits."""
        x = self.extract_features(waveform)  # [B, M, T]
        if self.feature_extractor == "melspec":

            x = x.unsqueeze(1)  # [B, 1, M, T]
            x = self.cnn_stem(x)
            x = self.cnn(x)  # [B, C, M', T]
            x = x.mean(dim=2)  # [B, C, T]
            x = x.transpose(1, 2)  # [B, T, C]
        elif self.feature_extractor == "wav2vec2":
            pass
        elif self.feature_extractor == "fbank":
            # x = x.unsqueeze(1)  # [B, 1, M, T]
            # x = self.cnn_stem(x)
            # x = self.cnn(x)  # [B, C, M', T]
            # x = x.mean(dim=2)  # [B, C, T]
            # x = x.transpose(1, 2)  # [B, T, C]
            pass
        if not self.use_wenet:
            x = self.proj(x)

            for blk in self.conformer:
                x = blk(x)
        else:
            x = self.conformer(x, torch.tensor([x.size(1)]).to(x.device))[0]
        logits = self.classifier(x)  # [B, T, num_classes]
        return logits


class EMA:
    """Exponential Moving Average (EMA) wrapper for model parameters."""

    def __init__(self, model, decay=0.999):
        """Initialize EMA state."""
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化影子参数：将当前模型参数克隆一份
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters after one optimization step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + \
                    self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA shadow weights to model (typically before validation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights after EMA evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
