"""
PyTorch port of the Keras/TensorFlow OCR CRNN model (model.py).

Architecture:
  - Input normalization (divide by 255)
  - 7x Residual CNN blocks (16 → 16 → 16 → 32 → 32 → 64 → 64 filters)
  - Flatten spatial dims → sequence
  - Bidirectional LSTM (64 hidden units)
  - Linear projection → vocab_size + 1 (CTC blank)
  - Softmax output

Input tensor shape : (N, C, H, W)  — PyTorch channel-first convention.
The DataLoader should transpose images from (H, W, C) → (C, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    """Return an activation module by name string (mirrors mltu convention)."""
    name = name.lower()
    if name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# Residual Block  (mirrors mltu.tensorflow.model_utils.residual_block)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Conv2D residual block equivalent to mltu's residual_block().

    Args:
        in_channels  : Number of input feature maps.
        out_channels : Number of output feature maps (filters).
        activation   : Activation name string (default: "leaky_relu").
        skip_conv    : If True, applies a 1×1 conv on the skip path to
                       match channel / spatial dimensions (required when
                       in_channels != out_channels OR strides > 1).
        strides      : Conv stride (applied to the main path; the skip
                       path uses the same stride when skip_conv=True).
        dropout      : Dropout probability applied after the residual sum.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "leaky_relu",
        skip_conv: bool = True,
        strides: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        # --- Main path ---
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=strides, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = _get_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Skip / shortcut path ---
        if skip_conv:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=strides, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # Identity: only valid when in_channels == out_channels AND strides == 1
            self.skip = nn.Identity()

        self.act2 = _get_activation(activation)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.act2(out)
        out = self.dropout(out)
        return out


# ---------------------------------------------------------------------------
# Full CRNN model
# ---------------------------------------------------------------------------

class OCRModel(nn.Module):
    """
    CRNN OCR model equivalent to the Keras train_model() function.

    Args:
        input_dim   : (H, W, C) — height, width, channels  (Keras convention).
                      Internally the model expects (N, C, H, W) tensors.
        output_dim  : Vocabulary size (number of unique characters).
        activation  : Activation name string (default: "leaky_relu").
        dropout     : Dropout probability (default: 0.2).
    """

    def __init__(
        self,
        input_dim: tuple,           # (H, W, C)
        output_dim: int,
        activation: str = "leaky_relu",
        dropout: float = 0.2,
    ):
        super().__init__()

        H, W, C = input_dim         # unpack Keras-style (height, width, channels)
        self.output_dim = output_dim

        # --- CNN backbone (residual blocks) ---
        # Block 1  : C  → 16,  stride 1,  skip_conv=True
        self.block1 = ResidualBlock(C,  16, activation, skip_conv=True,  strides=1, dropout=dropout)
        # Block 2  : 16 → 16,  stride 2,  skip_conv=True   → H/2, W/2
        self.block2 = ResidualBlock(16, 16, activation, skip_conv=True,  strides=2, dropout=dropout)
        # Block 3  : 16 → 16,  stride 1,  skip_conv=False
        self.block3 = ResidualBlock(16, 16, activation, skip_conv=False, strides=1, dropout=dropout)
        # Block 4  : 16 → 32,  stride 2,  skip_conv=True   → H/4, W/4
        self.block4 = ResidualBlock(16, 32, activation, skip_conv=True,  strides=2, dropout=dropout)
        # Block 5  : 32 → 32,  stride 1,  skip_conv=False
        self.block5 = ResidualBlock(32, 32, activation, skip_conv=False, strides=1, dropout=dropout)
        # Block 6  : 32 → 64,  stride 1,  skip_conv=True
        self.block6 = ResidualBlock(32, 64, activation, skip_conv=True,  strides=1, dropout=dropout)
        # Block 7  : 64 → 64,  stride 1,  skip_conv=False
        self.block7 = ResidualBlock(64, 64, activation, skip_conv=False, strides=1, dropout=dropout)

        # After 2× stride-2 downsampling: feature map is (H//4, W//4, 64)
        feat_h = H // 4
        feat_w = W // 4

        # --- Sequence model ---
        # Reshape (N, 64, feat_h, feat_w) → (N, feat_h * feat_w, 64)  ← sequence of length feat_h*feat_w
        lstm_input_size = feat_h * feat_w   # flattened spatial positions treated as seq length
        # Actually we want: seq_len = feat_h * feat_w, feature = 64
        # So we permute: (N, C, H, W) → (N, H*W, C)  after CNN.

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Output projection ---
        # output_dim + 1 for CTC blank token
        self.fc = nn.Linear(64 * 2, output_dim + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, C, H, W)  — float tensor, pixel values in [0, 255].
        Returns:
            log_probs : (T, N, vocab_size+1)  for use with nn.CTCLoss.
        """
        # Normalize to [0, 1]  (mirrors the Lambda(x/255) Keras layer)
        x = x / 255.0

        # CNN feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # x: (N, 64, H', W')
        N, C, H, W = x.shape

        # Flatten spatial dims and treat as sequence:
        # (N, C, H, W) → (N, H*W, C)
        x = x.permute(0, 2, 3, 1)          # (N, H', W', C)
        x = x.reshape(N, H * W, C)         # (N, T, C)  where T = H'*W'

        # BiLSTM
        x, _ = self.bilstm(x)              # (N, T, 128)

        # Dense projection → (N, T, vocab+1)
        x = self.fc(x)

        # Softmax (matches Keras Dense softmax output)
        x = F.softmax(x, dim=-1)

        # Transpose for CTCLoss: (T, N, vocab+1)
        x = x.permute(1, 0, 2)

        return x


# ---------------------------------------------------------------------------
# Convenience constructor  (mirrors the Keras train_model() API)
# ---------------------------------------------------------------------------

def train_model(
    input_dim: tuple,
    output_dim: int,
    activation: str = "leaky_relu",
    dropout: float = 0.2,
) -> OCRModel:
    """
    Factory function matching the original Keras API:

        model = train_model(
            input_dim  = (configs.height, configs.width, 3),
            output_dim = len(configs.vocab),
        )

    Returns an OCRModel instance moved to the best available device.
    """
    model = OCRModel(
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
        dropout=dropout,
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    H, W, C = 32, 128, 3
    vocab_size = 62      # 0-9 + A-Z + a-z

    model = train_model(input_dim=(H, W, C), output_dim=vocab_size).to(device)

    dummy_input = torch.randint(0, 256, (4, C, H, W), dtype=torch.float32).to(device)
    out = model(dummy_input)

    print(f"Output shape : {out.shape}")   # Expected: (T, N, vocab+1)
    print(f"T  = {out.shape[0]}  (sequence length = feat_h * feat_w = {H//4} * {W//4})")
    print(f"N  = {out.shape[1]}  (batch size)")
    print(f"V  = {out.shape[2]}  (vocab_size + 1 = {vocab_size + 1})")
