"""
PyTorch port of tensorTest.ipynb
IAM handwriting word recognition — CRNN (CNN + BiLSTM) + CTC loss.

Runs on GPU automatically when CUDA is available.

Usage:
    python train_torch.py
"""

import os
import math
import time
import random
from itertools import groupby

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Hyperparameters / constants  (mirrors the notebook's parameter cell)
# ---------------------------------------------------------------------------

ALPHABETS       = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
MAX_STR_LEN     = 19          # max label length (padded)
NUM_CHARACTERS  = len(ALPHABETS) + 1   # +1 for CTC blank
NUM_TIMESTAMPS  = 64          # max predicted sequence length (unused at inference)
DEFAULT_PATH    = "../Datasets/iam_words/"
BATCH_SIZE      = 512
IMG_H, IMG_W    = 32, 128
EPOCHS          = 150
LR              = 0.001
PATIENCE        = 15          # early stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Label helpers  (mirrors label_to_num / num_to_label / ctc_decoder)
# ---------------------------------------------------------------------------

def label_to_num(txt: str) -> list:
    """Encode a text label to a list of integer indices, padded to MAX_STR_LEN."""
    encoded = []
    for ch in txt:
        try:
            encoded.append(ALPHABETS.index(ch))
        except ValueError:
            pass   # skip unknown characters (mirrors the original try/except print)
    # Pad / truncate to MAX_STR_LEN with the blank index
    encoded = encoded[:MAX_STR_LEN]
    encoded += [len(ALPHABETS)] * (MAX_STR_LEN - len(encoded))
    return encoded


def num_to_label(num) -> str:
    """Decode integer indices back to text, stopping at -1 (CTC blank)."""
    ret = ""
    for ch in num:
        if ch == -1:
            break
        else:
            ret += ALPHABETS[ch]
    return ret


def ctc_decoder(predictions: np.ndarray) -> list:
    """
    Greedy CTC decode.

    Args:
        predictions: numpy array of shape (N, T, V)  — softmax probabilities.
    Returns:
        List of decoded text strings, one per sample.
    """
    text_list = []
    pred_indices = np.argmax(predictions, axis=2)   # (N, T)

    for i in range(pred_indices.shape[0]):
        ans = ""
        merged = [k for k, _ in groupby(pred_indices[i])]
        for p in merged:
            if p != len(ALPHABETS):
                ans += ALPHABETS[int(p)]
        text_list.append(ans)

    return text_list


# ---------------------------------------------------------------------------
# Data loading  (mirrors the words.txt parsing cells)
# ---------------------------------------------------------------------------

def load_iam_dataset(data_root: str):
    """
    Parse IAM words.txt and return a list of (image_path, label) tuples.
    Only includes samples where the image file actually exists on disk.
    """
    words_txt = os.path.join(data_root, "words.txt")
    samples = []

    with open(words_txt, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            word_id = parts[0]
            status  = parts[1]
            if status != "ok":
                continue
            label   = " ".join(parts[8:])
            folder1 = word_id.split("-")[0]
            folder2 = "-".join(word_id.split("-")[:2])
            img_path = os.path.join(
                data_root, "words", folder1, folder2, f"{word_id}.png"
            )
            if os.path.exists(img_path):
                samples.append((img_path, label))

    print(f"[INFO] Loaded {len(samples)} samples from {data_root}")
    return samples


def build_dataframes(data_root: str):
    """
    Mirrors the notebook's DataFrame construction + train/val split.
    Returns (train_df, valid_df).
    """
    samples = load_iam_dataset(data_root)
    data = pd.DataFrame(samples, columns=["Fpath", "Identify"]).astype(str)
    data.dropna(inplace=True)

    train = data.sample(frac=0.9, random_state=42)
    valid = data.drop(train.index)

    # Mirror the notebook slice (80 000 / 8 000)
    train = train.iloc[:80000].reset_index(drop=True)
    valid = valid.iloc[:8000].reset_index(drop=True)

    print(f"[INFO] Train: {len(train)}  |  Val: {len(valid)}")
    return train, valid


# ---------------------------------------------------------------------------
# PyTorch Dataset  (replaces tf.data.Dataset + process_single_sample)
# ---------------------------------------------------------------------------

class IAMDataset(Dataset):
    """
    Reads PNG word images, resizes to (32, 128), converts to grayscale float,
    and encodes labels as integer sequences.

    Returns:
        image : FloatTensor  (1, H, W)  in [0, 1]
        label : LongTensor   (MAX_STR_LEN,)
    """

    def __init__(self, df: pd.DataFrame):
        self.paths  = df["Fpath"].tolist()
        self.labels = [label_to_num(lbl) for lbl in df["Identify"].tolist()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label    = self.labels[idx]

        # Read as grayscale (mirrors decode_png channels=1)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        # Resize to (H=32, W=128)
        img = cv2.resize(img, (IMG_W, IMG_H))

        # Convert to float in [0, 1]  (mirrors convert_image_dtype float32)
        img = img.astype(np.float32) / 255.0

        # Add channel dim: (H, W) → (1, H, W)
        img = torch.from_numpy(img).unsqueeze(0)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img, label_tensor


# ---------------------------------------------------------------------------
# Model  (exact port of the Keras model in Cell 11)
#
# Keras input:  (N, H=32, W=128, C=1)
# PyTorch input: (N, C=1, H=32, W=128)
#
# CNN path:
#   conv1  32  3×3 selu  →  MaxPool 2×2   → (N,32,16,64)
#   conv2  64  3×3 selu  →  MaxPool 2×2   → (N,64, 8,32)
#   conv3  128 3×3 selu
#   conv4  128 3×3 selu                    → (N,128,8,32)
#   conv5  512 3×3 selu
#   conv6  512 3×3 selu  → Dropout 0.2    → (N,512,8,32)
#   conv7  512 3×3 selu
#   conv8  512 3×3 selu  → MaxPool 2×1    → (N,512,4,32)
#   conv9  256 3×3 selu  → BN → Drop 0.2
#   conv10 256 3×3 selu  → BN → MaxPool 2×1 → Drop 0.2 → (N,256,2,32)
#   conv11  64 2×2 selu  → Drop 0.2       → (N,64,1,31)
#
# Squeeze height dim (K.squeeze axis=1) → (N,31,64) — the sequence
# BiLSTM stack: 128 → 512 → 512 → 512 → 128
# Dense 128 (relu)
# Dense NUM_CHARACTERS (softmax)
# ---------------------------------------------------------------------------

class CRNN(nn.Module):

    def __init__(self, num_characters: int = NUM_CHARACTERS):
        super().__init__()

        # --- CNN backbone ---
        self.cnn = nn.Sequential(
            # conv1 + pool
            nn.Conv2d(1,   32,  3, padding=1), nn.SELU(),
            nn.MaxPool2d((2, 2)),

            # conv2 + pool
            nn.Conv2d(32,  64,  3, padding=1), nn.SELU(),
            nn.MaxPool2d((2, 2)),

            # conv3, conv4
            nn.Conv2d(64,  128, 3, padding=1), nn.SELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.SELU(),

            # conv5, conv6 + dropout
            nn.Conv2d(128, 512, 3, padding=1), nn.SELU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.SELU(),
            nn.Dropout2d(0.2),

            # conv7, conv8 + pool (2×1 — height halved, width kept)
            nn.Conv2d(512, 512, 3, padding=1), nn.SELU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.SELU(),
            nn.MaxPool2d((2, 1)),

            # conv9 + BN + dropout
            nn.Conv2d(512, 256, 3, padding=1), nn.SELU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.2),

            # conv10 + BN + pool (2×1) + dropout
            nn.Conv2d(256, 256, 3, padding=1), nn.SELU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),

            # conv11 (2×2, NO padding → height collapses to 1) + dropout
            nn.Conv2d(256, 64, (2, 2)), nn.SELU(),
            nn.Dropout2d(0.2),
        )

        # --- Sequence model (BiLSTM stack) ---
        # After CNN the spatial map is (N, 64, 1, W')
        # We squeeze the height dim → (N, W', 64) as the time-series input.
        self.lstm1 = nn.LSTM(64,  128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 512, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(1024,512, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(1024,512, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(1024,128, batch_first=True, bidirectional=True)

        # --- Output head ---
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, num_characters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, 1, H, W)  float in [0, 1]
        Returns:
            out : (T, N, num_characters)  log-softmax — ready for nn.CTCLoss
        """
        # CNN feature extraction
        feat = self.cnn(x)                  # (N, 64, 1, W')

        # Squeeze height → (N, 64, W')  then permute → (N, W', 64)
        feat = feat.squeeze(2)              # (N, 64, W')
        feat = feat.permute(0, 2, 1)        # (N, T, 64)

        # BiLSTM stack
        out, _ = self.lstm1(feat)           # (N, T, 256)
        out, _ = self.lstm2(out)            # (N, T, 1024)
        out, _ = self.lstm3(out)            # (N, T, 1024)
        out, _ = self.lstm4(out)            # (N, T, 1024)
        out, _ = self.lstm5(out)            # (N, T, 256)

        # Dense projection
        out = F.relu(self.dense1(out))      # (N, T, 128)
        out = self.dense2(out)              # (N, T, num_characters)

        # Log-softmax for nn.CTCLoss
        out = F.log_softmax(out, dim=-1)    # (N, T, num_characters)

        # CTCLoss expects (T, N, C)
        out = out.permute(1, 0, 2)          # (T, N, num_characters)
        return out


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_accuracy(log_probs: torch.Tensor, targets: torch.Tensor,
                     blank: int) -> float:
    """
    Greedy-decode log_probs and compute exact-match word accuracy vs targets.

    Args:
        log_probs : (T, N, V)
        targets   : (N, MAX_STR_LEN)  padded with blank index
        blank     : CTC blank token index
    Returns:
        Word-level accuracy (float, 0–1).
    """
    probs = log_probs.permute(1, 0, 2).exp().detach().cpu().numpy()  # (N, T, V)
    decoded = ctc_decoder(probs)        # list of N strings

    correct = 0
    for pred, tgt_row in zip(decoded, targets.cpu().tolist()):
        gt = num_to_label([t for t in tgt_row if t != blank])
        if pred == gt:
            correct += 1
    return correct / len(decoded)


def train_epoch(model, loader, optimizer, criterion, blank):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for images, labels in loader:
        images = images.to(DEVICE)      # (N, 1, 32, 128)
        labels = labels.to(DEVICE)      # (N, MAX_STR_LEN)

        log_probs = model(images)       # (T, N, num_characters)
        T, N, _ = log_probs.shape

        # Build CTC targets: concatenate non-blank tokens per sample
        target_lengths = (labels != blank).sum(dim=1).cpu()
        flat_targets   = torch.cat([
            labels[i, :tl] for i, tl in enumerate(target_lengths)
        ]).cpu()
        input_lengths = torch.full((N,), T, dtype=torch.long)

        loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # clipnorm=1.0
        optimizer.step()

        total_loss += loss.item()
        total_acc  += compute_accuracy(log_probs.detach(), labels, blank)
        n += 1

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, blank):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        log_probs = model(images)
        T, N, _ = log_probs.shape

        target_lengths = (labels != blank).sum(dim=1).cpu()
        flat_targets   = torch.cat([
            labels[i, :tl] for i, tl in enumerate(target_lengths)
        ]).cpu()
        input_lengths = torch.full((N,), T, dtype=torch.long)

        loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
        total_loss += loss.item()
        total_acc  += compute_accuracy(log_probs, labels, blank)
        n += 1

    return total_loss / n, total_acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Data ----
    train_df, valid_df = build_dataframes(DEFAULT_PATH)

    train_ds = IAMDataset(train_df)
    valid_ds = IAMDataset(valid_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=True,
    )

    # ---- Model ----
    model = CRNN(num_characters=NUM_CHARACTERS).to(DEVICE)
    print(model)

    # ---- Loss ----
    blank_idx = len(ALPHABETS)
    criterion = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    # ---- Optimiser  (mirrors Adam lr=0.001, beta1=0.9, beta2=0.999, clipnorm=1) ----
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))

    # ---- LR scheduler ----
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ---- Checkpointing / early stopping ----
    best_val_loss   = math.inf
    patience_cnt    = 0
    best_ckpt_path  = "best_model.pt"

    print(f"\n{'='*60}")
    print(f"  Starting training for {EPOCHS} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, blank_idx
        )
        val_loss, val_acc = eval_epoch(
            model, valid_loader, criterion, blank_idx
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"loss={train_loss:.4f}  acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} | "
            f"lr={lr_now:.2e}  [{elapsed:.0f}s]"
        )

        # Best checkpoint  (mirrors ModelCheckpoint monitor='val_loss')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "val_loss":            val_loss,
                "val_acc":             val_acc,
            }, best_ckpt_path)
            print(f"  ✔ Saved best model  (val_loss={val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    print(f"\n[INFO] Training complete. Best val_loss = {best_val_loss:.4f}")
    print(f"[INFO] Best checkpoint saved to: {best_ckpt_path}")

    # Save full model state (mirrors model.save('my_model.h5'))
    torch.save(model.state_dict(), "my_model.pt")
    print("[INFO] Final model saved to: my_model.pt")


if __name__ == "__main__":
    main()
