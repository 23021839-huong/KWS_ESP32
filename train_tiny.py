"""
train_tiny.py

Pipeline đúng (sau khi sửa):
  waveform (B,1,16000)
    → MelSpectrogram → squeeze → log → crop/pad → normalize(mean/std)
    → log_mel (B, MAX_LEN, N_MELS)
    → model.forward(mel):
        CNN 1 lần → feat (B, FLATTEN_SIZE)
        normalize [0,1] → direct encode → RLIF loop (recurrent) → mean membrane
    → logits (B, NUM_CLASSES) → CrossEntropyLoss
"""

import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from config_tiny import Config
from dataset_tiny import KWSDataset
from model_tiny   import KWS_SNN_Tiny


# ── Mel transform — phải khớp compute_log_mel() firmware ────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = Config.SAMPLE_RATE,
    n_fft       = Config.N_FFT,
    hop_length  = Config.HOP_LENGTH,
    n_mels      = Config.N_MELS,
    f_min       = Config.MEL_FMIN,
    f_max       = Config.MEL_FMAX,
).to(Config.DEVICE)


def extract_features(waveform_batch):
    """
    waveform_batch : (B, 1, 16000)
    returns        : (B, MAX_LEN, N_MELS)  — log-mel, normalize mean/std per sample
    """
    with torch.no_grad():
        # squeeze(1): (B,1,16000) → (B,16000) trước khi mel
        mel     = mel_transform(waveform_batch.squeeze(1))  # (B, N_MELS, T)
        log_mel = torch.log(mel + 1e-9)
        log_mel = log_mel.permute(0, 2, 1)                  # (B, T, N_MELS)

    # Crop / pad về MAX_LEN
    T = log_mel.size(1)
    if T > Config.MAX_LEN:
        log_mel = log_mel[:, :Config.MAX_LEN, :]
    elif T < Config.MAX_LEN:
        pad     = torch.zeros(log_mel.size(0), Config.MAX_LEN - T,
                              Config.N_MELS, device=log_mel.device)
        log_mel = torch.cat([log_mel, pad], dim=1)

    # Per-sample mean/std normalize — khớp firmware
    mean    = log_mel.mean(dim=(1, 2), keepdim=True)
    std     = log_mel.std(dim=(1, 2),  keepdim=True) + 1e-9
    return (log_mel - mean) / std   # (B, MAX_LEN, N_MELS)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for waveform, labels in loader:
        waveform = waveform.to(Config.DEVICE)
        labels   = labels.to(Config.DEVICE)
        feats    = extract_features(waveform)
        out      = model(feats)                  # (B, NUM_CLASSES)
        correct += (out.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)

    print("=== Datasets ===")
    train_set = KWSDataset("training",   augment_data=True)
    val_set   = KWSDataset("validation", augment_data=False)
    test_set  = KWSDataset("testing",    augment_data=False)

    # train_eval_set: không augment, không sampler → đo train acc thực
    train_eval_set = KWSDataset("training", augment_data=False)

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE,
                              sampler=train_set.get_sampler(),
                              num_workers=4, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_set, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2)
    val_loader   = DataLoader(val_set,  batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_set, batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=2)

    print("\n=== Model ===")
    model     = KWS_SNN_Tiny().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=Config.EPOCHS, eta_min=1e-5)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    print(f"\n=== Training {Config.EPOCHS} epochs ===")

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for waveform, labels in train_loader:
            waveform = waveform.to(Config.DEVICE)
            labels   = labels.to(Config.DEVICE)

            feats = extract_features(waveform)   # (B, MAX_LEN, N_MELS)
            out   = model(feats)                 # (B, NUM_CLASSES)
            loss  = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_acc = evaluate(model, train_eval_loader)
        val_acc   = evaluate(model, val_loader)
        lr_now    = optimizer.param_groups[0]["lr"]
        tag       = ""

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "val_acc":     best_acc,
                "epoch":       epoch,
                "config": {
                    "CONV1_CH":     Config.CONV1_CH,
                    "CONV2_CH":     Config.CONV2_CH,
                    "FC1_UNITS":    Config.FC1_UNITS,
                    "FLATTEN_SIZE": Config.FLATTEN_SIZE,
                    "TIME_STEPS":   Config.TIME_STEPS,
                    "BETA":         Config.BETA,
                    "MAX_LEN":      Config.MAX_LEN,
                    "N_MELS":       Config.N_MELS,
                    "NUM_CLASSES":  Config.NUM_CLASSES,
                    "ALL_LABELS":   Config.ALL_LABELS,
                }
            }, "checkpoints/tiny_best.pth")
            tag = "  ← saved ✓"

        print(f"Epoch {epoch:3d}/{Config.EPOCHS}  "
              f"Loss={total_loss/n_batches:.4f}  "
              f"Train={train_acc:.4f}  Val={val_acc:.4f}  "
              f"LR={lr_now:.2e}{tag}")

    # Test cuối
    ckpt = torch.load("checkpoints/tiny_best.pth", map_location=Config.DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_acc = evaluate(model, test_loader)
    print(f"\nBest Val = {best_acc:.4f}")
    print(f"Test Acc = {test_acc:.4f}")