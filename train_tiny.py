import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from config_tiny import Config
from dataset_tiny import KWSDataset
from model_tiny import KWS_SNN_Tiny
import os

# ── transforms ──────────────────────────────────────────────
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=Config.SAMPLE_RATE, n_mels=Config.N_MELS,
    n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
)

def extract_features(waveform):
    """waveform: (1, samples) → (MAX_LEN, N_MELS)"""
    mel = mel_transform(waveform)           # (1, N_MELS, time)
    log_mel = torch.log(mel + 1e-9).squeeze(0).permute(1, 0)  # (time, N_MELS)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    # pad / crop
    if log_mel.size(0) > Config.MAX_LEN:
        log_mel = log_mel[:Config.MAX_LEN]
    else:
        pad = torch.zeros(Config.MAX_LEN - log_mel.size(0), Config.N_MELS)
        log_mel = torch.cat([log_mel, pad], dim=0)
    return log_mel

def rate_encode(x):
    """x: (B, MAX_LEN, N_MELS) → (T, B, MAX_LEN, N_MELS)"""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    x_min = x_flat.min(1)[0].view(B, 1, 1)
    x_max = x_flat.max(1)[0].view(B, 1, 1)
    x = (x - x_min) / (x_max - x_min + 1e-9)
    return (torch.rand(Config.TIME_STEPS, *x.shape, device=x.device) < x).float()

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for waveform, labels in loader:
        labels = labels.to(Config.DEVICE)
        feats = torch.stack([extract_features(w) for w in waveform]).to(Config.DEVICE)
        spikes = rate_encode(feats)
        out = model(spikes).sum(dim=0)          # spike count
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for waveform, labels in loader:
            labels = labels.to(Config.DEVICE)
            feats = torch.stack([extract_features(w) for w in waveform]).to(Config.DEVICE)
            spikes = rate_encode(feats)
            out = model(spikes).sum(dim=0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ── main ────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)

    train_set = KWSDataset("training")
    test_set  = KWSDataset("testing")

    sampler       = train_set.get_sampler()
    class_weights = train_set.get_class_weights().to(Config.DEVICE)

    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=Config.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model     = KWS_SNN_Tiny().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR,
                                 weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=Config.EPOCHS, eta_min=1e-5)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Training on: {Config.DEVICE}")
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        loss, tr_acc = train_epoch(model, train_loader, optimizer, loss_fn)
        val_acc = evaluate(model, test_loader)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | Loss={loss:.4f} | Train={tr_acc:.4f} | Val={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/tiny_best.pth")
            print(f"  --> Saved (Val={best_acc:.4f})")

    print(f"\nBest ValAcc: {best_acc:.4f}")
