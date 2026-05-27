"""
debug_check.py v2 — chạy sau train để xem điểm yếu còn lại.
    python debug_check.py
"""
import torch
import torchaudio
from collections import Counter
from torch.utils.data import DataLoader
from config_tiny import Config
from dataset_tiny import KWSDataset
from model_tiny   import KWS_SNN_Tiny

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=Config.SAMPLE_RATE, n_fft=Config.N_FFT,
    hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS,
    f_min=Config.MEL_FMIN, f_max=Config.MEL_FMAX,
)

def extract_features(wb):
    mel = mel_transform(wb.squeeze(1))
    lm  = torch.log(mel + 1e-9).permute(0, 2, 1)
    T   = lm.size(1)
    if T > Config.MAX_LEN:   lm = lm[:, :Config.MAX_LEN, :]
    elif T < Config.MAX_LEN:
        lm = torch.cat([lm, torch.zeros(lm.size(0), Config.MAX_LEN-T, Config.N_MELS)], 1)
    mean = lm.mean(dim=(1,2), keepdim=True)
    std  = lm.std(dim=(1,2),  keepdim=True) + 1e-9
    return (lm - mean) / std

model = KWS_SNN_Tiny()
ckpt  = torch.load("checkpoints/tiny_best.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

val_ds     = KWSDataset("validation", augment_data=False)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# Per-class accuracy + confusion matrix
n   = Config.NUM_CLASSES
cm  = [[0]*n for _ in range(n)]
with torch.no_grad():
    for wb, lb in val_loader:
        out   = model(extract_features(wb))
        preds = out.argmax(1)
        for p, t in zip(preds.tolist(), lb.tolist()):
            cm[t][p] += 1

print(f"Val acc: {ckpt['val_acc']:.4f}  |  epoch {ckpt['epoch']}")
print(f"lif1.beta={ckpt['model_state']['lif1.beta'].item():.4f}  "
      f"lif2.beta={ckpt['model_state']['lif2.beta'].item():.4f}\n")

print("Per-class accuracy:")
for i, lbl in enumerate(Config.ALL_LABELS):
    tot = sum(cm[i]); hit = cm[i][i]
    bar = "█"*int(hit/max(tot,1)*20)
    print(f"  {lbl:8s}: {hit/max(tot,1):.3f}  {bar}  ({hit}/{tot})")

print("\nConfusion matrix (row=true, col=pred):")
hdr = "         " + "".join(f"{Config.ALL_LABELS[j][:5]:>8}" for j in range(n))
print(hdr)
for i, row in enumerate(cm):
    print(f"  {Config.ALL_LABELS[i][:8]:8s} " + "".join(f"{v:8d}" for v in row))

# TIME_STEPS sweep
print("\nTIME_STEPS sweep (val batch):")
wb, lb = next(iter(val_loader))
feats  = extract_features(wb)
with torch.no_grad():
    for T in [8, 16, 32, 64, 128]:
        out = model(feats, time_steps=T)
        acc = (out.argmax(1) == lb).float().mean().item()
        print(f"  T={T:4d}: {acc:.3f}")