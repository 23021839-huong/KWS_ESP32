import torch
import torchaudio
import random
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter, defaultdict
from config_tiny import Config


# Ngưỡng lọc chất lượng
_MIN_RMS        = 0.002   
_MAX_CLIP_RATIO = 0.01    

# Tất cả từ trong SpeechCommands v2
_UNKNOWN_WORDS = sorted({
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow",   "forward", "four", "go",   "happy", "house", "learn",
    "marvin",   "nine",    "no",   "one",  "seven", "sheila", "six",
    "stop",     "three",   "tree", "two",  "up",    "visual", "wow",
    "yes",      "zero",
} - set(Config.KEYWORDS))


# Helpers

def quality_check(waveform, sr):
    """
    Kiểm tra chất lượng waveform.
    Returns (ok: bool, reason: str)
    """
    w = waveform.mean(dim=0)

    if w.abs().max() < 1e-6:
        return False, "zero"

    rms = w.pow(2).mean().sqrt().item()
    if rms < _MIN_RMS:
        return False, "low_rms"

    clip_ratio = (w.abs() > 0.99).float().mean().item()
    if clip_ratio > _MAX_CLIP_RATIO:
        return False, "clipping"

    return True, "ok"


def normalize_waveform(waveform, sr):
    """
    Resample → mono → pad/crop về 1 giây.
    Chạy 1 lần khi load, lưu vào RAM.
    """
    if sr != Config.SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)    # (1, T)
    T = waveform.size(1)
    if T > Config.SAMPLE_RATE:
        waveform = waveform[:, :Config.SAMPLE_RATE]
    elif T < Config.SAMPLE_RATE:
        waveform = torch.nn.functional.pad(waveform, (0, Config.SAMPLE_RATE - T))
    return waveform   # (1, 16000)


def augment(waveform):
    """
    Augmentation cho môi trường trong nhà.
    Thêm speed perturbation để giúp on/off phân biệt tốt hơn.
    """
    # 1. Gaussian noise nhẹ
    if torch.rand(1) < 0.5:
        waveform = waveform + torch.randn_like(waveform) * 0.003

    # 2. Time shift ±125ms
    if torch.rand(1) < 0.5:
        shift = int(torch.randint(-2000, 2000, (1,)))
        waveform = torch.roll(waveform, shifts=shift, dims=1)

    # 3. Volume ×0.7–1.3
    if torch.rand(1) < 0.4:
        waveform = waveform * (0.7 + torch.rand(1).item() * 0.6)

    # 4. Speed perturbation ×0.9–1.1 (resample trick)
    #    Giúp on/off không bị nhầm do tốc độ nói khác nhau
    if torch.rand(1) < 0.4:
        speed = 0.9 + torch.rand(1).item() * 0.2   # [0.9, 1.1]
        orig_len = waveform.size(1)
        new_len  = int(orig_len / speed)
        # Resample về new_len rồi crop/pad về orig_len
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
        if waveform.size(1) > orig_len:
            waveform = waveform[:, :orig_len]
        elif waveform.size(1) < orig_len:
            waveform = torch.nn.functional.pad(waveform, (0, orig_len - waveform.size(1)))

    # 5. Frequency masking nhẹ trên raw waveform (band-stop ngẫu nhiên)
    #    Tăng robustness với giọng khác nhau
    if torch.rand(1) < 0.3:
        mask_start = int(torch.randint(0, 12000, (1,)))
        mask_len   = int(torch.randint(500, 2000, (1,)))
        waveform[:, mask_start:mask_start+mask_len] *= 0.1

    return waveform


# ── Dataset ───────────────────────────────────────────────────

class KWSDataset(Dataset):
    """
    Load Google Speech Commands:
      - 4 keyword (on/off/left/right): lấy toàn bộ, lọc chất lượng
      - unknown: lấy đều từ các từ lạ, tổng ≈ số keyword để cân bằng
      - WeightedRandomSampler cân bằng 5 class
      - Augmentation chỉ lúc training
    """

    def __init__(self, subset="training", augment_data=True):
        self.do_augment = augment_data and (subset == "training")

        raw = torchaudio.datasets.SPEECHCOMMANDS(
            root=Config.DATA_PATH,
            download=True,
            subset=subset,
        )

        self.label_to_idx = {lbl: i for i, lbl in enumerate(Config.ALL_LABELS)}
        unknown_idx       = self.label_to_idx["unknown"]

        self.samples = []
        self.labels  = []
        skipped      = Counter()
        unk_pool     = defaultdict(list)

        # Pass 1: keyword chính + gom unknown pool
        for waveform, sr, label, *_ in raw:

            if label in self.label_to_idx and label != "unknown":
                ok, reason = quality_check(waveform, sr)
                if not ok:
                    skipped[reason] += 1
                    continue
                self.samples.append(normalize_waveform(waveform, sr))
                self.labels.append(self.label_to_idx[label])

            elif label in _UNKNOWN_WORDS:
                if len(unk_pool[label]) < Config.UNKNOWN_PER_CLASS:
                    ok, _ = quality_check(waveform, sr)
                    if ok:
                        unk_pool[label].append((waveform, sr))

        # Pass 2: chọn unknown cân bằng từ pool
        n_keyword     = len(self.samples)
        n_unk_words   = len(unk_pool)
        if n_unk_words > 0:
            # Chia đều quota mỗi từ lạ để đa dạng nguồn
            quota = max(1, n_keyword // (len(Config.KEYWORDS) * n_unk_words))
            for word, items in unk_pool.items():
                chosen = random.sample(items, min(quota, len(items)))
                for waveform, sr in chosen:
                    self.samples.append(normalize_waveform(waveform, sr))
                    self.labels.append(unknown_idx)

        # Thống kê
        count   = Counter(self.labels)
        n_total = len(self.samples)
        n_bad   = sum(skipped.values())
        print(f"\n[{subset}] {n_total} samples  |  loại {n_bad}")
        for lbl, idx in self.label_to_idx.items():
            bar = "█" * (count[idx] // 80)
            print(f"  {lbl:8s} [{idx}]: {count[idx]:5d}  {bar}")
        if n_bad:
            print(f"  Bị loại: {dict(skipped)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform = self.samples[idx].clone()
        # Normalize amplitude per-sample
        peak = waveform.abs().max()
        if peak > 1e-9:
            waveform = waveform / peak
        if self.do_augment:
            waveform = augment(waveform)
        return waveform, self.labels[idx]

    def get_sampler(self):
        """WeightedRandomSampler — cân bằng 5 class."""
        count   = Counter(self.labels)
        weights = [1.0 / count[l] for l in self.labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)