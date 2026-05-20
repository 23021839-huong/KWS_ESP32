import torch
import torchaudio
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter
from config_tiny import Config

class KWSDataset(Dataset):
    """
    Chỉ lấy 6 keyword cần thiết, bỏ _unknown_ và _silence_
    để tránh class imbalance nghiêm trọng.
    """
    def __init__(self, subset="training"):
        raw = torchaudio.datasets.SPEECHCOMMANDS(
            root=Config.DATA_PATH, download=True, subset=subset
        )

        self.label_to_index = {l: i for i, l in enumerate(Config.KEYWORDS)}

        # Lọc chỉ giữ 6 keyword
        self.samples = []
        self.label_indices = []
        for item in raw:
            waveform, sr, label, *_ = item
            if label in self.label_to_index:
                self.samples.append((waveform, sr))
                self.label_indices.append(self.label_to_index[label])

        print(f"[{subset}] {len(self.samples)} samples | classes: {dict(Counter(self.label_indices))}")

    def get_sampler(self):
        count = Counter(self.label_indices)
        weights = [1.0 / count[l] for l in self.label_indices]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def get_class_weights(self):
        count = Counter(self.label_indices)
        total = len(self.label_indices)
        return torch.tensor(
            [total / (Config.NUM_CLASSES * max(count[i], 1)) for i in range(Config.NUM_CLASSES)],
            dtype=torch.float
        )

    def preprocess(self, waveform, sr):
        if sr != Config.SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, Config.SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)
        target_len = Config.SAMPLE_RATE
        if waveform.size(1) > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.size(1) < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.size(1)))
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, sr = self.samples[idx]
        waveform = self.preprocess(waveform, sr)
        return waveform, self.label_indices[idx]
