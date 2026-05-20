import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from config_tiny import Config

class KWS_SNN_Tiny(nn.Module):
    """
    Model tối giản cho ESP32:
    - 1 conv layer: 8 filters (thay vì 32)
    - FC: flatten → 64 → 6
    - Ước tính RAM (int8): ~60 KB
    """
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.pool  = nn.MaxPool2d(2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, Config.MAX_LEN, Config.N_MELS)
            out   = self.pool(torch.relu(self.bn1(self.conv1(dummy))))
            self.flatten_size = out.view(1, -1).shape[1]

        print(f"[*] Flatten size: {self.flatten_size}")
        print(f"[*] fc1 params: {self.flatten_size * 64} | Ước tính RAM float32: {self.flatten_size * 64 * 4 / 1024:.1f} KB")

        self.fc1  = nn.Linear(self.flatten_size, 64)
        self.lif1 = snn.Leaky(beta=Config.BETA, spike_grad=spike_grad)

        self.fc2  = nn.Linear(64, Config.NUM_CLASSES)
        self.lif2 = snn.Leaky(beta=Config.BETA, spike_grad=spike_grad)

    def forward(self, x):
        # x: (T, B, MAX_LEN, N_MELS)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        outputs = []

        for t in range(x.size(0)):
            cur = x[t].unsqueeze(1)                         # (B,1,time,feat)
            cur = self.pool(torch.relu(self.bn1(self.conv1(cur))))
            cur = cur.reshape(cur.size(0), -1)
            spk1, mem1 = self.lif1(self.fc1(cur), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            outputs.append(spk2)

        return torch.stack(outputs)  # (T, B, NUM_CLASSES)
