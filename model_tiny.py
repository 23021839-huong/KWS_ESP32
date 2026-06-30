import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from config_tiny import Config


class KWS_SNN_Tiny(nn.Module):

    def __init__(self):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, Config.CONV1_CH,
                               kernel_size=3, padding=1, bias=True)
        self.bn1   = nn.BatchNorm2d(Config.CONV1_CH)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(Config.CONV1_CH, Config.CONV2_CH,
                               kernel_size=3, padding=1, bias=True)
        self.bn2   = nn.BatchNorm2d(Config.CONV2_CH)
        self.pool2 = nn.MaxPool2d(2)

        # Kiểm tra FLATTEN_SIZE
        with torch.no_grad():
            dummy    = torch.zeros(1, 1, Config.MAX_LEN, Config.N_MELS)
            computed = self.cnn_extract(dummy).shape[1]
        assert computed == Config.FLATTEN_SIZE, (
            f"FLATTEN_SIZE lệch: config={Config.FLATTEN_SIZE}, thực tế={computed}."
        )

        # FC layers
        self.fc1 = nn.Linear(Config.FLATTEN_SIZE, Config.FC1_UNITS)
        self.fc2 = nn.Linear(Config.FC1_UNITS, Config.NUM_CLASSES)

        # RLIF layers 
        # all_to_all=True: recurrent weight là nn.Linear(FC1_UNITS, FC1_UNITS)
        # U[t+1] = β·U[t] + FC1(x[t]) + V·S[t]
        # learn_beta=True: β tự học
        # learn_threshold=True: threshold tự học thay vì cố định 1.0
        self.rlif1 = snn.RLeaky(
            beta             = Config.BETA,
            spike_grad       = spike_grad,
            all_to_all       = True,
            linear_features  = Config.FC1_UNITS,
            learn_beta       = True,
            learn_threshold  = True,
            reset_mechanism  = "subtract",
        )

        # Layer cuối dùng RLeaky với all_to_all=False (elementwise V)
        # để giảm số param và tránh overfit ở output layer
        self.rlif2 = snn.RLeaky(
            beta            = Config.BETA,
            spike_grad      = spike_grad,
            all_to_all      = False,
            learn_beta      = True,
            learn_threshold = True,
            reset_mechanism = "subtract",
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Model] flatten={computed} | params={n_params:,} "
              f"(~{n_params*4/1024:.1f} KB f32, ~{n_params/1024:.1f} KB int8)")

        # In thêm recurrent weight info
        print(f"  rlif1.recurrent: {self.rlif1.recurrent.weight.shape}  "
              f"(all_to_all Linear)")
        print(f"  rlif1.beta={Config.BETA} (learnable)  "
              f"threshold=1.0 (learnable)")

    def cnn_extract(self, x):
        """x: (B,1,H,W) → (B, FLATTEN_SIZE). Chạy 1 lần duy nhất."""
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        return x.reshape(x.size(0), -1)

    def forward(self, mel, time_steps=None):

        T = time_steps or Config.TIME_STEPS

        # 1. CNN feature — 1 lần, deterministic
        feat = self.cnn_extract(mel.unsqueeze(1))   # (B, FLATTEN_SIZE)

        # 2. Normalize về [0,1] — làm input ổn định cho FC1
        feat_min  = feat.min(1, keepdim=True)[0]
        feat_max  = feat.max(1, keepdim=True)[0]
        feat_norm = (feat - feat_min) / (feat_max - feat_min + 1e-9)

        # 3. Khởi tạo RLIF state
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()

        mem2_sum = torch.zeros(feat.size(0), Config.NUM_CLASSES,
                               device=feat.device)

        # 4. Direct encoding loop — feed cùng feat_norm vào T bước
        #    RLIF tự tạo dynamics qua recurrent connection
        for _ in range(T):
            cur1       = self.fc1(feat_norm)          # (B, FC1_UNITS)
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)

            cur2       = self.fc2(spk1)               # (B, NUM_CLASSES)
            spk2, mem2 = self.rlif2(cur2, spk2, mem2)

            mem2_sum = mem2_sum + mem2   # tích lũy membrane potential

        # 5. Mean membrane → logits
        return mem2_sum / T   # (B, NUM_CLASSES)