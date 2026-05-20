import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "./data"

    SAMPLE_RATE = 16000
    N_MELS     = 20        # Giảm từ 40 → 20 (tiết kiệm RAM)
    N_FFT      = 400
    HOP_LENGTH = 160
    MAX_LEN    = 32        # Giảm từ 64 → 32

    # Chỉ 6 keyword cần thiết cho demo LED
    KEYWORDS    = ['on', 'off', 'left', 'right', 'stop', 'go']
    NUM_CLASSES = 6

    BETA       = 0.9
    BATCH_SIZE = 64
    LR         = 1e-3
    EPOCHS     = 30
    WEIGHT_DECAY = 1e-4

    TIME_STEPS = 10        # Giảm từ 50 → 10 để inference nhanh trên ESP32
