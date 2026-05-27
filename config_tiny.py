import torch


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATA_PATH = "D:/lab/kws_sg_snn/data"

    SAMPLE_RATE = 16000

    # ── Feature extraction — khớp hoàn toàn với firmware ────────
    N_MELS     = 40
    N_FFT      = 400
    HOP_LENGTH = 160
    MEL_FMIN   = 80.0
    MEL_FMAX   = 7600.0

    # Số frame — phải khớp KWS_MAX_LEN trong firmware
    MAX_LEN = 64

    # ── Classes ──────────────────────────────────────────────────
    # Index 0-3: keyword phản hồi LED
    # Index 4  : unknown → firmware bỏ qua, không làm gì
    KEYWORDS    = ["on", "off", "left", "right"]
    ALL_LABELS  = KEYWORDS + ["unknown"]
    NUM_CLASSES = len(ALL_LABELS)   # = 5

    # Số samples unknown tối đa lấy từ MỖI từ lạ trong dataset gốc
    # ~29 từ × 80 = ~2320 unknown ≈ cân bằng với mỗi keyword (~2000–2500)
    UNKNOWN_PER_CLASS = 400

    # ── Kiến trúc model ──────────────────────────────────────────
    # Tăng nhẹ capacity so với 6-class vì bài toán đơn giản hơn
    # nhưng vẫn trong giới hạn DRAM ESP32 (320 KB)
    #
    # RAM tĩnh firmware ước tính:
    #   audio_buf (int16)  : 32  KB
    #   mel_buf (float)    : 10  KB
    #   pre_pool1 (float)  : CONV1_CH×64×40×4 = 12×64×40×4 = 120 KB
    #   pre_pool2 (float)  : CONV2_CH×32×20×4 = 24×32×20×4 =  60 KB
    #   feat_buf           :  15  KB
    #   SNN state          :   1  KB
    #   Tổng               : ~238 KB / 320 KB  → OK
    CONV1_CH  = 12
    CONV2_CH  = 24
    FC1_UNITS = 96

    # ── SNN ──────────────────────────────────────────────────────
    TIME_STEPS = 64   # T=64 cho ~89% vs T=16 chỉ 84% (xem debug sweep)
    BETA       = 0.90

    # ── Training ─────────────────────────────────────────────────
    BATCH_SIZE   = 64
    LR           = 2e-3
    EPOCHS       = 100
    WEIGHT_DECAY = 1e-4

    # ── Flatten size — single source of truth cho Python + firmware
    FLATTEN_H    = MAX_LEN // 4                        # = 16
    FLATTEN_W    = N_MELS  // 4                        # = 10
    FLATTEN_SIZE = FLATTEN_H * FLATTEN_W * CONV2_CH   # = 16×10×24 = 3840