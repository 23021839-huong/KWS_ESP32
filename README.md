# KWS SNN ESP32 — Hướng dẫn đầy đủ

## Sơ đồ nối dây

```
INMP441          ESP32 DevKit V1
-------          ---------------
VDD      →       3.3V
GND      →       GND
SCK      →       GPIO 26
WS       →       GPIO 25
SD       →       GPIO 33
L/R      →       GND  (chọn kênh trái)

LED 1 (on/off)   → GPIO 2  → 220Ω → GND
LED 2 (left)     → GPIO 4  → 220Ω → GND
LED 3 (right)    → GPIO 5  → 220Ω → GND
LED 4 (go/stop)  → GPIO 18 → 220Ω → GND

OLED SSD1306 (tùy chọn)
SDA      →       GPIO 21
SCL      →       GPIO 22
VCC      →       3.3V
GND      →       GND
```

## Các bước thực hiện

### Bước 1: Train model trên PC
```bash
# Cài thư viện
pip install torch torchaudio snntorch

# Copy các file sau vào 1 thư mục:
#   config_tiny.py, dataset_tiny.py, model_tiny.py, train_tiny.py

# Chạy training (~20-30 phút tùy CPU/GPU)
python train_tiny.py

# Kết quả: checkpoints/tiny_best.pth
```

### Bước 2: Export sang C header
```bash
# Copy export_to_c.py vào cùng thư mục
python export_to_c.py

# Kết quả: model_weights.h (~80 KB)
```

### Bước 3: Nạp firmware ESP32
1. Mở Arduino IDE
2. Cài board: ESP32 by Espressif Systems
3. Chọn board: **ESP32 Dev Module**
4. Copy `model_weights.h` vào thư mục `kws_esp32/`
5. Nếu có OLED: cài thư viện Adafruit SSD1306, bỏ comment `#define USE_OLED`
6. Upload sketch

### Bước 4: Test
Mở Serial Monitor (115200 baud), nói các keyword:
- **"on"**   → LED 1 sáng
- **"off"**  → LED 1 tắt
- **"left"** → LED 2 toggle
- **"right"**→ LED 3 toggle
- **"go"**   → LED 4 sáng
- **"stop"** → Tất cả tắt

## Điều chỉnh nếu không nhận đúng

**VAD threshold** (dòng `#define VAD_THRESHOLD 500`):
- Môi trường ồn → tăng lên 800-1000
- Mic yếu, nói xa → giảm xuống 200-300

**Confidence threshold** (trong `run_snn_inference()`):
- Dòng `if (spike_count[best] / total < 0.35f)` 
- Tăng 0.35 → 0.45 nếu nhận nhầm nhiều
- Giảm xuống 0.25 nếu không nhận ra

**Debounce** (`#define DEBOUNCE_MS 800`):
- Tăng nếu 1 lần nói bị nhận 2 lần
- Giảm nếu muốn phản hồi nhanh hơn

## Ước tính tài nguyên

| Thứ | Kích thước |
|-----|-----------|
| Model weights (int8) | ~60 KB flash |
| Audio buffer (1s)   | ~32 KB RAM   |
| Conv output buffer  | ~4 KB RAM    |
| Tổng RAM dùng       | ~80 KB / 520 KB |

## Ghi chú
- Inference time ước tính: 200-500ms trên ESP32 @ 240 MHz
- Nếu quá chậm: giảm `TIME_STEPS = 5` trong config_tiny.py
- Nếu accuracy thấp: tăng `EPOCHS = 50`, thêm data augmentation
