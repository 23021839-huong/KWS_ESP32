# KWS using RLIF-CSNN on ESP32

## Overview

Đây là dự án triển khai **Keyword Spotting (KWS)** trên nền tảng **ESP32** sử dụng **Convolutional Spiking Neural Network (CSNN)** kết hợp **Recurrent Leaky Integrate-and-Fire (RLIF)**.

Mục tiêu của dự án là nghiên cứu khả năng triển khai mô hình Spiking Neural Network trên vi điều khiển có tài nguyên hạn chế, đồng thời xây dựng một hệ thống nhận dạng từ khóa hoạt động theo thời gian thực.

Quy trình xử lý của hệ thống:

```
Speech
    ↓
INMP441 Microphone
    ↓
Log-Mel Spectrogram
    ↓
CNN Feature Extraction
    ↓
Spike Encoding
    ↓
RLIF-CSNN
    ↓
Keyword Prediction
    ↓
LED / OLED / Serial Monitor
```

---

# Hardware

* ESP32 DevKit V1
* INMP441 I2S Microphone
* OLED SSD1306 (Optional)
* 4 LEDs
* USB Serial

---

# Training Phase (PC)

Toàn bộ quá trình huấn luyện được thực hiện trên máy tính sử dụng **PyTorch** và **snnTorch**.

Pipeline huấn luyện:

```
Google Speech Commands
        ↓
Log-Mel Spectrogram
        ↓
CNN Feature Extractor
        ↓
Spike Encoding
        ↓
RLIF-CSNN Training
        ↓
model.pth
        ↓
Quantization
        ↓
Export Weights (.h/.cpp)
```

Sau khi mô hình hội tụ, các trọng số được lượng tử hóa và chuyển đổi sang file C Header để tích hợp vào firmware của ESP32.

---

# Deployment Phase (ESP32)

ESP32 chỉ thực hiện **Inference**, không thực hiện quá trình huấn luyện.

Pipeline xử lý trên ESP32:

```
INMP441
    ↓
I2S Driver
    ↓
PCM Audio Buffer
    ↓
compute_log_mel()
    ↓
run_cnn_features()
    ↓
rate_encode_feature()
    ↓
RLIF-CSNN Inference
    ↓
Keyword Prediction
    ↓
LED / OLED / Serial Monitor
```

Kết quả dự đoán sẽ được:

* Hiển thị trên Serial Monitor.
* Điều khiển LED tương ứng với từ khóa.
* Hiển thị trên OLED (nếu được bật).

---

# Repository Structure

```
.
├── checkpoints/
│   └── tiny_best.pth                # Best trained model
│
├── image/                           # Results images
│
├── config_tiny.py                   # Hyperparameters
├── dataset_tiny.py                  # Dataset loader
├── model_tiny.py                    # RLIF-CSNN architecture
├── train_tiny.py                    # Training script
├── export_to_c.py                   # Convert model to C header
├── model_weights.h                  # Quantized weights
├── kws_esp32.ino                    # ESP32 inference firmware
├── debug.py                         # Debug utilities
├── README.md
└── requirements.txt (optional)
```

---

# Experimental Results

Qua quá trình thử nghiệm, mô hình RLIF-CSNN cho kết quả tốt hơn so với mô hình SNN hai lớp ban đầu.

Một số cải tiến đã thực hiện:

* Bổ sung CNN Feature Extractor trước lớp Spiking.
* Thay neuron LIF bằng RLIF nhằm tăng khả năng khai thác thông tin theo thời gian.
* Lượng tử hóa trọng số để phù hợp với bộ nhớ của ESP32.

Mô hình đã có thể:

* Thu âm trực tiếp từ INMP441.
* Thực hiện suy luận trên ESP32.
* Nhận dạng từ khóa và điều khiển LED theo kết quả dự đoán.
* Xuất thông tin dự đoán qua Serial Monitor.

---

# Current Limitations

Đây vẫn là phiên bản nghiên cứu và chưa phải hệ thống hoàn thiện.

Một số hạn chế hiện tại:

* Độ chính xác trên ESP32 còn thấp hơn so với khi chạy trên PC.
* Mô hình vẫn cần tối ưu thêm về kích thước và thời gian suy luận.
* Chưa đánh giá đầy đủ về độ trễ (Latency), năng lượng tiêu thụ và độ ổn định trong nhiều môi trường khác nhau.
* Chưa triển khai các kỹ thuật tối ưu như Pruning hoặc Knowledge Distillation.

---

# Future Work

Trong các phiên bản tiếp theo, dự án sẽ tập trung vào:
* Tối ưu kiến trúc RLIF-CSNN.
* Giảm kích thước mô hình để phù hợp hơn với ESP32.
* Cải thiện Accuracy và Latency.
* Đánh giá mức tiêu thụ năng lượng trên thiết bị Edge.
* Triển khai trên Raspberry Pi và FPGA để so sánh hiệu năng.
* Phát triển giao diện hiển thị và hệ thống điều khiển hoàn chỉnh.

---

# Acknowledgements

* PyTorch
* snnTorch
* Google Speech Commands Dataset
* Espressif ESP32
