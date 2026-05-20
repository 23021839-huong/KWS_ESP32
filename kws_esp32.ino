/*
 * kws_esp32.ino
 * Keyword Spotting với SNN trên ESP32 DevKit V1
 *
 * Phần cứng:
 *   INMP441  → I2S  (SCK=GPIO26, WS=GPIO25, SD=GPIO33)
 *   LED 1    → GPIO2  (on/off)
 *   LED 2    → GPIO4  (left)
 *   LED 3    → GPIO5  (right)
 *   LED 4    → GPIO18 (go/stop)
 *   OLED     → I2C   (SDA=GPIO21, SCL=GPIO22) - tùy chọn
 *
 * Thư viện cần cài (Library Manager):
 *   - ESP32 Arduino core (espressif/arduino-esp32)
 *   - Adafruit SSD1306  (nếu dùng OLED)
 *   - Adafruit GFX
 *
 * Sau khi train xong Python, chạy export_to_c.py
 * rồi copy model_weights.h vào cùng thư mục này.
 */

#include <driver/i2s.h>
#include <math.h>
#include "model_weights.h"

// ── Nếu có OLED thì bỏ comment dòng dưới ──────────────────
// #define USE_OLED
#ifdef USE_OLED
  #include <Wire.h>
  #include <Adafruit_GFX.h>
  #include <Adafruit_SSD1306.h>
  Adafruit_SSD1306 display(128, 64, &Wire, -1);
#endif

// ── Pin config ─────────────────────────────────────────────
#define I2S_SCK   26
#define I2S_WS    25
#define I2S_SD    33

#define LED_ON    2    // keyword "on" / "off"
#define LED_LEFT  4    // keyword "left"
#define LED_RIGHT 5    // keyword "right"
#define LED_GO    18   // keyword "go" / "stop"

// ── Audio config ───────────────────────────────────────────
#define SAMPLE_RATE    16000
#define AUDIO_SAMPLES  16000   // 1 giây
#define HOP_LENGTH     160
#define N_FFT          400
#define FRAME_COUNT    (AUDIO_SAMPLES / HOP_LENGTH)  // ~100 frames

// ── VAD ────────────────────────────────────────────────────
#define VAD_THRESHOLD  500     // Điều chỉnh tùy mic và môi trường
#define DEBOUNCE_MS    800     // Không nhận lệnh liên tiếp quá nhanh

// ── Buffers ────────────────────────────────────────────────
int16_t audio_buf[AUDIO_SAMPLES];
float   mel_buf[KWS_MAX_LEN][KWS_N_MELS];

// SNN state
float mem1[64];   // membrane potential lif1
float mem2[KWS_NUM_CLASSES];  // membrane potential lif2
float spike_count[KWS_NUM_CLASSES];

// ── Dequantize helper ───────────────────────────────────────
inline float deq(uint8_t q, float scale, int zp) {
  return (q - zp) * scale;
}

// ── Forward pass (inference thủ công) ─────────────────────
// Conv2D: 1 channel in, 8 channel out, kernel 3x3, same padding
// Input: mel_buf[MAX_LEN][N_MELS]
// Output: conv_out[8][MAX_LEN/2][N_MELS/2] (sau pool)
#define CONV_OUT_H (KWS_MAX_LEN / 2)
#define CONV_OUT_W (KWS_N_MELS  / 2)
#define CONV_CH    8
#define FC1_OUT    64

float conv_out[CONV_CH][CONV_OUT_H][CONV_OUT_W];
float fc1_out[FC1_OUT];
float fc2_out[KWS_NUM_CLASSES];

void run_conv_bn_pool() {
  // Conv 3x3 same padding + BN + ReLU + MaxPool 2x2
  float tmp[CONV_CH][KWS_MAX_LEN][KWS_N_MELS];

  for (int c = 0; c < CONV_CH; c++) {
    // Dequantize BN params
    float bn_w = deq(bn1_weight[c], bn1_weight_scale, bn1_weight_zp);
    float bn_b = deq(bn1_bias[c],   bn1_bias_scale,   bn1_bias_zp);
    float bn_m = deq(bn1_mean[c],   bn1_mean_scale,   bn1_mean_zp);
    float bn_v = deq(bn1_var[c],    bn1_var_scale,    bn1_var_zp);
    float bn_std = sqrtf(bn_v + 1e-5f);

    for (int h = 0; h < KWS_MAX_LEN; h++) {
      for (int w = 0; w < KWS_N_MELS; w++) {
        float val = 0;
        // kernel 3x3
        for (int kh = -1; kh <= 1; kh++) {
          for (int kw = -1; kw <= 1; kw++) {
            int ih = h + kh, iw = w + kw;
            if (ih < 0 || ih >= KWS_MAX_LEN || iw < 0 || iw >= KWS_N_MELS) continue;
            // weight index: [out_ch, in_ch=0, kh+1, kw+1] → flat
            int widx = c * 9 + (kh + 1) * 3 + (kw + 1);
            float wval = deq(conv1_weight[widx], conv1_weight_scale, conv1_weight_zp);
            val += wval * mel_buf[ih][iw];
          }
        }
        float bval = deq(conv1_bias[c], conv1_bias_scale, conv1_bias_zp);
        val += bval;
        // BatchNorm
        val = (val - bn_m) / bn_std * bn_w + bn_b;
        // ReLU
        tmp[c][h][w] = val > 0 ? val : 0;
      }
    }
  }

  // MaxPool 2x2
  for (int c = 0; c < CONV_CH; c++) {
    for (int h = 0; h < CONV_OUT_H; h++) {
      for (int w = 0; w < CONV_OUT_W; w++) {
        float m = -1e9;
        for (int ph = 0; ph < 2; ph++)
          for (int pw = 0; pw < 2; pw++)
            m = fmaxf(m, tmp[c][h*2+ph][w*2+pw]);
        conv_out[c][h][w] = m;
      }
    }
  }
}

// LIF neuron: returns spike (0/1), updates mem in-place
inline float lif_step(float input, float *mem) {
  *mem = LIF_BETA * (*mem) + input;
  if (*mem >= 1.0f) { *mem -= 1.0f; return 1.0f; }
  return 0.0f;
}

int run_snn_inference() {
  // Reset state
  memset(mem1, 0, sizeof(mem1));
  memset(mem2, 0, sizeof(mem2));
  memset(spike_count, 0, sizeof(spike_count));

  int flat_size = CONV_CH * CONV_OUT_H * CONV_OUT_W;

  for (int t = 0; t < KWS_TIME_STEPS; t++) {
    // Rate encoding: spike nếu rand() < mel_buf[pixel]
    // (dùng conv_out đã tính 1 lần, encode theo thời gian)

    // FC1
    for (int j = 0; j < FC1_OUT; j++) {
      float sum = deq(fc1_bias[j], fc1_bias_scale, fc1_bias_zp);
      int flat_idx = 0;
      for (int c = 0; c < CONV_CH; c++)
        for (int h = 0; h < CONV_OUT_H; h++)
          for (int w = 0; w < CONV_OUT_W; w++, flat_idx++) {
            // Rate encode: spike với xác suất = conv_out (đã normalize [0,1])
            float prob = conv_out[c][h][w];
            float spike = ((float)rand() / RAND_MAX < prob) ? 1.0f : 0.0f;
            float wval = deq(fc1_weight[j * flat_size + flat_idx],
                             fc1_weight_scale, fc1_weight_zp);
            sum += wval * spike;
          }
      fc1_out[j] = lif_step(sum, &mem1[j]);
    }

    // FC2
    for (int k = 0; k < KWS_NUM_CLASSES; k++) {
      float sum = deq(fc2_bias[k], fc2_bias_scale, fc2_bias_zp);
      for (int j = 0; j < FC1_OUT; j++) {
        float wval = deq(fc2_weight[k * FC1_OUT + j],
                         fc2_weight_scale, fc2_weight_zp);
        sum += wval * fc1_out[j];
      }
      float spk = lif_step(sum, &mem2[k]);
      spike_count[k] += spk;
    }
  }

  // Argmax
  int best = 0;
  for (int k = 1; k < KWS_NUM_CLASSES; k++)
    if (spike_count[k] > spike_count[best]) best = k;

  // Confidence check: tổng spike phải đủ lớn
  float total = 0;
  for (int k = 0; k < KWS_NUM_CLASSES; k++) total += spike_count[k];
  if (total < 2.0f) return -1;  // không đủ tự tin
  if (spike_count[best] / total < 0.35f) return -1;

  return best;
}

// ── Feature extraction: waveform → Log-Mel ────────────────
// Dùng FFT đơn giản bằng Goertzel filter cho từng mel band
// (nhẹ hơn full FFT, phù hợp MCU)

// Mel frequency helpers
float hz_to_mel(float hz)  { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

void compute_log_mel() {
  float mel_min = hz_to_mel(80.0f);
  float mel_max = hz_to_mel(7600.0f);

  float mean_val = 0, std_val = 0;
  int count = 0;

  for (int frame = 0; frame < KWS_MAX_LEN; frame++) {
    int start = frame * HOP_LENGTH;
    if (start + N_FFT > AUDIO_SAMPLES) break;

    for (int m = 0; m < KWS_N_MELS; m++) {
      // Tần số trung tâm của mel band m
      float mel_center = mel_min + (m + 0.5f) * (mel_max - mel_min) / KWS_N_MELS;
      float f_center   = mel_to_hz(mel_center);

      // Goertzel algorithm — tính năng lượng tại tần số f_center
      float omega = 2.0f * M_PI * f_center / SAMPLE_RATE;
      float coeff = 2.0f * cosf(omega);
      float s0 = 0, s1 = 0, s2 = 0;

      for (int n = 0; n < N_FFT; n++) {
        float sample = audio_buf[start + n] / 32768.0f;
        // Hann window
        float win = 0.5f * (1.0f - cosf(2.0f * M_PI * n / (N_FFT - 1)));
        s0 = coeff * s1 - s2 + sample * win;
        s2 = s1; s1 = s0;
      }
      float power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
      mel_buf[frame][m] = logf(power + 1e-9f);
      mean_val += mel_buf[frame][m];
      count++;
    }
  }

  // Normalize
  mean_val /= count;
  for (int i = 0; i < KWS_MAX_LEN; i++)
    for (int j = 0; j < KWS_N_MELS; j++)
      std_val += (mel_buf[i][j] - mean_val) * (mel_buf[i][j] - mean_val);
  std_val = sqrtf(std_val / count + 1e-9f);

  for (int i = 0; i < KWS_MAX_LEN; i++)
    for (int j = 0; j < KWS_N_MELS; j++)
      mel_buf[i][j] = (mel_buf[i][j] - mean_val) / std_val;

  // Normalize conv_out sang [0,1] cho rate encoding
  run_conv_bn_pool();
  float cmax = 0;
  for (int c = 0; c < CONV_CH; c++)
    for (int h = 0; h < CONV_OUT_H; h++)
      for (int w = 0; w < CONV_OUT_W; w++)
        cmax = fmaxf(cmax, conv_out[c][h][w]);
  if (cmax > 0)
    for (int c = 0; c < CONV_CH; c++)
      for (int h = 0; h < CONV_OUT_H; h++)
        for (int w = 0; w < CONV_OUT_W; w++)
          conv_out[c][h][w] /= cmax;
}

// ── VAD: kiểm tra có tiếng nói không ──────────────────────
bool vad_detected() {
  long sum = 0;
  for (int i = 0; i < AUDIO_SAMPLES; i++)
    sum += abs(audio_buf[i]);
  return (sum / AUDIO_SAMPLES) > VAD_THRESHOLD;
}

// ── LED actions ─────────────────────────────────────────────
bool led_on_state    = false;
bool led_left_state  = false;
bool led_right_state = false;
bool led_go_state    = false;

void blink_led(int pin, int times = 3) {
  for (int i = 0; i < times; i++) {
    digitalWrite(pin, HIGH); delay(100);
    digitalWrite(pin, LOW);  delay(100);
  }
}

void handle_keyword(int kw_idx) {
  String kw = KWS_LABELS[kw_idx];
  Serial.printf(">>> Keyword: %s (spikes: ", kw.c_str());
  for (int k = 0; k < KWS_NUM_CLASSES; k++) Serial.printf("%.0f ", spike_count[k]);
  Serial.println(")");

  if      (kw == "on")    { led_on_state = true;  digitalWrite(LED_ON, HIGH); }
  else if (kw == "off")   { led_on_state = false; digitalWrite(LED_ON, LOW);  }
  else if (kw == "left")  { led_left_state = !led_left_state;  digitalWrite(LED_LEFT,  led_left_state  ? HIGH : LOW); }
  else if (kw == "right") { led_right_state = !led_right_state; digitalWrite(LED_RIGHT, led_right_state ? HIGH : LOW); }
  else if (kw == "go")    { led_go_state = true;  digitalWrite(LED_GO, HIGH); }
  else if (kw == "stop")  {
    // Tắt tất cả
    led_on_state = led_left_state = led_right_state = led_go_state = false;
    digitalWrite(LED_ON, LOW); digitalWrite(LED_LEFT, LOW);
    digitalWrite(LED_RIGHT, LOW); digitalWrite(LED_GO, LOW);
    blink_led(LED_GO, 2);
  }

#ifdef USE_OLED
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 20);
  display.println(kw);
  float conf = spike_count[kw_idx] / KWS_TIME_STEPS * 100;
  display.setTextSize(1);
  display.setCursor(10, 50);
  display.printf("conf: %.0f%%", conf);
  display.display();
#endif
}

// ── I2S setup ───────────────────────────────────────────────
void setup_i2s() {
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 4,
    .dma_buf_len          = 512,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  i2s_pin_config_t pins = {
    .bck_io_num   = I2S_SCK,
    .ws_io_num    = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

void read_audio() {
  int32_t raw[512];
  size_t bytes_read;
  int samples_read = 0;
  while (samples_read < AUDIO_SAMPLES) {
    int to_read = min(512, AUDIO_SAMPLES - samples_read);
    i2s_read(I2S_NUM_0, raw, to_read * sizeof(int32_t), &bytes_read, portMAX_DELAY);
    int got = bytes_read / sizeof(int32_t);
    for (int i = 0; i < got && samples_read < AUDIO_SAMPLES; i++, samples_read++)
      audio_buf[samples_read] = (int16_t)(raw[i] >> 16);
  }
}

// ── Setup & Loop ───────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("KWS SNN ESP32 starting...");

  pinMode(LED_ON,    OUTPUT); digitalWrite(LED_ON,    LOW);
  pinMode(LED_LEFT,  OUTPUT); digitalWrite(LED_LEFT,  LOW);
  pinMode(LED_RIGHT, OUTPUT); digitalWrite(LED_RIGHT, LOW);
  pinMode(LED_GO,    OUTPUT); digitalWrite(LED_GO,    LOW);

  setup_i2s();

#ifdef USE_OLED
  Wire.begin(21, 22);
  if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("KWS SNN Ready");
    display.println("Say: on/off/left");
    display.println("     right/go/stop");
    display.display();
  }
#endif

  srand(42);

  // Startup blink
  for (int p : {LED_ON, LED_LEFT, LED_RIGHT, LED_GO}) {
    digitalWrite(p, HIGH); delay(100);
  }
  delay(200);
  for (int p : {LED_ON, LED_LEFT, LED_RIGHT, LED_GO}) {
    digitalWrite(p, LOW);
  }

  Serial.println("Ready! Listening...");
}

unsigned long last_kw_time = 0;

void loop() {
  read_audio();

  if (!vad_detected()) {
    Serial.println("... (silence)");
    return;
  }

  Serial.println("Speech detected! Running inference...");
  unsigned long t0 = millis();

  compute_log_mel();
  int kw = run_snn_inference();

  unsigned long elapsed = millis() - t0;
  Serial.printf("Inference time: %lu ms\n", elapsed);

  if (kw >= 0) {
    unsigned long now = millis();
    if (now - last_kw_time > DEBOUNCE_MS) {
      handle_keyword(kw);
      last_kw_time = now;
    }
  } else {
    Serial.println("(không nhận ra keyword)");
  }
}
