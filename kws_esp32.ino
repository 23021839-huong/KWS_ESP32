/*
 * kws_esp32.ino — KWS RLIF trên ESP32 DevKit V1
 * RAM-optimized: không lưu pre_pool buffer đầy đủ
 *
 * Hardware:
 *   INMP441: SCK=26  WS=25  SD=33  L/R=GND
 *   LED on    : GPIO 2
 *   LED left  : GPIO 4
 *   LED right : GPIO 5
 *   OLED SSD1306: SDA=21  SCL=22
 */

#include <driver/i2s.h>
#include <math.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model_weights.h"

// ── Pins ────────────────────────────────────────────────────
#define PIN_I2S_SCK   26
#define PIN_I2S_WS    25
#define PIN_I2S_SD    33
#define PIN_LED_ON     2
#define PIN_LED_LEFT   4
#define PIN_LED_RIGHT  5

// ── Audio ────────────────────────────────────────────────────
#define SAMPLE_RATE   16000
#define AUDIO_SAMPLES 16000
#define N_FFT         400
#define HOP_LENGTH    160
#define MEL_FMIN      80.0f
#define MEL_FMAX      7600.0f

// ── Thresholds ───────────────────────────────────────────────
#define VAD_THRESHOLD  600
#define DEBOUNCE_MS    1500
#define CONF_THRESHOLD 0.55f

#define OLED_W 128
#define OLED_H  64

// ════════════════════════════════════════════════════════════
// Static buffers — tính toán cẩn thận để không tràn DRAM
//
// audio_buf          : 16000×2        = 32 KB
// mel_buf            : 64×40×4        = 10 KB
// after_pool1        : 12×32×20×4     = 30 KB  (sau pool1)
// after_pool2        : 24×16×10×4     = 15 KB  (sau pool2)
// feat_buf           : 3840×4         = 15 KB
// RLIF state         : (96+5)×4×4     =  1.5 KB
// Tổng               : ~104 KB / 320 KB  ← OK
//
// Không lưu pre_pool (trước MaxPool) — tính trực tiếp
// ════════════════════════════════════════════════════════════

static int16_t audio_buf[AUDIO_SAMPLES];              // 32 KB
static float   mel_buf[KWS_MAX_LEN][KWS_N_MELS];     // 10 KB

// Sau MaxPool1: int16_t (scale ×128) tiết kiệm 15 KB so với float
static int16_t pool1_out[KWS_CONV1_CH]
                        [KWS_CONV1_OUT_H]
                        [KWS_CONV1_OUT_W];            // 15 KB

// Sau MaxPool2: int16_t tiết kiệm 8 KB
static int16_t pool2_out[KWS_CONV2_CH]
                        [KWS_CONV2_OUT_H]
                        [KWS_CONV2_OUT_W];            //  8 KB

static float feat_buf[KWS_FLATTEN_SIZE];              // 15 KB

// RLIF state
static float mem1[KWS_FC1_UNITS];
static float mem2[KWS_NUM_CLASSES];
static float spk1[KWS_FC1_UNITS];
static float spk2[KWS_NUM_CLASSES];
static float mem2_sum[KWS_NUM_CLASSES];
static float fc1_out[KWS_FC1_UNITS];

static Adafruit_SSD1306 display(OLED_W, OLED_H, &Wire, -1);
static bool oled_ok = false;
static bool led_on_state = false, led_left_state = false, led_right_state = false;

// ── Helpers ──────────────────────────────────────────────────
static inline float deq(uint8_t q, float scale, int zp) {
    return (float)(q - zp) * scale;
}
static inline float rlif(float input, float *mem, float beta, float thresh) {
    *mem = beta * (*mem) + input;
    if (*mem >= thresh) { *mem -= thresh; return 1.0f; }
    return 0.0f;
}
static inline float hz2mel(float hz) { return 2595.0f * log10f(1.0f + hz/700.0f); }
static inline float mel2hz(float mel) { return 700.0f * (powf(10.0f, mel/2595.0f) - 1.0f); }

// ════════════════════════════════════════════════════════════
// 1. MEL SPECTROGRAM
// ════════════════════════════════════════════════════════════
void compute_log_mel() {
    const float mel_min = hz2mel(MEL_FMIN);
    const float mel_max = hz2mel(MEL_FMAX);
    float sum = 0.0f; int cnt = 0;

    for (int fr = 0; fr < KWS_MAX_LEN; fr++) {
        int start = fr * HOP_LENGTH;
        for (int m = 0; m < KWS_N_MELS; m++) {
            if (start + N_FFT > AUDIO_SAMPLES) { mel_buf[fr][m] = 0.0f; continue; }
            float f_c   = mel2hz(mel_min + (m+0.5f)*(mel_max-mel_min)/KWS_N_MELS);
            float omega = 2.0f*(float)M_PI*f_c/SAMPLE_RATE;
            float coeff = 2.0f*cosf(omega);
            float s1=0,s2=0;
            for (int n = 0; n < N_FFT; n++) {
                float win = 0.5f*(1.0f-cosf(2.0f*(float)M_PI*n/(N_FFT-1)));
                float s   = (audio_buf[start+n]/32768.0f)*win;
                float s0  = coeff*s1-s2+s; s2=s1; s1=s0;
            }
            mel_buf[fr][m] = logf(s1*s1+s2*s2-coeff*s1*s2+1e-9f);
            sum += mel_buf[fr][m]; cnt++;
        }
    }
    float mean = sum/cnt, var=0;
    for (int i=0;i<KWS_MAX_LEN;i++) for(int j=0;j<KWS_N_MELS;j++) var+=(mel_buf[i][j]-mean)*(mel_buf[i][j]-mean);
    float inv_std = 1.0f/sqrtf(var/cnt+1e-9f);
    for (int i=0;i<KWS_MAX_LEN;i++) for(int j=0;j<KWS_N_MELS;j++) mel_buf[i][j]=(mel_buf[i][j]-mean)*inv_std;
}

// ════════════════════════════════════════════════════════════
// 2. CNN: mel_buf → feat_buf
//    Tính MaxPool trực tiếp: với mỗi output cell (oh,ow),
//    tính conv trên 2×2 input cells rồi lấy max — không cần
//    lưu toàn bộ pre_pool buffer.
// ════════════════════════════════════════════════════════════
void run_cnn_features() {

    // ── Conv1 + BN1 + ReLU + MaxPool ──────────────────────
    for (int oc = 0; oc < KWS_CONV1_CH; oc++) {
        float bnW   = deq(bn1_weight[oc], bn1_weight_scale, bn1_weight_zp);
        float bnB   = deq(bn1_bias[oc],   bn1_bias_scale,   bn1_bias_zp);
        float bnM   = deq(bn1_mean[oc],   bn1_mean_scale,   bn1_mean_zp);
        float bnStd = sqrtf(deq(bn1_var[oc], bn1_var_scale, bn1_var_zp) + 1e-5f);

        for (int oh = 0; oh < KWS_CONV1_OUT_H; oh++) {
            for (int ow = 0; ow < KWS_CONV1_OUT_W; ow++) {
                float mx = -1e9f;
                // MaxPool 2×2: duyệt qua 4 pre-pool cells
                for (int ph = 0; ph < 2; ph++) {
                    for (int pw = 0; pw < 2; pw++) {
                        int h = oh*2 + ph;
                        int w = ow*2 + pw;
                        float val = deq(conv1_bias[oc], conv1_bias_scale, conv1_bias_zp);
                        for (int kh=-1; kh<=1; kh++) {
                            int ih = h+kh;
                            if (ih<0||ih>=KWS_MAX_LEN) continue;
                            for (int kw=-1; kw<=1; kw++) {
                                int iw = w+kw;
                                if (iw<0||iw>=KWS_N_MELS) continue;
                                val += deq(conv1_weight[oc*9+(kh+1)*3+(kw+1)],
                                           conv1_weight_scale, conv1_weight_zp)
                                       * mel_buf[ih][iw];
                            }
                        }
                        // BN + ReLU
                        val = (val - bnM) / bnStd * bnW + bnB;
                        if (val < 0) val = 0;
                        if (val > mx) mx = val;
                    }
                }
                // Lưu dạng int16: scale ×128, clamp [-256, 255]
                pool1_out[oc][oh][ow] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, mx * 128.0f));
            }
        }
    }

    // ── Conv2 + BN2 + ReLU + MaxPool ──────────────────────
    for (int oc = 0; oc < KWS_CONV2_CH; oc++) {
        float bnW   = deq(bn2_weight[oc], bn2_weight_scale, bn2_weight_zp);
        float bnB   = deq(bn2_bias[oc],   bn2_bias_scale,   bn2_bias_zp);
        float bnM   = deq(bn2_mean[oc],   bn2_mean_scale,   bn2_mean_zp);
        float bnStd = sqrtf(deq(bn2_var[oc], bn2_var_scale, bn2_var_zp) + 1e-5f);

        for (int oh = 0; oh < KWS_CONV2_OUT_H; oh++) {
            for (int ow = 0; ow < KWS_CONV2_OUT_W; ow++) {
                float mx = -1e9f;
                for (int ph = 0; ph < 2; ph++) {
                    for (int pw = 0; pw < 2; pw++) {
                        int h = oh*2+ph, w = ow*2+pw;
                        float val = deq(conv2_bias[oc], conv2_bias_scale, conv2_bias_zp);
                        for (int ic=0; ic<KWS_CONV1_CH; ic++) {
                            for (int kh=-1; kh<=1; kh++) {
                                int ih = h+kh;
                                if (ih<0||ih>=KWS_CONV1_OUT_H) continue;
                                for (int kw=-1; kw<=1; kw++) {
                                    int iw = w+kw;
                                    if (iw<0||iw>=KWS_CONV1_OUT_W) continue;
                                    val += deq(conv2_weight[oc*(KWS_CONV1_CH*9)+ic*9+(kh+1)*3+(kw+1)],
                                               conv2_weight_scale, conv2_weight_zp)
                                           * (pool1_out[ic][ih][iw] / 128.0f);
                                }
                            }
                        }
                        val = (val - bnM) / bnStd * bnW + bnB;
                        if (val < 0) val = 0;
                        if (val > mx) mx = val;
                    }
                }
                pool2_out[oc][oh][ow] = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, mx * 256.0f));
            }
        }
    }

    // ── Flatten + Normalize [0,1] ──────────────────────────
    int idx = 0;
    for (int c=0;c<KWS_CONV2_CH;c++)
        for (int h=0;h<KWS_CONV2_OUT_H;h++)
            for (int w=0;w<KWS_CONV2_OUT_W;w++)
                feat_buf[idx++] = pool2_out[c][h][w] / 256.0f;

    float fmax = 0;
    for (int i=0;i<KWS_FLATTEN_SIZE;i++) if(feat_buf[i]>fmax) fmax=feat_buf[i];
    if (fmax > 1e-9f) { float inv=1.0f/fmax; for(int i=0;i<KWS_FLATTEN_SIZE;i++) feat_buf[i]*=inv; }
}

// ════════════════════════════════════════════════════════════
// 3. RLIF INFERENCE
// ════════════════════════════════════════════════════════════
int run_rlif_inference() {
    memset(mem1,0,sizeof(mem1)); memset(mem2,0,sizeof(mem2));
    memset(spk1,0,sizeof(spk1)); memset(spk2,0,sizeof(spk2));
    memset(mem2_sum,0,sizeof(mem2_sum));

    for (int t = 0; t < KWS_TIME_STEPS; t++) {
        // FC1 + Recurrent1 + RLIF1
        for (int j=0; j<KWS_FC1_UNITS; j++) {
            float cur = deq(fc1_bias[j], fc1_bias_scale, fc1_bias_zp);
            for (int i=0;i<KWS_FLATTEN_SIZE;i++)
                cur += deq(fc1_weight[j*KWS_FLATTEN_SIZE+i], fc1_weight_scale, fc1_weight_zp) * feat_buf[i];
            float rec = deq(rlif1_rec_bias[j], rlif1_rec_bias_scale, rlif1_rec_bias_zp);
            for (int k=0;k<KWS_FC1_UNITS;k++)
                if (spk1[k]>0.5f)
                    rec += deq(rlif1_rec_weight[j*KWS_FC1_UNITS+k], rlif1_rec_weight_scale, rlif1_rec_weight_zp);
            fc1_out[j] = rlif(cur+rec, &mem1[j], RLIF1_BETA, RLIF1_THRESHOLD);
        }
        memcpy(spk1, fc1_out, sizeof(spk1));

        // FC2 + Recurrent2 elementwise + RLIF2
        for (int k=0; k<KWS_NUM_CLASSES; k++) {
            float cur = deq(fc2_bias[k], fc2_bias_scale, fc2_bias_zp);
            for (int j=0;j<KWS_FC1_UNITS;j++)
                if (fc1_out[j]>0.5f)
                    cur += deq(fc2_weight[k*KWS_FC1_UNITS+j], fc2_weight_scale, fc2_weight_zp);
            float rec = deq(rlif2_V[0], rlif2_V_scale, rlif2_V_zp) * spk2[k];
            spk2[k]    = rlif(cur+rec, &mem2[k], RLIF2_BETA, RLIF2_THRESHOLD);
            mem2_sum[k] += mem2[k];
        }
    }

    int best = 0;
    for (int k=1;k<KWS_NUM_CLASSES;k++) if(mem2_sum[k]>mem2_sum[best]) best=k;
    float total=0;
    for (int k=0;k<KWS_NUM_CLASSES;k++) if(mem2_sum[k]>0) total+=mem2_sum[k];
    if (total<1e-6f) return -1;
    if (mem2_sum[best]/total < CONF_THRESHOLD) return -1;
    if (strcmp(KWS_LABELS[best],"unknown")==0) return -1;
    return best;
}

// ════════════════════════════════════════════════════════════
// VAD
// ════════════════════════════════════════════════════════════
bool vad_detected() {
  long sum = 0;
  for (int i = 0; i < AUDIO_SAMPLES; i++)
    sum += abs(audio_buf[i]); 
  
  long avg = sum / AUDIO_SAMPLES;
  Serial.printf("VAD Avg: %ld\n", avg); // <--- Thêm dòng này để xem số thực tế
  return avg > VAD_THRESHOLD; 
}

// ════════════════════════════════════════════════════════════
// OLED
// ════════════════════════════════════════════════════════════
void oled_ready() {
    if (!oled_ok) return;
    display.clearDisplay(); display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0,0);  display.println("KWS RLIF  Ready");
    display.setCursor(0,14); display.println("Say: on / off");
    display.setCursor(0,24); display.println("     left / right");
    display.setCursor(0,44); display.println("Listening...");
    display.display();
}
void oled_processing() {
    if (!oled_ok) return;
    display.clearDisplay(); display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0,20); display.println("  Processing...");
    display.display();
}
void oled_result(int kw_idx, float conf) {
    if (!oled_ok) return;
    display.clearDisplay(); display.setTextColor(SSD1306_WHITE);
    display.setTextSize(3);
    const char* kw = KWS_LABELS[kw_idx];
    int16_t x1,y1; uint16_t tw,th;
    display.getTextBounds(kw,0,0,&x1,&y1,&tw,&th);
    display.setCursor((OLED_W-tw)/2,4); display.print(kw);
    display.setTextSize(1); display.setCursor(0,42);
    display.printf("conf: %d%%",(int)(conf*100));
    display.fillRect(0,54,(int)(conf*OLED_W),8,SSD1306_WHITE);
    display.display();
}
void oled_unknown() {
    if (!oled_ok) return;
    display.clearDisplay(); display.setTextSize(2);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(20,20); display.println("???");
    display.display();
}

// ════════════════════════════════════════════════════════════
// Keyword handler
// ════════════════════════════════════════════════════════════
void handle_keyword(int kw_idx) {
    const char* kw = KWS_LABELS[kw_idx];
    float total=0;
    for (int k=0;k<KWS_NUM_CLASSES;k++) if(mem2_sum[k]>0) total+=mem2_sum[k];
    float conf = (total>0) ? mem2_sum[kw_idx]/total : 0;

    Serial.printf(">>> [%s]  conf=%d%%\n", kw, (int)(conf*100));

    if      (strcmp(kw,"on")   ==0) { led_on_state=true;  digitalWrite(PIN_LED_ON,HIGH); }
    else if (strcmp(kw,"off")  ==0) { led_on_state=false; digitalWrite(PIN_LED_ON,LOW);  }
    else if (strcmp(kw,"left") ==0) { led_left_state=!led_left_state;   digitalWrite(PIN_LED_LEFT, led_left_state ?HIGH:LOW); }
    else if (strcmp(kw,"right")==0) { led_right_state=!led_right_state; digitalWrite(PIN_LED_RIGHT,led_right_state?HIGH:LOW); }

    oled_result(kw_idx, conf);
}

// ════════════════════════════════════════════════════════════
// I2S
// ════════════════════════════════════════════════════════════
void setup_i2s() {
    i2s_config_t cfg = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER|I2S_MODE_RX),
        .sample_rate          = SAMPLE_RATE,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = 512,
        .use_apll             = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num   = PIN_I2S_SCK,
        .ws_io_num    = PIN_I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num  = PIN_I2S_SD,
    };
    i2s_driver_install(I2S_NUM_0,&cfg,0,NULL);
    i2s_set_pin(I2S_NUM_0,&pins);
    i2s_zero_dma_buffer(I2S_NUM_0);
}

void read_audio() {
    static int32_t tmp[512]; size_t br; int got=0;
    while (got<AUDIO_SAMPLES) {
        int n=min(512,AUDIO_SAMPLES-got);
        i2s_read(I2S_NUM_0,tmp,n*sizeof(int32_t),&br,portMAX_DELAY);
        int rn=br/sizeof(int32_t);
        for(int i=0;i<rn&&got<AUDIO_SAMPLES;i++,got++) audio_buf[got]=(int16_t)(tmp[i]>>16);
    }
}

// ════════════════════════════════════════════════════════════
// Setup & Loop
// ════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200); delay(200);
    Serial.println("\n=== KWS RLIF ESP32 ===");
    Serial.printf("Classes=%d  T=%d  Flatten=%d\n", KWS_NUM_CLASSES, KWS_TIME_STEPS, KWS_FLATTEN_SIZE);

    pinMode(PIN_LED_ON,   OUTPUT); digitalWrite(PIN_LED_ON,  LOW);
    pinMode(PIN_LED_LEFT, OUTPUT); digitalWrite(PIN_LED_LEFT,LOW);
    pinMode(PIN_LED_RIGHT,OUTPUT); digitalWrite(PIN_LED_RIGHT,LOW);

    Wire.begin(21,22);
    oled_ok = display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
    if (oled_ok) { display.clearDisplay(); display.display(); Serial.println("OLED OK"); }
    else           Serial.println("OLED not found");

    setup_i2s();

    for (int p : {PIN_LED_ON,PIN_LED_LEFT,PIN_LED_RIGHT}) { digitalWrite(p,HIGH); delay(100); }
    delay(200);
    for (int p : {PIN_LED_ON,PIN_LED_LEFT,PIN_LED_RIGHT}) digitalWrite(p,LOW);

    Serial.printf("Free heap: %d bytes\n",(int)ESP.getFreeHeap());
    oled_ready();
    Serial.println("Ready. Listening...");
}

static unsigned long last_kw_ms = 0;

void loop() {
    read_audio();
    if (!vad_detected()) {
        static unsigned long last_idle=0;
        if (millis()-last_idle>8000) { Serial.println("... (silence)"); last_idle=millis(); }
        return;
    }
    Serial.println("Speech detected!");
    oled_processing();

    unsigned long t0=millis();
    compute_log_mel();
    run_cnn_features();
    int kw=run_rlif_inference();
    Serial.printf("Inference: %lu ms\n", millis()-t0);

    if (kw>=0) {
        unsigned long now=millis();
        if (now-last_kw_ms>DEBOUNCE_MS) { handle_keyword(kw); last_kw_ms=now; }
        else Serial.println("(debounce skip)");
    } else {
        Serial.println("(unknown / low conf)");
        oled_unknown(); delay(600); oled_ready();
    }
}
