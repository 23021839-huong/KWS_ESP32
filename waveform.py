import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import os

# =====================================================
# CONFIG
# =====================================================

AUDIO_PATH = "D:/lab/kws_sg_snn/data/SpeechCommands/speech_commands_v0.02/right/00b01445_nohash_0.wav"

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 40
FMIN = 80
FMAX = 7600

SAVE_DIR = "output_right"

os.makedirs(SAVE_DIR, exist_ok=True)
# LOAD AUDIO

y, sr = librosa.load(AUDIO_PATH, sr=SR)

# 1. Waveform

plt.figure(figsize=(10,3))

librosa.display.waveshow(
    y,
    sr=sr,
    color="blue"
)

plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR, "1_waveform.png"), dpi=300)

# 2. FFT

N = len(y)

fft = np.abs(rfft(y))

freq = rfftfreq(N, 1/sr)

plt.figure(figsize=(10,3))

plt.plot(freq, fft)

plt.title("FFT Spectrum")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0,8000)

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR,"2_fft.png"),dpi=300)

# 3. Spectrogram

D = librosa.amplitude_to_db(

    np.abs(

        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP
        )

    ),

    ref=np.max

)

plt.figure(figsize=(10,4))

librosa.display.specshow(

    D,

    sr=sr,

    hop_length=HOP,

    x_axis="time",

    y_axis="linear"

)

plt.colorbar(format="%+2.f dB")

plt.title("Spectrogram")

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR,"3_spectrogram.png"),dpi=300)

# 4. Mel Spectrogram

mel = librosa.feature.melspectrogram(

    y=y,

    sr=sr,

    n_fft=N_FFT,

    hop_length=HOP,

    n_mels=N_MELS,

    fmin=FMIN,

    fmax=FMAX

)

plt.figure(figsize=(10,4))

librosa.display.specshow(

    mel,

    sr=sr,

    hop_length=HOP,

    x_axis="time",

    y_axis="mel"

)

plt.colorbar()

plt.title("Mel Spectrogram")

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR,"4_mel.png"),dpi=300)

# 5. Log-Mel Spectrogram

logmel = np.log(mel + 1e-6)

plt.figure(figsize=(10,4))

librosa.display.specshow(

    logmel,

    sr=sr,

    hop_length=HOP,

    x_axis="time",

    y_axis="mel"

)

plt.colorbar()

plt.title("Log-Mel Spectrogram")

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR,"5_logmel.png"),dpi=300)

# 6. Figure tổng hợp

fig, ax = plt.subplots(3,2, figsize=(14,12))

# Waveform
librosa.display.waveshow(
    y,
    sr=sr,
    ax=ax[0,0]
)

ax[0,0].set_title("Waveform")

# FFT
ax[0,1].plot(freq, fft)

ax[0,1].set_xlim(0,8000)

ax[0,1].set_title("FFT Spectrum")

# Spectrogram
img = librosa.display.specshow(

    D,

    sr=sr,

    hop_length=HOP,

    x_axis="time",

    y_axis="linear",

    ax=ax[1,0]

)

ax[1,0].set_title("Spectrogram")

fig.colorbar(img, ax=ax[1,0])

# Mel

img = librosa.display.specshow(

    mel,

    sr=sr,

    hop_length=HOP,

    x_axis="time",

    y_axis="mel",

    ax=ax[1,1]

)

ax[1,1].set_title("Mel Spectrogram")

fig.colorbar(img, ax=ax[1,1])

# LogMel

img = librosa.display.specshow(

    logmel,

    sr=sr,

    hop_length=HOP,

    x_axis="time",

    y_axis="mel",

    ax=ax[2,0]

)

ax[2,0].set_title("Log-Mel Spectrogram")

fig.colorbar(img, ax=ax[2,0])

# Pipeline

ax[2,1].axis("off")

ax[2,1].text(

    0.05,

    0.5,

    "Speech\n↓\nWaveform\n↓\nFFT\n↓\nMel Filter Bank\n↓\nLog-Mel Spectrogram",

    fontsize=16

)

plt.tight_layout()

plt.savefig(os.path.join(SAVE_DIR,"6_pipeline.png"),dpi=300)

plt.show()

print("Done!")
print("Images saved in:", SAVE_DIR)