
from pathlib import Path
import librosa
import matplotlib.pyplot as plt

# 1. Dataset qovluğu
DATA_DIR = Path("data")

# 2. İlk audio faylını tap
audio_files = sorted(DATA_DIR.rglob("*.wav"))

if not audio_files:
    raise FileNotFoundError("No WAV files found. Check DATA_DIR.")

audio_path = audio_files[0]
print("Audio file:", audio_path)

# 3. Audio faylını oxu
signal, sr = librosa.load(audio_path, sr=None)

print("Sample rate:", sr)
print("Duration:", len(signal) / sr, "seconds")
print("Number of samples:", len(signal))

# 4. Audio dalğa formasını göstər
time = [i / sr for i in range(len(signal))]

plt.figure(figsize=(12, 4))
plt.plot(time, signal)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.tight_layout()
plt.show()

# 5. MFCC xüsusiyyətlərini çıxar
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=13
)

# 6. Nəticənin ölçüsünü göstər
print("MFCC shape:", mfcc.shape)