import librosa
import librosa.display
import matplotlib.pyplot as plt

# Librosa-nın daxili nümunəsi (ilk dəfə internetdən yüklənir)
path = librosa.example("trumpet")

y, sr = librosa.load(path, sr=22050)

print("Sample rate:", sr)
print("Siqnal uzunluğu (nümunə sayı):", len(y))
print("Müddət (saniyə):", round(len(y) / sr, 2))

plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.xlabel("Vaxt (s)")
plt.ylabel("Amplituda")
plt.tight_layout()
plt.show()