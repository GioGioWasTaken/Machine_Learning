import librosa
import matplotlib.pyplot as plt
import numpy as np

song_path_1= r"C:\Users\USER1\PycharmProjects\Machine_Learning\Learning_Librosa\Songs\Live And Let Die.mp3"
song_path_2=r"C:\Users\USER1\PycharmProjects\Machine_Learning\Learning_Librosa\Songs\It's A Sin.mp3"

def extract_from_song(song_path):
    audio_signal, sample_rate=librosa.load(song_path)
    y, sr = librosa.load(song_path)
    print('Audio signal shape:', audio_signal.shape)
    print('Sample rate:', sample_rate)
    Fourier_trans=librosa.stft(audio_signal)
    Fourier_trans=np.absolute(Fourier_trans)
    spectrogram=librosa.power_to_db(Fourier_trans)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return audio_signal, sample_rate, spectrogram, tempo

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# extract and plot spectrogram for first song on first subplot
audio_signal_1, sample_rate_1, spectrogram_1,tempo_1 = extract_from_song(song_path_1)
ax1.imshow(spectrogram_1)
# extract and plot spectrogram for second song on second subplot
audio_signal_2, sample_rate_2, spectrogram_2, tempo_2 = extract_from_song(song_path_2)
ax2.imshow(spectrogram_2)

# set title for each subplot
ax1.set_title('Live and Let Die')
ax2.set_title("It's a Sin")
# shortening it so i can check for hamming distance
audio_signal_2=audio_signal_2[0:4029952]
print(len(audio_signal_1),len(audio_signal_2))

# hamming distance between the audio signals
hamming_dist = np.count_nonzero(audio_signal_1 != audio_signal_2)
print(f"Hamming distance: {hamming_dist}")

# show the figure
plt.show()

# This code is nothing but an excuse to do something with hamming distance, and to check how librosa operates.