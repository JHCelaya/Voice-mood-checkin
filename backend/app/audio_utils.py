# Starting with this file. imports and simple test to
# make sure we can process the audio clip samples for testing

import librosa
import numpy as np

# test loading an audio file
audio, sr = librosa.load("03-01-01-01-01-01-01.wav", sr=22050)
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sr}")

# Test feature extraction
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print(f"MFCCs shape: {mfccs.shape}")
print(f"MFCCs mean: {np.mean(mfccs, axis=1)}")