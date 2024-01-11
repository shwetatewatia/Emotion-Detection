import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import pyaudio
from tensorflow.keras.models import load_model, model_from_json
import msvcrt


# Load your existing emotion recognition model
# Replace 'your_model_path.h5' with the actual path to your trained model

def SpeechEmotionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

model = SpeechEmotionModel("model_a_on_5th_jan.json","model_on_5th_jan.h5")

#data augmentation 
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data,sr=sampling_rate,n_steps=pitch_factor)

def frequency_masking(data):

    stft = librosa.stft(data)
    stft_magnitude, _ = librosa.magphase(stft)
    audio_after_masking = librosa.istft(stft * stft_magnitude)
    
    return audio_after_masking






# Function to extract features from audio
# Function to extract features from audio
# def extract_features(audio, sr):
#     audio = audio.astype(np.float32)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=162)
#     features = np.mean(mfccs, axis=1)
#     return np.expand_dims(features, axis=1)

def extract_features(data,sample_rate):
    
    audio_data_float = librosa.util.normalize(data.astype(np.float32))
    # ZCR
    
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_data_float).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(audio_data_float))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data_float, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    
    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=audio_data_float).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=audio_data_float, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    #spectral centroid
    # spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data_float,sr = sample_rate).T,axis=0)
    # result = np.hstack((result,spec_centroid))
    
    #spectral bandwidth
    # spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data_float,sr=sample_rate).T ,axis=0)
    # result = np.hstack((result,spectral_bandwidth))
    
    return result

def get_features(data,sample_rate):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    # data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    data_float = librosa.util.normalize(data.astype(float))
    # without augmentation
    res1 = extract_features(data_float,sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data_float)
    res2 = extract_features(noise_data,sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
       
    # data with stretching and pitching
    new_data = stretch(data_float)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch,sample_rate)
    result = np.vstack((result, res3)) # stacking vertically
    
    # data_ = frequency_masking(data_float)
    # res4 = extract_features(data_,sample_rate)
    # result=np.vstack((result,res4))
    
    
    return result


# Function to predict emotion from audio
def predict_emotion(audio_data,sr):
    features = get_features(audio_data,sr)
    features = np.expand_dims(features, axis=-1)  # Add batch dimension
    emotion_probabilities = model.predict(features)
    emotion_index = np.argmax(emotion_probabilities)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral","Surprise", "Sad","Calm"]
    if 0 <= emotion_index < len(emotions):
        predicted_emotion = emotions[emotion_index]
    else:
        # Handle the case where emotion_index is out of range
        predicted_emotion = "Unknown"
    # predicted_emotion = emotions[emotion_index]
    return predicted_emotion

# Real-time emotion recognition using microphone
def real_time_emotion_recognition():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8025

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for emotion...")

    try:
        while True:
            audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
            predicted_emotion = predict_emotion(audio_data, RATE)
            print("Predicted Emotion:", predicted_emotion)
            
            # Check if any key is pressed to exit the program
            if msvcrt.kbhit() and msvcrt.getch() == b'q':
                print("Key 'q' pressed. Exiting...")
                break


    except KeyboardInterrupt:
        print("Stopped by user.")
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    real_time_emotion_recognition()