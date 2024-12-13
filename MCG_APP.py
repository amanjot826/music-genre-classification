import streamlit as st
import numpy as np
from keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

# Load the pre-trained model
MODEL_PATH = '/Users/srivenikunduru/Downloads/model_cnn3.keras'
model = load_model(MODEL_PATH)

# Define genre mapping based on training logic
genres = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]
genre_map = dict(zip(genres, range(len(genres))))
reversed_genre_map = {v: k for k, v in genre_map.items()}

# Function to preprocess audio file
def preprocess_audio(file_path, fs=22050, duration=30, n_fft=2048, hop_length=512, n_mfcc=13, num_segments=10):
    samples_per_track = fs * duration
    samps_per_segment = int(samples_per_track / num_segments)
    mfccs_per_segment = int(np.ceil(samps_per_segment / hop_length))

    audio, _ = librosa.load(file_path, sr=fs, duration=duration)
    mfccs = []

    for seg in range(num_segments):
        start_sample = seg * samps_per_segment
        end_sample = start_sample + samps_per_segment
        mfcc = librosa.feature.mfcc(
            y=audio[start_sample:end_sample], sr=fs, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
        )
        mfcc = mfcc.T
        if len(mfcc) == mfccs_per_segment:
            mfccs.append(mfcc.tolist())

    return np.array(mfccs)

# Streamlit app setup
st.title("Music Genre Classification with Keras CNN Model")
st.write("Upload an audio file to classify its music genre using the CNN model.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    file_path = f"temp_audio_file.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Preprocess the audio file
    processed_audio = preprocess_audio(file_path)

    # Reshape for model prediction
    processed_audio = np.expand_dims(processed_audio, axis=-1)

    # Make predictions
    predictions = model.predict(processed_audio)
    st.write("Raw Predictions:", predictions)  # Debugging

    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_genre = reversed_genre_map.get(predicted_class, "Unknown")

    # Display result
    st.write("Predicted Genre:", predicted_genre)
    st.write("Prediction Confidence:", predictions[0])

    # Optional: Visualize the audio waveform
    audio, sr = librosa.load(file_path, sr=22050, duration=30.0)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    st.pyplot(plt)
else:
    st.write("Please upload an audio file.")
