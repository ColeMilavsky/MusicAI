import librosa
import numpy as np
import tensorflow as tf

# Load the two keras models
genre_model = tf.keras.models.load_model('models/modelFinals/GTZANGenreClassifier.keras', safe_mode=False)
metadata_model = tf.keras.models.load_model('models/modelFinals/GTZANMetadataClassifier.keras', safe_mode=False)

# genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def run_song_analysis(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, mono=True, duration=30)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=256, n_fft=512, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec**2)
        mel_spec_db = resize_spectrogram(mel_spec_db, 250)  # custom function to pad/crop width
        input_data = mel_spec_db.reshape((1, 64, 250, 1))

        # Predict genre
        genre_probs = genre_model.predict(input_data)
        predicted_genre_idx = np.argmax(genre_probs)
        predicted_genre = genres[predicted_genre_idx]

        # Predict metadata
        metadata_pred = metadata_model.predict(input_data).flatten()

        # Print results
        print(f"🎵 **Predicted Genre:** {predicted_genre}")
        print(f"🎧 **Metadata Predictions:**")
        for i, value in enumerate(metadata_pred):
            print(f"  Feature {i + 1}: {value:.4f}")

    except Exception as e:
        print(f"Error analyzing file: {e}")

# Ensure proper size for input spectrogram
def resize_spectrogram(spec, target_width):
    current_width = spec.shape[1]
    if current_width < target_width:
        pad_width = target_width - current_width
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spec = spec[:, :target_width]
    return spec