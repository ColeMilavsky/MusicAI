import librosa
import numpy as np
import tensorflow as tf
#import keras

# Load the two keras models
# genre_model = tf.keras.models.load_model('models/modelFinals/GTZANGenreClassifier.keras', compile=False, safe_mode=False)
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
        # genre_probs = genre_model.predict(input_data)
        # predicted_genre_idx = np.argmax(genre_probs)
        # predicted_genre = genres[predicted_genre_idx]

        # Predict metadata
        metadata_pred = metadata_model.predict(input_data).flatten()

        feature_labels = [
            'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
            'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean', 'mfcc3_var',
            'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var',
            'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean', 'mfcc9_var',
            'mfcc10_mean', 'mfcc10_var', 'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var',
            'mfcc13_mean', 'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
            'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean', 'mfcc18_var',
            'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
        ]

        # Print results
        #print(f"**Predicted Genre:** {predicted_genre}")
        print(f"**Metadata Predictions:**")
        for label, value in zip(feature_labels, metadata_pred):
            print(f"{label}: {value:.4f}")

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