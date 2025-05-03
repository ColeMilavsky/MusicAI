import numpy as np

import tensorflow as tf
from keras.models import load_model
import joblib

CNN_model = load_model('keras_model.keras')
RandForest_model = joblib.load('rf_model.joblib')

# Prepare inputs
audio_input = prepare_audio_input()   # shape e.g., (1, 64, 173, 1)
metadata_input = prepare_metadata_input()  # shape e.g., (1, 60)

# Get predictions
keras_pred = CNN_model.predict(audio_input)
rf_pred = RandForest_model.predict_proba(metadata_input)

# Combine predictions (weighted average)
final_pred = 0.6 * keras_pred + 0.4 * rf_pred

# Get final label
final_label = np.argmax(final_pred)