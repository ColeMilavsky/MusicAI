import sys

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
from pathlib import Path
import csv

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

import keras
from keras import backend
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda, LSTM, Reshape, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

import random

def change_width(spec, width):
    if spec.shape[1] > width:
        return spec[:, :width]
    elif spec.shape[1] < width:
        pad_width = width - spec.shape[1]
        return np.pad(spec, ((0,0), (0, pad_width)), mode='constant')
    else:
        return spec

def main(args):
    return 0

if __name__ == '__main__':

    GTZAN_WAV_data = f'data/input/GTZAN/genres_original'
    model_path = f'models/modelFinals'

    # Import metadata and drop filename + genre
    metadata = pd.read_csv('data/input/GTZAN/features_3_sec.csv')
    #metadata = metadata.drop(columns=['filename', 'label'])

    # Array of genres
    genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

    # Collect dataset of audio files to create a regression CNN model of
    dataset = []
    successful_files = []
    for genre in genres.keys():
        for filename in os.listdir(f'{GTZAN_WAV_data}/{genre}'):
            songname = f'{GTZAN_WAV_data}/{genre}/{filename}'
            # cut into 3 sec slices
            for index in range(10):
                try:
                    y, sr = librosa.load(songname, mono=True, duration=3, offset=index*3)
                except Exception as e:
                    print(f"Error loading {songname}: {e}")
                    continue
                ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=256, n_fft=512, n_mels=64)
                ps = librosa.power_to_db(ps**2)
                dataset.append(ps)

                successful_files.append(f"{filename[:-4]}.{index}.wav")

    # Filter metadata to keep only rows with successful audio
    metadata_filtered = metadata[metadata['filename'].isin(successful_files)]

    # Combine audio file dataset and metadata for shuffling, metadata stored row by row in a dictionary
    combined = list(zip(dataset, metadata_filtered.to_dict(orient='records')))
    random.shuffle(combined)

    # Seperate shuffled information and restructure data
    dataset_shuffled, metadata_shuffled = zip(*combined)
    dataset_shuffled = list(dataset_shuffled)
    metadata_shuffled = pd.DataFrame(metadata_shuffled)

    # No longer need id column
    metadata_shuffled = metadata_shuffled.drop(columns=['filename', 'label'])

    # Normalize metadata targets
    scaler = StandardScaler()
    metadata_scaled = pd.DataFrame(scaler.fit_transform(metadata_shuffled), columns=metadata_shuffled.columns)

    # Split metadata into train and test
    cut = int(len(metadata_shuffled) * 0.8)
    Y_train = metadata_scaled.iloc[:cut].to_numpy()
    Y_test = metadata_scaled.iloc[cut:].to_numpy()

    # Define train and test arrays, ensure all 3 sec slices are the same width
    X_train = np.array([change_width(x, 250).reshape((64, 250, 1)) for x in dataset_shuffled[:cut]])
    X_test = np.array([change_width(x, 250).reshape((64, 250, 1)) for x in dataset_shuffled[cut:]])

    #####################################################################################################################################
    ### Create Functional CNN model ############################################################################################################
    #####################################################################################################################################    
    inputs = Input(shape=(64, 250, 1))
    x = Conv2D(20, (5, 5), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(50, (5, 5))(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(20, activation="relu")(x)
    x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)
    x = LSTM(512)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(58, activation="linear")(x)

    CNNmodel = Model(inputs=inputs, outputs=outputs)

    CNNmodel.summary()

    # Compile the CNN model
    CNNmodel.compile(optimizer='adam', loss='mse', metrics=['mae'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),   # Stops training when validation loss doesn't improve for 5 epochs
        ReduceLROnPlateau(patience=3)                           # If the model gets stuck lower learning rate to help escape plateaus
    ]

    # Fit the model to our data, save the accuracy and loss
    history = CNNmodel.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=64,
        validation_data= (X_test, Y_test),
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    CNNmodel.save(model_path + '/GTZANMetadataClassifier.keras')

    ######################################## Analyze Model Performance ##############################################
    # Get a preidction and rescale (un-normalize) values
    y_pred = CNNmodel.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/figures/figureFinals/GenreClassifierAccuracy.png')

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/figures/figureFinals/GenreClassifierLoss.png')
    

    sys.exit(main(sys.argv))