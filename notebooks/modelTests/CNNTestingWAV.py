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
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda, LSTM, Reshape
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

import random

def main(args):
    return 0

if __name__ == '__main__':

    GTZAN_WAV_data = f'data/input/GTZAN/genres_original'

    # Array of genres
    # genres = [f for f in os.listdir(GTZAN_WAV_data) if os.path.isdir(os.path.join(GTZAN_WAV_data, f))]
    genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

    dataset = []

    err_count = 0

    for genre, genre_number in genres.items():
        for filename in os.listdir(f'{GTZAN_WAV_data}/{genre}'):
            songname = f'{GTZAN_WAV_data}/{genre}/{filename}'
            for index in range(14):
                try:
                    y, sr = librosa.load(songname, mono=True, duration=2, offset=index*2)
                except Exception as e:
                    print(f"Error loading {songname}: {e}")
                    err_count += 1
                    continue
                ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length = 256, n_fft = 512, n_mels=64)
                ps = librosa.power_to_db(ps**2)
                dataset.append( (ps, genre_number) )

    random.shuffle(dataset)

    cut_start = 10000 - err_count
    cut_end = 12000 - err_count

    train = dataset[:cut_start]
    valid = dataset[cut_start:cut_end]
    test = dataset[cut_end:]

    X_train, Y_train = zip(*train)
    X_valid, Y_valid = zip(*valid)
    X_test, Y_test = zip(*test)

    # Reshape for CNN input
    X_train = np.array([x.reshape( (64, 173, 1) ) for x in X_train])
    X_valid = np.array([x.reshape( (64, 173, 1) ) for x in X_valid])
    X_test = np.array([x.reshape( (64, 173, 1) ) for x in X_test])

    # One-Hot encoding for classes
    Y_train = np.array(keras.utils.to_categorical(Y_train, 10))
    Y_valid = np.array(keras.utils.to_categorical(Y_valid, 10))
    Y_test = np.array(keras.utils.to_categorical(Y_test, 10))

    #####################################################################################################################################
    ### Create new CNN model ############################################################################################################
    #####################################################################################################################################
    CNNmodel = Sequential()

    CNNmodel.add(Conv2D(20, (5, 5), input_shape=(64, 173, 1), activation="relu", strides=1, padding="valid"))
    CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))

    CNNmodel.add(Conv2D(50, (5, 5), use_bias=50))
    CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))

    CNNmodel.add(Flatten())
                 
    CNNmodel.add(Dense(20, activation="relu"))
    CNNmodel.add(Lambda(lambda x: backend.expand_dims(x, axis=-1)))

    CNNmodel.add(LSTM(512, activation="relu", return_sequences=False))

    CNNmodel.add(Dense(10, activation = "softmax"))
    ######################################################################################################################################

    CNNmodel.summary()

    # Compile the CNN model
    CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),   # Stops training when validation loss doesn't improve for 5 epochs
        ReduceLROnPlateau(patience=3)   # If the model gets stuck lower learning rate to help escape plateaus
    ]

    # Fit the model to our data, save the accuracy and loss
    CNNmodel.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=64,
        validation_data= (X_test, Y_test),
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    ######################################## Analyze Model Performance ##############################################
    

    sys.exit(main(sys.argv))