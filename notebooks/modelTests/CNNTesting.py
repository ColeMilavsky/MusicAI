import sys
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

def main(args):
    return 0

if __name__ == '__main__':

    # Check if TensorFlow was built with CUDA
    print("CUDA built:", tf.test.is_built_with_cuda())
    # Check if GPU is available
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

    if tf.config.list_physical_devices('GPU') == []:
        sys.exit()

    # Data Path
    data_path = Path("data")

    # Input/Output Paths
    input_path = Path(data_path/"input")
    output_path = Path(data_path/"output")

    # GTZAN Paths
    GTZAN = Path(input_path/"GTZAN")
    GTZAN_img_data = Path(GTZAN/"images_original")

    # Array of genres
    genres = [f for f in os.listdir(GTZAN_img_data) if os.path.isdir(os.path.join(GTZAN_img_data, f))]

    # Get the number of folders in the genre folder location
    num_genres = len([f for f in os.listdir(GTZAN_img_data) if os.path.isdir(os.path.join(GTZAN_img_data, f))])

    img_height, img_width = 288, 432
    GTZAN_img_train, GTZAN_ID_train = [], []
    GTZAN_img_test, GTZAN_ID_test = [], []

    for classID, genre in enumerate(genres):
        genre_path = GTZAN_img_data / genre
        img_files = sorted(glob(str(genre_path / '*.png')))  # auto-detects all .png files

        for j, img_path in enumerate(img_files):
            img = Image.open(img_path).resize((img_width, img_height))
            img_array = np.array(img).astype('float32')

            if j < 90:
                GTZAN_img_train.append(img_array)
                GTZAN_ID_train.append(classID)
            else:
                GTZAN_img_test.append(img_array)
                GTZAN_ID_test.append(classID)

    # Create new CNN model
    CNNmodel = Sequential([
        # Input shape: (288, 432, 4) — match your resized image + RGBA
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(288, 432, 4)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        #Conv2D(64, (3, 3), activation='relu', padding='same'),
        #BatchNormalization(),
        #MaxPooling2D(pool_size=(2, 2)),

        #Conv2D(128, (3, 3), activation='relu', padding='same'),
        #BatchNormalization(),
        #MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        #Dense(128, activation='relu'),
        #Dropout(0.5),

        Dense(32, activation='relu'),
        Dropout(0.3),

        Dense(10, activation='softmax')
    ])

    # Convert arrays to np arrays
    GTZAN_img_train = np.array(GTZAN_img_train).astype('float32')
    GTZAN_img_test = np.array(GTZAN_img_test).astype('float32')
    GTZAN_ID_train = np.array(GTZAN_ID_train)
    GTZAN_ID_test = np.array(GTZAN_ID_test)

    # One hot encode ID arrays
    GTZAN_ID_train = to_categorical(GTZAN_ID_train, num_classes=10)
    GTZAN_ID_test = to_categorical(GTZAN_ID_test, num_classes=10)

    # Compile the CNN model
    CNNmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),   # Stops training when validation loss doesn't improve for 5 epochs
        ReduceLROnPlateau(patience=3)   # If the model gets stuck lower learning rate to help escape plateaus
    ]

    # Fit the model to our data, save the accuracy and loss
    CNNmodel.fit(
        GTZAN_img_train,
        GTZAN_ID_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    ######################################## Test on Testing Data ##############################################
    preds = CNNmodel.predict(GTZAN_img_test, batch_size=64)

    sys.exit(main(sys.argv))