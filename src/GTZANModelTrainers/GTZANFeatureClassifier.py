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
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf

import random

import joblib

def main(args):
    return 0

if __name__ == '__main__':

    GTZAN_WAV_data = f'data/input/GTZAN/genres_original'
    model_path = f'models/modelFinals'

    # Import GTZAN metadata features and drop filename as it won't be needed
    metadata = pd.read_csv('data/input/GTZAN/features_3_sec.csv')
    metadata = metadata.drop(columns=['filename'])

    # Encode genre labels as digits
    encoder = LabelEncoder()
    metadata['label'] = encoder.fit_transform(metadata['label'])

    # shuffle metadata rows to improve learning
    metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

    cut = int(len(metadata) * 0.8)

    train = metadata.iloc[cut:]
    test = metadata.iloc[:cut]

    # Separate features and labels
    X_train = train.drop(columns=['label'])
    Y_train = train['label']

    X_test = test.drop(columns=['label'])
    Y_test = test['label']

    # Normalize all features
    X_train_norm = StandardScaler().fit_transform(X_train)
    X_test_norm = StandardScaler().fit_transform(X_test)

    ######################################################################################################################################
    ### Begin Random Forest Training #####################################################################################################
    ######################################################################################################################################
    RandForestModel = RandomForestClassifier(n_estimators=100, random_state=42)
    RandForestModel.fit(X_train, Y_train)

    # Create predictions, test accuracy
    Y_pred = RandForestModel.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(Y_test, Y_pred, target_names=encoder.classes_))

    # Save model
    joblib.dump(RandForestModel, 'models/modelFinals/GTZANGenreFeatureClassifier.joblib')

    sys.exit(main(sys.argv))