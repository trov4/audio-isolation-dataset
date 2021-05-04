import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import glob
import os
import math
import tensorflow as tf

from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

#Code credit: Leciano Strika 
#https://www.kdnuggets.com/2019/07/convolutional-neural-networks-python-tutorial-tensorflow-keras.html

songs = './data/images/songs'
vox = './data/images/vox'
IMG_SIZE = (640, 480)
def pixels_from_path(file_path):
    im = Image.open(file_path)
    im = im.resize(IMG_SIZE)
    np_im = np.array(im)
    #matrix of pixel RGB values
    return np_im


SAMPLE_SIZE = 131
print("loading training songs images...")
songs_train_set = np.asarray([pixels_from_path(songs) for songs in glob.glob('./data/images/songs/*')[:SAMPLE_SIZE]])
print("loading training vox images...")
vox_train_set = np.asarray([pixels_from_path(vox) for vox in glob.glob('./data/images/vox/*')[:SAMPLE_SIZE]])

valid_size = 22
print("loading validation songs images...")
songs_valid_set = np.asarray([pixels_from_path(songs) for songs in glob.glob('./data/images/songs/*')[-valid_size:]])
print("loading validation vox images...")
vox_valid_set = np.asarray([pixels_from_path(vox) for vox in glob.glob('./data/images/vox/*')[-valid_size:]])

print("vox_train", songs_train_set)

# generate X and Y (inputs and labels).
x_train = np.concatenate([songs_train_set, vox_train_set])
labels_train = np.asarray([1 for _ in range(SAMPLE_SIZE)]+[0 for _ in range(SAMPLE_SIZE)])

x_valid = np.concatenate([songs_valid_set, vox_valid_set])
labels_valid = np.asarray([1 for _ in range(valid_size)]+[0 for _ in range(valid_size)])

fc_layer_size = 128
img_size = IMG_SIZE

conv_inputs = keras.Input(shape=(img_size[1], img_size[0],4), name='ani_image')
conv_layer = layers.Conv2D(24, kernel_size=3, activation='relu')(conv_inputs)
conv_layer = layers.MaxPool2D(pool_size=(2,2))(conv_layer)
conv_x = layers.Flatten(name = 'flattened_features')(conv_layer) #turn image to vector.

conv_x = layers.Dense(fc_layer_size, activation='relu', name='first_layer')(conv_x)
conv_x = layers.Dense(fc_layer_size, activation='relu', name='second_layer')(conv_x)
conv_outputs = layers.Dense(1, activation='sigmoid', name='class')(conv_x)

conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)

customAdam = keras.optimizers.Adam(lr=1e-6)
conv_model.compile(optimizer=customAdam,  # Optimizer
              # Loss function to minimize
              loss="binary_crossentropy",
              # List of metrics to monitor
              metrics=["binary_crossentropy","mean_squared_error", "accuracy"])


history = conv_model.fit(x_train, 
                    labels_train, 
                    batch_size=32, 
                    shuffle = True, #important since we loaded cats first, dogs second.
                    epochs=3,
                    validation_data=(x_valid, labels_valid))

print("DONE\n")


