import os
from pathlib import Path, PureWindowsPath
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
plt.style.use("ggplot")


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
import sklearn.metrics as metrics

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



it = 1
vox_list = os.listdir('vox\orginal_stft')
songs_list = os.listdir('songs\original_stft')

# Set some parameters
im_width = 640
im_height = 560
border = 5

ids = next(os.walk("songs\original_stft"))[2] # list of names all images in the given path
print("No. of songs = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)


# tqdm is used to display the progress bar
for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
    # Load images
    img = load_img('.\\songs\\original_stft\\'+id_, color_mode = "grayscale")
    x_img = img_to_array(img)
    x_img = resize(x_img, (560, 640, 1), mode = 'constant', preserve_range = True)
    # Load masks
    extension = ".jpg"
    insert = "-vox-mask"
    idx = id_.index(extension)
    id2_ = id_[:idx] + insert + id_[idx:]
    mask = img_to_array(load_img('vox/masked_stft/' + id2_, color_mode="grayscale"))
    mask = resize(mask, (560, 640, 1), mode = 'constant', preserve_range = True)
    # Save/normalize images
    X[n] = x_img/255.0
    y[n] = mask/255.0

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)



model =  tf.keras.models.load_model("vocal-iso-model.h5")
model.summary()


# load the best model
model.load_weights('vocal-iso-weights.h5')


# Evaluate on validation set (this must be equals to the best log_loss)
scores = model.evaluate(X_valid, y_valid, verbose=1)
print("Accuracy: ", scores[1])
print("Loss: ", scores[0])


# Predict on train, val and test
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

print("train pred shape ", preds_train.shape)
print("val pred shape ", preds_val.shape)


# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

'''
def plot_sample(X, y, preds, binary_preds, ix=None):
    it = 1
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0])
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Overlayed graph')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Vox of ' + str(vox_list[ix]))

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Vox Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('Vox Predicted binary')
    it = it +1
    fig.savefig('fig\\ax' + str(it) +'-index-' + songs_list[ix] + '.jpg')



# Check if training data looks all right
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=14)

plot_sample(X_train, y_train, preds_train, preds_train_t, ix =212)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)
plot_sample(X_train, y_train, preds_train, preds_train_t)

plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=19)

plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
plot_sample(X_valid, y_valid, preds_val, preds_val_t)
'''