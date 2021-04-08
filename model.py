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
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from keras_unet.models import custom_unet




def build_model(input_layer, start_neurons):
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    # Middle
    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    deconv4 = layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = layers.Conv2D(1, (1,1), padding="same", activation="linear")(uconv1)
    
    return output_layer




#Code credit: Leciano Strika 
#https://www.kdnuggets.com/2019/07/convolutional-neural-networks-python-tutorial-tensorflow-keras.html

songs = './songs'
vox = './vox'
#1548 x 1168
IMG_SIZE = (1280, 1120)
def pixels_from_path(file_path):
    im = Image.open(file_path)
    im = im.resize(IMG_SIZE)
    np_im = np.array(im)
    #matrix of pixel RGB values
    return np_im


SAMPLE_SIZE = 294
print("loading training songs images...")
songs_train_set = np.asarray([pixels_from_path(songs) for songs in glob.glob('./songs/*')[:SAMPLE_SIZE]])
print("loading training vox images...")
vox_train_set = np.asarray([pixels_from_path(vox) for vox in glob.glob('./vox/*')[:SAMPLE_SIZE]])
print(np.shape(songs_train_set), " ", np.shape(vox_train_set))
valid_size = 74
print("loading validation songs images...")
songs_valid_set = np.asarray([pixels_from_path(songs) for songs in glob.glob('./songs/*')[-valid_size:]])
print("loading validation vox images...")
vox_valid_set = np.asarray([pixels_from_path(vox) for vox in glob.glob('./vox/*')[-valid_size:]])

print("SHAPE OF VOX VALID (LABELS VALID)", vox_valid_set.shape)

# generate X and Y (inputs and labels).
x_train = songs_train_set #np.concatenate([songs_train_set, vox_train_set])
labels_train = vox_train_set #np.asarray([1 for _ in range(SAMPLE_SIZE)]+[0 for _ in range(SAMPLE_SIZE)])

x_valid = songs_valid_set #np.concatenate([songs_valid_set, vox_valid_set])

labels_valid = vox_valid_set #np.asarray([1 for _ in range(valid_size)]+[0 for _ in range(valid_size)])

x_train = np.concatenate([x_train, x_valid])
y_train = np.concatenate([labels_train, labels_valid])
fc_layer_size = 128
img_size = IMG_SIZE

#conv_inputs = keras.Input(shape=(img_size[1], img_size[0],4), name='ani_image')
'''
conv_layer = layers.Conv2D(24, kernel_size=3, activation='relu')(conv_inputs)
conv_layer = layers.MaxPool2D(pool_size=(2,2))(conv_layer)

conv_layer = layers.Conv2D(24, kernel_size=3, activation='relu')(conv_layer)
conv_layer = layers.MaxPool2D(pool_size=(2,2))(conv_layer)

conv_x = layers.Flatten(name = 'flattened_features')(conv_layer) #turn image to vector.

conv_x = layers.Dense(fc_layer_size, activation='relu', name='first_layer')(conv_x)
conv_x = layers.Dense(fc_layer_size, activation='relu', name='second_layer')(conv_x)
conv_outputs = layers.Dense(1, activation='linear', name='class')(conv_x)
'''

'''
conv_outputs = build_model(conv_inputs, 16)

conv_model = keras.Model(inputs=conv_inputs, outputs=conv_outputs)

customAdam = keras.optimizers.Adam(lr=1e-6)
conv_model.compile(optimizer=customAdam,  # Optimizer
              # Loss function to minimize
              loss=keras.losses.MeanSquaredError(),
              # List of metrics to monitor
              metrics=["mean_squared_error", "accuracy"])


conv_model.fit(x_train, 
                    labels_train, 
                    batch_size=32, 
                    shuffle = True, #important since we loaded cats first, dogs second.
                    epochs=3,
                    validation_data=(x_valid, labels_valid))

'''
model = custom_unet(
    input_shape=(1120, 1280, 3),
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')

model.compile(optimizer='Adam', loss='MeanSquaredError')

print(np.shape(x_train), " ", np.shape(y_train))

model.fit(x=x_train, 
          y=y_train,
          batch_size=32,
          epochs=5,
          validation_split=.2
          )


#input_img = image.load_img('./love-me-do.png', target_size=(640,480))
input_img = pixels_from_path('./love-me-do.jpg')
print("input shape", np.array(input_img).shape)
img_array = image.img_to_array(input_img)
print("input array", np.array(img_array).shape)
img_batch = np.expand_dims(img_array, axis =0)
print("input batch", np.array(img_batch).shape)
img_preprocessed = preprocess_input(img_batch)
print("input pre", np.array(img_preprocessed).shape)


predictions = model.predict(img_preprocessed)

print("predictions shape is ", predictions.shape)
print("prediction is ", predictions)

test = np.reshape(predictions, (1120,1280))

print("Test ", test.shape)
print(test)

plt.imshow(test)
plt.colorbar()
plt.show()

out_img = Image.fromarray(predictions)
out_img.save("PooPoo.jpg")
img.show()
print("DONE\n")


