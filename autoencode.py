from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow as tf
import numpy as np
import glob
from PIL import Image

##########
## data ##
##########
songs = './songs'
vox = './vox'
#1548 x 1168
IMG_SIZE = (1276, 1116)
NET_INPUT = (1116, 1276, 1)
def pixels_from_path(file_path):
    im = Image.open(file_path).convert('L')
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

# have to reshape
x_train = tf.expand_dims(x_train, axis=-1)
y_train = tf.expand_dims(y_train, axis=-1)

print("New shapes: ", np.shape(x_train), " ", np.shape(y_train))

# # The encoding process
# input_img = Input(shape=NET_INPUT) 

# ############
# # Encoding #
# ############

# # Conv1 #
# x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

# # Conv2 #
# x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
# x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 

# # Conv 3 #
# x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
# encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

# # Note:
# # padding is a hyper-arameter for either 'valid' or 'same'. 
# # "valid" means "no padding". 
# # "same" results in padding the input such that the output has the same length as the original input.

# ############
# # Decoding #
# ############

# # DeConv1
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2, 2))(x)

# # DeConv2
# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)

# # Deconv3
# x = Conv2D(16, (3, 3), activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


# # Declare the model
# autoencoder = Model(input_img, decoded)
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

inputs = Input(NET_INPUT)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv10)

model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model
autoencoder.fit(x_train, y_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_split=.2
               )

decoded_img = autoencoder.predict(x_train[0,:])
my_img = Image.fromarray(decoded_img)
my_img.save('./poopoo.jpg')