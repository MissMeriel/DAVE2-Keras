import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json

#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.layers import Activation, Flatten, Lambda, Input, ELU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

import h5py
import os
from PIL import Image
import PIL

class Model:

    def __init__(self):
        self.model = Sequential()
        self.input_shape = (150, 200, 3) #(960,1280,3)

    def define_model(self):
        # Start of MODEL Definition
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(1, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # filename = '{}/model.h5'.format(dir_path)
        # self.model.load_weights(filename)
        return self.model

    def define_model_BeamNG(self, h5_filename):
        # Start of MODEL Definition
        self.model = Sequential()
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(1, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        self.load_weights(h5_filename)
        return self.model


    # outputs vector [steering, throttle]
    def define_dual_model_BeamNG(self):
        # Start of MODEL Definition
        self.model = Sequential()
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(2, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        # self.load_weights(h5_filename)
        return self.model

    def atan_layer(self, x):
        return tf.multiply(tf.atan(x), 2)

    def atan_layer_shape(self, input_shape):
        return input_shape

    def define_model_DAVEorig(self):
        input_tensor = Input(shape=(100, 100, 3))
        model = Sequential()
        model.add(Conv2D(24, 5, 2, padding='valid', activation='relu', name='block1_conv1'))
        model.add(Conv2D(36, 5, 2, padding='valid', activation='relu',  name='block1_conv2'))
        model.add(Conv2D(48, 5, 2, padding='valid', activation='relu', name='block1_conv3'))
        model.add(Conv2D(64, 3, 1, padding='valid', activation='relu', name='block1_conv4'))
        model.add(Conv2D(64, 3, 1, padding='valid', activation='relu', name='block1_conv5'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(1164, activation='relu', name='fc1'))
        model.add(Dense(100, activation='relu', name='fc2'))
        model.add(Dense(50, activation='relu', name='fc3'))
        model.add(Dense(10, activation='relu', name='fc4'))
        model.add(Dense(1, name='before_prediction'))
        model.add(Lambda(lambda x: tf.multiply(tf.atan(x), 2), output_shape=input_tensor, name='prediction'))
        model.compile(loss='mse', optimizer='adadelta')
        self.model = model
        return model

    def load_weights(self, h5_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = '{}/{}'.format(dir_path, h5_file)
        self.model.load_weights(filename)
        return self.model

    def process_image(self, image):
        # image = image.crop((0, 200, 512, 369))
        # image = image.resize((self.input_shape[1], self.input_shape[0]), Image.ANTIALIAS)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image = np.array(image).reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return image

    # Functions to read and preprocess images
    def readProcess(self, image_file):
        """Function to read an image file and crop and resize it for input layer

        Args:
          image_file (str): Image filename (expected in 'data/' subdirectory)

        Returns:
          numpy array of size 66x200x3, for the image that was read from disk
        """
        # Read file from disk
        image = mpimg.imread('data/' + image_file.strip())
        # Remove the top 20 and bottom 20 pixels of 160x320x3 images
        image = image[20:140, :, :]
        # Resize the image to match input layer of the model
        resize = (self.input_shape[0], self.input_shape[1])
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        return image

    def randBright(self, image, br=0.25):
        """Function to randomly change the brightness of an image

        Args:
          image (numpy array): RGB array of input image
          br (float): V-channel will be scaled by a random between br to 1+br

        Returns:
          numpy array of brighness adjusted RGB image of same size as input
        """
        rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rand_bright = br + np.random.uniform()
        rand_image[:,:,2] = rand_image[:,:,2]*rand_bright
        rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
        return rand_image
