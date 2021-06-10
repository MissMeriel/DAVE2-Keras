import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from keras.layers import Activation, Flatten, Lambda, Input, ELU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import h5py

from PIL import Image
import PIL

def define_model():
    # Start of MODEL Definition
    input_shape = (66, 200, 3)
    model = Sequential()
    # Input normalization layer
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, name='lambda_norm'))

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Conv2D(24, 5, 2, input_shape=input_shape, name='conv1'))
    model.add(ELU(name='elu1'))
    model.add(Conv2D(36, 5, 2, input_shape=input_shape, name='conv2'))
    model.add(ELU(name='elu2'))
    model.add(Conv2D(48, 5, 2, name='conv3'))
    model.add(ELU(name='elu3'))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Conv2D(64, 3, 1, input_shape=input_shape, name='conv4'))
    model.add(ELU(name='elu4'))
    model.add(Conv2D(64, 3, 1, input_shape=input_shape, name='conv5'))
    model.add(ELU(name='elu5'))

    # Flatten before passing to the fully connected layers
    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(100, name='fc1'))
    model.add(Dropout(.5, name='do1'))
    model.add(ELU(name='elu6'))
    model.add(Dense(50, name='fc2'))
    model.add(Dropout(.5, name='do2'))
    model.add(ELU(name='elu7'))
    model.add(Dense(10, name='fc3'))
    model.add(Dropout(.5, name='do3'))
    model.add(ELU(name='elu8'))

    # Output layer with tanh activation
    model.add(Dense(1, activation='tanh', name='output'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer="adam", loss="mse")
    return model

# Functions to read and preprocess images
def readProcess(image_file):
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
    resize = (200, 66)
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    return image

def randBright(image, br=0.25):
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

def main():
    # Convert training dataframe into images and labels arrays

    # Training data generator with random shear and random brightness
    datagen = ImageDataGenerator(shear_range=0.1, preprocessing_function=randBright)

    filename = 'model.h5'
    h5 = h5py.File(filename, 'r')
    print(h5.keys())
    for k in h5.keys():
        print("h5[{}] = {}".format(k, h5[k]))

    # Start of MODEL Definition
    model = define_model()
    model.load_weights(filename)
    image = Image.open("C:\\Users\\merie\\Documents\\BeamNGpy-master\\BeamNGpy-master\\examples\\training_images\\etk800_westcoast_1.bmp")
    w, h = image.size
    print("{} {}".format(w,h))
    image = image.crop((0, 0, 200,  66))
    image = np.array(image).reshape(1, 66, 200, 3)
    prediction = model.predict(image)
    print(prediction)


#model.save_weights('model.h5')
#with open('model.json', 'w') as outfile:
#    json.dump(model.to_json(), outfile)

if __name__ == '__main__':
    main()
