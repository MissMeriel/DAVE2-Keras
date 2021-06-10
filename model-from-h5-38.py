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
import os
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import csv
from DAVE2 import Model

sample_count = 0
path_to_trainingdir = 'H:/BeamNG_DAVE2'

# def define_model():
#     # Start of MODEL Definition
#     input_shape = (66, 200, 3)
#     model = Sequential()
#     # Input normalization layer
#     model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, name='lambda_norm'))
#
#     # 5x5 Convolutional layers with stride of 2x2
#
#     # 5x5 Convolutional layers with stride of 2x2
#     model.add(Conv2D(24, 5, 2, name='conv1'))
#     model.add(ELU(name='elu1'))
#     model.add(Conv2D(36, 5, 2, name='conv2'))
#     model.add(ELU(name='elu2'))
#     model.add(Conv2D(48, 5, 2, name='conv3'))
#     model.add(ELU(name='elu3'))
#
#     # 3x3 Convolutional layers with stride of 1x1
#     model.add(Conv2D(64, 3, 1, name='conv4'))
#     model.add(ELU(name='elu4'))
#     model.add(Conv2D(64, 3, 1, name='conv5'))
#     model.add(ELU(name='elu5'))
#
#     # Flatten before passing to the fully connected layers
#     model.add(Flatten())
#     # Three fully connected layers
#     model.add(Dense(100, name='fc1'))
#     model.add(Dropout(.5, name='do1'))
#     model.add(ELU(name='elu6'))
#     model.add(Dense(50, name='fc2'))
#     model.add(Dropout(.5, name='do2'))
#     model.add(ELU(name='elu7'))
#     model.add(Dense(10, name='fc3'))
#     model.add(Dropout(.5, name='do3'))
#     model.add(ELU(name='elu8'))
#
#     # Output layer with tanh activation
#     model.add(Dense(1, activation='tanh', name='output'))
#
#     adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     model.compile(optimizer="adam", loss="mse")
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     filename = '{}/model.h5'.format(dir_path)
#     model.load_weights(filename)
#     return model

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

def process_image1(image):
    size = (200, 66)
    plt.imshow(np.array(image))
    w, h = image.size
    print("{} {}".format(w,h))
    plt.pause(1)
    image = image.crop((0, 200, 512, 369))
    plt.imshow(np.array(image))
    plt.pause(1)
    image = image.resize(size, Image.ANTIALIAS)
    plt.imshow(np.array(image))
    plt.pause(1)
    image = np.array(image).reshape(1, 66, 200, 3)
    return image

def process_image(image):
    size = (200, 66)
    image = image.crop((0, 200, 512, 369))
    image = image.resize(size, Image.ANTIALIAS)
    image = np.array(image).reshape(1, 66, 200, 3)
    return image

def process_csv(filename):
    global path_to_trainingdir
    hashmap = {}
    with open(filename) as csvfile:
        metadata = csv.reader(csvfile, delimiter=',')
        for row in metadata:
            imgfile = "{}/{}".format(path_to_trainingdir, row[0].replace("\\", "/"))
            hashmap[imgfile] = row[1:]
    return hashmap

def process_training_dir(trainingdir):
    global sample_count, path_to_trainingdir
    td = os.listdir(trainingdir)
    td.remove("data.csv")
    X_train = []; Y_train = []
    hashmap = process_csv("{}\\data.csv".format(trainingdir))
    for img in td:
        img_file = "{}{}".format(trainingdir, img)
        image = Image.open(img_file)
        image = process_image(image)
        X_train.append(image.copy())
        y = float(hashmap[img_file][1])
        Y_train.append(y)
        sample_count += 1
    return np.asarray(X_train), np.asarray(Y_train)

def redo_csv(readfile):
    temp = readfile.split("/")
    writefile = "/".join(temp[:-1]) + "data1.csv"
    #print("writefile:{}".format(writefile))
    with open(readfile) as csvfile:
        with open(writefile, 'w') as csvfile1:
            metadata = csvfile.readlines()
            #metadata = csv.reader(csvfile, delimiter=',')
            for row in metadata:
                #row = row.replace('C:\\Users\\merie\\Documents\\BeamNGpy-master\\BeamNGpy-master\\examples/', '')
                row = row.replace('H:/BeamNG_DAVE2/', '')
                csvfile1.write(row)
                #print("row:{}".format(row))
    os.remove(readfile)
    os.rename(writefile, readfile)

def main():
    global sample_count, path_to_trainingdir
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    # Start of MODEL Definition
    m = Model()
    model = m.define_model()

    # prep training set
    print(os.listdir(path_to_trainingdir))
    t = os.listdir(path_to_trainingdir)
    training_dirs = []
    for training_dir in t:
        training_dirs.append("{}/{}/".format(path_to_trainingdir, training_dir))
    print(training_dirs)
    shape = (0, 1, 66, 200, 3)
    X_train = np.array([]).reshape(shape); y_train = np.array([])
    for d in training_dirs:
        print("Processing {}".format(d))
        redo_csv("{}/data.csv".format(d))
        x_temp, y_temp = process_training_dir(d)
        print("X_train shape:{} x_temp shape:{}".format(X_train.shape, x_temp.shape))
        X_train = np.concatenate((X_train, x_temp), axis=0)
        y_train = np.concatenate((y_train,y_temp), axis=0)
    print("Final X_train shape:{} Final y_train shape:{}".format(X_train.shape, y_train.shape))

    # Train and save the model

    BATCH_SIZE = 100
    NB_EPOCH = 9
    NB_SAMPLES = 2*len(X_train)
    model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH)

    model.save_weights('BeamNGmodel-5.h5')
    with open('BeamNGmodel-5.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print("All done :)")
    print("Total training samples: {}".format(sample_count))

if __name__ == '__main__':
    main()
