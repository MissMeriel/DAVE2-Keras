################################
import numpy as np
import pandas as pd
import csv
from tensorflow.python.keras.utils.data_utils import Sequence
from keras.utils import np_utils
import os, cv2
from PIL import Image

class DatasetGenerator(Sequence):

    def __init__(self, batch_size, training_dir, input_shape):
        self.batch_size = batch_size
        self.training_dir = training_dir
        self.input_shape = input_shape
        self.shape = input_shape

    def __getitem__(self, index):
        yield self.load_training_dir(self.training_dir, self.input_shape)

    def __len__(self):
        # return len(self.indices) // self.batch_size
        t = os.listdir(self.training_dir)
        training_dirs = []
        count = 0
        for td in t:
            count += len(os.listdir("{}/{}".format(self.training_dir, td))) - 1
        return count


    # def on_epoch_end(self):
    #     self.index = np.arange(len(self.indices))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.index)

    def load_data(self, Train_df, idx, batch_size):
        df = pd.read_csv(
            Train_df, skiprows=idx * batch_size,
            nrows=batch_size)
        x = df.iloc[:, 1:]

        y = df.iloc[:, 0]
        return (np.array(x), np_utils.to_categorical(y))

    def process_csv(self, filename):
        global path_to_trainingdir
        hashmap = {}
        with open(filename) as csvfile:
            metadata = csv.reader(csvfile, delimiter=',')
            for row in metadata:
                imgfile = row[0].replace("\\", "/")
                hashmap[imgfile] = row[1:]
        return hashmap

    def process_training_dir(self, trainingdir, m):
        td = os.listdir(trainingdir)
        td.remove("data.csv")
        X_train = [];
        steering_Y_train = [];
        throttle_Y_train = []
        hashmap = self.process_csv("{}\\data.csv".format(trainingdir))
        for img in td:
            img_file = "{}{}".format(trainingdir, img)
            # image = Image.open(img_file)
            image = cv2.imwrite(img_file)
            image = m.process_image(image)
            # plt.imshow(np.reshape(image, m.input_shape))
            # plt.pause(0.00001)
            X_train.append(image.copy())
            y = float(hashmap[img][2])
            throttle_Y_train.append(y)
            y = float(hashmap[img][0])
            steering_Y_train.append(y)
        return np.asarray(X_train), np.asarray(steering_Y_train), np.asarray(throttle_Y_train)

    def load_training_data(self, path_to_trainingdir, input_shape, batch_size):
        t = os.listdir(path_to_trainingdir)
        training_dirs = []
        for training_dir in t:
            training_dirs.append("{}/{}/".format(path_to_trainingdir, training_dir))
        # print(training_dirs)

        shape = (0, 1, input_shape[0],input_shape[1], input_shape[2])
        X_train = np.array([]).reshape(shape);
        y_train = np.array([])
        current_batch_size = 0
        for d in training_dirs:
            print("Processing {}".format(d))
            x_temp, steering_y_temp, throttle_y_temp = self.process_training_dir(d, input_shape)
            print("Concatenating X_train shape:{} x_temp shape:{}".format(X_train.shape, x_temp.shape))
            X_train = np.concatenate((X_train, x_temp), axis=0)
            steering_y_train = np.concatenate((y_train, steering_y_temp), axis=0)
            throttle_y_train = np.concatenate((y_train, throttle_y_temp), axis=0)
            current_batch_size += 1
            if current_batch_size == batch_size:
                yield X_train, steering_y_train
        # print("Final X_train shape:{} Final y_train shape:{}".format(X_train.shape, steering_y_train.shape))