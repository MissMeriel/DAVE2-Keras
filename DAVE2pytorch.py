import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json

import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.layers import Activation, Flatten, Lambda, Input, ELU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate

import h5py
import os
from PIL import Image
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda


class DAVE2PytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (150,200)
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        self.dropout = nn.Dropout()
        # self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=13824, out_features=100, bias=True)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin2 = nn.Linear(in_features=100, out_features=50, bias=True)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin3 = nn.Linear(in_features=50, out_features=10, bias=True)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        # self.lin4 = nn.Linear(in_features=10, out_features=2, bias=True)
        self.lin4 = nn.Linear(in_features=10, out_features=1, bias=True)
        torch.nn.init.xavier_uniform_(self.lin4.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x, inplace=True)
        x = self.conv2(x)
        x = F.elu(x, inplace=True)
        x = self.conv3(x)
        x = F.elu(x, inplace=True)
        x = self.conv4(x)
        x = F.elu(x, inplace=True)
        x = self.conv5(x)
        x = F.elu(x, inplace=True)
        x = x.flatten(1)
        x = self.lin1(x)
        x = self.dropout(x)
        x = F.elu(x, inplace=True)
        x = self.lin2(x)
        x = self.dropout(x)
        x = F.elu(x, inplace=True)
        x = self.lin3(x)
        x = self.dropout(x)
        x = F.elu(x, inplace=True)
        x = self.lin4(x)
        x = torch.tanh(x)
        # x = 2 * torch.atan(x)
        return x

    def load(self, path="test-model.pt"):
        return torch.load(path)

    # process PIL image to Tensor
    def process_image(self, image, transform=Compose([ToTensor()])):
        # image = image.resize((self.input_shape[1], self.input_shape[0]), Image.ANTIALIAS)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        # add a single dimension to the front of the matrix -- [...,None] inserts dimension in index 1
        image = np.array(image)[None]#.reshape(1, self.input_shape[0], self.input_shape[1], 3)
        # use transpose instead of reshape -- reshape doesn't change representation in memory
        image = image.transpose((0,3,1,2))
        # ToTensor() normalizes data between 0-1 but torch.from_numppy just casts to Tensor
        if transform:
            image = transform(image) #torch.from_numpy(image)/255.0 #transform(image)
        return image #.permute(2, 1, 0)

class DAVE2v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (150,200)

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(0.5)

        self.lin1 = nn.Linear(8320, 512)
        self.lin2 = nn.Linear(512, 1)

        self.max_action = 1.0

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)
        #print(x.shape)# flatten
        x = self.dropout(x)
        # print(x.shape)
        x = self.lr(self.lin1(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        # x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        # x[:, 1] = self.tanh(x[:, 1])
        x = torch.tanh(x)
        return x

    # process PIL image to Tensor
    def process_image(self, image, transform=Compose([ToTensor()])):
        # image = image.resize((self.input_shape[1], self.input_shape[0]), Image.ANTIALIAS)
        image = cv2.resize(image, (150,200))
        # add a single dimension to the front of the matrix -- [...,None] inserts dimension in index 1
        image = np.array(image)[None]#.reshape(1, self.input_shape[0], self.input_shape[1], 3)
        # use transpose instead of reshape -- reshape doesn't change representation in memory
        image = image.transpose((0,3,1,2))
        # ToTensor() normalizes data between 0-1 but torch.from_numppy just casts to Tensor
        if transform:
            image = torch.from_numpy(image)/255.0 #transform(image)
        return image #.permute(2, 1, 0)

    def load(self, path="test-model.pt"):
        return torch.load(path)
