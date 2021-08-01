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
from torch.autograd import Variable

import h5py
import os
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import csv
from DAVE2 import DAVE2Model
from DatasetGenerator import DatasetGenerator, DataSequence, MultiDirectoryDataSequence
import time

from DAVE2pytorch import DAVE2PytorchModel, DAVE2v2, ConvNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize

sample_count = 0
path_to_trainingdir = 'H:/BeamNG_DAVE2_racetracks'
X = None; X_kph = None; y_all = None; y_steering = None; y_throttle = None

def process_csv(filename):
    global path_to_trainingdir
    hashmap = {}
    with open(filename) as csvfile:
        metadata = csv.reader(csvfile, delimiter=',')
        for row in metadata:
            imgfile = row[0].replace("\\", "/")
            hashmap[imgfile] = row[1:]
    return hashmap

def process_training_dir(trainingdir, m):
    global sample_count, path_to_trainingdir
    td = os.listdir(trainingdir)
    td.remove("data.csv")
    X_train = []; steering_Y_train = []; throttle_Y_train = []
    hashmap = process_csv("{}\\data.csv".format(trainingdir))
    for img in td:
        img_file = "{}{}".format(trainingdir, img)
        image = cv2.imread(img_file)
        if "bmp" in img_file:
            print("img_file {}".format(img_file))
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            temp = image
            plt.imshow(temp)
            plt.pause(0.001)
            # cv2.imwrite(img_file.replace("bmp", "png"), temp, compression_params)
            img = img.replace("bmp", "png")
            Image.fromarray(image).save(img_file.replace("bmp", "png"))
            os.remove(img_file)
        image = m.process_image(image)
        # plt.imshow(np.reshape(image, m.input_shape))
        # plt.pause(0.00001)
        X_train.append(image.copy())
        y = float(hashmap[img][2])
        throttle_Y_train.append(y)
        y = float(hashmap[img][0])
        steering_Y_train.append(y)
        sample_count += 1
    return np.asarray(X_train), np.asarray(steering_Y_train),np.asarray(throttle_Y_train)

def save_model(model, model_name):
    model.save_weights('{}-weights.h5'.format(model_name))
    model.save('{}-model.h5'.format(model_name))
    with open('{}.json'.format(model_name), 'w') as outfile:
        json.dump(model.to_json(), outfile)

def characterize_steering_distribution(y_steering, generator):
    turning = []; straight = []
    for i in y_steering:
        if abs(i) < 0.1:
            straight.append(abs(i))
        else:
            turning.append(abs(i))
    # turning = [i for i in y_steering if i > 0.1]
    # straight = [i for i in y_steering if i <= 0.1]
    try:
        print("Moments of abs. val'd turning steering distribution:", generator.get_distribution_moments(turning))
        print("Moments of abs. val'd straight steering distribution:", generator.get_distribution_moments(straight))
    except Exception as e:
        print(e)
        print("len(turning)", len(turning))
        print("len(straight)", len(straight))

def main_compare_dual_model():
    global sample_count, path_to_trainingdir
    global X, X_kph, y_all, y_steering, y_throttle

    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m = DAVE2Model()
    model = m.define_dual_model_BeamNG()
    print(model.summary())
    dirlist = [0,1,2]
    generator = DatasetGenerator(dirlist, batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    # 2D output
    if X is None and X_kph is None and y_all is None:
        X, y_all, y_steering, y_throttle = generator.process_all_training_dirs_with_2D_output()
    print("dataset size: {} output size: {}".format(X.shape, y_all.shape))
    print("time to load dataset: {}".format(time.time() - start_time))

    BATCH_SIZE = 64
    NB_EPOCH = 20

    # Train steering
    it = "-comparison100K-PIDcontrolset-3-sanitycheckretraining"
    model_name = 'BeamNGmodel-racetrackdual{}'.format(it)
    model.fit(x=X, y=y_all, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    save_model(model, model_name)
    print("Finished dual model")

    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))

def main_restructure_dataset():
    generator = DatasetGenerator([0,1,2], batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    generator.restructure_training_set()

def main_multi_input_model():
    global sample_count, path_to_trainingdir
    global X, X_kph, y_all, y_steering, y_throttle
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m1 = DAVE2Model()
    m2 = DAVE2Model()
    m3 = DAVE2Model()
    model1 = m1.define_model()
    model2 = m2.define_model()
    model3 = m3.define_multi_input_model_BeamNG([150, 200, 3])
    dirlist = [0,1,2]
    generator = DatasetGenerator(dirlist, batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    # 2D output
    if X is None and X_kph is None and y_all is None:
        X, X_kph, y_all, y_steering, y_throttle = generator.process_all_training_dirs_with_2D_output_and_multi_input(m1)
    print("dataset size: {} output size: {}".format(X.shape, y_all.shape))
    print("time to load dataset: {}".format(time.time() - start_time))
    print("X.shape", X.shape)
    print("X_kph.shape", X_kph.shape)
    print("y_all.shape", y_all.shape)


    BATCH_SIZE = 32
    NB_EPOCH = 20

    # Train steering
    it = "comparison100K-PIDcontrolset-2"
    # model1_name = 'BeamNGmodel-racetracksteering-{}'.format(it)
    # model2_name = 'BeamNGmodel-racetrackthrottle-{}'.format(it)
    model3_name = 'BeamNGmodel-racetrack-multiinput-dualoutput-{}'.format(it)
    # model1.fit(x=X, y=y_steering, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    # save_model(model1, model1_name)
    # print("Finished steering model")
    # # delete the previous model so that you don't max out the memory
    # del model1
    # model2.fit(x=[X, y_steering, y_throttle], y=y_throttle, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    # save_model(model2, model2_name)
    # print("Finished throttle model")
    # del model2
    model3.fit(x=[X, X_kph], y=y_all, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    save_model(model3, model3_name)
    print("Finished multi-input, dual-output model")

    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))

def main_pytorch_model():
    global sample_count, path_to_trainingdir
    global X, X_kph, y_all, y_steering, y_throttle
    start_time = time.time()
    # Start of MODEL Definition
    model = DAVE2PytorchModel() #DAVE2v2() #
    print(model)
    BATCH_SIZE = 64
    NB_EPOCH = 50
    # turn np.array to pytorch Tensor
    # if X is None and X_kph is None and y_all is None:
    #     X, y_steering = generator.process_all_training_dirs_pytorch() #process_first_half_of_dataset_for_pytorch()
    # X = torch.from_numpy(X).permute(0,3,1,2).float()/255.0
    # y_steering = torch.from_numpy(y_steering).float()/255.0
    # X = torch.as_tensor(X).permute(0,3,1,2).float()/255.0 #, device=torch.device('cuda'))
    # y_steering = torch.as_tensor(y_steering).float()/255.0 #, device=torch.device('cuda'))
    # dataset = DataSequence(training_directory, transform=Compose([ToTensor()]))
    dataset = MultiDirectoryDataSequence("H:/BeamNG_DAVE2_racetracks_all/PID/", transform=Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
    print("time to load dataset: {}".format(time.time() - start_time))

    # train model
    iteration = '7-trad-50epochs-64batch-1e4lr-ORIGDATASET-singleoutput'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4) #, betas=(0.9, 0.999), eps=1e-08)
    for epoch in range(NB_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image'].float().to(device)
            y = hashmap['steering_input'].float().to(device)
            x = Variable(x, requires_grad=True)
            y = Variable(y, requires_grad=False)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            # loss = F.mse_loss(outputs.flatten(), y)
            loss = F.mse_loss(outputs, y)
            # loss = F.gaussian_nll_loss(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            logfreq = 20
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq))
                running_loss = 0.0
                # from torchvision.utils import save_image
                # save_image(x, "test_david.png", nrow=8)
            # if len(running_loss) > 10:
            #     running_loss[-10:]
        print(f"Finished {epoch=}")
        print(f"Saving model to H:/GitHub/DAVE2-Keras/test-{iteration}-model-epoch-{epoch}.pt")
        torch.save(model, f'H:/GitHub/DAVE2-Keras/test-{iteration}-model-epoch-{epoch}.pt')
    print('Finished Training')

    # save model
    # torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    torch.save(model, f'H:/GitHub/DAVE2-Keras/test{iteration}-model.pt')
    print("Finished dual model")

    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))


if __name__ == '__main__':
    # to train Keras model:
    # main_compare_dual_model()
    # to train pytorch model:
    main_pytorch_model()
