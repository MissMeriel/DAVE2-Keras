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
from DAVE2 import DAVE2Model
from DatasetGenerator import DatasetGenerator, DataSequence, MultiDirectoryDataSequence
import time

from DAVE2pytorch import DAVE2PytorchModel, DAVE2v2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda


sample_count = 0
path_to_trainingdir = 'H:/BeamNG_DAVE2_racetracks'
X = None; X_kph = None; y_all = None; y_steering = None; y_throttle = None

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
                row = row.replace('H:/BeamNG_DAVE2_racetracks/', '')
                row = row.replace('bmp', 'png')
                csvfile1.write(row)
                #print("row:{}".format(row))
    os.remove(readfile)
    os.rename(writefile, readfile)

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

def main2():
    global sample_count, path_to_trainingdir
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m = DAVE2Model()
    model = m.define_model()

    # prep training set
    t = os.listdir(path_to_trainingdir)
    training_dirs = ["{}/{}/".format(path_to_trainingdir, training_dir) for training_dir in t]
    # for training_dir in t:
    #     training_dirs.append("{}/{}/".format(path_to_trainingdir, training_dir))

    shape = (0, 1, m.input_shape[0], m.input_shape[1], m.input_shape[2])
    X_train = np.array([]).reshape(shape); y_train = np.array([])
    for d in training_dirs[-1:]:
        print("Processing {}".format(d))
        redo_csv("{}/data.csv".format(d))
        x_temp, steering_y_temp, throttle_y_temp = process_training_dir(d, m)
        print("Concatenating X_train shape:{} x_temp shape:{}".format(X_train.shape, x_temp.shape))
        X_train = np.concatenate((X_train, x_temp), axis=0)
        steering_y_train = np.concatenate((y_train,steering_y_temp), axis=0)
        throttle_y_train = np.concatenate((y_train,throttle_y_temp), axis=0)
    print("Final X_train shape:{} Final y_train shape:{}".format(X_train.shape, steering_y_train.shape))

    # Train and save the model
    BATCH_SIZE = 100
    NB_EPOCH = 9
    NB_SAMPLES = 2*len(X_train)
    # Train steering
    model.fit(x=X_train, y=steering_y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    model_name = 'BeamNGmodel-racetracksteering'
    model.save_weights('{}.h5'.format(model_name))
    with open('{}.json'.format(model_name), 'w') as outfile:
        json.dump(model.to_json(), outfile)
    # Train throttle
    model.fit(x=X_train, y=throttle_y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    model_name = 'BeamNGmodel-racetrackthrottle'
    model.save_weights('{}.h5'.format(model_name))
    with open('{}.json'.format(model_name), 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print("All done :)")
    print("Total training samples: {}".format(sample_count))
    print("Time to train: {}".format(time.time() - start_time))

def main3():
    global sample_count, path_to_trainingdir
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m = DAVE2Model()
    model = m.define_model()
    generator = DatasetGenerator([0,1,2], batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1,
                        callbacks=None, validation_data=None, validation_steps=None,
                        validation_freq=1, class_weight=None, max_queue_size=10,
                        workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    model_name = 'BeamNGmodel-racetracksteering2'
    model.save_weights('{}.h5'.format(model_name))
    with open('{}.json'.format(model_name), 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))

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

def main():
    global sample_count, path_to_trainingdir
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m1 = DAVE2Model()
    m2 = DAVE2Model()
    model1 = m1.define_model()
    model2 = m2.define_model()
    dirlist = [0,1,2]
    generator = DatasetGenerator(dirlist, batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    # model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1,
    #                     callbacks=None, validation_data=None, validation_steps=None,
    #                     validation_freq=1, class_weight=None, max_queue_size=10,
    #                     workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    # for d in dirlist:
    #     print("Batch training on dir {}".format(d))
    #     X,y1, y2 = generator.data_generation(d)
    #     model1.train_on_batch(X, y1)
    #     model2.train_on_batch(X, y2)
    filename_root = "H:/BeamNG_DAVE2_racetracks/training_images_industrial-racetrackstartinggate"
    # X, y_steering, y_throttle = generator.process_enumerated_training_dirs(filename_root, dirlist, m1)

    # 1D output
    X, y_steering, y_throttle = generator.process_all_training_dirs(m1)
    print("X .shape", X.shape, "y_steering.shape", y_steering.shape, "y_throttle.shape", y_throttle.shape)
    characterize_steering_distribution(y_steering, generator)
    print("Moments of steering distribution:", generator.get_distribution_moments(y_steering))
    print("Moments of throttle distribution:", generator.get_distribution_moments(y_throttle))

    # 2D output
    # X, y = generator.process_all_training_dirs_with_2D_output(m1)

    print("time to load dataset: {}".format(time.time() - start_time))

    BATCH_SIZE = 64
    NB_EPOCH = 20

    # Train steering
    it = "1Doutput"
    model1_name = 'BeamNGmodel-racetracksteering{}'.format(it)
    model2_name = 'BeamNGmodel-racetrackthrottle{}'.format(it)
    model1.fit(x=X, y=y_steering, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    save_model(model1, model1_name)
    print("Finished steering model")
    # delete the previous model so that you don't max out the memory
    del model1; del y_steering
    model2.fit(x=X, y=y_throttle, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    save_model(model2, model2_name)
    print("Finished throttle model")

    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))

def main_compare_dual_model():
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
    model3 = m3.define_dual_model_BeamNG()
    print(model3.summary())
    dirlist = [0,1,2]
    generator = DatasetGenerator(dirlist, batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    # 2D output
    if X is None and X_kph is None and y_all is None:
        X, y_all, y_steering, y_throttle = generator.process_all_training_dirs_with_2D_output(m1)
    print("dataset size: {} output size: {}".format(X.shape, y_all.shape))
    print("time to load dataset: {}".format(time.time() - start_time))

    BATCH_SIZE = 64
    NB_EPOCH = 20

    # Train steering
    it = "-comparison100K-PIDcontrolset-3"
    model1_name = 'BeamNGmodel-racetracksteering{}'.format(it)
    model2_name = 'BeamNGmodel-racetrackthrottle{}'.format(it)
    model3_name = 'BeamNGmodel-racetrackdual{}'.format(it)
    # model1.fit(x=X, y=y_steering, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    # save_model(model1, model1_name)
    # print("Finished steering model")
    # # delete the previous model so that you don't max out the memory
    # del model1
    # model2.fit(x=X, y=y_throttle, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    # save_model(model2, model2_name)
    # print("Finished throttle model")
    # del model2
    model3.fit(x=X, y=y_all, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    save_model(model3, model3_name)
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


    BATCH_SIZE = 64
    NB_EPOCH = 20

    # Train steering
    it = "comparison100K-PIDcontrolset-2"
    model1_name = 'BeamNGmodel-racetracksteering-{}'.format(it)
    model2_name = 'BeamNGmodel-racetrackthrottle-{}'.format(it)
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
    NB_EPOCH = 20
    # turn np.array to pytorch Tensor
    # if X is None and X_kph is None and y_all is None:
    #     X, y_steering = generator.process_all_training_dirs_pytorch() #process_first_half_of_dataset_for_pytorch()
    # X = torch.from_numpy(X).permute(0,3,1,2).float()/255.0
    # y_steering = torch.from_numpy(y_steering).float()/255.0
    # X = torch.as_tensor(X).permute(0,3,1,2).float()/255.0 #, device=torch.device('cuda'))
    # y_steering = torch.as_tensor(y_steering).float()/255.0 #, device=torch.device('cuda'))
    # dataset = DataSequence(training_directory, transform=Compose([ToTensor()]))
    dataset = MultiDirectoryDataSequence("H:/BeamNG_DAVE2_racetracks_all/PID/", transform=Compose([ToTensor()]))
    print("Retrieving output distribution....")
    print("Moments of distribution:", dataset.get_outputs_distribution())

    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("time to load dataset: {}".format(time.time() - start_time))

    # train model
    iteration = '7-trad-20epochs-100Ksamples-singleoutput'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}")
    print(f"{device=}")
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, eithr inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    for epoch in range(NB_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, hashmap in enumerate(trainloader, 0):
            x = hashmap['image'].float().to(device)
            y = hashmap['steering_input'].float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)
            # loss = F.mse_loss(outputs.flatten(), y)
            loss = F.mse_loss(outputs, y)
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
    torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    torch.save(model, f'H:/GitHub/DAVE2-Keras/test{iteration}-model.pt')
    print("Finished dual model")

    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))


if __name__ == '__main__':
    # main_compare_dual_model()
    main_pytorch_model()
    # main_multi_input_model()
