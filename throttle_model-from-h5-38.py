import numpy as np
import argparse
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json
from torch.autograd import Variable

# import h5py
import os
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import csv
# from DAVE2 import DAVE2Model
from ResNet import ResNet152, ResNet50, ResNet101
from DAVE2pytorch import DAVE2PytorchModel, DAVE2v1, DAVE2v2, DAVE2v3, Epoch
from BaseDatasetGenerator import MultiDirectoryDataSequence
import time

# from VAEbasic import VAEbasic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize, Lambda, Normalize

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(prog='ProgramName', description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument("-p", '--path_to_trainingdir', type=str, default='/p/autosoft/Meriel/BeamNG_DAVE2_racetracks',
                        help='path to training data parentdir')
    parser.add_argument('-n', '--noisevar', type=float, default=20,
                        help='max noisevar to sample')
    parser.add_argument('-o', '--outdir_id', type=str, default="out",
                        help='identifier or slurm job id')
    args = parser.parse_args()
    print(args.path_to_trainingdir, args.noisevar, args.outdir_id, flush=True)
    return args

args = parse_args()

sample_count = 0
path_to_trainingdir = args.path_to_trainingdir #'H:/BeamNG_DAVE2_racetracks'
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
            print("img_file {}".format(img_file), flush=True)
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
    # with open('{}.json'.format(model_name), 'w') as outfile:
    #     json.dump(model.to_json(), outfile)

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
        print("Moments of abs. val'd turning steering distribution:", generator.get_distribution_moments(turning), flush=True)
        print("Moments of abs. val'd straight steering distribution:", generator.get_distribution_moments(straight), flush=True)
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
    start_time = time.time()
    # Start of MODEL Definition
    m3 = DAVE2Model()
    model3 = m3.define_multi_input_model_BeamNG([150, 200, 3])
    dirlist = [0,1,2]
    generator = DatasetGenerator(dirlist, batch_size=10000, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False)
    # 2D output
    if X is None and X_kph is None and y_all is None:
        X, X_kph, y_all, y_steering, y_throttle = generator.process_all_training_dirs_with_2D_output_and_multi_input(m3)
    print("dataset size: {} output size: {}".format(X.shape, y_all.shape))
    print("time to load dataset: {}".format(time.time() - start_time))
    print("X.shape", X.shape)
    print("X_kph.shape", X_kph.shape)
    print("y_all.shape", y_all.shape)

    BATCH_SIZE = 32
    NB_EPOCH = 20

    # Train steering
    it = "comparison100K-PIDcontrolset-2"
    model3_name = 'BeamNGmodel-racetrack-multiinput-dualoutput-{}'.format(it)

    model3.fit(x=[X, X_kph], y=y_all, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    save_model(model3, model3_name)
    print("Finished multi-input, dual-output model")

    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))

def main_pytorch_model():
    global sample_count, path_to_trainingdir
    global X, X_kph, y_all, y_steering, y_throttle
    import time
    start_time = time.time()
    # model = DAVE2v1(input_shape=(135,240))
    # model = VAEbasic(3, 100, input_shape=(135,240))
    # model = DAVE2PytorchModel(input_shape=(225,400))
    # model = DAVE2PytorchModel(input_shape=(67,120))
    input_shape = (135,240)
    model = DAVE2v3(input_shape=input_shape)
    # model = Epoch(input_shape=input_shape)
    # model = ResNet152(1)
    # model = ResNet101(1)
    # print(f"new model:{model}")
    BATCH_SIZE = 64
    NB_EPOCH = 500
    lr = 1e-4
    robustification = True
    noise_level = 15
    print(args, flush=True)
    dataset = MultiDirectoryDataSequence(args.path_to_trainingdir, image_size=(input_shape[::-1]), transform=Compose([ToTensor()]),\
                                         robustification=robustification, noise_level=noise_level) #, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    print("Retrieving output distribution....", flush=True)
    print("Moments of distribution:", dataset.get_outputs_distribution(), flush=True)
    print("Total samples:", dataset.get_total_samples(), flush=True)
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=worker_init_fn)
    newdir = f""
    import time 
    localtime = time.localtime()
    timestr = "{}_{}-{}_{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
    import random, string, shutil
    randstr = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    newdir = f"./{model._get_name()}-{NB_EPOCH}epoch-{args.outdir_id}-{timestr}-{randstr}"
    if not os.path.exists(newdir):
        # original_umask = os.umask(0)
        os.mkdir(newdir,  mode=0o777)
        #shutil.copy(f"{__file__}", newdir)
        shutil.copyfile(__file__, f"{newdir}/{__file__.split('/')[-1]}", follow_symlinks=False)
    iteration = f'{model._get_name()}-randomblurnoise-{input_shape[0]}x{input_shape[1]}-lr1e4-{NB_EPOCH}epoch-{BATCH_SIZE}batch-lossMSE-{int(dataset.get_total_samples()/1000)}Ksamples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{iteration=}", flush=True)
    print(f"{device=}", flush=True)
    model = model.to(device)
    # if loss doesnt level out after 20 epochs, either inc epochs or inc learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr) #, betas=(0.9, 0.999), eps=1e-08)
    lowest_loss = 1e5
    logfreq = 20
    best_model_count = 0
    for epoch in range(NB_EPOCH):
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
            running_loss += loss.item()
            if i % logfreq == logfreq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, i + 1, running_loss / logfreq), flush=True)
                if (running_loss / logfreq) < lowest_loss:
                    print(f"New best model! MSE loss: {running_loss / logfreq}", flush=True)
                    model_name = f"./{newdir}/model-{iteration}-best{best_model_count:03d}.pt"
                    print(f"Saving model to {model_name}", flush=True)
                    torch.save(model, model_name)
                    best_model_count += 1
                    lowest_loss = running_loss / logfreq
                running_loss = 0.0
        print(f"Finished {epoch=}", flush=True)
        model_name = f"./{newdir}/model-{iteration}-epoch{epoch:03d}.pt"
        print(f"Saving model to {model_name}", flush=True)
        torch.save(model, model_name)
        # if loss < 0.002:
        #     print(f"Loss at {loss}; quitting training...")
        #     break
    print('Finished Training', flush=True)

    # save model
    # torch.save(model.state_dict(), f'H:/GitHub/DAVE2-Keras/test{iteration}-weights.pt')
    model_name = f'./{newdir}/model-{iteration}.pt'
    #torch.save(model, model_name)
    torch.save(model, model_name)

    # delete models from previous epochs
    print("Deleting models from previous epochs...", flush=True)
    for epoch in range(NB_EPOCH):
        os.remove(f"./{newdir}/model-{iteration}-epoch{epoch:03d}.pt")
    print(f"Saving model to {model_name}", flush=True)
    time_to_train=time.time() - start_time
    print("Time to train: {}".format(time_to_train), flush=True)
    # save metainformation about training
    with open(f'./{newdir}/model-{iteration}-metainfo.txt', "w") as f:
        f.write(f"{model_name=}\n"
                f"total_samples={dataset.get_total_samples()}\n"
                f"{NB_EPOCH=}\n"
                f"{lr=}\n"
                f"{BATCH_SIZE=}\n"
                f"{optimizer=}\n"
                f"final_loss={running_loss / logfreq}\n"
                f"{device=}\n"
                f"{robustification=}\n"
                f"{noise_level=}\n"
                f"dataset_moments={dataset.get_outputs_distribution()}\n"
                f"{time_to_train=}\n"
                f"dirs={dataset.get_directories()}")


if __name__ == '__main__':
    # to train Keras model:
    # main_compare_dual_model()
    # to train pytorch model:
    main_pytorch_model()