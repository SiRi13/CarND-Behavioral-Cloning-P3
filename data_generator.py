import os
import tarfile
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATA_PATH = './data/'
IMG_PATH = os.path.join(DATA_PATH, 'IMG')

POSITION_STEERING_CORRECTION = [0.2, 0.0, -0.2]

def load_log(file_name='driving_log.csv'):
    # first make sure data exists
    if not os.path.exists(DATA_PATH):
        # otherwise extract files from archive
        with tarfile.open('./data.tar', mode='r') as archive:
            archive.extractall(path='./')

    lines = []
    with open(os.path.join(DATA_PATH, file_name), mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            if 'steering' in line:
                continue
            lines.append(line)

    return lines

def split_to_sets(data):
    return train_test_split(data, test_size=0.3)

def data_generator(data, batch_size=128):
    num_samples = len(data)
    while True: # Loop forever so the generator never terminates
        data = shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # load center, left and right camera images
                center_angle = float(batch_sample[3])
                for i in range(3):
                    img_file_name = batch_sample[i].split('/')[-1]
                    image = cv2.imread(os.path.join(IMG_PATH, img_file_name))
                    images.append(image)
                    angles.append(center_angle + POSITION_STEERING_CORRECTION[i])

            # TODO: trim image to only see section with road
            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)
