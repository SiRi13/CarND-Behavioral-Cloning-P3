import os
import zipfile
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

UDACITY_DATA_PATH = './udacity_data/data/'
MY_DATA_PATH = './my_data/data/'
LOG_FILE_NAME = 'driving_log.csv'
IMG_PATH = 'IMG'

POSITION_STEERING_CORRECTION = [0.2, 0.0, -0.2]

def load_logs():
    # first make sure data exists
    if not os.path.exists(MY_DATA_PATH):
        # otherwise extract files from archive
        with zipfile.ZipFile('./my_data.zip', mode='r') as archive:
            archive.extractall(path='./my_data/')
    if not os.path.exists(UDACITY_DATA_PATH):
        # otherwise extract files from archive
        with zipfile.ZipFile('./udacity_data.zip', mode='r') as archive:
            archive.extractall(path='./udacity_data/')

    lines = []
    with open(os.path.join(MY_DATA_PATH, LOG_FILE_NAME), mode='r') as csv_file:
        reader = csv.reader(csv_file)
        counter = 0
        for line in reader:
            steering_angle = float(line[3])
            if steering_angle == 0.0:
                if counter < 5:
                    counter += 1
                    continue
                else:
                    counter = 0

            for i in range(3):
                line[i] = line[i].rsplit('/', n=1).str[-1].apply(lambda fileName: os.path.join(MY_DATA_PATH, IMG_PATH, fileName))
            lines.append(line)

    with open(os.path.join(UDACITY_DATA_PATH, LOG_FILE_NAME), mode='r') as csv_file:
        reader = csv.reader(csv_file)
        counter = 0
        for line in reader:
            if 'steering' in line:
                continue

            steering_angle = float(line[3])
            if steering_angle == 0.0:
                if counter < 5:
                    counter += 1
                    continue
                else:
                    counter = 0

            for i in range(3):
                line[i] = line[i].split('/', n=1).str[-1].apply(lambda fileName: os.path.join(MY_DATA_PATH, IMG_PATH, fileName))
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
                # load center, left and right camera image randomly
                cam_position = np.random.randint(0, 3)
                center_angle = float(batch_sample[3])
                img_file_path = batch_sample[cam_position].split('/')[-1]
                print("img_file_name: ", img_file_path)
                image = cv2.imread(img_file_path)
                images.append(image)
                angles.append(center_angle + POSITION_STEERING_CORRECTION[cam_position])

            # TODO: trim image to only see section with road
            X = np.array(images)
            y = np.array(angles)
            print(len(X))
            yield sklearn.utils.shuffle(X, y)

tmpBatch = np.array([])
for batch in data_generator(load_logs()):
    tmpBatch = batch
    break
tmpBatch
