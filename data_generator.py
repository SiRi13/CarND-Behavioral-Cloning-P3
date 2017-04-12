import os
import zipfile
import csv
import cv2
import numpy as np
import pandas as pd
import sklearn
import random
from sklearn.model_selection import train_test_split
import skimage.transform as sktransform
from scipy.stats import bernoulli

IMAGE_SIZE = (160, 320, 3)

DATA_UDACITY_PATH = './udacity_data/'

DATA_TR1_BRIDGE_PATH = './tr1_bridge/'
DATA_TR1_LAP_PATH = './tr1_lap/'
DATA_TR1_LAP_CTR_PATH = './tr1_lap_ctr/'
DATA_TR1_TURNS_PATH = './tr1_turns/'
DATA_TR1_TURNS2_PATH = './tr1_turns2/'

LOG_FILE_NAME = 'balanced_driving_log.csv'
IMG_PATH = 'IMG'

def __extract_data(base_path=DATA_UDACITY_PATH, archive_name='./udacity_data.zip', output_path='./udacity_data/'):
    # first make sure data exists
    if not os.path.exists(base_path):
        # otherwise extract files from archive
        with zipfile.ZipFile(archive_name, mode='r') as archive:
            archive.extractall(path=output_path)

def __read_logs(lines=[], base_path=DATA_UDACITY_PATH, log_file_name=LOG_FILE_NAME):
    DROP_ANGLE = 0.1
    KEEP_RATIO = 6
    lines_added = 0
    with open(os.path.join(base_path, log_file_name), mode='r') as csv_file:
        reader = csv.reader(csv_file)
        counter = 0
        for line in reader:
            if 'steering' in line:
                continue

            steering_angle = abs(float(line[3]))
            if steering_angle <= DROP_ANGLE:
                if counter < KEEP_RATIO:
                    counter += 1
                    continue
                else:
                    counter = 0

            for i in range(3):
                line[i] = os.path.join(base_path, IMG_PATH, line[i].rsplit('/', 1)[-1])

            lines.append(line)
            lines_added += 1

    cCounter = 0
    lCounter = 0
    rCounter = 0
    for line in lines:
        cCounter += 1 if float(line[3])==0.0 else 0
        lCounter += 1 if float(line[3])<0.0 else 0
        rCounter += 1 if float(line[3])>0.0 else 0
    print("center count: {}\tleft count: {}\tright count: {}".format(cCounter, lCounter, rCounter))

    return lines_added, lines

def __add_random_shadow(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = -k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)

    return image

def __crop_image(image, top=0, bottom=0, left=0, right=0):
    image = image[top:bottom, left:right]
    return cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), cv2.INTER_CUBIC)

def __random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2]*(np.random.uniform(0.35, 1.0))
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

POSITION_STEERING_CORRECTION_1 = [0, 1, -1]
def __steering_correction(steering_angle, cam_pos):
    correction = 0.12
    # correct angle depending on which camera an steering direction
    if steering_angle < 0 and cam_pos == 1:
        angle = steering_angle * .928 + correction
    elif steering_angle < 0 and cam_pos == 2:
        angle = steering_angle * 1.78846 - correction
    elif steering_angle > 0 and cam_pos == 1:
        angle = steering_angle * 1.78846 + correction
    elif steering_angle > 0 and cam_pos == 2:
        angle = steering_angle * .928 - correction
    else:
        # going straight on ...
        # ... center cam => no correction
        # ... left hand cam => slight correction towards right
        # ... right hand cam => slight correction towards left
        angle = steering_angle + (correction * POSITION_STEERING_CORRECTION_1[cam_pos])

    return angle

def load_logs(data_folder_list=[DATA_UDACITY_PATH]):
    lines = list()
    for path in data_folder_list:
        count, lines = __read_logs(lines=lines, base_path=path)
        print("Data Points for '{}': {}".format(path, count))

    print("Data Points Total: ", len(lines))

    return lines

def split_to_sets(data, test_size=0.2):
    return train_test_split(data, test_size=test_size)

def data_generator(data, batch_size=128):
    num_samples = len(data)
    batch_counter = 0
    counter = 0
    while True: # Loop forever so the generator never terminates
        random.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                cam_pos = np.random.randint(3)
                image_path = batch_sample[cam_pos]
                steering_angle = float(batch_sample[3])
                steering_angle_corrected = __steering_correction(steering_angle, cam_pos)
                image = np.array(cv2.imread(image_path))

                counter += 1
                images.append(image)
                angles.append(steering_angle_corrected)

                # add random shadow
                if bernoulli.rvs(0.5):
                    counter += 1
                    images.append(__add_random_shadow(image))
                    angles.append(steering_angle_corrected)
                # change brightness randomly
                elif bernoulli.rvs(0.5):
                    counter += 1
                    images.append(__random_brightness(image))
                    angles.append(steering_angle_corrected)

                # remove parts of image and resize to IMAGE_SIZE
                if bernoulli.rvs(0.5):
                    counter += 1
                    images.append(__crop_image(image, int(np.random.uniform(0, 10)), int(np.random.uniform(-11, -1)), 0, IMAGE_SIZE[0]))
                    angles.append(steering_angle_corrected)

                # flip horizontally
                if bernoulli.rvs(0.5):
                    counter += 1
                    images.append(cv2.flip(image, 1))
                    angles.append(-steering_angle_corrected)


            X = np.array(images)
            y = np.array(angles)

            batch_counter += 1
            # counter += 1
            # cv2.imwrite('./batch_test/batch_image' + str(counter) + '.jpeg', images[np.random.randint(len(images))])
            yield sklearn.utils.shuffle(X, y)
    print('Batch #{} contains {} data points'.format(batch_counter, counter))
    counter = 0
