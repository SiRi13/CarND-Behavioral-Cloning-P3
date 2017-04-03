import tarfile
from os.path import exists
import os
import csv
import cv2
import numpy as np

# extract files from archive
if not exists('./data'):
    with tarfile.open('./data.tar', mode='r') as archive:
        archive.extractall(path='./')

lines = []
with open('./data/driving_log.csv') as log:
    log_reader = csv.reader(log)
    for line in log_reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    if 'steering' in line:
        continue
    # extract filename from path
    img_filename = line[0].split('/')[-1]
    img_filename_left = line[1].split('/')[-1]
    img_filename_right = line[2].split('/')[-1]
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image_left = cv2.imread(os.path.join('./data/IMG', img_filename_left))
    image_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2YUV)
    image_right = cv2.imread(os.path.join('./data/IMG', img_filename_right))
    image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2YUV)
    steering_angle = float(line[3])
    steering_angle_left = steering_angle + 0.25
    steering_angle_right = steering_angle - 0.25
    images.append(image)
    measurements.append(steering_angle)
    images.append(image_left)
    measurements.append(steering_angle_left)
    images.append(image_right)
    measurements.append(steering_angle_right)

    # images.append(cv2.flip(image, 1))
    # measurements.append(steering_angle * -1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout

model = Sequential()
# simple normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
# conv layer 1 - 24 filters / 2x2 stride / 5x5 kernel
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2)))
# conv layer 2 - 36 filters / 2x2 stride / 5x5 kernel
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2)))
# conv layer 3 - 48 filters / 2x2 stride / 5x5 kernel
model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2)))
# conv layer 4 - 64 filters / none stride / 3x3 kernel
model.add(Conv2D(64, kernel_size=(3,3)))
# conv layer 5 - 64 filters / none stride / 5x5 kernel
model.add(Conv2D(64, kernel_size=(3,3)))
# flatten
model.add(Flatten())
# fully connected layer 1 - 100
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
# fully connected layer 2 - 50
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
# fully connected layer 3 - 10
model.add(Dense(10, activation='relu'))
# output
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('cArI.h5')
