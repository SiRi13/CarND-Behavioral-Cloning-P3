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
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    steering_angle = float(line[3])
    images.append(image)
    measurements.append(steering_angle)
    images.append(cv2.flip(image, 1))
    measurements.append(steering_angle * -1.0)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=1)

model.save('cArI.h5')
