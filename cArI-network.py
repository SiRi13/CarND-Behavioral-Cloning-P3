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
    # extract filename from path
    img_filename = line[0].split('/')[-1]
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    images.append(image)
    steering_angle = line[3]
    if "," in steering_angle:
        steering_angle = steering_angle.replace(',', '.')
    measurements.append(float(steering_angle))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('cArI.h5')
