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

ch, row, col = 3, 160, 320 # image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPooling2D, ELU
from keras.optimizers import Adam

model = Sequential()
# simple normalization
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(row, col, ch), output_shape=(row, col, ch)))
# model.add(Cropping2D(cropping=((60,25), (0,0))))
# conv layer 1
model.add(Conv2D(16, kernel_size=(8,8), strides=(4,4), padding='same'))
model.add(ELU())
# conv layer 2
model.add(Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same'))
model.add(ELU())
# conv layer 3 - 64
model.add(Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same'))
# flatten layer
model.add(Flatten())
model.add(Dropout(0.2))
model.add(ELU())
# fully connected layer
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(ELU())
# output
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit(X_train, y_train, validation_split=0.3, batch_size=256, shuffle=True, epochs=3, verbose=2)

model.save('cArI.h5')

history_obj.history.keys()

import matplotlib.pyplot as plt
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
