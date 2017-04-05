from data_generator import load_log, split_to_sets, data_generator
import numpy as np

data = load_log()
train, valid = split_to_sets(data)

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
history_obj = model.fit_generator(data_generator(train), steps_per_epoch=len(train),
                                  validation_data=data_generator(valid), validation_steps=len(valid),
                                  verbose=2, epochs=8)

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
