import data_generator
from data_generator import IMAGE_SIZE
import numpy as np

data = data_generator.load_logs()
train, valid = data_generator.split_to_sets(data)

row, col, ch = IMAGE_SIZE # image format
batch_size=64

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, ELU
from keras.optimizers import Adam

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(row, col, ch)))
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
# fully connected layer 1
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(ELU())
# fully connected layer 2
model.add(Dense(200))
model.add(Dropout(0.5))
model.add(ELU())
# fully connected layer 3
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(ELU())
# output
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_obj = model.fit_generator(data_generator.data_generator(train, batch_size=batch_size),
                                  steps_per_epoch=(len(train)*1.5)//batch_size,
                                  validation_data=data_generator.data_generator(valid, batch_size=batch_size),
                                  validation_steps=(len(valid)*1.5)//batch_size,
                                  epochs=5)

model.save('cArI.h5')

import matplotlib.pyplot as plt
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
