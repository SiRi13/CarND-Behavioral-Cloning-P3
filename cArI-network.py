import data_generator
from data_generator import IMAGE_SIZE
import time
import numpy as np
import matplotlib.pyplot as plt

start = time.time()

data = data_generator.load_logs()
train, valid = data_generator.split_to_sets(data)

row, col, ch = IMAGE_SIZE # image format
batch_size = 32

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, ELU, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def get_nvidia_pooling(model):
    model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    return 'nvidia_with_pooling', model

def get_nvidia(model):
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

    return 'nvidia', model

def get_comma_ai(model):
    # conv layer 0
    model.add(Conv2D(8, kernel_size=(8,8), strides=(4,4), padding='same'))
    model.add(ELU())
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
    # fully connected layer 0
    model.add(Dense(1024))
    model.add(Dropout(0.5))
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

    return 'comma_ai', model

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=IMAGE_SIZE))
model.add(Cropping2D(cropping=((60, 23), (0, 0))))
# get model architecture
model_arch, model = get_nvidia_pooling(model)
# output
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=1e-04))

earlyStopper = EarlyStopping(min_delta=0.009, patience=4, mode='min')
checkpointer = ModelCheckpoint(filepath="./weights.h5", verbose=1, save_best_only=True)
history = LossHistory()

history_obj = model.fit_generator(data_generator.data_generator(train, batch_size=batch_size),
                                  steps_per_epoch=(len(train)*1.5)//batch_size,
                                  validation_data=data_generator.data_generator(valid, batch_size=batch_size),
                                  validation_steps=(len(valid)*1.5)//batch_size,
                                  epochs=50, verbose=2,
                                  callbacks=[history, earlyStopper, checkpointer])

# model.save('cArI.h5')
end = time.time()

current_time_formatted =  str(time.strftime('%Y%m%d_%H%M%S'))
duration = end - start
duration / 60
model.summary()
# with open('./model_architecture_{}.txt'.format(model_arch), mode='w') as arch_file:
#     arch_file.write('Model: {} trained @ {}'.format(model_arch, time.time()))
#     arch_file.writelines(summary.split('\n'))

plt.plot(history.losses)
plt.savefig('./images/loss_plot_{}_{}.jpeg'.format(model_arch,current_time_formatted))
plt.show()

plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./images/val_loss_plot_{}_{}.jpeg'.format(model_arch, current_time_formatted))
plt.show()
