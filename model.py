import data_generator
from data_generator import IMAGE_SIZE
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# set start time
start = time.time()

# load log files and merge them to one list
data = data_generator.load_logs([data_generator.DATA_TR1_LAP_PATH,
                                 data_generator.DATA_TR1_LAP_CTR_PATH,
                                 data_generator.DATA_TR1_BRIDGE_PATH,
                                 data_generator.DATA_TR1_TURNS2_PATH,
                                 data_generator.DATA_TR1_TURNS3_PATH,
                                 data_generator.DATA_TR1_BRIDGE2_PATH,
                                 data_generator.DATA_UDACITY_PATH])

# shuffle and split data in training and validation set
train, valid = data_generator.split_to_sets(data)

# image format from data_generator
row, col, ch = IMAGE_SIZE
# 32 on aws instance, 16 on local machine
batch_size = 32

# importing keras dependencies
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# LossHistory class extends Callback
# saves losses of each batch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# creating model
model = Sequential()
# normalizing pixel values
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=IMAGE_SIZE))
# cropping hood and scenery of image
model.add(Cropping2D(cropping=((60, 23), (0, 0))))
# model architecture
# first conv layer with 16 filters, 3x3 kernel and relu activation
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
# 2x2 pooling layer
model.add(MaxPooling2D())
# second conv layer with 32 filters, 3x3 kernel and relu activation
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
# 2x2 pooling layer
model.add(MaxPooling2D())
# third conv layer with 64 filters, 3x3 kernel and relu activation
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# 2x2 pooling layer
model.add(MaxPooling2D())
# flatten layer
model.add(Flatten())
# first fully connected layer with relu activation and 500 units
model.add(Dense(500, activation='relu'))
# 50% dropout
model.add(Dropout(0.5))
# second fully connected layer with relu activation and 100 units
model.add(Dense(100, activation='relu'))
# 50% dropout
model.add(Dropout(0.5))
# third fully connected layer with relu activation and 20 units
model.add(Dense(20, activation='relu'))
# output layer with one unit and no activation
model.add(Dense(1))

# Mean Average Error as loss function and Adam as optimizer
model.compile(loss='mae', optimizer='adam')

# stops after 4 epochs of less than 0.008 improvement
earlyStopper = EarlyStopping(min_delta=0.008, patience=4, mode='min')
# saves the best weights after each epoch
checkpointer = ModelCheckpoint(filepath="./model.h5", verbose=1, save_best_only=True)
history = LossHistory()

# fit_generator uses data_generator for training and validation data
history_obj = model.fit_generator(data_generator.data_generator(train, batch_size=batch_size),
                                  steps_per_epoch=(len(train)*2.5)//batch_size,
                                  validation_data=data_generator.data_generator(valid, batch_size=batch_size),
                                  validation_steps=(len(valid)*2.5)//batch_size,
                                  epochs=50, verbose=2,
                                  callbacks=[history, earlyStopper, checkpointer])

# set end time
end = time.time()

# save model as json-file to restore it for visualization
json.dump(model.to_json(), open('model.json', 'w'))

# current time formatted
current_time_formatted =  str(time.strftime('%Y%m%d_%H%M%S'))
# display duration of training
duration = end - start
print(duration / 60)

model_arch = 'traffic_sign_net'
# plot diagram of losses per epoch
plt.plot(history.losses)
plt.savefig('./images/loss_plot_{}_{}.jpeg'.format(model_arch,current_time_formatted))
plt.show()

# plot progress of training and validation loss over all epochs
plt.plot(history_obj.history['loss'])
plt.plot(history_obj.history['val_loss'])
plt.title('model mean average error loss')
plt.ylabel('mean average error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./images/val_loss_plot_{}_{}.jpeg'.format(model_arch, current_time_formatted))
plt.show()
