import data_generator
from data_generator import IMAGE_SIZE
import time
import json
import numpy as np
import matplotlib.pyplot as plt

start = time.time()

data = data_generator.load_logs(['./simulator_data/tr1_lap/', './simulator_data/tr1_lap_ctr/',
                                 './simulator_data/tr1_bridge/', './simulator_data/tr1_turns2/',
                                 './simulator_data/tr1_turns3/', './simulator_data/tr1_bridge2/'])

train, valid = data_generator.split_to_sets(data)

row, col, ch = IMAGE_SIZE # image format
batch_size = 16

from keras.models import Sequential, model_from_json
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, ELU, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import plot_model

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
    model.add(Droptout(0.25))
    # fully connected layer 0 - 500
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
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

model = model_from_json(json.load(open('./cArI_model.json')))
model.load_weights('cArI_weights.h5')

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=IMAGE_SIZE))
model.add(Cropping2D(cropping=((60, 23), (0, 0))))
# get model architecture
model_arch, model = get_nvidia_pooling(model)
# output
model.add(Dense(1))

model.compile(loss='mae', optimizer=Adam(lr=1e-04))

earlyStopper = EarlyStopping(min_delta=0.008, patience=4, mode='min')
checkpointer = ModelCheckpoint(filepath="./weights_mae.h5", verbose=1, save_best_only=True)
history = LossHistory()

history_obj = model.fit_generator(data_generator.data_generator(train, batch_size=batch_size),
                                  steps_per_epoch=(len(train)*1.5)//batch_size,
                                  validation_data=data_generator.data_generator(valid, batch_size=batch_size),
                                  validation_steps=(len(valid)*1.5)//batch_size,
                                  epochs=1, verbose=2,
                                  callbacks=[history, earlyStopper, checkpointer])

# model.save('cArI.h5')
end = time.time()

current_time_formatted =  str(time.strftime('%Y%m%d_%H%M%S'))
duration = end - start
print(duration / 60)

json.dump(model.to_json(), open('cArI_model.json', 'w'))

plot_model(model=model, show_shapes=True, to_file='./images/cArI_visualized.png')

import cv2
from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation, visualize_cam

vis_path = './batch_test/batch_image1.jpeg'
layer_idx = len(model.layers)-1
seed_img = cv2.imread(vis_path)
# out = visualize_saliency(model, 14, 0, seed_img=seed_img)
out = visualize_activation(model, 14, filter_indices=0, seed_img=seed_img)
out2 = visualize_cam(model, 14, 0, seed_img)
out3 = visualize_saliency(model, 14, 0, seed_img, overlay=False)

import matplotlib.pyplot as plt
plt.imshow(out)
plt.show()
# with open('./model_architecture_{}.txt'.format(model_arch), mode='w') as arch_file:
#     arch_file.write('Model: {} trained @ {}'.format(model_arch, time.time()))
#     arch_file.writelines(summary.split('\n'))
# model.summary()

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
