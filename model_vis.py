import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, model_from_json
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation, visualize_cam

model = model_from_json(json.load(open('./cArI_model.json')))
model.load_weights('model.h5')

plot_model(model=model, show_shapes=True, to_file='./images/model_plot.png')

vis_path = './batch_test/batch_image2.jpeg'
img = cv2.imread(vis_path)

layer_names = [l.name for l in model.layers]
print(layer_names)
layer_idx = len(model.layers)-6
print(model.layers[layer_idx].name)

# out = visualize_saliency(model, 14, 0, seed_img=seed_img)
out = visualize_activation(model, 2, 0, img)
plt.imshow(out)
plt.show()

# out2 = visualize_cam(model, 2, 1, img[0])
# out3 = visualize_saliency(model, 2, 1, img[0])

layer1 = Model(inputs=model.input, outputs=model.get_layer('conv2d_4').output)
layer2 = Model(inputs=model.input, outputs=model.get_layer('conv2d_5').output)
layer3 = Model(inputs=model.input, outputs=model.get_layer('conv2d_6').output)

img = cv2.imread('/home/simon/udacity/carnd/simulator_data/tr1_bridge/IMG/center_2017_04_10_19_15_38_624.jpg')
img2 = cv2.imread('/home/simon/Pictures/IMG/center_2017_04_13_19_21_18_924.jpg')
img = np.expand_dims(img, axis=0)
img2 = np.expand_dims(img2, axis=0)

def visualize_layer(img, img_filename):
    visual_layer1, visual_layer2, visual_layer3 = layer1.predict(img), layer2.predict(img), layer3.predict(img)

    arr_1, arr_2, arr_3, layer_1, layer_2, layer_2 = [], [], [], [], [], []
    for i in range(16):
        arr_1.append(visual_layer1[0, :, :, i])
    for i in range(32):
        arr_2.append(visual_layer2[0, :, :, i])
    for i in range(64):
        arr_3.append(visual_layer3[0, :, :, i])

    plt.figure(figsize=(10,8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        temp = arr_1[i]
        temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
        plt.imshow(temp)
        plt.axis('off')
    plt.savefig('./images/plots/layer1_' + img_filename)
    plt.show()

    plt.figure(figsize=(12,8))
    for i in range(32):
        plt.subplot(8, 4, i+1)
        temp = arr_2[i]
        temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
        plt.imshow(temp)
        plt.axis('off')
    plt.savefig('./images/plots/layer2_' + img_filename)
    plt.show()

    plt.figure(figsize=(20,10))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        temp = arr_3[i]
        temp = cv2.resize(temp, (200, 66), cv2.INTER_AREA)
        plt.imshow(temp)
        plt.axis('off')
    plt.savefig('./images/plots/layer3_' + img_filename)
    plt.show()

visualize_layer(img, 'feature_map_with_street.png')
visualize_layer(img2, 'feature_map_withou_street.png')

def visualize_conv_layers(image, img_name, layer=2):
    layerOutput = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer].output])
    # output in test mode = 0, train mode = 1
    layerOutputSample = layerOutput([image.reshape(1,image.shape[0],image.shape[1],image.shape[2]), 1])[0]
    layerOutputSample = layerOutputSample.reshape(layerOutputSample.shape[1],layerOutputSample.shape[2],layerOutputSample.shape[3])
    print(layerOutputSample.shape)
    figure = plt.figure(figsize=(24,8))
    factors = [8,8]
    for ind in range(layerOutputSample.shape[2]):
        img = figure.add_subplot(factors[0],factors[1],ind + 1)
        #plt.subplot(4, 4, ind + 1)
        val = layerOutputSample[:,:,ind]
        plt.axis("off")
        plt.imshow(val, cmap='gray', interpolation='nearest')
    plt.savefig('./images/plots/vis_con_layer_{}_{}.png'.format(layer, img_name))
    plt.show()

visualize_conv_layers(img[0], 'with_street', 3)
visualize_conv_layers(img[0], 'with_street', 5)
visualize_conv_layers(img[0], 'with_street', 7)

visualize_conv_layers(img2[0], 'without_street', 3)
visualize_conv_layers(img2[0], 'without_street', 5)
visualize_conv_layers(img2[0], 'without_street', 7)
