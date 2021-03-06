import os
import cv2
import csv
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with zipfile.ZipFile('./udacity_data.zip', mode='r') as dataZip:
    dataZip.extractall('./udacity_data/')

with zipfile.ZipFile('./my_data.zip', mode='r') as archive:
    archive.extractall(path='./my_data/')

lines = []
with open('./my_data/data/driving_log.csv') as log:
    log_reader = csv.reader(log)
    for line in log_reader:
        lines.append(line)

with open('./udacity_data/data/driving_log.csv') as log:
    log_reader = csv.reader(log)
    for line in log_reader:
        lines.append(line)

img_center = []
img_left = []
img_right = []
angles = []
for line in lines:
    if 'steering' in line:
        continue

    steering_angle = float(line[3])
    if steering_angle == 0.0 and len(np.where(angles==0.0))%5!=0:
        continue

    # get steering angle
    angles.append(steering_angle)
    # extract filename from path
    # center image
    img_filename = line[0].split('/')[-1]
    image = cv2.imread(os.path.join('./my_data/data/IMG', img_filename))
    img_center.append(image)
    # left image
    img_filename = line[1].split('/')[-1]
    image = cv2.imread(os.path.join('./my_data/data/IMG', img_filename))
    img_left.append(image)
    # left image
    img_filename = line[2].split('/')[-1]
    image = cv2.imread(os.path.join('./my_data/data/IMG', img_filename))
    img_right.append(image)

print(len(angles))

plt.hist(angles, 50, facecolor='green', alpha=0.75)
plt.show()

rnd_img = np.random.randint(len(angles))
center_img = cv2.cvtColor(img_center[rnd_img],cv2.COLOR_RGB2YUV)
left_img = cv2.cvtColor(img_left[rnd_img],cv2.COLOR_RGB2YUV)
right_img = cv2.cvtColor(img_right[rnd_img],cv2.COLOR_RGB2YUV)
plt.figure(figsize=(15,5))
plt.subplot(231)
plt.imshow(left_img)
plt.subplot(234)
plt.imshow(left_img[60:-23,:])
plt.subplot(232)
plt.imshow(center_img)
plt.subplot(235)
plt.imshow(center_img[60:-23,:])
plt.subplot(233)
plt.imshow(right_img)
plt.subplot(236)
plt.imshow(right_img[60:-23,:])
plt.show()

center_YUV = cv2.cvtColor(center_img, cv2.COLOR_RGB2HLS)
left_YUV = cv2.cvtColor(img_left[rnd_img], cv2.COLOR_RGB2HLS)
right_YUV = cv2.cvtColor(img_right[rnd_img], cv2.COLOR_RGB2HLS)
plt.figure(figsize=(15,5))
plt.subplot(232)
plt.imshow(cv2.cvtColor(center_img, cv2.COLOR_YUV2BGR))
plt.subplot(234)
plt.imshow(left_YUV)
plt.subplot(235)
plt.imshow(center_YUV)
plt.subplot(236)
plt.imshow(right_YUV)
plt.show()
print(angles[rnd_img])

last = plt.imread('./data/IMG/center_2017_03_30_19_42_00_473.jpg')
last_l = plt.imread('./data/IMG/left_2017_03_30_19_42_00_473.jpg')

line = ['/home/simon/udacity/carnd/CarND-Behavioral-Cloning-P3/data/IMG/center_2017_03_30_19_39_46_898.jpg',
        '/home/simon/udacity/carnd/CarND-Behavioral-Cloning-P3/data/IMG/left_2017_03_30_19_39_46_898.jpg',
        '/home/simon/udacity/carnd/CarND-Behavioral-Cloning-P3/data/IMG/right_2017_03_30_19_39_46_898.jpg',
        -0.1924883,0.4,0,16.5762]
line[0].rsplit('/',1)[-1].apply(lambda fileName: os.path.join(MY_DATA_PATH, IMG_PATH, fileName))


from scipy.stats import bernoulli
bernoulli.rvs(0.5)

import data_generator

lines = data_generator.load_logs(['./simulator_data/tr1_lap/'])
for batch in data_generator.data_generator(lines):
    X_batch, Y_batch = batch
    idxList = np.random.randint(0, len(X_batch), size=6)

    plt.figure(figsize=(15,5))
    for i in range(len(idxList)):
        plt.subplot(2,3,i+1)
        plt.title("angle: {}".format(Y_batch[i]))
        plt.imshow(cv2.cvtColor(X_batch[i], cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    plt.savefig('./images/plots/random_batch.png')
    plt.show()

    break

rnd_line = lines[np.random.randint(len(lines))]
c_img, l_img, r_img = rnd_line[:3]
c_img = cv2.cvtColor(cv2.imread(c_img), cv2.COLOR_BGR2RGB)
l_img = cv2.cvtColor(cv2.imread(l_img), cv2.COLOR_BGR2RGB)
r_img = cv2.cvtColor(cv2.imread(r_img), cv2.COLOR_BGR2RGB)
c_img_shadow = data_generator.__add_random_shadow(np.copy(c_img))
l_img_brightness = data_generator.__random_brightness(np.copy(l_img))
r_img_cropped = data_generator.__crop_image(np.copy(r_img), int(np.random.uniform(0, 10)), int(np.random.uniform(-11, -1)), 0, data_generator.IMAGE_SIZE[0])
c_img_shadow_flip = cv2.flip(np.copy(c_img_shadow), 1)

plt.figure(figsize=(8,5))
plt.subplot(221)
plt.title("center image w/ random shadow")
plt.imshow(np.vstack((c_img, c_img_shadow)))
plt.subplot(222)
plt.title("left hand image w/ random brightness")
plt.imshow(np.vstack((l_img, l_img_brightness)))
plt.subplot(223)
plt.title("right hand image w/ cropped slightly")
plt.imshow(np.vstack((r_img, r_img_cropped)))
plt.subplot(224)
plt.title("center image w/ shadow and flipped")
plt.imshow(np.vstack((c_img_shadow, c_img_shadow_flip)))
plt.tight_layout()
plt.savefig('./images/plots/random_augmented_images.png')
plt.show()
