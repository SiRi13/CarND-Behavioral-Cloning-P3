import os
import cv2
import csv
import tarfile
import zipfile
import pandas as pd
import matplotlib.pyplot as plt

with zipfile.ZipFile('./data.zip', mode='r') as dataZip:
    dataZip.extractall('./data_udacity/')

with tarfile.open('./data.tar', mode='r') as archive:
    archive.extractall(path='./my_data/')

lines = []
with open('./my_data/data/driving_log.csv') as log:
    log_reader = csv.reader(log)
    for line in log_reader:
        lines.append(line)

with open('./data_udacity/data/driving_log.csv') as log:
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
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    img_center.append(image)
    # left image
    img_filename = line[1].split('/')[-1]
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    img_left.append(image)
    # left image
    img_filename = line[2].split('/')[-1]
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    img_right.append(image)

print(len(angles))

plt.hist(angles, 50, facecolor='green', alpha=0.75)
plt.show()

rnd_img = np.random.randint(len(angles))
center_ref = img_center[rnd_img]
center_YUV = cv2.cvtColor(img_center[rnd_img], cv2.COLOR_RGB2YUV)
left_YUV = cv2.cvtColor(img_left[rnd_img], cv2.COLOR_RGB2YUV)
right_YUV = cv2.cvtColor(img_right[rnd_img], cv2.COLOR_RGB2YUV)
plt.subplot(232)
plt.imshow(center_ref)
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
last_r = plt.imread('./data/IMG/right_2017_03_30_19_42_00_473.jpg')
plt.figure(figsize=(20, 80))
plt.subplot(131)
plt.imshow(last_l)
plt.subplot(132)
plt.imshow(last)
plt.subplot(133)
plt.imshow(last_r)
