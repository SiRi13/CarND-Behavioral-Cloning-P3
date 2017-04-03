import os
import cv2
import csv
import matplotlib.pyplot as plt

lines = []
with open('./data/driving_log.csv') as log:
    log_reader = csv.reader(log)
    for line in log_reader:
        lines.append(line)

images = []
for line in lines:
    # extract filename from path
    img_filename = line[0].split('/')[-1]
    image = cv2.imread(os.path.join('./data/IMG', img_filename))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    images.append(image)

len(images)
first = plt.imread('./data/IMG/center_2017_03_30_19_39_45_324.jpg')
# first = cv2.cvtColor(first, cv2.COLOR_RGB2RGB)
first_YUV = cv2.cvtColor(first, cv2.COLOR_RGB2YUV)
first_HLS = cv2.cvtColor(first, cv2.COLOR_RGB2HLS)
plt.subplot(131)
plt.imshow(first)
plt.subplot(132)
plt.imshow(first_YUV)
plt.subplot(133)
plt.imshow(first_HLS)

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
