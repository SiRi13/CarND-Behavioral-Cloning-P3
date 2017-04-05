import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def load_log(file_path='./driving_log.csv'):
    lines = []
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    return lines

def data_generator(data, batch_size=128):
    # TODO
    pass
