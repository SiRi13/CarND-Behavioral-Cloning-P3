import argparse
import select
import sys
import os
import numpy as np
import pandas as pd
import cv2
import csv

def silent_delete(file):
    """
    This method delete the given file from the file system if it is available
    Source: http://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
    :param file:
        File to be deleted
    :return:
        None
    """
    try:
        os.remove(file)

    except OSError as error:
        if error.errno != errno.ENOENT:
            raise

def do(base_folder):
    with open(os.path.join(base_folder, 'driving_log.csv'), mode='r') as csv_file:
        if not os.path.exists('./new_data/'):
            os.mkdir('./new_data')

        with open('./new_data/driving_log_new.csv', mode='a') as csv_out:
            reader = csv.reader(csv_file)
            reader_list = list(reader)
            writer = csv.writer(csv_out)
            new_data_list = list()
            counter = -1
            curr_counter = -1
            last_existing = -1
            while True:
                counter += 1
                if counter >= len(reader_list):
                    counter = 0
                line = reader_list[counter]
                center_image = os.path.join(base_folder, 'IMG', line[0].rsplit('/', 1)[-1])
                left_image = os.path.join(base_folder, 'IMG', line[1].rsplit('/', 1)[-1])
                right_image = os.path.join(base_folder, 'IMG', line[2].rsplit('/', 1)[-1])
                print(center_image)
                if not os.path.exists(center_image) or not os.path.exists(left_image) or not os.path.exists(right_image):
                    last_existing = counter - 2 if last_existing < 0 else last_existing
                    continue
                center_img = cv2.imread(center_image)
                left_img = cv2.imread(left_image)
                right_img = cv2.imread(right_image)
                angle = float(line[3])
                out_img = np.hstack((left_img, center_img, right_img))
                out_img = cv2.putText(out_img, 'angle: {}'.format(angle), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, 0xFFFFFF)
                cv2.imshow('a == keep; d == delete', out_img)
                retVal = cv2.waitKey(0) & 0xFF
                print('key code: ', retVal)
                if retVal == ord('d'):
                    # delete images and remove from csv
                    print("Delete image: ", line[0])
                    silent_delete(left_image)
                    silent_delete(center_image)
                    silent_delete(right_image)
                    curr_counter = -1
                    continue
                elif retVal == 83: # right
                    print("next")
                    curr_counter = -1
                    continue
                elif retVal == 81: #left
                    print("back")
                    curr_counter = -1
                    counter = counter - 2 if last_existing < 0 else last_existing
                    last_existing = -1
                    continue
                elif retVal == 32: # space
                    print('skip 5 images')
                    # keep curr_counter until backspace
                    curr_counter = counter if curr_counter < 0 else curr_counter
                    counter += 5
                    print('counter: {}\t curr_counter: {}'.format(counter, curr_counter))
                    continue
                elif retVal == 8: #backspace
                    print('back after space')
                    print('counter: {}\t curr_counter: {}'.format(counter, curr_counter))
                    counter = curr_counter if curr_counter > 0 else counter - 1
                    curr_counter = -1
                    continue
                elif retVal == 80:
                    # jump to start
                    counter = 0
                    continue
                elif retVal == ord('a') :
                    new_data_list.append(line)
                    continue
                elif retVal == ord('p'):
                    # persist csv
                    writer.writerow(line)
                    new_data_list = list()
                    counter -= 1
                    continue
                elif retVal == ord('r'):
                    # reset new data list
                    new_data_list = list()
                    counter -= 1
                    last_existing = -1
                    curr_counter = -1
                    continue
                elif retVal == ord('q'):
                    break

            cv2.destroyAllWindows()

def clean_logs(base_path='~/data/', log_file_name='driving_log.csv'):
    df = pd.read_csv(os.path.join(base_path, log_file_name), names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    # remove missing images from csv
    indices2remove = list()
    for idx, row in df.iterrows():
        file_path_split = row.center.rsplit('/', 2)[-2:]
        if not os.path.exists(os.path.join(base_path, file_path_split[0].strip(), file_path_split[1].strip())):
            print('missing: ', os.path.join(base_path, file_path_split[0].strip(), file_path_split[1].strip()))
            indices2remove.append(idx)
            continue
        file_path_split = row.left.rsplit('/', 2)[-2:]
        if not os.path.exists(os.path.join(base_path, file_path_split[0].strip(), file_path_split[1].strip())):
            print('missing: ', os.path.join(base_path, file_path_split[0].strip(), file_path_split[1].strip()))
            indices2remove.append(idx)
            continue
        file_path_split = row.right.rsplit('/', 2)[-2:]
        if not os.path.exists(os.path.join(base_path, file_path_split[0].strip(), file_path_split[1].strip())):
            print('missing: ', os.path.join(base_path, file_path_split[0].strip(), file_path_split[1].strip()))
            indices2remove.append(idx)
            continue

    print("df count: ", len(df), " 2remove: ", len(indices2remove))
    df = df.drop(df.index[indices2remove])
    df.to_csv(os.path.join(base_path, 'balanced_driving_log.csv'), header=False, index=False)

def balance_logs(base_path='~/data/', log_file_name='driving_log.csv'):
    df = pd.read_csv(os.path.join(base_path, log_file_name), names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    balanced = pd.DataFrame() 	# Balanced dataset
    bins = 1000 				# N of bins
    bin_n = 200 				# N of examples to include in each bin (at most)

    start = 0
    for end in np.linspace(0, 1, num=bins):
        if end == start:
            continue
        print('start: ', start, ' end: ', end)
        df_range = df[(np.absolute(df.steering) >= start) & (np.absolute(df.steering) < end)]
        print('df_range.shape[0]: ', df_range.shape[0])
        range_n = min(bin_n, df_range.shape[0])
        if range_n > 0:
            balanced = pd.concat([balanced, df_range.sample(range_n)])
        start = end

    balanced.to_csv(os.path.join(base_path, 'balanced_driving_log.csv'), header=False, index=False)

def test(base_path="./data/", log_file_name="balanced_driving_log.csv"):
    df = pd.read_csv(os.path.join(base_path, log_file_name), names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    for idx, row in df.iterrows():
        center_img = cv2.imread(os.path.join(base_path, 'IMG', row.center.rsplit('/', 1)[-1]))
        left_img = cv2.imread(os.path.join(base_path, 'IMG', row.left.rsplit('/', 1)[-1]))
        right_img = cv2.imread(os.path.join(base_path, 'IMG', row.right.rsplit('/', 1)[-1]))
        angle = float(row.steering)
        out_img = np.hstack((left_img, center_img, right_img))
        out_img = cv2.putText(out_img, 'angle: {}'.format(angle), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, 0xFFFFFF)
        cv2.imshow('a == keep; d == delete', out_img)
        retVal = cv2.waitKey(2) & 0xFF
        if retVal == ord('q'):
            break
    cv2.destroyAllWindows()

def __load_logs(base_path, log_file_name):
    DROP_ANGLE = 0.1
    KEEP_RATIO = 6
    lines = list()
    with open(os.path.join(base_path, log_file_name), mode='r') as csv_file:
        reader = csv.reader(csv_file)
        counter = 0
        for line in reader:
            if 'steering' in line:
                continue

            steering_angle = abs(float(line[3]))
            if steering_angle <= DROP_ANGLE:
                if counter < KEEP_RATIO:
                    counter += 1
                    continue
                else:
                    counter = 0

            for i in range(3):
                line[i] = os.path.join(base_path, 'IMG', line[i].rsplit('/', 1)[-1])
            lines.append(line)
    return np.array(lines)

def print_stats(base_path, log_file_name):
    base_path = './udacity_data/'
    log_file_name = 'driving_log.csv'
    df = pd.read_csv(os.path.join(base_path, log_file_name), names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    print(len(df[df.steering > 0.0]))
    print(len(df[df.steering < 0.0]))
    print(len(df[df.steering == 0.0]))

    lines = __load_logs(base_path, log_file_name)
    cCounter = 0
    lCounter = 0
    rCounter = 0
    for line in lines:
        cCounter += 1 if float(line[3])==0.0 else 0
        lCounter += 1 if float(line[3])<0.0 else 0
        rCounter += 1 if float(line[3])>0.0 else 0
    print('c: {}\tl: {}\tr:{}'.format(cCounter, lCounter, rCounter))

def ParseBoolean(b):
    # ...
    if len(b) < 1:
        raise ValueError ('Cannot parse empty string into boolean.')
    b = b[0].lower()
    if b == 't' or b == 'y' or b == '1':
        return True
    if b == 'f' or b == 'n' or b == '0':
        return False
    raise ValueError ('Cannot parse string into boolean.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleaning Data')
    parser.add_argument(
        '--source_folder',
        default='~/data/',
        type=str,
        help='Path to files.'
    )
    parser.add_argument(
        '--log_file',
        default='driving_log.csv',
        type=str,
        help='Name of log file'
    )
    parser.add_argument(
        '--clean_only',
        default=False,
        type=ParseBoolean,
        nargs='?',
        help='Clean dataset'
    )
    parser.add_argument(
        '--balance_only',
        default=False,
        type=ParseBoolean,
        nargs='?',
        help='Balance dataset'
    )
    parser.add_argument(
        '--test_only',
        default=False,
        type=ParseBoolean,
        nargs='?',
        help='Test dataset'
    )
    parser.add_argument(
        '--stats',
        default=False,
        type=ParseBoolean,
        nargs='?',
        help='stats of dataset'
    )
    args = parser.parse_args()

    if args.stats:
        print_stats(args.source_folder, args.log_file)
    elif args.test_only:
        test(args.source_folder)
    elif args.clean_only:
        clean_logs(args.source_folder)
    elif args.balance_only:
        balance_logs(args.source_folder, args.log_file)
    else:
        do(args.source_folder)
