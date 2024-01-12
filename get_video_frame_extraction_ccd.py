import cv2
import numpy as np
import os
import pickle
import argparse
import cv2
from config import *


path_dir_compile = './data/CrashBest'
select_frame = ['01', '05', '09', '13',
               '17', '21', '25', '29',
               '33', '37', '41', '45',
               '46', '47', '48', '49']


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.jpg'):
                    path_list.append(file_absolute_path)
    return path_list


def get_video_np(path_dir_compile):

    path_list = get_path(path_dir_compile)
    path_list = [i[:-6] for i in path_list]
    path_list = sorted(list(set(path_list)))
    key_list = [i.split('_')[1] for i in path_list]
    label = [dic_ccd[i] for i in key_list]

    crop_size = (112, 112)
    all_frame_list = []
    j = 0
    for path in path_list:
        frame_list = []
        for i in select_frame:
            frame_path = path+i+'.jpg'
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
            frame_list.append(frame)
        all_frame_list.append(frame_list)
        write_result(str(j), 'ccd.txt')
        j += 1
    all_frame_np = np.array(all_frame_list)
    label_np = np.array(label)
    return all_frame_np, label_np


def main():
    all_frame_np, label_np = get_video_np(path_dir_compile)
    pickle.dump(all_frame_np, open('./pkl_data/ccd/ccd_x.pkl', 'wb'), protocol=4)
    pickle.dump(label_np, open('./pkl_data/ccd/ccd_y.pkl', 'wb'), protocol=4)
    print(all_frame_np.shape)
    print(label_np.shape)


if __name__ == '__main__':
    main()
