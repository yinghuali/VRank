import cv2
import numpy as np
import os
import pickle
import argparse
from config import *
ap = argparse.ArgumentParser()
ap.add_argument("--data_name", type=str)
ap.add_argument("--video_dir", type=str)
ap.add_argument("--save_path_x", type=str)
ap.add_argument("--save_path_y", type=str)
args = ap.parse_args()

# python get_video_frame_extraction.py --data_name 'ucf101' --video_dir './data/UCF-101' --save_path_x './data/pkl_data/ucf101/ucf101_x.pkl' --save_path_y './data/pkl_data/ucf101/ucf101_y.pkl'

# mac
# python get_video_frame_extraction.py --data_name 'ucf101' --video_dir './data' --save_path_x './data/pkl_data/ucf101/ucf101_x.pkl' --save_path_y './data/pkl_data/ucf101/ucf101_y.pkl'


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.avi'):
                    path_list.append(file_absolute_path)
    return path_list


def get_pic_list(path_video):

    interval = 4
    crop_size = (112, 112)

    video_capture = cv2.VideoCapture(path_video)

    i = 0
    pic_list = []
    while True:
        success, frame = video_capture.read()
        i += 1
        if i % interval == 0:
            frame = cv2.resize(frame, crop_size, interpolation=cv2.INTER_CUBIC)
            pic_list.append(frame)
        if len(pic_list) >= 16:
            break
    if len(pic_list) < 16:
        add_pic_list = [pic_list[-1]] * (16-len(pic_list))
        pic_list += add_pic_list
    video_pic_np = np.array(pic_list)
    return video_pic_np


def get_all_pic_list(path_video_list, label_np):
    all_video_pic_np = []
    all_label_np = []
    for i in range(len(path_video_list)):
        try:
            video_pic_np = get_pic_list(path_video_list[i])
            all_video_pic_np.append(video_pic_np)
            all_label_np.append(label_np[i])
        except:
            continue
    all_video_pic_np = np.array(all_video_pic_np)
    all_label_np = np.array(all_label_np)
    return all_video_pic_np, all_label_np


def main():
    if args.data_name == 'ucf101':
        dic = dic_ucf101

    path_video_list = get_path(args.video_dir)
    video_name_list = [i.split('/')[-2] for i in path_video_list]
    label_np = np.array([dic[i] for i in video_name_list])

    all_video_pic_np, all_label_np = get_all_pic_list(path_video_list, label_np)

    pickle.dump(all_video_pic_np, open(args.save_path_x, 'wb'), protocol=4)
    pickle.dump(all_label_np, open(args.save_path_y, 'wb'), protocol=4)

    print(len(all_label_np))


if __name__ == '__main__':
    main()
