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

# python get_video_frame_extraction_other.py --data_name 'ucf101' --video_dir '/raid/yinghua/VRank/data/UCF-101' --save_path_x '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x_32.pkl' --save_path_y '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_y.pkl'
# python get_video_frame_extraction_other.py --data_name 'hmdb51' --video_dir '/raid/yinghua/VRank/data/hmdb51' --save_path_x '/raid/yinghua/VRank/data/pkl_data/hmdb51/hmdb51_x_4.pkl' --save_path_y '/raid/yinghua/VRank/data/pkl_data/hmdb51/hmdb51_y.pkl'
# python get_video_frame_extraction_other.py --data_name 'accident' --video_dir '/raid/yinghua/VRank/data/hwid12/Video-Accident-Dataset' --save_path_x '/raid/yinghua/VRank/data/pkl_data/accident/accident_x_8.pkl' --save_path_y '/raid/yinghua/VRank/data/pkl_data/accident/accident_y.pkl'


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.mp4'):
                # if file_absolute_path.endswith('.avi'):
                    path_list.append(file_absolute_path)
    return path_list


def get_pic_list_8(path_video):

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
        if len(pic_list) >= 8:
            break
    if len(pic_list) < 8:
        add_pic_list = [pic_list[-1]] * (8-len(pic_list))
        pic_list += add_pic_list
    video_pic_np = np.array(pic_list)
    return video_pic_np


def get_pic_list_4(path_video):

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
        if len(pic_list) >= 4:
            break
    if len(pic_list) < 4:
        add_pic_list = [pic_list[-1]] * (4-len(pic_list))
        pic_list += add_pic_list
    video_pic_np = np.array(pic_list)
    return video_pic_np


def get_pic_list_32(path_video):

    interval = 2
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
        if len(pic_list) >= 32:
            break
    if len(pic_list) < 32:
        add_pic_list = [pic_list[-1]] * (32-len(pic_list))
        pic_list += add_pic_list
    video_pic_np = np.array(pic_list)
    return video_pic_np


def get_all_pic_list(path_video_list, label_np):
    all_video_pic_np = []
    all_label_np = []
    for i in range(len(path_video_list)):
        try:
            # video_pic_np = get_pic_list_4(path_video_list[i])
            # video_pic_np = get_pic_list_8(path_video_list[i])
            video_pic_np = get_pic_list_32(path_video_list[i])
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
    if args.data_name == 'hmdb51':
        dic = dic_hmdb51
    if args.data_name == 'accident':
        dic = dic_accident

    path_video_list_ = get_path(args.video_dir)
    video_name_list_ = [i.split('/')[-2] for i in path_video_list_] # hmdb51, Video-Accident-Dataset

    path_video_list = []
    video_name_list = []
    for i in range(len(video_name_list_)):
        if video_name_list_[i] in dic:
            video_name_list.append(video_name_list_[i])
            path_video_list.append(path_video_list_[i])
    print(len(video_name_list))

    label_np = np.array([dic[i] for i in video_name_list])

    all_video_pic_np, all_label_np = get_all_pic_list(path_video_list, label_np)

    pickle.dump(all_video_pic_np, open(args.save_path_x, 'wb'), protocol=4)


if __name__ == '__main__':
    main()


