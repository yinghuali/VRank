import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
import numpy as np
import argparse

path_x = './pkl_data/accident/accident_x.pkl'


def get_euclidean_frame(x):
    all_feature = []
    for video in x:
        tmp_feature = []
        n_frame = len(video)
        for i in range(1, n_frame):
            current_vec = video[i].reshape(-1)
            previous_vec = video[i-1].reshape(-1)
            diff = np.linalg.norm(current_vec - previous_vec)
            tmp_feature.append(diff)
        all_feature.append(tmp_feature)
    all_feature = np.array(all_feature)
    return all_feature


def get_manhattan_frame(x):
    all_feature = []
    for video in x:
        tmp_feature = []
        n_frame = len(video)
        for i in range(1, n_frame):
            current_vec = video[i].reshape(-1)
            previous_vec = video[i-1].reshape(-1)
            diff = np.sum(np.abs(np.array(current_vec) - np.array(previous_vec)))
            tmp_feature.append(diff)
        all_feature.append(tmp_feature)
    all_feature = np.array(all_feature)
    return all_feature


def get_squared_frame(x):
    all_feature = []
    for video in x:
        tmp_feature = []
        n_frame = len(video)
        for i in range(1, n_frame):
            current_vec = video[i].reshape(-1)
            previous_vec = video[i-1].reshape(-1)
            diff = np.sum((current_vec - previous_vec) ** 2)
            tmp_feature.append(diff)
        all_feature.append(tmp_feature)
    all_feature = np.array(all_feature)
    return all_feature


def main():
    x = pickle.load(open(path_x, 'rb'))

    print(x.shape)
    feature = get_manhattan_frame(x)
    print(feature.shape)


if __name__ == '__main__':
    main()

