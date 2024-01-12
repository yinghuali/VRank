import pickle
import numpy as np


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


def get_pearson_frame(x):
    all_feature = []
    for video in x:
        tmp_feature = []
        n_frame = len(video)
        for i in range(1, n_frame):
            current_vec = video[i].reshape(-1)
            previous_vec = video[i-1].reshape(-1)
            diff = np.corrcoef(current_vec, previous_vec)[0, 1]
            tmp_feature.append(diff)
        all_feature.append(tmp_feature)
    all_feature = np.array(all_feature)
    return all_feature


def get_average_frame(x):
    all_feature = []
    for video in x:
        tmp_feature = []
        for frame in video:
            tmp_feature.append(np.mean(frame))
        all_feature.append(tmp_feature)
    all_feature = np.array(all_feature)
    return all_feature


def get_statistic_frame(x):
    all_feature = []
    for video in x:
        tmp_feature = []
        for frame in video:
            tmp_feature.append(np.mean(frame))
        var_feaure = np.var(tmp_feature)
        mean_feature = np.mean(tmp_feature)
        median_feaure = np.median(tmp_feature)
        video_feature = [var_feaure, mean_feature, median_feaure]
        all_feature.append(video_feature)
    all_feature = np.array(all_feature)
    return all_feature


def get_all_time_feaure(x):
    feature_euclidean = get_euclidean_frame(x)
    feature_manhattan = get_manhattan_frame(x)
    feature_squared = get_squared_frame(x)
    feature_pearson = get_pearson_frame(x)
    feature_average = get_average_frame(x)
    featur_statistic = get_statistic_frame(x)
    frame_diff = np.hstack((feature_euclidean, feature_manhattan, feature_squared, feature_pearson, feature_average, featur_statistic))
    return frame_diff


# if __name__ == '__main__':
#     path_x = './pkl_data/accident/accident_x.pkl'
#     x = pickle.load(open(path_x, 'rb'))
#     print(x.shape)
#     frame_diff = get_all_time_feaure(x)
#     print(frame_diff.shape)

