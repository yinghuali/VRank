import pickle
import numpy as np
import argparse
from PIL import Image
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from utils import *
from get_rank_idx import *

path_x = '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x.pkl'
path_y = '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_y.pkl'
path_val_pre = './target_models/ucf101_slowfastnet_24_val_pre.pkl'
path_test_pre = './target_models/ucf101_slowfastnet_24_test_pre.pkl'
path_x_embedding = '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x_embedding.pkl'


def get_uncertainty_feature(x):
    margin_score = np.sort(x)[:, -1] - np.sort(x)[:, -2]
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    least_score = x.max(1)
    VanillaSoftmax_score = 1 - x.max(1)
    PCS_score = 1 - (np.sort(x)[:, -1] - np.sort(x)[:, -2])
    entropy_score = entropy(np.array([i / np.sum(i) for i in x]), axis=1)

    feature_vec = np.vstack((margin_score, gini_score, least_score, VanillaSoftmax_score, PCS_score, entropy_score))
    return feature_vec.T


def get_compress_feature(video):
    """
    return compress feature
    :param video: video shape = [48, 112, 112]
    :return: compress_feature shape = [1, 100]
    """
    x_mean = np.mean(video, axis=0).astype(np.int8)
    image = Image.fromarray(x_mean)
    resized_image = image.resize((10, 10))
    compress_feature = np.array(resized_image)
    compress_feature = compress_feature.flatten()
    return compress_feature


def main():
    x = pickle.load(open(path_x, 'rb'))  # (13038, 16, 112, 112, 3)
    x_embedding = pickle.load(open(path_x_embedding, 'rb'))  # (13038, 2048)
    y = pickle.load(open(path_y, 'rb'))

    shape = x.shape
    new_x = x.reshape(shape[0], shape[1] * shape[4], shape[2], shape[3]) # (13038, 48, 112, 112)
    compress_feature = []
    for video in new_x:
        feature = get_compress_feature(video)
        compress_feature.append(feature)
    compress_feature = np.array(compress_feature)
    print(compress_feature.shape)

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    train_x_embedding_, test_x_embedding, _, _ = train_test_split(x_embedding, y, test_size=0.3, random_state=17)
    train_x_embedding, val_x_embedding, _, _ = train_test_split(train_x_embedding_, train_y_, test_size=0.3, random_state=17)

    compress_feature_train_x_, compress_feature_test_x, _, _ = train_test_split(compress_feature, y, test_size=0.3, random_state=17)
    compress_feature_train_x, compress_feature_val_x, _, _ = train_test_split(compress_feature_train_x_, train_y_, test_size=0.3, random_state=17)

    val_pre_vec = pickle.load(open(path_val_pre, 'rb'))
    test_pre_vec = pickle.load(open(path_test_pre, 'rb'))

    uncertainty_feature_val = get_uncertainty_feature(val_pre_vec)
    uncertainty_feature_test = get_uncertainty_feature(test_pre_vec)

    y = np.array(list(val_y) + list(test_y))
    pre_vec = np.vstack((val_pre_vec, test_pre_vec))
    compress_feature = np.vstack((compress_feature_val_x, compress_feature_test_x))
    uncertainty_feature = np.vstack((uncertainty_feature_val, uncertainty_feature_test))
    embedding_feature = np.vstack((val_x_embedding, test_x_embedding))

    train_pre_vec, test_pre_vec, train_y, test_y = train_test_split(pre_vec, y, test_size=0.3, random_state=17)
    train_compress_feature, test_compress_feature, _, _ = train_test_split(compress_feature, y, test_size=0.3, random_state=17)
    train_uncertainty_feature, test_uncertainty_feature, _, _ = train_test_split(uncertainty_feature, y, test_size=0.3, random_state=17)
    train_embedding_feature, test_embedding_feature, _, _ = train_test_split(embedding_feature, y, test_size=0.3, random_state=17)

    concat_train_all_feature = np.hstack((train_pre_vec, train_compress_feature, train_uncertainty_feature, train_embedding_feature))
    concat_test_all_feature = np.hstack((test_pre_vec, test_compress_feature, test_uncertainty_feature, test_embedding_feature))

    train_pre = train_pre_vec.argsort()[:, -1]
    test_pre = test_pre_vec.argsort()[:, -1]

    print('train:', len(train_pre))
    print('test:', len(test_pre))

    train_rank_label = get_rank_label(train_pre, train_y)
    idx_miss_list = get_idx_miss_class(test_pre, test_y)

    model = LGBMClassifier(n_estimators=300, max_depth=5)
    model.fit(concat_train_all_feature, train_rank_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    deepGini_rank_idx = DeepGini_rank_idx(test_pre_vec)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(test_pre_vec)
    pcs_rank_idx = PCS_rank_idx(test_pre_vec)
    entropy_rank_idx = Entropy_rank_idx(test_pre_vec)
    random_rank_idx = Random_rank_idx(test_pre_vec)

    lgb_apfd = apfd(idx_miss_list, lgb_rank_idx)
    deepGini_apfd = apfd(idx_miss_list, deepGini_rank_idx)
    random_apfd = apfd(idx_miss_list, random_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_list, entropy_rank_idx)

    dic = {'lgb': lgb_apfd,
           'deepGini': deepGini_apfd,
           'vanillasoftmax': vanillasoftmax_apfd,
           'pcs': pcs_apfd,
           'entropy': entropy_apfd,
           'random': random_apfd
           }
    print(dic)


if __name__ == '__main__':
    main()

