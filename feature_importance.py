import argparse
from xgboost import XGBClassifier
from get_rank_idx import *
from scipy.stats import entropy
from get_frame_fearure import get_all_frame_feaure
import json


path_x = './pkl_data/ucf101/ucf101_x.pkl'
path_y = './pkl_data/ucf101/ucf101_y.pkl'
path_val_pre = './target_models/ucf101_slowfastnet_24_val_pre.pkl'
path_test_pre = './target_models/ucf101_slowfastnet_24_test_pre.pkl'
path_x_embedding = './pkl_data/ucf101/ucf101_x_embedding.pkl'
save_path_name = 'ucf101_slowfastnet_24'


def get_uncertainty_feature(x):
    margin_score = np.sort(x)[:, -1] - np.sort(x)[:, -2]
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    least_score = x.max(1)
    VanillaSoftmax_score = 1 - x.max(1)
    PCS_score = 1 - (np.sort(x)[:, -1] - np.sort(x)[:, -2])
    entropy_score = entropy(np.array([i / np.sum(i) for i in x]), axis=1)

    feature_vec = np.vstack((margin_score, gini_score, least_score, VanillaSoftmax_score, PCS_score, entropy_score))
    return feature_vec.T



def main():
    x = pickle.load(open(path_x, 'rb'))
    x_embedding = pickle.load(open(path_x_embedding, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    val_pre_vec = pickle.load(open(path_val_pre, 'rb'))
    test_pre_vec = pickle.load(open(path_test_pre, 'rb'))

    frame_feature = get_all_frame_feaure(x)

    uncertainty_feature_val = get_uncertainty_feature(val_pre_vec)
    uncertainty_feature_test = get_uncertainty_feature(test_pre_vec)

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    train_x_embedding_, test_x_embedding, _, _ = train_test_split(x_embedding, y, test_size=0.3, random_state=17)
    train_x_embedding, val_x_embedding, _, _ = train_test_split(train_x_embedding_, train_y_, test_size=0.3, random_state=17)

    train_frame_feature_, test_frame_feature, _, _ = train_test_split(frame_feature, y, test_size=0.3, random_state=17)
    train_frame_feature, val_frame_feature, _, _ = train_test_split(train_frame_feature_, train_y_, test_size=0.3, random_state=17)


    all_pre_vec = np.vstack((uncertainty_feature_val, uncertainty_feature_test))
    all_uncertainty_feature = np.vstack((val_pre_vec, test_pre_vec))
    all_embedding = np.vstack((val_x_embedding, test_x_embedding))
    all_frame_feature = np.vstack((val_frame_feature, test_frame_feature))
    all_y = np.hstack((val_y, test_y))

    uncertainty_feature_train, uncertainty_feature_test, train_y, test_y = train_test_split(all_pre_vec, all_y, test_size=0.3, random_state=17)
    train_pre_vec, test_pre_vec, _, _ = train_test_split(all_uncertainty_feature, all_y, test_size=0.3, random_state=17)
    train_x_embedding, test_x_embedding, _, _ = train_test_split(all_embedding, all_y, test_size=0.3, random_state=17)
    train_frame_feature, test_frame_feature, _, _ = train_test_split(all_frame_feature, all_y, test_size=0.3, random_state=17)


    train_pre = train_pre_vec.argsort()[:, -1]
    test_pre = test_pre_vec.argsort()[:, -1]

    print(len(train_pre))
    print(len(test_pre))

    train_rank_label = get_rank_label(train_pre, train_y)
    idx_miss_list = get_idx_miss_class(test_pre, test_y)

    concat_train_all_feature = np.hstack((uncertainty_feature_train, train_pre_vec, train_x_embedding, train_frame_feature))
    concat_test_all_feature = np.hstack((uncertainty_feature_test, test_pre_vec, test_x_embedding, test_frame_feature))

    model = XGBClassifier(importance_type='cover')
    model.fit(concat_train_all_feature, train_rank_label)
    importance = model.get_booster().get_score(importance_type='cover')
    dic = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    json.dump(dic, open('./results/' + save_path_name + '_cover.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
