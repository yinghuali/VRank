# source activate py37
import json
import pickle
import argparse
from utils import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from models.video_transformer import get_vt_model, build_feature_extractor
from tensorflow.keras.models import load_model
from models.video_transformer import PositionalEmbedding, TransformerEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from get_rank_idx import *
from scipy.stats import entropy
from get_frame_fearure import get_all_frame_feaure


path_model_save = './target_models/accident_vt.h5'
path_frame = './pkl_data/accident/accident_x.pkl'
path_x = './pkl_data/accident/vt_accident_x.pkl'
path_y = './pkl_data/accident/accident_y.pkl'
path_x_embedding = './pkl_data/accident/accident_x_embedding.pkl'
save_path = './results/vt_accident_x.json'
num_classes = 12


model = load_model(path_model_save, custom_objects={'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder})


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
    frame_video = pickle.load(open(path_frame, 'rb'))

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    val_x = tf.stack(val_x, axis=0)
    test_x = tf.stack(test_x, axis=0)

    M = load_model(path_model_save, custom_objects={'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder})
    val_pre_vec = M.predict(val_x)
    test_pre_vec = M.predict(test_x)

    frame_feature = get_all_frame_feaure(frame_video)

    uncertainty_feature_val = get_uncertainty_feature(val_pre_vec)
    uncertainty_feature_test = get_uncertainty_feature(test_pre_vec)


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

    percentile_95 = np.percentile(concat_train_all_feature, 95, axis=0)
    concat_train_all_feature = concat_train_all_feature / percentile_95
    concat_test_all_feature = concat_test_all_feature / percentile_95

    model = XGBClassifier()
    model.fit(concat_train_all_feature, train_rank_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    xgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = LGBMClassifier(n_estimators=300)
    model.fit(concat_train_all_feature, train_rank_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    lgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = RandomForestClassifier()
    model.fit(concat_train_all_feature, train_rank_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    rf_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = LogisticRegression(solver='liblinear')
    model.fit(concat_train_all_feature, train_rank_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    lr_rank_idx = y_concat_all.argsort()[::-1].copy()

    xgb_apfd = apfd(idx_miss_list, xgb_rank_idx)
    lgb_apfd = apfd(idx_miss_list, lgb_rank_idx)
    rf_apfd = apfd(idx_miss_list,  rf_rank_idx)
    lr_apfd = apfd(idx_miss_list, lr_rank_idx)

    dic = {'xgb_apfd': xgb_apfd,
           'lgb_apfd': lgb_apfd,
           'rf_apfd': rf_apfd,
           'lr_apfd': lr_apfd
           }

    json.dump(dic, open(save_path, 'w'), sort_keys=True, indent=4)
    print(dic)

if __name__ == '__main__':
    main()




