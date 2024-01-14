from utils import *
import pickle
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from get_rank_idx import *
from scipy.stats import entropy
from get_frame_fearure import get_all_frame_feaure


ap = argparse.ArgumentParser()
ap.add_argument("--data_name", type=str, default='')
ap.add_argument("--model_name", type=str, default='')
args = vars(ap.parse_args())

data_name = args['data_name']
model_name = args['model_name']

save_path_name = 'apfd_{}_{}.csv'.format(data_name, model_name)


path_x_1 = './pkl_data/{}/{}_x.pkl'.format(data_name, data_name)
path_y_1 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_1 = './target_models/{}_{}_val_pre.pkl'.format(data_name, model_name)
path_test_pre_1 = './target_models/{}_{}_test_pre.pkl'.format(data_name, model_name)
path_x_embedding_1 = './pkl_data/{}/{}_x_embedding.pkl'.format(data_name, data_name)

path_x_2 = './pkl_data/{}_noise/augmentation_channel_shift_range_x.pkl'.format(data_name)
path_y_2 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_2 = './pkl_data/{}_noise/{}_{}_val_augmentation_channel_shift_range_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_2 = './pkl_data/{}_noise/{}_{}_test_augmentation_channel_shift_range_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_2 = './pkl_data/{}_noise/augmentation_channel_shift_range_x_embedding.pkl'.format(data_name)

path_x_3 = './pkl_data/{}_noise/augmentation_featurewise_std_normalization_x.pkl'.format(data_name)
path_y_3 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_3 = './pkl_data/{}_noise/{}_{}_val_augmentation_featurewise_std_normalization_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_3 = './pkl_data/{}_noise/{}_{}_test_augmentation_featurewise_std_normalization_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_3 = './pkl_data/{}_noise/augmentation_featurewise_std_normalization_x_embedding.pkl'.format(data_name)

path_x_4 = './pkl_data/{}_noise/augmentation_height_shift_x.pkl'.format(data_name)
path_y_4 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_4 = './pkl_data/{}_noise/{}_{}_val_augmentation_height_shift_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_4 = './pkl_data/{}_noise/{}_{}_test_augmentation_height_shift_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_4 = './pkl_data/{}_noise/augmentation_height_shift_x_embedding.pkl'.format(data_name)

path_x_5 = './pkl_data/{}_noise/augmentation_horizontal_flip_x.pkl'.format(data_name)
path_y_5 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_5 = './pkl_data/{}_noise/{}_{}_val_augmentation_horizontal_flip_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_5 = './pkl_data/{}_noise/{}_{}_test_augmentation_horizontal_flip_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_5 = './pkl_data/{}_noise/augmentation_horizontal_flip_x_embedding.pkl'.format(data_name)

path_x_6 = './pkl_data/{}_noise/augmentation_shear_range_x.pkl'.format(data_name)
path_y_6 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_6 = './pkl_data/{}_noise/{}_{}_val_augmentation_shear_range_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_6 = './pkl_data/{}_noise/{}_{}_test_augmentation_shear_range_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_6 = './pkl_data/{}_noise/augmentation_shear_range_x_embedding.pkl'.format(data_name)

path_x_7 = './pkl_data/{}_noise/augmentation_width_shift_x.pkl'.format(data_name)
path_y_7 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_7 = './pkl_data/{}_noise/{}_{}_val_augmentation_width_shift_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_7 = './pkl_data/{}_noise/{}_{}_test_augmentation_width_shift_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_7 = './pkl_data/{}_noise/augmentation_width_shift_x_embedding.pkl'.format(data_name)

path_x_8 = './pkl_data/{}_noise/augmentation_zca_whitening_x.pkl'.format(data_name)
path_y_8 = './pkl_data/{}/{}_y.pkl'.format(data_name, data_name)
path_val_pre_8 = './pkl_data/{}_noise/{}_{}_val_augmentation_zca_whitening_x_pre.pkl'.format(data_name, data_name, model_name)
path_test_pre_8 = './pkl_data/{}_noise/{}_{}_test_augmentation_zca_whitening_x_pre.pkl'.format(data_name, data_name, model_name)
path_x_embedding_8 = './pkl_data/{}_noise/augmentation_zca_whitening_x_embedding.pkl'.format(data_name)


def get_all_data(path_x, path_x_embedding, path_y, path_val_pre, path_test_pre):

    x = pickle.load(open(path_x, 'rb'))
    x_embedding = pickle.load(open(path_x_embedding, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    val_pre_vec = pickle.load(open(path_val_pre, 'rb'))
    test_pre_vec = pickle.load(open(path_test_pre, 'rb'))


    uncertainty_feature_val = get_uncertainty_feature(val_pre_vec)
    uncertainty_feature_test = get_uncertainty_feature(test_pre_vec)

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    train_x_embedding_, test_x_embedding, _, _ = train_test_split(x_embedding, y, test_size=0.3, random_state=17)
    train_x_embedding, val_x_embedding, _, _ = train_test_split(train_x_embedding_, train_y_, test_size=0.3, random_state=17)

    frame_feature = get_all_frame_feaure(x)
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

    nan_exists = np.isnan(concat_train_all_feature).any()
    inf_exists = np.isinf(concat_train_all_feature).any()
    if nan_exists:
        concat_train_all_feature = np.nan_to_num(concat_train_all_feature, nan=0.0)
    if inf_exists:
        concat_train_all_feature[concat_train_all_feature == np.inf] = np.finfo(np.float32).max
        concat_train_all_feature[concat_train_all_feature == -np.inf] = np.finfo(np.float32).min

    nan_exists = np.isnan(concat_test_all_feature).any()
    inf_exists = np.isinf(concat_test_all_feature).any()
    if nan_exists:
        concat_test_all_feature = np.nan_to_num(concat_test_all_feature, nan=0.0)
    if inf_exists:
        concat_test_all_feature[concat_test_all_feature == np.inf] = np.finfo(np.float32).max
        concat_test_all_feature[concat_test_all_feature == -np.inf] = np.finfo(np.float32).min

    # percentile_95 = np.percentile(concat_train_all_feature, 95, axis=0)
    # concat_train_all_feature = concat_train_all_feature / percentile_95
    # concat_test_all_feature = concat_test_all_feature / percentile_95

    return train_rank_label, idx_miss_list, concat_train_all_feature, concat_test_all_feature, test_pre_vec


def get_uncertainty_feature(x):
    margin_score = np.sort(x)[:, -1] - np.sort(x)[:, -2]
    gini_score = 1 - np.sum(np.power(x, 2), axis=1)
    least_score = x.max(1)
    VanillaSoftmax_score = 1 - x.max(1)
    PCS_score = 1 - (np.sort(x)[:, -1] - np.sort(x)[:, -2])
    entropy_score = entropy(np.array([i / np.sum(i) for i in x]), axis=1)

    feature_vec = np.vstack((margin_score,gini_score, least_score, VanillaSoftmax_score, PCS_score, entropy_score))
    return feature_vec.T


def main():
    train_rank_label_1, idx_miss_list_1, concat_train_all_feature_1, concat_test_all_feature_1, test_pre_vec_1 = \
        get_all_data(path_x_1, path_x_embedding_1, path_y_1, path_val_pre_1, path_test_pre_1)
    print('===1===')
    train_rank_label_2, idx_miss_list_2, concat_train_all_feature_2, concat_test_all_feature_2, test_pre_vec_2 = \
        get_all_data(path_x_2, path_x_embedding_2, path_y_2, path_val_pre_2, path_test_pre_2)
    print('===2===')
    train_rank_label_3, idx_miss_list_3, concat_train_all_feature_3, concat_test_all_feature_3, test_pre_vec_3 = \
        get_all_data(path_x_3, path_x_embedding_3, path_y_3, path_val_pre_3, path_test_pre_3)
    print('===3===')
    train_rank_label_4, idx_miss_list_4, concat_train_all_feature_4, concat_test_all_feature_4, test_pre_vec_4 = \
        get_all_data(path_x_4, path_x_embedding_4, path_y_4, path_val_pre_4, path_test_pre_4)
    print('===4===')
    train_rank_label_5, idx_miss_list_5, concat_train_all_feature_5, concat_test_all_feature_5, test_pre_vec_5 = \
        get_all_data(path_x_5, path_x_embedding_5, path_y_5, path_val_pre_5, path_test_pre_5)
    print('===5===')
    train_rank_label_6, idx_miss_list_6, concat_train_all_feature_6, concat_test_all_feature_6, test_pre_vec_6 = \
        get_all_data(path_x_6, path_x_embedding_6, path_y_6, path_val_pre_6, path_test_pre_6)
    print('===6===')
    train_rank_label_7, idx_miss_list_7, concat_train_all_feature_7, concat_test_all_feature_7, test_pre_vec_7 = \
        get_all_data(path_x_7, path_x_embedding_7, path_y_7, path_val_pre_7, path_test_pre_7)
    print('===7===')
    train_rank_label_8, idx_miss_list_8, concat_train_all_feature_8, concat_test_all_feature_8, test_pre_vec_8 = \
        get_all_data(path_x_8, path_x_embedding_8, path_y_8, path_val_pre_8, path_test_pre_8)
    print('===8===')
    concat_test_all_feature = np.concatenate((concat_test_all_feature_1, concat_test_all_feature_2, concat_test_all_feature_3, concat_test_all_feature_4, concat_test_all_feature_5, concat_test_all_feature_6, concat_test_all_feature_7, concat_test_all_feature_8), axis=0)
    concat_train_all_feature = np.concatenate((concat_train_all_feature_1, concat_train_all_feature_2, concat_train_all_feature_3, concat_train_all_feature_4, concat_train_all_feature_5, concat_train_all_feature_6, concat_train_all_feature_7, concat_train_all_feature_8), axis=0)
    train_rank_label = np.concatenate((train_rank_label_1, train_rank_label_2, train_rank_label_3, train_rank_label_4, train_rank_label_5, train_rank_label_6, train_rank_label_7, train_rank_label_8), axis=0)
    test_pre_vec = np.concatenate((test_pre_vec_1, test_pre_vec_2, test_pre_vec_3, test_pre_vec_4, test_pre_vec_5, test_pre_vec_6, test_pre_vec_7, test_pre_vec_8), axis=0)

    def get_model_idx(model_name):
        if model_name=='xgb':
            model = XGBClassifier()
        if model_name=='lgb':
            model = LGBMClassifier(n_estimators=300)
        if model_name=='rf':
            model = RandomForestClassifier()
        if model_name=='lr':
            model = LogisticRegression(solver='liblinear')
        model.fit(concat_train_all_feature, train_rank_label)
        y_concat_all_1 = model.predict_proba(concat_test_all_feature_1)[:, 1]
        y_concat_all_2 = model.predict_proba(concat_test_all_feature_2)[:, 1]
        y_concat_all_3 = model.predict_proba(concat_test_all_feature_3)[:, 1]
        y_concat_all_4 = model.predict_proba(concat_test_all_feature_4)[:, 1]
        y_concat_all_5 = model.predict_proba(concat_test_all_feature_5)[:, 1]
        y_concat_all_6 = model.predict_proba(concat_test_all_feature_6)[:, 1]
        y_concat_all_7 = model.predict_proba(concat_test_all_feature_7)[:, 1]
        y_concat_all_8 = model.predict_proba(concat_test_all_feature_8)[:, 1]

        rank_idx_1 = y_concat_all_1.argsort()[::-1].copy()
        rank_idx_2 = y_concat_all_2.argsort()[::-1].copy()
        rank_idx_3 = y_concat_all_3.argsort()[::-1].copy()
        rank_idx_4 = y_concat_all_4.argsort()[::-1].copy()
        rank_idx_5 = y_concat_all_5.argsort()[::-1].copy()
        rank_idx_6 = y_concat_all_6.argsort()[::-1].copy()
        rank_idx_7 = y_concat_all_7.argsort()[::-1].copy()
        rank_idx_8 = y_concat_all_8.argsort()[::-1].copy()
        return rank_idx_1, rank_idx_2, rank_idx_3, rank_idx_4, rank_idx_5, rank_idx_6, rank_idx_7, rank_idx_8



    xgb_rank_idx_1, xgb_rank_idx_2, xgb_rank_idx_3, xgb_rank_idx_4, xgb_rank_idx_5, xgb_rank_idx_6, xgb_rank_idx_7, xgb_rank_idx_8 = get_model_idx('xgb')
    lgb_rank_idx_1, lgb_rank_idx_2, lgb_rank_idx_3, lgb_rank_idx_4, lgb_rank_idx_5, lgb_rank_idx_6, lgb_rank_idx_7, lgb_rank_idx_8 = get_model_idx('lgb')
    rf_rank_idx_1, rf_rank_idx_2, rf_rank_idx_3, rf_rank_idx_4, rf_rank_idx_5, rf_rank_idx_6, rf_rank_idx_7, rf_rank_idx_8 = get_model_idx('rf')
    lr_rank_idx_1, lr_rank_idx_2, lr_rank_idx_3, lr_rank_idx_4, lr_rank_idx_5, lr_rank_idx_6, lr_rank_idx_7, lr_rank_idx_8 = get_model_idx('lr')

    deepGini_rank_idx_1 = DeepGini_rank_idx(test_pre_vec_1)
    deepGini_rank_idx_2 = DeepGini_rank_idx(test_pre_vec_2)
    deepGini_rank_idx_3 = DeepGini_rank_idx(test_pre_vec_3)
    deepGini_rank_idx_4 = DeepGini_rank_idx(test_pre_vec_4)
    deepGini_rank_idx_5 = DeepGini_rank_idx(test_pre_vec_5)
    deepGini_rank_idx_6 = DeepGini_rank_idx(test_pre_vec_6)
    deepGini_rank_idx_7 = DeepGini_rank_idx(test_pre_vec_7)
    deepGini_rank_idx_8 = DeepGini_rank_idx(test_pre_vec_8)

    VanillaSoftmax_rank_idx_1 = VanillaSoftmax_rank_idx(test_pre_vec_1)
    VanillaSoftmax_rank_idx_2 = VanillaSoftmax_rank_idx(test_pre_vec_2)
    VanillaSoftmax_rank_idx_3 = VanillaSoftmax_rank_idx(test_pre_vec_3)
    VanillaSoftmax_rank_idx_4 = VanillaSoftmax_rank_idx(test_pre_vec_4)
    VanillaSoftmax_rank_idx_5 = VanillaSoftmax_rank_idx(test_pre_vec_5)
    VanillaSoftmax_rank_idx_6 = VanillaSoftmax_rank_idx(test_pre_vec_6)
    VanillaSoftmax_rank_idx_7 = VanillaSoftmax_rank_idx(test_pre_vec_7)
    VanillaSoftmax_rank_idx_8 = VanillaSoftmax_rank_idx(test_pre_vec_8)

    PCS_rank_idx_1 = PCS_rank_idx(test_pre_vec_1)
    PCS_rank_idx_2 = PCS_rank_idx(test_pre_vec_2)
    PCS_rank_idx_3 = PCS_rank_idx(test_pre_vec_3)
    PCS_rank_idx_4 = PCS_rank_idx(test_pre_vec_4)
    PCS_rank_idx_5 = PCS_rank_idx(test_pre_vec_5)
    PCS_rank_idx_6 = PCS_rank_idx(test_pre_vec_6)
    PCS_rank_idx_7 = PCS_rank_idx(test_pre_vec_7)
    PCS_rank_idx_8 = PCS_rank_idx(test_pre_vec_8)

    Entropy_rank_idx_1 = Entropy_rank_idx(test_pre_vec_1)
    Entropy_rank_idx_2 = Entropy_rank_idx(test_pre_vec_2)
    Entropy_rank_idx_3 = Entropy_rank_idx(test_pre_vec_3)
    Entropy_rank_idx_4 = Entropy_rank_idx(test_pre_vec_4)
    Entropy_rank_idx_5 = Entropy_rank_idx(test_pre_vec_5)
    Entropy_rank_idx_6 = Entropy_rank_idx(test_pre_vec_6)
    Entropy_rank_idx_7 = Entropy_rank_idx(test_pre_vec_7)
    Entropy_rank_idx_8 = Entropy_rank_idx(test_pre_vec_8)

    Random_rank_idx_1 = Random_rank_idx(test_pre_vec_1)
    Random_rank_idx_2 = Random_rank_idx(test_pre_vec_2)
    Random_rank_idx_3 = Random_rank_idx(test_pre_vec_3)
    Random_rank_idx_4 = Random_rank_idx(test_pre_vec_4)
    Random_rank_idx_5 = Random_rank_idx(test_pre_vec_5)
    Random_rank_idx_6 = Random_rank_idx(test_pre_vec_6)
    Random_rank_idx_7 = Random_rank_idx(test_pre_vec_7)
    Random_rank_idx_8 = Random_rank_idx(test_pre_vec_8)

    xgb_apfd_1 = apfd(idx_miss_list_1, xgb_rank_idx_1)
    lgb_apfd_1 = apfd(idx_miss_list_1, lgb_rank_idx_1)
    rf_apfd_1 = apfd(idx_miss_list_1,  rf_rank_idx_1)
    lr_apfd_1 = apfd(idx_miss_list_1, lr_rank_idx_1)
    deepGini_apfd_1 = apfd(idx_miss_list_1, deepGini_rank_idx_1)
    random_apfd_1 = apfd(idx_miss_list_1, Random_rank_idx_1)
    vanillasoftmax_apfd_1 = apfd(idx_miss_list_1, VanillaSoftmax_rank_idx_1)
    pcs_apfd_1 = apfd(idx_miss_list_1, PCS_rank_idx_1)
    entropy_apfd_1 = apfd(idx_miss_list_1, Entropy_rank_idx_1)

    xgb_apfd_2 = apfd(idx_miss_list_2, xgb_rank_idx_2)
    lgb_apfd_2 = apfd(idx_miss_list_2, lgb_rank_idx_2)
    rf_apfd_2 = apfd(idx_miss_list_2,  rf_rank_idx_2)
    lr_apfd_2 = apfd(idx_miss_list_2, lr_rank_idx_2)
    deepGini_apfd_2 = apfd(idx_miss_list_2, deepGini_rank_idx_2)
    random_apfd_2 = apfd(idx_miss_list_2, Random_rank_idx_2)
    vanillasoftmax_apfd_2 = apfd(idx_miss_list_2, VanillaSoftmax_rank_idx_2)
    pcs_apfd_2 = apfd(idx_miss_list_2, PCS_rank_idx_2)
    entropy_apfd_2 = apfd(idx_miss_list_2, Entropy_rank_idx_2)

    xgb_apfd_3 = apfd(idx_miss_list_3, xgb_rank_idx_3)
    lgb_apfd_3 = apfd(idx_miss_list_3, lgb_rank_idx_3)
    rf_apfd_3 = apfd(idx_miss_list_3,  rf_rank_idx_3)
    lr_apfd_3 = apfd(idx_miss_list_3, lr_rank_idx_3)
    deepGini_apfd_3 = apfd(idx_miss_list_3, deepGini_rank_idx_3)
    random_apfd_3 = apfd(idx_miss_list_3, Random_rank_idx_3)
    vanillasoftmax_apfd_3 = apfd(idx_miss_list_3, VanillaSoftmax_rank_idx_3)
    pcs_apfd_3 = apfd(idx_miss_list_3, PCS_rank_idx_3)
    entropy_apfd_3 = apfd(idx_miss_list_3, Entropy_rank_idx_3)

    xgb_apfd_4 = apfd(idx_miss_list_4, xgb_rank_idx_4)
    lgb_apfd_4 = apfd(idx_miss_list_4, lgb_rank_idx_4)
    rf_apfd_4 = apfd(idx_miss_list_4,  rf_rank_idx_4)
    lr_apfd_4 = apfd(idx_miss_list_4, lr_rank_idx_4)
    deepGini_apfd_4 = apfd(idx_miss_list_4, deepGini_rank_idx_4)
    random_apfd_4 = apfd(idx_miss_list_4, Random_rank_idx_4)
    vanillasoftmax_apfd_4 = apfd(idx_miss_list_4, VanillaSoftmax_rank_idx_4)
    pcs_apfd_4 = apfd(idx_miss_list_4, PCS_rank_idx_4)
    entropy_apfd_4 = apfd(idx_miss_list_4, Entropy_rank_idx_4)

    xgb_apfd_5 = apfd(idx_miss_list_5, xgb_rank_idx_5)
    lgb_apfd_5 = apfd(idx_miss_list_5, lgb_rank_idx_5)
    rf_apfd_5 = apfd(idx_miss_list_5,  rf_rank_idx_5)
    lr_apfd_5 = apfd(idx_miss_list_5, lr_rank_idx_5)
    deepGini_apfd_5 = apfd(idx_miss_list_5, deepGini_rank_idx_5)
    random_apfd_5 = apfd(idx_miss_list_5, Random_rank_idx_5)
    vanillasoftmax_apfd_5 = apfd(idx_miss_list_5, VanillaSoftmax_rank_idx_5)
    pcs_apfd_5 = apfd(idx_miss_list_5, PCS_rank_idx_5)
    entropy_apfd_5 = apfd(idx_miss_list_5, Entropy_rank_idx_5)

    xgb_apfd_6 = apfd(idx_miss_list_6, xgb_rank_idx_6)
    lgb_apfd_6 = apfd(idx_miss_list_6, lgb_rank_idx_6)
    rf_apfd_6 = apfd(idx_miss_list_6,  rf_rank_idx_6)
    lr_apfd_6 = apfd(idx_miss_list_6, lr_rank_idx_6)
    deepGini_apfd_6 = apfd(idx_miss_list_6, deepGini_rank_idx_6)
    random_apfd_6 = apfd(idx_miss_list_6, Random_rank_idx_6)
    vanillasoftmax_apfd_6 = apfd(idx_miss_list_6, VanillaSoftmax_rank_idx_6)
    pcs_apfd_6 = apfd(idx_miss_list_6, PCS_rank_idx_6)
    entropy_apfd_6 = apfd(idx_miss_list_6, Entropy_rank_idx_6)

    xgb_apfd_7 = apfd(idx_miss_list_7, xgb_rank_idx_7)
    lgb_apfd_7 = apfd(idx_miss_list_7, lgb_rank_idx_7)
    rf_apfd_7 = apfd(idx_miss_list_7,  rf_rank_idx_7)
    lr_apfd_7 = apfd(idx_miss_list_7, lr_rank_idx_7)
    deepGini_apfd_7 = apfd(idx_miss_list_7, deepGini_rank_idx_7)
    random_apfd_7 = apfd(idx_miss_list_7, Random_rank_idx_7)
    vanillasoftmax_apfd_7 = apfd(idx_miss_list_7, VanillaSoftmax_rank_idx_7)
    pcs_apfd_7 = apfd(idx_miss_list_7, PCS_rank_idx_7)
    entropy_apfd_7 = apfd(idx_miss_list_7, Entropy_rank_idx_7)

    xgb_apfd_8 = apfd(idx_miss_list_8, xgb_rank_idx_8)
    lgb_apfd_8 = apfd(idx_miss_list_8, lgb_rank_idx_8)
    rf_apfd_8 = apfd(idx_miss_list_8,  rf_rank_idx_8)
    lr_apfd_8 = apfd(idx_miss_list_8, lr_rank_idx_8)
    deepGini_apfd_8 = apfd(idx_miss_list_8, deepGini_rank_idx_8)
    random_apfd_8 = apfd(idx_miss_list_8, Random_rank_idx_8)
    vanillasoftmax_apfd_8 = apfd(idx_miss_list_8, VanillaSoftmax_rank_idx_8)
    pcs_apfd_8 = apfd(idx_miss_list_8, PCS_rank_idx_8)
    entropy_apfd_8 = apfd(idx_miss_list_8, Entropy_rank_idx_8)

    df_apfd = pd.DataFrame(columns=['name'])
    df_apfd['name'] = ['1', '2', '3', '4', '5', '6', '7', '8']
    df_apfd['xgb_apfd'] = [xgb_apfd_1,xgb_apfd_2,xgb_apfd_3,xgb_apfd_4,xgb_apfd_5,xgb_apfd_6,xgb_apfd_7,xgb_apfd_8]
    df_apfd['lgb_apfd'] = [lgb_apfd_1,lgb_apfd_2,lgb_apfd_3,lgb_apfd_4,lgb_apfd_5,lgb_apfd_6,lgb_apfd_7,lgb_apfd_8]
    df_apfd['rf_apfd'] = [rf_apfd_1,rf_apfd_2,rf_apfd_3,rf_apfd_4,rf_apfd_5,rf_apfd_6,rf_apfd_7,rf_apfd_8]
    df_apfd['lr_apfd'] = [lr_apfd_1,lr_apfd_2,lr_apfd_3,lr_apfd_4,lr_apfd_5,lr_apfd_6,lr_apfd_7,lr_apfd_8]
    df_apfd['deepGini_apfd'] = [deepGini_apfd_1,deepGini_apfd_2,deepGini_apfd_3,deepGini_apfd_4,deepGini_apfd_5,deepGini_apfd_6,deepGini_apfd_7,deepGini_apfd_8]
    df_apfd['vanillasoftmax_apfd'] = [vanillasoftmax_apfd_1,vanillasoftmax_apfd_2,vanillasoftmax_apfd_3,vanillasoftmax_apfd_4,vanillasoftmax_apfd_5,vanillasoftmax_apfd_6,vanillasoftmax_apfd_7,vanillasoftmax_apfd_8]
    df_apfd['pcs_apfd'] = [pcs_apfd_1,pcs_apfd_2,pcs_apfd_3,pcs_apfd_4,pcs_apfd_5,pcs_apfd_6,pcs_apfd_7,pcs_apfd_8]
    df_apfd['entropy_apfd'] = [entropy_apfd_1,entropy_apfd_2,entropy_apfd_3,entropy_apfd_4,entropy_apfd_5,entropy_apfd_6,entropy_apfd_7,entropy_apfd_8]
    df_apfd['random_apfd'] = [random_apfd_1,random_apfd_2,random_apfd_3,random_apfd_4,random_apfd_5,random_apfd_6,random_apfd_7,random_apfd_8]
    print(df_apfd)
    df_apfd.to_csv('results/'+save_path_name, header=True, index=False)
    print('finished')


if __name__ == '__main__':
    main()




