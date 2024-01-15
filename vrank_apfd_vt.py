# source activate py37
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from models.video_transformer import PositionalEmbedding, TransformerEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from get_rank_idx import *
from scipy.stats import entropy
from get_frame_fearure import get_all_frame_feaure


path_model_save = './target_models/hmdb51_vt.h5'
path_frame = './pkl_data/hmdb51/hmdb51_x.pkl'
path_x = './pkl_data/hmdb51/vt_hmdb51_x.pkl'
path_y = './pkl_data/hmdb51/hmdb51_y.pkl'
path_x_embedding = './pkl_data/hmdb51/hmdb51_x_embedding.pkl'
save_path = './results/vt_hmdb51_x.json'
num_classes = 51


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

    deepGini_rank_idx = DeepGini_rank_idx(test_pre_vec)
    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(test_pre_vec)
    pcs_rank_idx = PCS_rank_idx(test_pre_vec)
    entropy_rank_idx = Entropy_rank_idx(test_pre_vec)
    random_rank_idx = Random_rank_idx(test_pre_vec)
    deepGini_apfd = apfd(idx_miss_list, deepGini_rank_idx)
    random_apfd = apfd(idx_miss_list, random_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_list, entropy_rank_idx)

    dic = {'xgb_apfd': xgb_apfd,
           'lgb_apfd': lgb_apfd,
           'rf_apfd': rf_apfd,
           'lr_apfd': lr_apfd,
           'deepGini_apfd': deepGini_apfd,
           'vanillasoftmax_apfd': vanillasoftmax_apfd,
           'pcs_apfd': pcs_apfd,
           'entropy_apfd': entropy_apfd,
           'random_apfd': random_apfd
           }

    json.dump(dic, open(save_path, 'w'), sort_keys=True, indent=4)
    print(dic)


if __name__ == '__main__':
    main()

