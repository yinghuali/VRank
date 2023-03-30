from utils import *
from vrank import ConvLstm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from get_rank_idx import *
from scipy.stats import entropy
import pandas as pd


path_x = './data/pkl_data/ucf101/ucf101_x.pkl'
path_y = './data/pkl_data/ucf101/ucf101_y.pkl'
path_val_pre = './target_models/ucf101_C3D_24_val_pre.pkl'
path_test_pre = './target_models/ucf101_C3D_24_test_pre.pkl'
path_x_embedding = './data/pkl_data/ucf101/ucf101_x_embedding.pkl'


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

    all_pre_vec = np.vstack((uncertainty_feature_val, uncertainty_feature_test))
    all_uncertainty_feature = np.vstack((val_pre_vec, test_pre_vec))
    all_embedding = np.vstack((val_x_embedding, test_x_embedding))
    all_y = np.hstack((val_y, test_y))

    uncertainty_feature_train, uncertainty_feature_test, train_y, test_y = train_test_split(all_pre_vec, all_y, test_size=0.3, random_state=17)
    train_pre_vec, test_pre_vec, _, _ = train_test_split(all_uncertainty_feature, all_y, test_size=0.3, random_state=17)
    train_x_embedding, test_x_embedding, _, _ = train_test_split(all_embedding, all_y, test_size=0.3, random_state=17)

    train_pre = train_pre_vec.argsort()[:, -1]
    test_pre = test_pre_vec.argsort()[:, -1]

    print(len(train_pre))
    print(len(test_pre))

    train_rank_label = get_rank_label(train_pre, train_y)
    idx_miss_list = get_idx_miss_class(test_pre, test_y)

    concat_train_all_feature = np.hstack((uncertainty_feature_train, train_pre_vec, train_x_embedding))
    concat_test_all_feature = np.hstack((uncertainty_feature_test, test_pre_vec, test_x_embedding))

    model = XGBClassifier()
    model.fit(concat_train_all_feature, train_rank_label)
    y_concat_all = model.predict_proba(concat_test_all_feature)[:, 1]
    xgb_rank_idx = y_concat_all.argsort()[::-1].copy()

    model = LGBMClassifier()
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
    
    margin_rank_idx = Margin_rank_idx(test_pre_vec)
    deepGini_rank_idx = DeepGini_rank_idx(test_pre_vec)
    leastConfidence_rank_idx = LeastConfidence_rank_idx(test_pre_vec)

    vanillasoftmax_rank_idx = VanillaSoftmax_rank_idx(test_pre_vec)
    pcs_rank_idx = PCS_rank_idx(test_pre_vec)
    entropy_rank_idx = Entropy_rank_idx(test_pre_vec)
    random_rank_idx = Random_rank_idx(test_pre_vec)

    xgb_apfd = apfd(idx_miss_list, xgb_rank_idx)
    lgb_apfd = apfd(idx_miss_list, lgb_rank_idx)
    rf_apfd = apfd(idx_miss_list,  rf_rank_idx)
    lr_apfd = apfd(idx_miss_list, lr_rank_idx)

    deepGini_apfd = apfd(idx_miss_list, deepGini_rank_idx)
    leastConfidence_apfd = apfd(idx_miss_list, leastConfidence_rank_idx)
    margin_apfd = apfd(idx_miss_list, margin_rank_idx)
    random_apfd = apfd(idx_miss_list, random_rank_idx)
    vanillasoftmax_apfd = apfd(idx_miss_list, vanillasoftmax_rank_idx)
    pcs_apfd = apfd(idx_miss_list, pcs_rank_idx)
    entropy_apfd = apfd(idx_miss_list, entropy_rank_idx)

    df_apfd = pd.DataFrame(columns=['name'])
    df_apfd['name'] = ['clean_data']
    df_apfd['xgb_apfd'] = [xgb_apfd]
    df_apfd['lgb_apfd'] = [lgb_apfd]
    df_apfd['rf_apfd'] = [rf_apfd]
    df_apfd['lr_apfd'] = [lr_apfd]
    df_apfd['deepGini_apfd'] = [deepGini_apfd]
    df_apfd['leastConfidence_apfd'] = [leastConfidence_apfd]
    df_apfd['margin_apfd'] = [margin_apfd]
    df_apfd['vanillasoftmax_apfd'] = [vanillasoftmax_apfd]
    df_apfd['pcs_apfd'] = [pcs_apfd]
    df_apfd['entropy_apfd'] = [entropy_apfd]
    df_apfd['random_apfd'] = [random_apfd]
    print(df_apfd)
    # df_apfd.to_csv('res/repeat_apfd.csv', mode='a', header=False, index=False)
    print('finished')


if __name__ == '__main__':
    main()









