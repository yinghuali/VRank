# source activate py37
import pickle
import os
import numpy as np
from models.video_transformer import get_vt_model, build_feature_extractor

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_x", type=str)
ap.add_argument("--save_path", type=str)
args = ap.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

path_x = args.path_x
save_path = args.save_path

# python get_vt_extractor_x.py --path_x './pkl_data/accident/accident_x.pkl' --save_path './pkl_data/accident/vt_accident_x.pkl'

# path_x = './pkl_data/accident/accident_x.pkl'
# save_path = './pkl_data/accident/vt_accident_x.pkl'


def write_result(content, file_name):
    re = open(file_name, 'a')
    re.write('\n' + content)
    re.close()


def get_feature_extractor_x(x):
    feature_extractor = build_feature_extractor()
    frame_features = []
    for i in range(len(x)):
        frame_features.append(feature_extractor(x[i]))
        print(i)
        if i % 50 == 0:
            write_result(str(i)+'->'+str(len(x)), name)
    pickle.dump(frame_features, open(save_path, 'wb'), protocol=4)


if __name__ == '__main__':
    name = path_x.split('/')[-1].replace('.pkl', '.log')
    x = pickle.load(open(path_x, 'rb'))
    get_feature_extractor_x(x)
