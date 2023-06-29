import pickle
import numpy as np
import argparse
from PIL import Image

path_x = '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x.pkl'


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
    shape = x.shape
    x = x.reshape(shape[0], shape[1] * shape[4], shape[2], shape[3]) # (13038, 48, 112, 112)
    compress_feature = []
    for video in x:
        feature = get_compress_feature(video)
        compress_feature.append(feature)
    compress_feature = np.array(compress_feature)
    print(compress_feature.shape)


if __name__ == '__main__':
    main()

