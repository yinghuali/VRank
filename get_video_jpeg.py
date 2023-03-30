import os
from towhee import pipeline
import cv2
import pickle
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
path_x = './data/pkl_data/ucf101/ucf101_x.pkl'
save_dir = './data/pkl_data/ucf101/pic/'
save_video_embedding = '/home/yinghua/pycharm/VRank/data/pkl_data/ucf101/vec/'
p = pipeline('image-embedding')


def main():
    x = pickle.load(open(path_x, 'rb'))

    for i in range(len(x)):
        for j in range(len(x[i])):
            pic_path = save_dir + str(i)+'_'+str(j)+'.jpeg'
            cv2.imwrite(pic_path, x[i][j])


if __name__ == '__main__':
    main()