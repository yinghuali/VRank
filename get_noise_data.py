import argparse
import pickle
import cv2
import random
import math
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

ap = argparse.ArgumentParser()
ap.add_argument("--video_np_path", type=str, default='')
ap.add_argument("--select_method", type=str, default='')
ap.add_argument("--save_path_pkl", type=str, default='')

args = vars(ap.parse_args())
video_np_path = args['video_np_path']
select_method = args['select_method']
save_path_pkl = args['save_path_pkl']

# python get_noise_data.py  --select_method 'contrast' --save_path_pkl '/raid/yinghua/VRank/data/pkl_data/ucf_noise/contrast_x.pkl' --video_np_path '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x.pkl'


# image 0-255
# (480, 714, 3)


def augmentation_width_shift(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(width_shift_range=0.2)
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_height_shift(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(height_shift_range=0.2)
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_horizontal_flip(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(horizontal_flip=True)
    it = datagen.flow(samples, batch_size=1, seed=11)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_vertical_flip(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(vertical_flip=True)
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_rotation(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(rotation_range=45)
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_brightness(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(brightness_range=[0.2, 3.0])
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_zoom(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(zoom_range=[0.5,1.0])
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_featurewise_std_normalization(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(featurewise_std_normalization=True)
    it = datagen.flow(samples, batch_size=1, seed=11)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_zca_whitening(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(zca_whitening=True)
    it = datagen.flow(samples, batch_size=1, seed=12)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_shear_range(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(shear_range=0.5)
    it = datagen.flow(samples, batch_size=1, seed=9)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def augmentation_channel_shift_range(img):
    samples = expand_dims(img, 0)
    datagen = ImageDataGenerator(channel_shift_range=80)
    it = datagen.flow(samples, batch_size=1, seed=10)
    batch = it.next()
    image = batch[0].astype('int')
    return image


def noise_salt_pepper(image, prob=0.01):
    output = np.zeros(image.shape, np.uint8)
    noise_out = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                noise_out[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
                noise_out[i][j] = 255
            else:
                output[i][j] = image[i][j]
                noise_out[i][j] = 100
    return output


def noise_gasuss(image, mean=0, var=0.001):
    '''
    添加高斯噪声
    image:原始图像
    mean : 均值
    var : 方差,越大，噪声越大
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out


def fog(img):
    if np.max(img) > 5:
        img_f = img / 255.0
    else:
        img_f = img.copy()
    (row, col, chs) = img.shape
    A = 0.5  # 亮度
    beta = 0.02  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    img_f = np.uint8(img_f * 255)
    return img_f


def contrast(img):
    img_ = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    return img_


def run_video(video_path, select_method, save_path_pkl):
    if select_method == 'augmentation_width_shift':
        method = augmentation_width_shift
    elif select_method == 'augmentation_height_shift':
        method = augmentation_height_shift
    elif select_method == 'augmentation_horizontal_flip':
        method = augmentation_horizontal_flip
    elif select_method == 'augmentation_vertical_flip':
        method = augmentation_vertical_flip
    elif select_method == 'augmentation_rotation':
        method = augmentation_rotation
    elif select_method == 'augmentation_brightness':
        method = augmentation_brightness
    elif select_method == 'augmentation_zoom':
        method = augmentation_zoom
    elif select_method == 'augmentation_featurewise_std_normalization':
        method = augmentation_featurewise_std_normalization
    elif select_method == 'augmentation_zca_whitening':
        method = augmentation_zca_whitening
    elif select_method == 'augmentation_shear_range':
        method = augmentation_shear_range
    elif select_method == 'augmentation_channel_shift_range':
        method = augmentation_channel_shift_range
    elif select_method == 'noise_salt_pepper':
        method = noise_salt_pepper
    elif select_method == 'noise_gasuss':
        method = noise_gasuss
    elif select_method == 'fog':
        method = fog
    elif select_method == 'contrast':
        method = contrast

    video_np = np.array(pickle.load(open(video_path, 'rb')))
    new_video_list = []
    for video in video_np:
        img_list = []
        for img in video:
            tmp_img = method(img)
            img_list.append(tmp_img)
        new_video_list.append(img_list)
    new_video_np = np.array(new_video_list, dtype='uint8')
    pickle.dump(new_video_np, open(save_path_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    run_video(video_np_path, select_method, save_path_pkl)


# ucf101_x.pkl   (13038, 16, 112, 112, 3)