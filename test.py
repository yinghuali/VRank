
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import pickle
import cv2
import random
import math
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from get_noise_data import *



# video = pickle.load(open('contrast_x.pkl', 'rb'))
# print(video.shape)

# print(img.shape)
# plt.imshow(img)
# plt.show()

# for img in video[0]:
#     tmp_img = contrast(img)
#     plt.imshow(tmp_img)
#     plt.show()
#     print(tmp_img[0])

img = mpimg.imread('/Users/yinghua.li/Documents/Pycharm/VRank/data/bird.jpeg')
print(img.dtype)
# print(img.shape)
# plt.imshow(img)
# plt.show()

res = []
for i in range(5):
    tmp_img = augmentation_width_shift(img)
    res.append(tmp_img)
res = np.array(res)

print(res.dtype)


