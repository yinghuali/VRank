# source activate py37
import tensorflow as tf
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from models.video_transformer import get_vt_model, build_feature_extractor


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# video_frames = np.zeros((50, 16, 112, 112, 3), dtype=np.float32)

num_classes = 12
path_x = './pkl_data/accident/accident_x.pkl'
path_y = './pkl_data/accident/accident_y.pkl'
batch_size = 32
epochs = 20


x = pickle.load(open(path_x, 'rb'))
y = pickle.load(open(path_y, 'rb'))


def get_feature_extractor_x(x):
    feature_extractor = build_feature_extractor()
    frame_features = []
    for i in range(len(x)):
        frame_features.append(feature_extractor(x[i]))
        print(i, len(x))

get_feature_extractor_x(x)

# feature_extractor = build_feature_extractor()
# frame_features = [feature_extractor(frame) for frame in x]
# x = tf.stack(frame_features, axis=0)   # (50, 16, 1024)

# train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
# train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)
#
# train_y = tf.keras.utils.to_categorical(train_y, num_classes)
# val_y = tf.keras.utils.to_categorical(val_y, num_classes)
# test_y = tf.keras.utils.to_categorical(test_y, num_classes)
#
#
# model = get_vt_model(train_x.shape[1:], num_classes)
# sgd = SGD(lr=0.01, decay=0, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
# model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(train_x, train_y), shuffle=True)
#
# scores = model.evaluate(val_x, val_y, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1] * 100))

# model.save(path_save)

