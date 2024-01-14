# source activate py37
import tensorflow as tf
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from models.video_transformer import get_vt_model, build_feature_extractor
from tensorflow.keras.models import load_model
from models.video_transformer import PositionalEmbedding, TransformerEncoder


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# video_frames = np.zeros((50, 16, 112, 112, 3), dtype=np.float32)

num_classes = 101
path_x = './pkl_data/ucf101/vt_ucf101_x.pkl'
path_y = './pkl_data/ucf101/ucf101_y.pkl'
batch_size = 32
epochs = 5
path_save = './target_models/ucf101_vt.h5'


def main():
    x = pickle.load(open(path_x, 'rb'))
    y = pickle.load(open(path_y, 'rb'))
    x = np.array(x)

    all_x = tf.stack(x, axis=0)
    all_y = tf.keras.utils.to_categorical(y, num_classes)

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    train_y = tf.keras.utils.to_categorical(train_y, num_classes)
    val_y = tf.keras.utils.to_categorical(val_y, num_classes)
    test_y = tf.keras.utils.to_categorical(test_y, num_classes)

    train_x = tf.stack(train_x, axis=0)
    val_x = tf.stack(val_x, axis=0)
    test_x = tf.stack(test_x, axis=0)

    model = get_vt_model(train_x.shape[1:], num_classes)
    sgd = SGD(lr=0.01, decay=0, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y), shuffle=True)

    scores = model.evaluate(val_x, val_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    scores = model.evaluate(test_x, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save(path_save)

    ##############load_model####################
    M = load_model(path_save, custom_objects={'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder})
    scores = M.evaluate(val_x, val_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    scores = M.evaluate(test_x, test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    model.save(path_save)


if __name__ == '__main__':
    main()

