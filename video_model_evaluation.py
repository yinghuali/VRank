import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn

# python video_model_evaluation.py --cuda 'cuda:3' --path_x './data/pkl_data/ucf101/ucf101_x.pkl' --path_y './data/pkl_data/ucf101/ucf101_y.pkl' --model_path './target_models/ucf101_R3D_10.pt'

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cuda", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--model_path", type=str)
args = ap.parse_args()


def main():
    model = torch.load(args.model_path)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model.to(device)

    x = pickle.load(open(args.path_x, 'rb'))
    x = np.transpose(x, (0, 4, 1, 2, 3))
    y = pickle.load(open(args.path_y, 'rb'))

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    miss_idx = []
    correct_n = 0
    model.eval()
    for i in range(len(train_x)):
        x_t = train_x[i]
        x_t = np.array([x_t])
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            outputs = model(x_t)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        pre_vec = probs.cpu().numpy()[0]
        pre = preds.cpu().numpy()[0]
        label = train_y[i]
        if pre != label:
            miss_idx.append(i)
        else:
            correct_n += 1
    print('train_acc:', correct_n*1.0/len(train_x))

    miss_idx = []
    correct_n = 0
    for i in range(len(val_x)):
        x_t = val_x[i]
        x_t = np.array([x_t])
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            outputs = model(x_t)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        pre_vec = probs.cpu().numpy()[0]
        pre = preds.cpu().numpy()[0]
        label = val_y[i]
        if pre != label:
            miss_idx.append(i)
        else:
            correct_n += 1
    print('val_acc:', correct_n*1.0/len(val_x))

    miss_idx = []
    correct_n = 0
    for i in range(len(test_x)):
        x_t = test_x[i]
        x_t = np.array([x_t])
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            outputs = model(x_t)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        pre_vec = probs.cpu().numpy()[0]
        pre = preds.cpu().numpy()[0]
        label = test_y[i]
        if pre != label:
            miss_idx.append(i)
        else:
            correct_n += 1
    print('test_acc:', correct_n*1.0/len(test_x))


if __name__ == '__main__':
    main()
