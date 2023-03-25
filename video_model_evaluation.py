import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

# python video_model_evaluation.py --cuda 'cuda:2' --path_x './data/pkl_data/ucf101/ucf101_x.pkl' --path_y './data/pkl_data/ucf101/ucf101_y.pkl' --model_path './target_models/ucf101_R3D_40.pt'

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
    x.resize((len(x), 3, 16, 112, 112))
    y = pickle.load(open(args.path_y, 'rb'))

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)

    miss_idx = []
    for i in range(len(train_x)):
        x_t = train_x[i]
        x_t.shape = (1, 3, 16, 112, 112)
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            out = model(x_t)

        _, predict = torch.max(out.data, 1)
        pre = predict.cpu().numpy()[0]
        label = train_y[i]
        if pre != label:
            miss_idx.append(i)
    print('train_acc:', len(miss_idx)*1.0/len(train_x))

    miss_idx = []
    for i in range(len(test_x)):
        x_t = test_x[i]
        x_t.shape = (1, 3, 16, 112, 112)
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            out = model(x_t)

        _, predict = torch.max(out.data, 1)
        pre = predict.cpu().numpy()[0]
        label = test_y[i]
        if pre != label:
            miss_idx.append(i)
    print('test_acc:', len(miss_idx)*1.0/len(test_x))




if __name__ == '__main__':
    main()
