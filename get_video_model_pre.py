import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn

# python get_video_model_pre.py --cuda 'cuda:1' --path_x '/raid/yinghua/VRank/data/pkl_data/ucf_noise/augmentation_width_shift_x.pkl' --path_y '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_y.pkl'  --model_path './target_models/ucf101_C3D_18.pt' --save_val_vec '/raid/yinghua/VRank/data/pkl_data/ucf_noise/ucf101_C3D_18_val_augmentation_width_shift_x_pre.pkl' --save_test_vec '/raid/yinghua/VRank/data/pkl_data/ucf_noise/ucf101_C3D_18_test_augmentation_width_shift_x_pre.pkl'

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cuda", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--model_path", type=str)
ap.add_argument("--save_val_vec", type=str)
ap.add_argument("--save_test_vec", type=str)
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
    all_pre_vec = []
    for i in range(len(val_x)):
        x_t = val_x[i]
        x_t = np.array([x_t])
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            outputs = model(x_t)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        pre_vec = probs.cpu().numpy()[0]
        all_pre_vec.append(pre_vec)
        pre = preds.cpu().numpy()[0]
        label = val_y[i]
        if pre != label:
            miss_idx.append(i)
        else:
            correct_n += 1
    all_pre_vec = np.array(all_pre_vec)
    pickle.dump(all_pre_vec, open(args.save_val_vec, 'wb'), protocol=4)
    print('val_acc:', correct_n*1.0/len(val_x))


    miss_idx = []
    correct_n = 0
    all_pre_vec = []
    for i in range(len(test_x)):
        x_t = test_x[i]
        x_t = np.array([x_t])
        x_t = torch.from_numpy(x_t).float().to(device)

        with torch.no_grad():
            outputs = model(x_t)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        pre_vec = probs.cpu().numpy()[0]
        all_pre_vec.append(pre_vec)
        pre = preds.cpu().numpy()[0]
        label = test_y[i]
        if pre != label:
            miss_idx.append(i)
        else:
            correct_n += 1
    all_pre_vec = np.array(all_pre_vec)
    pickle.dump(all_pre_vec, open(args.save_test_vec, 'wb'), protocol=4)
    print('test_acc:', correct_n*1.0/len(test_x))


if __name__ == '__main__':
    main()

