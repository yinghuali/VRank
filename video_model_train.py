import torch
import pickle
from sklearn.model_selection import train_test_split
from models.R3D_model import R3DClassifier
import torch.utils.data as Data

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--cuda", type=str)
ap.add_argument("--model_name", type=str)
ap.add_argument("--epochs", type=int)
ap.add_argument("--data_name", type=str)
ap.add_argument("--path_x", type=str)
ap.add_argument("--path_y", type=str)
ap.add_argument("--batch_size", type=int)
ap.add_argument("--save_model_path", type=str)
args = ap.parse_args()

# python video_model_train.py --cuda 'cuda:1' --model_name 'R3D' --epochs 100 --data_name 'ucf101' --path_x './data/pkl_data/ucf101/ucf101_x.pkl' --path_y './data/pkl_data/ucf101/ucf101_y.pkl' --batch_size 8 --save_model_path './target_models/ucf101_R3D'


lr = 0.01


def get_correct_num(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    correct_num = (labels == predict).sum().item()
    return correct_num


def train_model():
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    x = pickle.load(open(args.path_x, 'rb'))
    x.resize((len(x), 3, 16, 112, 112))
    y = pickle.load(open(args.path_y, 'rb'))

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    x_train_t = torch.from_numpy(train_x).float()
    x_test_t = torch.from_numpy(test_x).float()
    y_train_t = torch.from_numpy(train_y).float()
    y_test_t = torch.from_numpy(test_y).float()

    dataset = Data.TensorDataset(x_train_t, y_train_t)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)


    if args.data_name == 'hmdb51':
        num_classes = 51
    elif args.data_name == 'ucf101':
        num_classes = 101


    if args.model_name == 'R3D':
        model = R3DClassifier(num_classes, (2, 2, 2, 2))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fun = nn.CrossEntropyLoss()

    for e in range(args.epochs):
        epoch_correct_num = 0
        total_samples = 0
        for x_t, y_t in dataloader:
            y_t = y_t.type(torch.LongTensor)
            x_t = x_t.to(device)
            y_t = y_t.to(device)
            optimizer.zero_grad()
            out = model(x_t)
            loss = loss_fun(out, y_t)
            loss.backward()
            optimizer.step()
            epoch_correct_num += get_correct_num(out, y_t)
            total_samples += y_t.size(0)
        if e % 10 == 0:
            torch.save(model, args.save_model_path+'_'+str(e)+'.pt')


        # with torch.no_grad():
        #     train_out = model(x_train_t)
        # acc_train = get_acc(train_out, y_train_t)
        # with torch.no_grad():
        #     test_out = model(x_test_t)
        # acc_test = get_acc(test_out, y_test_t)

        # print('acc_train', acc_train)
        # print('acc_test', acc_test)

    # y_pred_test = model(x_test_t).detach().numpy()[:, 1]


if __name__ == '__main__':
    train_model()
