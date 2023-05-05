import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from models.R3D_model import R3DClassifier
from models import C3D_model, R2Plus1D_model, R3D_model
import torch.utils.data as Data
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.slowfastnet import resnet50


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

# nohup python video_model_train.py --cuda 'cuda:2' --model_name 'R2Plus1D' --epochs 51 --data_name 'ucf101' --path_x '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x.pkl' --path_y '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_y.pkl' --batch_size 24 --save_model_path './target_models/ucf101_R2Plus1D' > train_R2Plus1D.log 2>&1 &
# nohup python video_model_train.py --cuda 'cuda:2' --model_name 'slowfastnet' --epochs 51 --data_name 'ucf101' --path_x '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_x.pkl' --path_y '/raid/yinghua/VRank/data/pkl_data/ucf101/ucf101_y.pkl' --batch_size 24 --save_model_path './target_models/ucf101_slowfastnet' > train_slowfastnet.log 2>&1 &
lr = 0.001


def get_correct_num(outputs, labels):
    _, predict = torch.max(outputs.data, 1)
    correct_num = (labels == predict).sum().item()
    return correct_num


def train_model():
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    x = pickle.load(open(args.path_x, 'rb'))
    x = np.transpose(x, (0, 4, 1, 2, 3))
    y = pickle.load(open(args.path_y, 'rb'))

    train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
    train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)

    x_train_t = torch.from_numpy(train_x)
    y_train_t = torch.from_numpy(train_y)

    dataset = Data.TensorDataset(x_train_t, y_train_t)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    if args.data_name == 'ucf101':
        num_classes = 101

    if args.data_name == 'charades':
        num_classes = 16

    if args.data_name == 'hmdb51':
        num_classes = 51

    if args.data_name == 'hollywood2':
        num_classes = 12

    if args.data_name == 'accident':
        num_classes = 12

    if args.model_name == 'R3D':
        model = R3DClassifier(num_classes, (2, 2, 2, 2))
        train_params = model.parameters()
    if args.model_name == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = model.parameters()
    if args.model_name == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    if args.model_name == 'slowfastnet':
        model = resnet50(class_num=num_classes)
        train_params = model.parameters()

    model.to(device)
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for e in range(args.epochs):
        scheduler.step()
        model.train()
        running_corrects = 0.0
        i = 0
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.float()
            inputs = Variable(inputs, requires_grad=True).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_corrects += torch.sum(preds == labels.data)
            acc = running_corrects.double() / len(train_x)

            i += 1
            if i % 50 == 0:
                print(acc, 'epochs =', e)

        epoch_acc = running_corrects.double() / len(train_x)
        print('epoch_acc =', epoch_acc)

        if e % 5 == 0 and e > 0:
            torch.save(model, args.save_model_path+'_'+str(e)+'.pt')


if __name__ == '__main__':
    train_model()


