#coding:utf8
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from models import C3D_model, R2Plus1D_model, R3D_model

modelName = 'R3D'
dataset = 'ucf101'
model_path = '/home/yinghua/pycharm/VRank/target_models/models/R3D-ucf101_epoch-19.pth.tar'
device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
lr = 1e-2 # Learning rate

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError



def model_prediction():


    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError

    checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=24, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',  clip_len=16), batch_size=8, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=8, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    model.eval()
    running_corrects = 0.0
    start_time = timeit.default_timer()

    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]

        running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / test_size
    print('=========')
    print('acc:', acc) # tensor(0.8193, device='cuda:1', dtype=torch.float64)
    print('============')

model_prediction()




