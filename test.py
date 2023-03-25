import pickle
import torch
import numpy as np
from torch import nn, optim
from models import C3D_model, R2Plus1D_model, R3D_model

modelName = 'R3D'
dataset = 'ucf101'
model_path = '/home/yinghua/pycharm/VRank/target_models/models/R3D-ucf101_epoch-19.pth.tar'
device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")
num_classes = 101
save_path_test = './data/pkl_data/R3D_ucf101_test_x_pre.pkl'
save_path_val = './data/pkl_data/R3D_ucf101_val_x_pre.pkl'
save_path_train = './data/pkl_data/R3D_ucf101_train_x_pre.pkl'

test_x = pickle.load(open('./data/pkl_data/ucf101_test_x.pkl', 'rb'))
val_x = pickle.load(open('./data/pkl_data/ucf101_val_x.pkl', 'rb'))
train_x = pickle.load(open('./data/pkl_data/ucf101_train_x.pkl', 'rb'))


if modelName == 'C3D':
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
if modelName == 'R2Plus1D':
    model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
if modelName == 'R3D':
    model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))

checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

model.eval()


def save_pre(save_path, x):
    pre_probability_vec = []
    for i in range(len(x)):
        inputs = torch.from_numpy(np.array([x[i]])).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        probability = list(probs.cpu().numpy()[0])
        pre_probability_vec.append(probability)
    pre_probability_vec = np.array(pre_probability_vec)
    pickle.dump(pre_probability_vec, open(save_path, 'wb'), protocol=4)


inputs = torch.from_numpy(test_x).to(device)
print(inputs.size())









