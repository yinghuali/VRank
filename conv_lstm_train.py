
from utils import *
from models import ConvLstm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.autograd import Variable


path_x = './data/pkl_data/ucf101/ucf101_x.pkl'
path_y = './data/pkl_data/ucf101/ucf101_y.pkl'
path_val_pre = './target_models/ucf101_R3D_30_val_pre.pkl'
path_test_pre = './target_models/ucf101_R3D_30_test_pre.pkl'
cuda = 'cuda:1'
batch_size = 8
latent_dim = 512 # The dim of the Conv FC output (default:512), LSTM的输入
hidden_size = 256 # The number of features in the LSTM hidden state (default:256)
lstm_layers = 10 # Number of recurrent layers (default:2)
num_classes = 101
lr = 0.001
epochs = 61
save_model_path = './vrank_models/ucf101_R3D_vrank'


## 前向传播forward函数，用于纯推理，输出标签
def foward_step_no_labels(model, images):
    ## 开启新的batch前必须重新设置隐藏层状态，否则会被认为是上一个序列
    model.Lstm.reset_hidden_state()
    with torch.no_grad():
        output = model(images)
    predicted_labels = output.detach().cpu().argmax(dim=1)
    return predicted_labels


def foward_step(model, images, labels, criterion, mode=''):
    model.Lstm.reset_hidden_state()
    if mode == 'test':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)

    loss = criterion(output, labels)
    probs = nn.Softmax(dim=1)(output)
    preds = torch.max(probs, 1)[1]

    return loss, probs, preds


x = pickle.load(open(path_x, 'rb'))
x = np.transpose(x, (0, 1, 4, 2, 3))  # (13038, 16, 3, 112, 112)

y = pickle.load(open(path_y, 'rb'))
val_pre_vec = pickle.load(open(path_val_pre, 'rb'))
test_pre_vec = pickle.load(open(path_test_pre, 'rb'))

device = torch.device(cuda if torch.cuda.is_available() else "cpu")
train_x_, test_x, train_y_, test_y = train_test_split(x, y, test_size=0.3, random_state=17)
train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, test_size=0.3, random_state=17)
val_pre = val_pre_vec.argsort()[:, -1]
test_pre = test_pre_vec.argsort()[:, -1]

val_rank_label = get_rank_label(val_pre, val_y)
test_rank_label = get_rank_label(test_pre, test_y)

x_train_t = torch.from_numpy(val_x)
y_train_t = torch.from_numpy(val_rank_label)

dataset = Data.TensorDataset(x_train_t, y_train_t)
dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


model = ConvLstm(latent_dim, hidden_size, lstm_layers, bidirectional=True, n_class=num_classes)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

for e in range(epochs):
    model.train()
    running_corrects = 0.0
    i = 0
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.float() / 255.0
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss, probs, preds = foward_step(model, inputs, labels, criterion, mode='train')
        loss.backward()
        optimizer.step()
        running_corrects += torch.sum(preds == labels.data)
        acc = running_corrects.double() / len(train_x)

        i += 1
        if i % 50 == 0:
            print(acc, 'epochs =', e)

    epoch_acc = running_corrects.double() / len(train_x)
    print('epoch_acc =', epoch_acc)

    if e % 10 == 0 and e > 0:
        torch.save(model, save_model_path + '_' + str(e) + '.pt')






