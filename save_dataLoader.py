import pickle
import torch
from torch.utils.data import DataLoader
from dataloaders.dataset import VideoDataset

dataset = 'ucf101'


if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101


def save_data():

    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=24, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=8, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=8, num_workers=4)
    pickle.dump(train_dataloader, open('./data/pkl_data/ucf101_train_dataloader.pkl', 'wb'), protocol=4)
    pickle.dump(val_dataloader, open('./data/pkl_data/ucf101_val_dataloader.pkl', 'wb'), protocol=4)
    pickle.dump(test_dataloader, open('./data/pkl_data/ucf101_test_dataloader.pkl', 'wb'), protocol=4)


save_data()




