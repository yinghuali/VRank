import pickle
import numpy as np

train_dataloader = pickle.load(open('./data/pkl_data/ucf101_train_dataloader.pkl', 'rb'))
val_dataloader = pickle.load(open('./data/pkl_data/ucf101_val_dataloader.pkl', 'rb'))
test_dataloader = pickle.load(open('./data/pkl_data/ucf101_test_dataloader.pkl', 'rb'))


def save_dataloader_to_numpy(save_path_x, save_path_y, dataloader):
    x = []
    y = []
    for inputs, labels in dataloader:
        tmp_x = inputs.numpy()
        for video in tmp_x:
            x.append(video)
        tmp_y = list(labels.numpy())
        y += tmp_y
    x = np.array(x)
    y = np.array(y)

    pickle.dump(x, open(save_path_x, 'wb'), protocol=4)
    pickle.dump(y, open(save_path_y, 'wb'), protocol=4)


if __name__ == '__main__':
    save_dataloader_to_numpy('./data/pkl_data/ucf101_test_x.pkl', './data/pkl_data/ucf101_test_x.pkl', test_dataloader)
    save_dataloader_to_numpy('./data/pkl_data/ucf101_val_x.pkl', './data/pkl_data/ucf101_val_x.pkl', val_dataloader)
    save_dataloader_to_numpy('./data/pkl_data/ucf101_train_x.pkl', './data/pkl_data/ucf101_train_x.pkl', train_dataloader)



