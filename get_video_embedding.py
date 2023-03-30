import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle
from PIL import Image
import numpy as np

path_x = './data/pkl_data/ucf101/ucf101_x.pkl'
save_video_embedding = '/home/yinghua/pycharm/VRank/data/pkl_data/ucf101/ucf101_x_embedding.pkl'
save_dir = './data/pkl_data/ucf101/pic/'
device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")


def main():
    x = pickle.load(open(path_x, 'rb'))
    x = np.transpose(x, (0, 1, 4, 2, 3))
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.float()
    model.to(device)
    model.eval()
    x_embedding = []
    for i in range(len(x)):
        input = x[i] / 255
        input = torch.from_numpy(input)
        input = input.to(device)

        with torch.no_grad():
            output = model(input.float())
        vector = output.cpu().squeeze().numpy()
        video_vec = np.mean(vector, axis=0)
        x_embedding.append(video_vec)
        print('======', i)
    x_embedding = np.array(x_embedding)
    pickle.dump(x_embedding, open(save_video_embedding, 'wb'), protocol=4)


if __name__ == '__main__':
    main()

