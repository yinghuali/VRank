import pickle
import matplotlib.pyplot as plt

x_path = './data/pkl_data/ucf101/ucf101_x.pkl'
y_path = './data/pkl_data/ucf101/ucf101_y.pkl'
x = pickle.load(open(x_path, 'rb'))
y = pickle.load(open(y_path, 'rb'))
print(x.shape)
print(y.shape)












