
python video_model_train.py --cuda 'cuda:0' --model_name 'C3D' --epochs 101 --data_name 'hmdb51' --path_x './pkl_data/hmdb51/hmdb51_x.pkl' --path_y './pkl_data/hmdb51/hmdb51_y.pkl' --batch_size 24 --save_model_path './target_models/hmdb51_C3D'
python video_model_train.py --cuda 'cuda:0' --model_name 'R3D' --epochs 101 --data_name 'hmdb51' --path_x './pkl_data/hmdb51/hmdb51_x.pkl' --path_y './pkl_data/hmdb51/hmdb51_y.pkl' --batch_size 24 --save_model_path './target_models/hmdb51_R3D'
python video_model_train.py --cuda 'cuda:0' --model_name 'R2Plus1D' --epochs 101 --data_name 'hmdb51' --path_x './pkl_data/hmdb51/hmdb51_x.pkl' --path_y './pkl_data/hmdb51/hmdb51_y.pkl' --batch_size 24 --save_model_path './target_models/hmdb51_R2Plus1D'
python video_model_train.py --cuda 'cuda:0' --model_name 'slowfastnet' --epochs 101 --data_name 'hmdb51' --path_x './pkl_data/hmdb51/hmdb51_x.pkl' --path_y './pkl_data/hmdb51/hmdb51_y.pkl' --batch_size 24 --save_model_path './target_models/hmdb51_slowfastnet'

python video_model_train.py --cuda 'cuda:0' --model_name 'C3D' --epochs 51 --data_name 'accident' --path_x './pkl_data/accident/accident_x.pkl' --path_y './pkl_data/accident/accident_y.pkl' --batch_size 24 --save_model_path './target_models/accident_C3D'
python video_model_train.py --cuda 'cuda:0' --model_name 'R3D' --epochs 51 --data_name 'accident' --path_x './pkl_data/accident/accident_x.pkl' --path_y './pkl_data/accident/accident_y.pkl' --batch_size 24 --save_model_path './target_models/accident_R3D'
python video_model_train.py --cuda 'cuda:0' --model_name 'R2Plus1D' --epochs 51 --data_name 'accident' --path_x './pkl_data/accident/accident_x.pkl' --path_y './pkl_data/accident/accident_y.pkl' --batch_size 24 --save_model_path './target_models/accident_R2Plus1D'
python video_model_train.py --cuda 'cuda:0' --model_name 'slowfastnet' --epochs 51 --data_name 'accident' --path_x './pkl_data/accident/accident_x.pkl' --path_y './pkl_data/accident/accident_y.pkl' --batch_size 24 --save_model_path './target_models/accident_slowfastnet'


python video_model_train.py --cuda 'cuda:0' --model_name 'C3D' --epochs 11 --data_name 'ucf101' --path_x './pkl_data/ucf101/ucf101_x.pkl' --path_y './pkl_data/ucf101/ucf101_y.pkl' --batch_size 24 --save_model_path './target_models/ucf101_C3D'
python video_model_train.py --cuda 'cuda:0' --model_name 'R3D' --epochs 11 --data_name 'ucf101' --path_x './pkl_data/ucf101/ucf101_x.pkl' --path_y './pkl_data/ucf101/ucf101_y.pkl' --batch_size 24 --save_model_path './target_models/ucf101_R3D'
python video_model_train.py --cuda 'cuda:0' --model_name 'R2Plus1D' --epochs 11 --data_name 'ucf101' --path_x './pkl_data/ucf101/ucf101_x.pkl' --path_y './pkl_data/ucf101/ucf101_y.pkl' --batch_size 24 --save_model_path './target_models/ucf101_R2Plus1D'
python video_model_train.py --cuda 'cuda:0' --model_name 'slowfastnet' --epochs 11 --data_name 'ucf101' --path_x './pkl_data/ucf101/ucf101_x.pkl' --path_y './pkl_data/ucf101/ucf101_y.pkl' --batch_size 24 --save_model_path './target_models/ucf101_slowfastnet'

