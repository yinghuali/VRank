#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --mem 300G


python get_noise_data.py  --select_method 'augmentation_width_shift' --save_path_pkl './pkl_data/ucf101_noise/augmentation_width_shift_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'
python get_noise_data.py  --select_method 'augmentation_height_shift' --save_path_pkl './pkl_data/ucf101_noise/augmentation_height_shift_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'
python get_noise_data.py  --select_method 'augmentation_horizontal_flip' --save_path_pkl './pkl_data/ucf101_noise/augmentation_horizontal_flip_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'
python get_noise_data.py  --select_method 'augmentation_featurewise_std_normalization' --save_path_pkl './pkl_data/ucf101_noise/augmentation_featurewise_std_normalization_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'
python get_noise_data.py  --select_method 'augmentation_zca_whitening' --save_path_pkl './pkl_data/ucf101_noise/augmentation_zca_whitening_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'
python get_noise_data.py  --select_method 'augmentation_shear_range' --save_path_pkl './pkl_data/ucf101_noise/augmentation_shear_range_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'
python get_noise_data.py  --select_method 'augmentation_channel_shift_range' --save_path_pkl './pkl_data/ucf101_noise/augmentation_channel_shift_range_x.pkl' --video_np_path './pkl_data/ucf101/ucf101_x.pkl'

python get_noise_data.py  --select_method 'augmentation_width_shift' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_width_shift_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'
python get_noise_data.py  --select_method 'augmentation_height_shift' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_height_shift_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'
python get_noise_data.py  --select_method 'augmentation_horizontal_flip' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_horizontal_flip_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'
python get_noise_data.py  --select_method 'augmentation_featurewise_std_normalization' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_featurewise_std_normalization_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'
python get_noise_data.py  --select_method 'augmentation_zca_whitening' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_zca_whitening_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'
python get_noise_data.py  --select_method 'augmentation_shear_range' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_shear_range_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'
python get_noise_data.py  --select_method 'augmentation_channel_shift_range' --save_path_pkl './pkl_data/hmdb51_noise/augmentation_channel_shift_range_x.pkl' --video_np_path './pkl_data/hmdb51/hmdb51_x.pkl'

python get_noise_data.py  --select_method 'augmentation_width_shift' --save_path_pkl './pkl_data/accident_noise/augmentation_width_shift_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'
python get_noise_data.py  --select_method 'augmentation_height_shift' --save_path_pkl './pkl_data/accident_noise/augmentation_height_shift_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'
python get_noise_data.py  --select_method 'augmentation_horizontal_flip' --save_path_pkl './pkl_data/accident_noise/augmentation_horizontal_flip_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'
python get_noise_data.py  --select_method 'augmentation_featurewise_std_normalization' --save_path_pkl './pkl_data/accident_noise/augmentation_featurewise_std_normalization_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'
python get_noise_data.py  --select_method 'augmentation_zca_whitening' --save_path_pkl './pkl_data/accident_noise/augmentation_zca_whitening_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'
python get_noise_data.py  --select_method 'augmentation_shear_range' --save_path_pkl './pkl_data/accident_noise/augmentation_shear_range_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'
python get_noise_data.py  --select_method 'augmentation_channel_shift_range' --save_path_pkl './pkl_data/accident_noise/augmentation_channel_shift_range_x.pkl' --video_np_path './pkl_data/accident/accident_x.pkl'


