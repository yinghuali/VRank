#!/bin/bash -l
#SBATCH -N 1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --time=1-12:00:00

source activate py37
python get_vt_extractor_x.py --path_x './pkl_data/accident/accident_x.pkl' --save_path './pkl_data/accident/vt_accident_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51/hmdb51_x.pkl' --save_path './pkl_data/hmdb51/vt_hmdb51_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101/ucf101_x.pkl' --save_path './pkl_data/ucf101/vt_ucf101_x.pkl'


python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_channel_shift_range_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_channel_shift_range_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_featurewise_std_normalization_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_featurewise_std_normalization_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_height_shift_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_height_shift_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_horizontal_flip_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_horizontal_flip_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_shear_range_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_shear_range_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_width_shift_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_width_shift_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/accident_noise/augmentation_zca_whitening_x.pkl' --save_path './pkl_data/accident_noise/vt_augmentation_zca_whitening_x.pkl'


python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_channel_shift_range_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_channel_shift_range_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_featurewise_std_normalization_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_featurewise_std_normalization_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_height_shift_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_height_shift_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_horizontal_flip_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_horizontal_flip_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_shear_range_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_shear_range_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_width_shift_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_width_shift_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/hmdb51_noise/augmentation_zca_whitening_x.pkl' --save_path './pkl_data/hmdb51_noise/vt_augmentation_zca_whitening_x.pkl'


python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_channel_shift_range_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_channel_shift_range_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_featurewise_std_normalization_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_featurewise_std_normalization_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_height_shift_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_height_shift_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_horizontal_flip_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_horizontal_flip_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_shear_range_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_shear_range_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_width_shift_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_width_shift_x.pkl'
python get_vt_extractor_x.py --path_x './pkl_data/ucf101_noise/augmentation_zca_whitening_x.pkl' --save_path './pkl_data/ucf101_noise/vt_augmentation_zca_whitening_x.pkl'

