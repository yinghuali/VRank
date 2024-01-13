#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-23:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=yinghua.li@uni.lu
#SBATCH -p bigmem
#SBATCH --output=/dev/null
#SBATCH --mem 100G

python vrank_apfd_all.py --data_name 'ucf101' --model_name 'C3D_18'
python vrank_apfd_all.py --data_name 'ucf101' --model_name 'R2Plus1D_21'
python vrank_apfd_all.py --data_name 'ucf101' --model_name 'R3D_10'
python vrank_apfd_all.py --data_name 'ucf101' --model_name 'slowfastnet_24'

python vrank_apfd_all.py --data_name 'hmdb51' --model_name 'C3D_40'
python vrank_apfd_all.py --data_name 'hmdb51' --model_name 'R2Plus1D_90'
python vrank_apfd_all.py --data_name 'hmdb51' --model_name 'R3D_40'
python vrank_apfd_all.py --data_name 'hmdb51' --model_name 'slowfastnet_40'

python vrank_apfd_all.py --data_name 'accident' --model_name 'C3D_10'
python vrank_apfd_all.py --data_name 'accident' --model_name 'R2Plus1D_50'
python vrank_apfd_all.py --data_name 'accident' --model_name 'R3D_10'
python vrank_apfd_all.py --data_name 'accident' --model_name 'slowfastnet_40'
