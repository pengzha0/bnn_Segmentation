#!/bin/bash
#SBATCH --job-name=react
#SBATCH --account=Project_2002243
#SBATCH --partition=gpusmall
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:a100:1,nvme:180
#SBATCH --output test_64.txt
#SBATCH --error err_test_64.txt

export PATH="/scratch/project_2004772/conda/mae/bin:$PATH"

# pip list
# srun nvidia-smi
python3 /scratch/project_2004772/medical_BNN/bnn_Segmentation/tools_zp/train.py
# python3 train.py -c ./configs/datasets/cifar100.yml --model cct_6_3x1_32_c100 --output ./base_results /scratch/project_2004772/BinViT/Compact-Transformers/data/
#python3 train.py -c ./configs/datasets/cifar100.yml --model cct_7_3x1_32 /scratch/project_2004772/BinViT/Compact-Transformers/data/
#
