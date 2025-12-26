#!/bin/bash
#SBATCH --job-name=p2pselftraining
#SBATCH --nodes=1
#SBATCH --time=2-17:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=rek94@pitt.edu    
#SBATCH --mail-type=END,FAIL 

## SBATCH --constraint=80g

module purge
module load gcc/8.2.0
module load python/ondemand-jupyter-python3.11

source activate /ix/dmosse/rek94/envs/pytorch_gpu

python3 /ihome/dmosse/rek94/p2p-cce-training-script/main.py --train-path /ihome/dmosse/rek94/new_data/dataset_bio/train_set --val-path /ihome/dmosse/rek94/new_data/dataset_bio/dev_set --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set \
--self-label-1-path /ihome/dmosse/rek94/new_data/dataset_bio/self_training_1 --self-label-2-path /ihome/dmosse/rek94/new_data/dataset_bio/self_training_2 -o /ix/dmosse/rek94/models > /ihome/dmosse/rek94/results.txt

#python3 /ihome/dmosse/rek94/p2p-cce-training-script/main.py --labeled-path /ihome/dmosse/rek94/data/dataset_bio/labeled --val-path /ihome/dmosse/rek94/data/dataset_bio/dev_set \
# --self-label-1-path /ihome/dmosse/rek94/data/dataset_bio/self_training_1 --self-label-2-path /ihome/dmosse/rek94/data/dataset_bio/self_training_2 --distant-path /ihome/dmosse/rek94/data/dataset_bio/distant -o /ix/dmosse/rek94/models -m > /ihome/dmosse/rek94/results.txt
