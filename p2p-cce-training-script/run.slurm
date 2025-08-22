#!/bin/bash
#SBATCH --job-name=p2pselftraining
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=rek94@pitt.edu    
#SBATCH --mail-type=END,FAIL 

module purge
module load gcc/8.2.0
module load python/anaconda3.10-2022.10

source activate /ix/dmosse/rek94/envs/pytorch_gpu
module unload python/anaconda3.10-2022.10

# python3 /ihome/dmosse/rek94/p2p-cce-training-script/main.py --train-path /ihome/dmosse/rek94/data/dataset_bio_uniform/train_set --val-path /ihome/dmosse/rek94/data/dataset_bio_uniform/dev_set --test-path /ihome/dmosse/rek94/data/dataset_bio_uniform/test_set \
# --self-label-1-path /ihome/dmosse/rek94/data/dataset_bio_uniform/self_training_1 --self-label-2-path /ihome/dmosse/rek94/data/dataset_bio_uniform/self_training_2 > /ihome/dmosse/rek94/results.txt

python3 /ihome/dmosse/rek94/p2p-cce-training-script/main.py --labeled-path /ihome/dmosse/rek94/data/dataset_bio_uniform/labeled --val-path /ihome/dmosse/rek94/data/dataset_bio_uniform/dev_set \
 --self-label-1-path /ihome/dmosse/rek94/data/dataset_bio_uniform/self_training_1 --self-label-2-path /ihome/dmosse/rek94/data/dataset_bio_uniform/self_training_2 --distant-path /ihome/dmosse/rek94/data/dataset_bio_uniform/distant > /ihome/dmosse/rek94/results.txt

conda deactivate
