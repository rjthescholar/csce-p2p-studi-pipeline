#!/bin/bash
#SBATCH --job-name=p2pselftraining
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --constraint=80g
#SBATCH --mail-user=rek94@pitt.edu    
#SBATCH --mail-type=END,FAIL 

module purge
module load gcc/8.2.0
module load python/ondemand-jupyter-python3.11

source activate /ix/dmosse/rek94/envs/pytorch_gpu

# python3 /ihome/dmosse/rek94/p2p-cce-training-script/main.py --train-path /ihome/dmosse/rek94/new_data/dataset_bio/train_set --val-path /ihome/dmosse/rek94/new_data/dataset_bio/dev_set --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set \
# --self-label-1-path /ihome/dmosse/rek94/new_data/dataset_bio/self_training_1 --self-label-2-path /ihome/dmosse/rek94/new_data/dataset_bio/self_training_2 > /ihome/dmosse/rek94/results.txt

echo =============== MODEL 1 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/base/1 >> /ihome/dmosse/rek94/results.txt
echo =============== P2P MODEL 1 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/p2p/1 >> /ihome/dmosse/rek94/results.txt

echo =============== MODEL 2 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/base/2 >> /ihome/dmosse/rek94/results.txt
echo =============== P2P MODEL 2 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/p2p/2 >> /ihome/dmosse/rek94/results.txt

echo =============== MODEL 3 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/base/3 >> /ihome/dmosse/rek94/results.txt
echo =============== P2P MODEL 3 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/p2p/3 >> /ihome/dmosse/rek94/results.txt

echo =============== MODEL 4 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/base/4 >> /ihome/dmosse/rek94/results.txt
echo =============== P2P MODEL 5 EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/test_set --model /ix/dmosse/rek94/models/2025-12-26_20-10-51/p2p/4 >> /ihome/dmosse/rek94/results.txt
