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

echo =============== CS-1502 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1502 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs1502 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-1502 P2P MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1502 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs1502 >> /ihome/dmosse/rek94/results.txt

echo =============== CS-0449 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs0449 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs0449 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-0449 P2P MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs0449 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs0449 >> /ihome/dmosse/rek94/results.txt

echo =============== CS-0441 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs0441 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs0441 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-0441 P2P MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs0441 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs0441 >> /ihome/dmosse/rek94/results.txt

echo =============== CS-1622 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1622 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs1622 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-1622 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1622 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs1622 >> /ihome/dmosse/rek94/results.txt

echo =============== CS-0007 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs0007 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs0007 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-0007 P2P MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs0007 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs0007 >> /ihome/dmosse/rek94/results.txt

echo =============== CS-1567 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1567 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs1567 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-1567 P2P MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1567 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs1567 >> /ihome/dmosse/rek94/results.txt

echo =============== CS-1550 MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1550 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/base/cs1550 >> /ihome/dmosse/rek94/results.txt
echo =============== CS-1550 P2P MODEL EVALUATION ================ >> /ihome/dmosse/rek94/results.txt
python3 /ihome/dmosse/rek94/p2p-cce-training-script/main_predict.py --test-path /ihome/dmosse/rek94/new_data/dataset_bio/labeled/cs1550 --model /ix/dmosse/rek94/models/2025-09-03_13-16-56/p2p/cs1550 >> /ihome/dmosse/rek94/results.txt