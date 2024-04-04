#!/bin/sh
#SBATCH --job-name=indcrop
#SBATCH --partition gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate exa-dm_env

# Setup experiments
MAX_STEPS=5000
MAX_CKPTS=2

# Setting of lexical-enhanced DR
method=ind-cropping
learn=bce
negative=random
exp=${method}-${learn}-${negative}

# Go
torchrun --nproc_per_node 2 \
    unsupervised_learning/train_ind_cropping.py \
    --model_name bert-base-uncased \
    --output_dir models/ckpt/monobert-${exp}-trec-covid \
    --temperature 1.0 \
    --per_device_train_batch_size 8 \
    --curr_mask_ratio 0.0 \
    --chunk_length 128 \
    --max_length 256 \
    --save_steps 2500 \
    --save_total_limit $MAX_CKPTS \
    --max_steps $MAX_STEPS \
    --warmup_steps 500 \
    --fp16 \
    --report_to wandb --run_name ${exp} \
    --train_data_dir /home/dju/datasets/test_collection/ind_cropping

echo done
