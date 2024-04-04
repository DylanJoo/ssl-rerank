# contriever with default independent cropping
torchrun --nproc_per_node 2 unsupervised_learning/train_rand_cropping.py \
    --model_name facebook/contriever \
    --output_dir models/ckpt/contriever-trec-covid \
    --per_device_train_batch_size 32 \
    --num_train_epochs 3 \
    --report_to wandb --run_name random_cropping_wo_mlm \
    --train_data_dir /home/dju/datasets/test_collection/random_cropping
