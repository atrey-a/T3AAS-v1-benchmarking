#! /bin/bash

python main.py \
    --path "/mnt/MIG_store/Datasets/t3aas-v1/6-column-no-aug" \
    --model "twostream_slitcnn" --dataset "t3aas" --mode "raw_form" \
    --epochs 80 --batch_size 32 --num_workers 16 --lr 0.00001 \
    --verbose_print --verify_random_forgery \
    --verify_skilled_forgery
