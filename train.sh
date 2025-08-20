#!/bin/bash

# 默认：不使用 nohup
USE_NOHUP=false

# 如果传入 --nohup 就启用
for arg in "$@"; do
    if [ "$arg" == "--nohup" ]; then
        USE_NOHUP=true
    fi
done

export CUDA_VISIBLE_DEVICES=1

# 使用数组构建命令
CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node=1
    --master_port=29502
    train_copy.py
    --model DiT-L/4
    --global-batch-size 12
    --data-path /data0/lmy/imagenette2/train
    --num-classes 10
    --max_gen_len 1000
    --epochs 5
    --ckpt-every 500
    --detailed-log-every 500
    --detailed-log-pic-print
    --detailed-log-middle-vars-print
    --rand-t 20
    --loss-type huber
    --use-real-target
    --use_ss
    --results-dir /data3/lmy/train_style-check
    --vae-path '/data0/dit-assets/sd-vae-ft-ema'
    --description 'use ss,mse loss type test; 1g dataset, unfixed sequence, print all middle vars; do not predict eos patch; t = 20'
)

if $USE_NOHUP; then
    echo "Running with nohup..."
    nohup "${CMD[@]}" > ./training_log_KL.txt 2>&1 &
    echo "Process started with PID: $!"
else
    echo "Running in foreground..."
    "${CMD[@]}"
fi


# --use-real-target \
# --detailed-log-loss-analysis \
# --detailed-log-middle-vars-print \
# --vae-path "/data0/dit-assets/sd-vae-ft-ema" \

# things to test:
#1. difference between training loss and inference loss
#2. difference between posterior loss and true label loss

# --data-path /data0/lmy/imagenet2012/train \
# --use-wandb \
# --wandb-project "dit" \
# --wandb-team "nju-ics" \
# --wandb-name "DiT-L/4 posterior large-trainset retrain" \
