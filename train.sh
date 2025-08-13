CUDA_VISIBLE_DEVICES=0,1 nohup torchrun  \
--nnodes=1 \
--nproc_per_node=2 \
train.py \
--model DiT-L/4 \
--global-batch-size 8 \
--data-path /data0/lmy/imagenet2012/train \
--num-classes 1000 \
--max_gen_len 1000 \
--epochs 100 \
--ckpt-every 5000 \
> ./training_log.txt 2>&1 &


# --use-wandb \
# --wandb-project "dit" \
# --wandb-team "nju-ics" \
# --wandb-name "DiT-L/4 posterior large-trainset retrain" \