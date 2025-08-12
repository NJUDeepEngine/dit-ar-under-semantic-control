CUDA_VISIBLE_DEVICES=0 nohup torchrun  \
--nnodes=1 \
--nproc_per_node=1 \
train.py \
--model DiT-L/4 \
--global-batch-size 1 \
--data-path /data0/lmy/imagenette2/train \
--num-classes 10 \
--max_gen_len 1000 \
--use-wandb \
--wandb-project "dit" \
--wandb-team "nju-ics" \
--wandb-name "DiT-L/4 posterior 2round" \
--epochs 100 \
--ckpt-every 2000 \
> ./training_log.txt 2>&1 &