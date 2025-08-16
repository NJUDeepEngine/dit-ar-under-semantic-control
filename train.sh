CUDA_VISIBLE_DEVICES=1 nohup torchrun  \
--nnodes=1 \
--nproc_per_node=1  \
train.py \
--model DiT-L/4 \
--global-batch-size 4 \
--data-path /data0/lmy/imagenet2012/train \
--num-classes 1000 \
--max_gen_len 1000 \
--epochs 100 \
--ckpt-every 2000 \
--detailed-log-every 1000 \
--detailed-log-pic-print \
--use-real-target \
--results-dir /data3/xdk/dit-results \
--vae-path "/data0/dit-assets/sd-vae-ft-ema" \
--description "large dataset, unfixed sequence, print pics, Use real next patch as target" \
> ./training_log.txt 2>&1 &

# --detailed-log-loss-analysis \
# --detailed-log-middle-vars-print \
# --detailed-log-target-print \
# --vae-path "/data0/dit-assets/sd-vae-ft-ema" \

# things to test:
#1. difference between training loss and inference loss
#2. difference between posterior loss and true label loss

# --data-path /data0/lmy/imagenet2012/train \
# --use-wandb \
# --wandb-project "dit" \
# --wandb-team "nju-ics" \
# --wandb-name "DiT-L/4 posterior large-trainset retrain" \