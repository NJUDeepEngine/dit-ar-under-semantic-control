CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun  \
--nnodes=1 \
--nproc_per_node=4  \
train.py \
--model DiT-L/4 \
--global-batch-size 16 \
--data-path /data0/lmy/imagenette2/train \
--num-classes 10 \
--max_gen_len 1000 \
--epochs 100 \
--ckpt-every 2000 \
--detailed-log-every 1000 \
--detailed-log-pic-print \
--detailed-log-middle-vars-print \
--use-real-target \
--results-dir /data3/xdk/dit-results \
--vae-path "/data0/dit-assets/sd-vae-ft-ema" \
--description "1g dataset, unfixed sequence, print all middle vars, Use real next patch as target" \
> ./training_log.txt 2>&1 &

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