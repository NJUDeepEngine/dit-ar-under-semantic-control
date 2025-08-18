CUDA_VISIBLE_DEVICES=0 nohup torchrun  \
--nnodes=1 \
--nproc_per_node=1 --master_port=29509 \
train.py \
--model DiT-L/4 \
--global-batch-size 8 \
--data-path /data0/lmy/imagenette2/train \
--num-classes 1000 \
--max_gen_len 1000 \
--epochs 5 \
--ckpt-every 500 \
--detailed-log-every 500 \
--detailed-log-pic-print \
--detailed-log-middle-vars-print \
--rand-t 20 \
--loss-type KL \
--results-dir /data3/xdk/loss-type-check \
--vae-path "/data0/dit-assets/sd-vae-ft-ema" \
--description "KL loss type test; 1g dataset, unfixed sequence, print all middle vars; don't predict eos patch; t = 20" \
> ./training_log_KL.txt 2>&1 &

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
