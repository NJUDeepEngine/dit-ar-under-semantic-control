CUDA_VISIBLE_DEVICES=0 torchrun   --nnodes=1   --nproc_per_node=1  --master_port=29502  sample_copy.py \
--model DiT-L/4 \
--ckpt /data3/xdk/dit-results/001-DiT-L-4/checkpoints/0006000.pt \
--image-size 256 \
--num-classes 1000 \
#--data-path /data0/lmy/imagenet2012/train