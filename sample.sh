CUDA_VISIBLE_DEVICES=0 torchrun   --nnodes=1   --nproc_per_node=1  --master_port=29506  sample_copy.py \
--model DiT-L/4 \
--ckpt /data3/lmy/ss_ksu_result/single_step/checkpoints/0011500.pt \
--image-size 256 \
--num-classes 10 \
--data-path /data0/lmy/imagenet2012/train
