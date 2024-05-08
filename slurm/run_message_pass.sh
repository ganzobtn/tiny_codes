# 192.168.52.102 master node
CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
--nproc_per_node=1 --nnodes=2 --node_rank=0 \
--master_addr=192.168.52.102 --master_port=1234 \
resnet_ddp.py \
