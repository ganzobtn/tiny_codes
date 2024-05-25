CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 4 --nproc_per_node 1 --node_rank 2 \
        --master_addr=192.168.52.103 --master_port=1234 \
        supervised_finetuning.py \
        --base_model 'baffo32/decapoda-research-llama-7B-hf'     \
        --dataset_name './data/alpaca_gpt4_data.json' \
        --streaming     --lr_scheduler_type 'cosine'   \
        --learning_rate 1e-5     --max_steps 4000     \
        --output_dir '/l/users/ganzorig.batnasan/results/supervised_llama/' \
        --resume_from_checkpoint '/l/users/ganzorig.batnasan/results/supervised_llama/checkpoint-2000' \