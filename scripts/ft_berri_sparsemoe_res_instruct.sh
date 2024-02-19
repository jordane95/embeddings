
export CUDA_VISIBLE_DEVICES=0

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

# bert-large
# 1024bs => 26G 42h

# bert-base
# 512bs x 512msl => 33G x 31h
# export CUDA_LAUNCH_BLOCKING=1

# deepspeed --include localhost:1 train.py \
#     --deepspeed config/ds_config.json \

pretrained_model_or_path=/home/lzh/data/gte-base
output_dir=ckpt/ft_berri_sparsemoe_res_instruct
add_instruction=True

# deepspeed finetune_mnkd.py --deepspeed config/ds_config.json \
python finetune_mnkd.py \
    --model_name_or_path $pretrained_model_or_path \
    --output_dir $output_dir \
    --finetune_data_config config/ft_data_config_instruct.yaml \
    --add_instruction $add_instruction \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --save_steps 1000 \
    --warmup_steps 2000 \
    --per_device_train_batch_size 3 \
    --q_max_len 512 \
    --d_max_len 512 \
    --normalize True \
    --temperature 0.02 \
    --negatives_x_device True \
    --fp16 \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1 \
    --add_pooler sparsemoe \
    --n_experts 8 \
    --topk 2 \
    --peft \
    --residual_pooler \
    --load_balancing_loss_ratio 1 \
    --residual
