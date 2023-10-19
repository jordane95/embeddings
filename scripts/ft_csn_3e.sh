
# export CUDA_VISIBLE_DEVICES=0,1

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \


# w/ gradckpt: 6G x 32h
# w/o gradckpt: 24G x 24h

pretrained_model_or_path=../models/starencoder
output_dir=ckpt/ft_csn_4a64bs512msl3e2e-5_starencoder

add_instruction=False

deepspeed finetune_mnkd.py --deepspeed config/ds_config.json \
    --model_name_or_path $pretrained_model_or_path \
    --output_dir $output_dir \
    --finetune_data_config config/ft_data_config_csn.yaml \
    --add_instruction $add_instruction \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --save_strategy epoch \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 64 \
    --q_max_len 512 \
    --d_max_len 512 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1
