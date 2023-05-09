
export CUDA_VISIBLE_DEVICES=2

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \


# w/ gradckpt: 6G x 32h
# w/o gradckpt: 24G x 24h

pretrained_model_or_path=bert-base-uncased
output_dir=ckpt/ft_medi

data_path=/data01/lizehan/embeddings/instructor-embedding/cache/samples.100.json
add_instruction=False

deepspeed finetune.py --deepspeed config/ds_config.json \
    --model_name_or_path $pretrained_model_or_path \
    --output_dir $output_dir \
    --finetune_data_path $data_path \
    --add_instruction $add_instruction \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --logging_steps 500 \
    --save_steps 1000 \
    --warmup_ratio 0.1 \
    --per_device_train_batch_size 16 \
    --q_max_len 512 \
    --d_max_len 512 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 \
    --gradient_checkpointing False \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1
