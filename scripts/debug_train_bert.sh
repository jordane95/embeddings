
export CUDA_VISIBLE_DEVICES=1

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
python train.py \
    --model_name_or_path models/gte-base \
    --output_dir debug \
    --train_dir /home/lzh/stackoverflow/pls \
    --data_config config/data_config.json \
    --query_column question \
    --doc_column answer \
    --add_prompt \
    --q_max_len 128 \
    --d_max_len 128 \
    --max_steps 100000 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 12 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1 \
    --add_pooler moe \
    --peft
