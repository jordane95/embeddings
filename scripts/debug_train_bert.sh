
export CUDA_VISIBLE_DEVICES=2,3

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

# bert-large
# 1024bs => 26G 42h

deepspeed train.py --deepspeed config/ds_config.json \
    --model_name_or_path bert-large-uncased \
    --output_dir debug \
    --train_dir /data01/lizehan/proqa/pls \
    --data_config config/data_config.json \
    --query_column question \
    --doc_column answer \
    --add_prompt \
    --max_steps 20000 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 2048 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1
