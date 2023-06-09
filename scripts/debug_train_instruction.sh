
export CUDA_VISIBLE_DEVICES=2

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

deepspeed train.py --deepspeed config/ds_config.json \
    --model_name_or_path bert-base-uncased \
    --output_dir debug \
    --train_dir /data01/lizehan/embeddings/data \
    --data_config config/data_instruction_config.json \
    --add_instruction \
    --mask_instruction_pooling False \
    --query_column query \
    --doc_column pos \
    --max_steps 20000 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1024 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device False \
    --fp16 \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1
