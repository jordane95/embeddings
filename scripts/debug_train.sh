
export CUDA_VISIBLE_DEVICES=0

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed ds_config.json \

# deepspeed train.py --deepspeed config/ds_config.json \
python train.py \
    --model_name_or_path bert-base-uncased \
    --output_dir debug \
    --train_dir /data01/lizehan/proqa/pls \
    --data_config config/data_config.json \
    --query_column question \
    --doc_column answer \
    --max_steps 20000 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 2 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device False \
    --fp16 \
    --gradient_checkpointing \
    --grad_cache True \
    --seed 42
