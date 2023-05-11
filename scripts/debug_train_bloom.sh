
export CUDA_VISIBLE_DEVICES=0,1

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

# bloom-560m
# fp16 crushed..., 512bs + deepspeed + gradckpt => 37G
# with weighted mean pooling, fp16 sucess!??, now => 22G
# 1024bs => 35G. 66h


# bloom-1b1
# 512bs => 34G, 53h


deepspeed train.py --deepspeed config/ds_config.json \
    --model_name_or_path bigscience/bloom-560m \
    --output_dir debug \
    --train_dir /data01/lizehan/proqa/pls \
    --data_config config/data_config.json \
    --query_column question \
    --doc_column answer \
    --max_steps 20000 \
    --learning_rate 2e-4 \
    --logging_steps 2 \
    --save_steps 500 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1024 \
    --normalize True \
    --pooling weightedmean \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 True \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1
