
export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

# bloom-560m
# fp16 crushed..., 512bs + deepspeed + gradckpt => 37G
# with weighted mean pooling, fp16 sucess!??, now => 22G
# 1024bs => 35G. 66h


# bloom-1b1
# 512bs => 34G, 53h

model_path="/data01/lizehan/llm/llama_hf/7B"

deepspeed train.py --deepspeed config/ds_config.json \
    --model_name_or_path ${model_path} \
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
    --per_device_train_batch_size 1 \
    --normalize True \
    --pooling last \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 False \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1 \
    --add_pooler \
    --embedding_dim 768 \
    --bf16 True 