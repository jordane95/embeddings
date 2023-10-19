
export CUDA_VISIBLE_DEVICES=0,1,2

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

# bloom-560m
# fp16 crushed..., 512bs + deepspeed + gradckpt => 37G
# with weighted mean pooling, fp16 sucess!??, now => 22G
# 1024bs => 35G. 66h
# 128bs + bitfit => 7G
# 128bs => 13G

# bloom-1b1
# 512bs => 34G, 53h


deepspeed train.py --deepspeed config/ds_config.json \
    --model_name_or_path ../models/starencoder \
    --output_dir ckpt/starencoder_so_3a3096bs128msl10k2e-4 \
    --train_dir ../pretrain_data/ \
    --data_config config/pt_data_config_so.json \
    --query_column query \
    --doc_column doc \
    --max_steps 10000 \
    --save_steps 1000 \
    --learning_rate 2e-4 \
    --logging_steps 10 \
    --warmup_steps 1000 \
    --per_device_train_batch_size 3096 \
    --normalize True \
    --pooling mean \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 True \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1 \
    --mix_coefficient 0.5
