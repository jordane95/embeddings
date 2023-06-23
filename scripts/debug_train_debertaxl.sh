
export CUDA_VISIBLE_DEVICES=2,3

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \

# bert-large
# 1024bs => 26G 42h

# bert-base
# 512bs x 512msl => 33G x 31h

# for deberta-v2-xl, 1e-4 diverge, 5e-5 diverge, 2e-5 works well, also 3e-5
# 512bs x 128msl x 2GPU x 50k => 32/33G gpu x 140h
# actual on 2 node each with 8 A100 x 80G cards, 
# 1024bs x 128msl x 16GPU x 50k => 51/54G x 260h

# for deberta-v2-xxlarge, 3e-5 works well, 
# 256bs x 128msl x 2GPU x 50k => 33G per gpu x 140h


torchrun --nproc_per_node 2 train.py \
    --deepspeed config/ds_config.json \
    --model_name_or_path ../models/deberta-v2-xxlarge \
    --output_dir debug \
    --train_dir /data01/lizehan/proqa/pls \
    --data_config config/data_config.json \
    --query_column question \
    --doc_column answer \
    --add_prompt \
    --q_max_len 128 \
    --d_max_len 128 \
    --max_steps 50000 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --save_steps 500 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 256 \
    --normalize True \
    --temperature 0.01 \
    --negatives_x_device True \
    --fp16 \
    --gradient_checkpointing True \
    --grad_cache False \
    --seed 42 \
    --dataloader_num_workers 1 \
    --loss_scale 1
