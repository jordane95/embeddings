
export CUDA_VISIBLE_DEVICES=0,1

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \


# w/ gradckpt: 6G x 32h
# w/o gradckpt: 24G x 24h

lang=$1

pretrained_model_or_path=ckpt/pt_graphcodebert_so_dec_2a3096bs128msl10k2e-4/checkpoint-10000
output_dir=ckpt/ft_${lang}_2a64bs512msl3e2e-5_pt_graphcodebert_so_dec_2a3096bs128msl10k2e-4
data_config_path=config/ft_data_config_${lang}.yaml

add_instruction=False

deepspeed finetune_mnkd.py --deepspeed config/ds_config.json \
    --model_name_or_path $pretrained_model_or_path \
    --output_dir $output_dir \
    --finetune_data_config $data_config_path \
    --add_instruction $add_instruction \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --logging_steps 10 \
    --save_strategy epoch \
    --warmup_ratio 0.1 \
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

model_dir=$output_dir

output_dir=results/${lang}/${model_dir}

mkdir -p $output_dir

python eval_codesearch_${lang}.py \
    --model-name-or-path $model_dir \
    --prompt '' \
    --pool-type avg \
    --batch-size 32 \
    --l2-normalize \
    --output-dir $output_dir

output_dir=results/${lang}_concat/${model_dir}

mkdir -p $output_dir
python eval_codesearch_${lang}_concat.py \
    --model-name-or-path $model_dir \
    --prompt '' \
    --pool-type avg \
    --batch-size 32 \
    --l2-normalize \
    --output-dir $output_dir
