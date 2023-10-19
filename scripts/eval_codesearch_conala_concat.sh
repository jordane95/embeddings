# export CUDA_VISIBLE_DEVICES=1,3,5,6,7
export CUDA_VISIBLE_DEVICES=2

model_dir=$1

output_dir=results/conala_concat/${model_dir}

mkdir -p $output_dir

python eval_codesearch_conala_concat.py \
    --model-name-or-path $model_dir \
    --prompt '' \
    --pool-type avg \
    --batch-size 32 \
    --l2-normalize \
    --output-dir $output_dir
