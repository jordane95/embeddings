# export CUDA_VISIBLE_DEVICES=1,3,5,6,7
export CUDA_VISIBLE_DEVICES=0,1

model_dir=$1

output_dir=results_code_dev/${model_dir}

mkdir -p $output_dir

python eval_codesearch_dev.py \
    --model-name-or-path $model_dir \
    --prompt '' \
    --pool-type avg \
    --batch-size 32 \
    --l2-normalize \
    --output-dir $output_dir
