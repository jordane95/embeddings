# export CUDA_VISIBLE_DEVICES=1,3,5,6,7


model_dir='/data01/lizehan/proqa/embeddings/ckpt/starencoder_4a3096bs128msl20k2e-4/checkpoint-20000'

output_dir=results_code/${model_dir}

mkdir -p $output_dir

python eval_codesearch.py \
    --model-name-or-path $model_dir \
    --prompt '' \
    --pool-type avg \
    --batch-size 128 \
    --l2-normalize \
    --output-dir $output_dir
