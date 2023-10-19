export CUDA_VISIBLE_DEVICES=2

# model_path=ckpt/starencoder_no_csn_4a3096bs128msl10k2e-4
model_path=ckpt/starencoder_4a3096bs128msl20k2e-4

for ckpt in `ls $model_path`; do
    echo $ckpt
    output_dir=results_code/${model_path}/${ckpt}
    mkdir -p $output_dir
    python eval_codesearch.py \
        --model-name-or-path ${model_path}/${ckpt} \
        --prompt '' \
        --pool-type avg \
        --batch-size 32 \
        --l2-normalize \
        --output-dir $output_dir
done
