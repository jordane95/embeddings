
model_path=ckpt/starencoder_no_csn_4a3096bs128msl20k2e-4

for ckpt in `ls $model_path`; do
    echo $ckpt
    output_dir=results_code_dev/${model_path}/${ckpt}
    mkdir -p $output_dir
    python eval_codesearch_dev.py \
        --model-name-or-path ${model_path}/${ckpt} \
        --prompt '' \
        --pool-type avg \
        --batch-size 32 \
        --l2-normalize \
        --output-dir $output_dir
done
