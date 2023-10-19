export CUDA_VISIBLE_DEVICES=3

# model_path=ckpt/starencoder_no_csn_4a3096bs128msl10k2e-4
model_path=ckpt/pt_graphcodebert_so_v3_2a3096bs128msl10k2e-4

for ckpt in `ls $model_path`; do
    echo $ckpt
    output_dir=results/csn_hard/${model_path}/${ckpt}
    mkdir -p $output_dir
    python eval_codesearch_hard.py \
        --model-name-or-path ${model_path}/${ckpt} \
        --prompt '' \
        --pool-type avg \
        --batch-size 32 \
        --l2-normalize \
        --output-dir $output_dir
done
