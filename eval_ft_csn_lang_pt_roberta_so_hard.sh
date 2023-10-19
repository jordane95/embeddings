export CUDA_VISIBLE_DEVICES=2

lang=$1
# model_path=ckpt/starencoder_no_csn_4a3096bs128msl10k2e-4
model_path=ckpt/ft_csn_${lang}_2a64bs512msl3e2e-5_pt_roberta_so_2a3096bs128msl10k2e-4


output_dir=results/csn_hard/${model_path}/${ckpt}
mkdir -p $output_dir
python eval_codesearch_hard.py \
    --model-name-or-path ${model_path}/${ckpt} \
    --prompt '' \
    --pool-type avg \
    --batch-size 32 \
    --l2-normalize \
    --output-dir $output_dir \
    --langs $lang

