
langs=('so-ds' 'staqc')

for lang in "${langs[@]}"; do
    echo "Evaluating $lang"
    bash scripts/eval_codesearch_${lang}_concat.sh ckpt/ft_${lang}_2a64bs512msl3e2e-5_pt_graphcodebert_so_v3_csn_2a3096bs128msl10k2e-4/
done
