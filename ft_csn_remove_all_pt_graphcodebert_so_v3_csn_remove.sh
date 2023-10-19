
export CUDA_VISIBLE_DEVICES=0,1

# torchrun --nproc_per_node 2 train.py \
# python train.py \
# deepspeed train.py --deepspeed config/ds_config.json \


# w/ gradckpt: 6G x 32h
# w/o gradckpt: 24G x 24h

langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')

for lang in ${langs[@]}; do
    bash scripts/ft_csn_lang_pt_graphcodebert_so_v3_csn_remove.sh ${lang}_remove
done
