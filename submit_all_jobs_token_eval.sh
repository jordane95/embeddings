
# finetune so v3 pretrained graphcodebert on csn + adv

# lang='adv'
# bash scripts/ft_cs_other_pt_graphcodebert_so_v3.sh $lang # adv

# bash ft_csn_all_pt_graphcodebert_so_v3.sh # csn



# 10.5 what about remove comment and docstring in code?
# lang='adv_remove'
# bash scripts/ft_cs_other_pt_graphcodebert_so_v3.sh $lang # adv_remove

# clean, ' '.join(code.split())
# lang='adv_clean'
# bash scripts/ft_cs_other_pt_graphcodebert_so_v3.sh $lang # adv_clean


# bash scripts/ft_cs_other_pt_graphcodebert_so_v3.sh solidity
# bash scripts/ft_cs_other_pt_graphcodebert_so_v3.sh sql


# langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')
# langs=('ruby')

# for lang in ${langs[@]}; do
#     bash scripts/ft_csn_lang_pt_graphcodebert_so_v3.sh ${lang}_remove
#     bash scripts/ft_csn_lang_pt_graphcodebert_so_v3.sh ${lang}_clean
# done


# bash scripts/pt_graphcodebert_so_v3_csn_remove.sh

# lang='adv_token'
# bash scripts/ft_cs_other_pt_graphcodebert_so_v3_csn_remove.sh $lang # adv_remove

# bash scripts/ft_cs_other_pt_graphcodebert_so_v3_csn.sh $lang # adv remove

# pretraining on remove set does not help


# # pt on so dec + csn
# bash scripts/pt_graphcodebert_so_dec_csn.sh

# langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')

# for lang in ${langs[@]}; do
#     bash scripts/ft_csn_lang_pt_graphcodebert_so_dec_csn.sh ${lang}_token
# done

# langs=('conala' 'so-ds' 'adv_token' 'cosqa' 'staqc' 'poj')

# for lang in ${langs[@]}; do
#     bash scripts/ft_cs_other_pt_graphcodebert_so_dec_csn.sh $lang
# done

# pt on csn
# bash scripts/pt_graphcodebert_csn.sh
# langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')
# for lang in ${langs[@]}; do
#     bash scripts/ft_csn_lang_pt_graphcodebert_csn.sh ${lang}_token
# done

# langs=('conala' 'so-ds' 'adv_token' 'cosqa' 'staqc' 'poj')
# langs=('adv_token')
# for lang in ${langs[@]}; do
#     bash scripts/ft_cs_other_pt_graphcodebert_csn.sh $lang
# done
langs=('conala' 'so-ds' 'staqc')
for lang in ${langs[@]}; do
    bash scripts/eval_codesearch_${lang}_concat.sh ckpt/ft_${lang}_2a64bs512msl3e2e-5_pt_graphcodebert
    bash scripts/eval_codesearch_${lang}_concat.sh ckpt/ft_${lang}_2a64bs512msl3e2e-5_pt_graphcodebert_csn_2a3096bs128msl10k2e-4
done


# pt on so dec
# bash scripts/pt_graphcodebert_so_dec.sh

# langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')
# for lang in ${langs[@]}; do
#     bash scripts/ft_csn_lang_pt_graphcodebert_so_dec.sh ${lang}_token
# done

# langs=('conala' 'so-ds' 'adv_token' 'cosqa' 'staqc' 'poj')
# for lang in ${langs[@]}; do
#     bash scripts/ft_cs_other_pt_graphcodebert_so_dec.sh $lang
# done

# # vanilla
# bash scripts/pt_graphcodebert.sh

# langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')
# for lang in ${langs[@]}; do
#     bash scripts/ft_csn_lang_pt_graphcodebert.sh ${lang}_token
# done

# langs=('conala' 'so-ds' 'adv_token' 'cosqa' 'staqc' 'poj')
# langs=('adv_token')
# for lang in ${langs[@]}; do
#     bash scripts/ft_cs_other_pt_graphcodebert.sh $lang
# done

