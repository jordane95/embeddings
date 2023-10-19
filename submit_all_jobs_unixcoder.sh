
# bash scripts/pt_unixcoder_so_dec_2e-5.sh

# langs=('go' 'java' 'javascript' 'php' 'python' 'ruby')

# for lang in ${langs[@]}; do
#     bash scripts/ft_csn_lang_pt_unixcoder_so_dec.sh ${lang}_token
# done

# langs=('conala' 'so-ds' 'adv_token' 'cosqa' 'staqc' 'poj')

# for lang in ${langs[@]}; do
#     bash scripts/ft_cs_other_pt_unixcoder_so_dec.sh $lang
# done


langs=('java' 'python' 'php')

for lang in ${langs[@]}; do
    bash scripts/ft_csn_lang_pt_unixcoder.sh ${lang}_token
done
