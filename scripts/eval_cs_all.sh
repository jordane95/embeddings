# export CUDA_VISIBLE_DEVICES=1,3,5,6,7
export CUDA_VISIBLE_DEVICES=2,3

model_dir=$1


bash scripts/eval_codesearch_hard.sh $model_dir


langs=('conala' 'so-ds' 'adv' 'cosqa' 'staqc' 'poj' 'codenet')

for lang in ${langs[@]}; do
    bash scripts/eval_codesearch_${lang}.sh $model_dir
done


