
model_path=$1
output_dir=$2


python eval_mteb.py \
    --task-types "STS" "Summarization" "PairClassification" "Classification" "Reranking" "Clustering" "BitextMining" \
    --output-dir ${output_dir} \
    --model-name-or-path ${model_path} \
    --pooling mean \
    --normalize \
    --add-pooler dense 
