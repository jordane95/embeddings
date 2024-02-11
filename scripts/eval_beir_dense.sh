
model_path=$1
output_dir=$2


python eval_beir.py \
    --task-types "Retrieval" \
    --output-dir ${output_dir} \
    --model-name-or-path ${model_path} \
    --pooling mean \
    --add-pooler dense 
