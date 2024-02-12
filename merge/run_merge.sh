
result_dir=$1
save_dir=$2

python merge/merge_cqadupstack.py ${result_dir}

python merge/mteb_meta.py ${result_dir} ${save_dir}

python merge/merge_mteb.py --meta-data-path ${save_dir}/mteb_metadata.md
