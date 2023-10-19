import os
import sys
import json
from datasets import load_dataset

lang = sys.argv[1]

input_path = f"data/the-valut/{lang}.function/full_train.jsonl"

output_path = f"pretrain_data"
os.makedirs(output_path, exist_ok=True)


output_path = f"{output_path}/thevalut_{lang}_docstring_code.jsonl"

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        item = json.loads(line)
        pair = {"query": item['docstring'], "doc": item['code']}
        fout.write(json.dumps(pair) + "\n")
