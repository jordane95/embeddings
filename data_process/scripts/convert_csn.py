import os
import sys
import gzip
import json

lang = sys.argv[1]

train_path = f"data/codesearch/{lang}/final/jsonl/train"

output_path = f"pretrain_data/codesearchnet_{lang}_docstring_code.jsonl"


with open(output_path, 'w') as fout:
    for root, dirs, files in os.walk(train_path):
        # print(root, dirs, files)
        for filename in files:
            if filename.endswith(".gz"):
                with gzip.open(os.path.join(root, filename), 'r') as fin:
                    for line in fin:
                        item = json.loads(line)
                        pair = {"query": item['docstring'], "doc": item['code']}
                        fout.write(json.dumps(pair) + "\n")
