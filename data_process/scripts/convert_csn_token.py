import os
import sys
import gzip
import json

lang = sys.argv[1]

train_path = f"/data01/lizehan/proqa/raw_data/codesearch/{lang}/final/jsonl/train"

output_path = f"/data01/lizehan/proqa/data/pretrain/csn_token_{lang}_docstring_code.jsonl"


with open(output_path, 'w') as fout:
    for root, dirs, files in os.walk(train_path):
        # print(root, dirs, files)
        for filename in files:
            if filename.endswith(".gz"):
                with gzip.open(os.path.join(root, filename), 'r') as fin:
                    for line in fin:
                        item = json.loads(line)
                        pair = {"query": ' '.join(item['docstring_tokens']), "doc": ' '.join(item['code_tokens'])}
                        fout.write(json.dumps(pair) + "\n")
