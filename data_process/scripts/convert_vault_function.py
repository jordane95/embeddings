import sys
from datasets import load_dataset
import json
from tqdm import tqdm

lang = sys.argv[1]

the_vault_inline = load_dataset("/data01/lizehan/proqa/raw_data/the-vault-function", languages=[lang], split="train")

print(the_vault_inline)

output_path = f"/data01/lizehan/proqa/data/pretrain/thevault_{lang}_docstring_function.jsonl"

with open(output_path, 'w') as fout:
    for item in tqdm(the_vault_inline):
        pair = {"query": item['docstring'], "doc": item['code']}
        fout.write(json.dumps(pair) + "\n")
