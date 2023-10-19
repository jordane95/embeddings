import sys
from datasets import load_dataset
import json

lang = sys.argv[1]

the_vault_inline = load_dataset("data/the-vault-inline", languages=[lang], split="train")

print(the_vault_inline)


output_path = f"pretrain_data/thevault_{lang}_comment_code.jsonl"

with open(output_path, 'w') as fout:
    for item in the_vault_inline:
        pair = {"query": item['comment'], "doc": item['code']}
        fout.write(json.dumps(pair) + "\n")
