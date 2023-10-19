from datasets import load_dataset
import json
from tqdm import tqdm

jupyter_text_code = load_dataset("/data01/lizehan/proqa/raw_data/jupyter-code-text-pairs/data", split="train")

print(jupyter_text_code)

output_path = f"/data01/lizehan/proqa/data/pretrain/jupyter_text_code.jsonl"

with open(output_path, 'w') as fout:
    for item in tqdm(jupyter_text_code):
        pair = {"query": item['markdown'], "doc": item['code']}
        fout.write(json.dumps(pair) + "\n")
