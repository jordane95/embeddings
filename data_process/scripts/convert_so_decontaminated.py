import os
import sys
import json
from datasets import load_dataset

lang = sys.argv[1]

input_path = f"/data01/lizehan/proqa/data/substring/so-{lang}/data/decontanimated_qa.jsonl"

output_path = f"/data01/lizehan/proqa/data/pretrain/so_decontaminated/"
os.makedirs(output_path, exist_ok=True)


output_path_qd = f"{output_path}/stackoverflow_dec_{lang}_question_description.jsonl"

output_path_qa = f"{output_path}/stackoverflow_dec_{lang}_question_answer.jsonl"

output_path_da = f"{output_path}/stackoverflow_dec_{lang}_description_answer.jsonl"

raw_data = load_dataset("json", data_files=input_path, split='train')

def map2qd(item):
    query = item['title']
    doc = item['question']
    return {"query": query, "doc": doc}

def map2qa(item):
    query = item['title']
    doc = item['answer']
    return {"query": query, "doc": doc}

def map2da(item):
    query = item['question']
    doc = item['answer']
    return {"query": query, "doc": doc}


qd_pairs = raw_data.map(map2qd, batched=False, num_proc=16, remove_columns=raw_data.column_names)
qd_pairs.to_json(output_path_qd)

qa_pairs = raw_data.map(map2qa, batched=False, num_proc=16, remove_columns=raw_data.column_names)
qa_pairs.to_json(output_path_qa)


da_pairs = raw_data.map(map2da, batched=False, num_proc=16, remove_columns=raw_data.column_names)
da_pairs.to_json(output_path_da)

# f_qd = open(output_path_qd, 'w')
# f_qa = open(output_path_qa, 'w')

# for item in train_set:
#     question = item['title']
#     description = item['question']
#     answer = item['answer']
#     f_qd.write(json.dumps({"query": question, "doc": description}) + "\n")
#     f_qa.write(json.dumps({"query": question, "doc": answer}) + "\n")

# f_qd.close()
# f_qa.close()
