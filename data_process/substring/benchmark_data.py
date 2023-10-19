
"""data to filter out of the dataset"""
import json
import itertools
from pathlib import Path


def csn_lang(lang: str):
    data_path = f'/data01/lizehan/proqa/CodeBERT/GraphCodeBERT/codesearch/dataset/{lang}/test.jsonl'
    with open(data_path) as f:
        data = [json.loads(line) for line in f]
    return data



def conala_query():
    data_path = "/data01/lizehan/proqa/raw_data/conala/conala-test-curated-0.5.jsonl"
    with open(data_path) as f:
        data = [json.loads(line)['query'] for line in f]
    return data

def sods_query():
    data_path = "/data01/lizehan/proqa/raw_data/so-ds/so-ds-feb20-test.jsonl"
    with open(data_path) as f:
        data = [json.loads(line)['query'] for line in f]
    return data

def staqc_query():
    data_path = "/data01/lizehan/proqa/raw_data/staqc/staqc-py-test-raw.jsonl"
    with open(data_path) as f:
        data = [json.loads(line)['query'] for line in f]
    return data


def csn_python_query():
    data_path = '/data01/lizehan/proqa/CodeBERT/GraphCodeBERT/codesearch/dataset/python/test.jsonl'
    with open(data_path) as f:
        data = [json.loads(line)['docstring'] for line in f]
        code = [json.loads(line)['code'] for line in f]
    return data + code

FILTER_OUT = {
    # "conala": conala_query(),
    # "sods": sods_query(),
    # "staqc": staqc_query(),
    "csn_python": csn_python_query(),
}

for benchmark, values in FILTER_OUT.items():
    print(f"num strings from {benchmark}: {len(values)}")
