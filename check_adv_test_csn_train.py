import os
import gzip
import json

adv_path = '/data01/lizehan/proqa/code/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset/test.txt'

adv_test_urls = set()
with open(adv_path, 'r') as fin:
    for line in fin:
        adv_test_urls.add(line.strip())



csn_path = "/data01/lizehan/proqa/data/codesearch/python/final/jsonl/train"


csn_train_urls = set()
for root, dirs, files in os.walk(csn_path):
    # print(root, dirs, files)
    for filename in files:
        if filename.endswith(".gz"):
            with gzip.open(os.path.join(root, filename), 'r') as fin:
                for line in fin:
                    item = json.loads(line)
                    csn_train_urls.add(item['url'])


import pdb; pdb.set_trace();
