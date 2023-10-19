from utils import remove_comments_and_docstrings

from transformers import AutoTokenizer

data_path = '/data01/lizehan/proqa/code/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset/test.jsonl'

tok = AutoTokenizer.from_pretrained('../models/graphcodebert-base')

import json

f = open(data_path)


item = json.loads(next(f))

tokened = ' '.join(item['function_tokens'])
removed = remove_comments_and_docstrings(item['function'], 'python')
cleaned = ' '.join(removed.split())


print("tokened:")
print(tokened)
print(tok(tokened)['input_ids'])
print("removed:")
print(removed)
print(tok(removed)['input_ids'])
print("cleaned:")
print(cleaned)
print(tok(cleaned)['input_ids'])


print('tokned:')
print(' '.join(item['docstring_tokens']))
print(tok(' '.join(item['docstring_tokens']))['input_ids'])
print('original:')
print(item['docstring'])
print(tok(item['docstring'])['input_ids'])

import pdb; pdb.set_trace()
