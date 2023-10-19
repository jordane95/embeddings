import json


# langs = ['ruby', 'javascript', 'go', 'python', 'java', 'php']
langs = ['java']

data_path_template = '/data01/lizehan/proqa/data/pretrain/codesearchnet_{lang}_docstring_code.jsonl'

removed_path_template = '/data01/lizehan/proqa/data/pretrain/csn_remove_{lang}_docstring_code.jsonl'

for lang in langs:
    data_path = data_path_template.format(lang=lang)
    removed_path = removed_path_template.format(lang=lang)
    with open(data_path) as f, open(removed_path) as f2:
            for line1, line2 in zip(f, f2):
                code = json.loads(line1)['doc']
                code_removed = json.loads(line2)['doc']
                print(json.loads(line1)['query'])
                print('-' * 50)
                print(code)
                print('-' * 50)
                print(code_removed)
                print()

                input()
