import os
import sys
import gzip
import json

import re
from io import StringIO
import tokenize


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)


lang = sys.argv[1]

train_path = f"/data01/lizehan/proqa/raw_data/codesearch/{lang}/final/jsonl/train"

output_path = f"/data01/lizehan/proqa/data/pretrain/csn_clean_{lang}_docstring_code.jsonl"


with open(output_path, 'w') as fout:
    for root, dirs, files in os.walk(train_path):
        # print(root, dirs, files)
        for filename in files:
            if filename.endswith(".gz"):
                with gzip.open(os.path.join(root, filename), 'r') as fin:
                    for line in fin:
                        item = json.loads(line)
                        try:
                            code = ' '.join(remove_comments_and_docstrings(item['code'], lang).split())
                        except:
                            print("bug code")
                            code = item['code']
                        pair = {"query": item['docstring'], "doc": code}
                        fout.write(json.dumps(pair) + "\n")