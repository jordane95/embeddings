import json
import torch
import torch.distributed as dist
from torch import Tensor

import logging
from typing import List, Union, Optional, Tuple, Mapping, Dict

import re
from io import StringIO
import tokenize


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        try: # if python code does not contain any bug
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
        except:
            print("bug code")
            return source
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


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def pool(last_hidden_states: Tensor,
         attention_mask: Tensor,
         pool_type: str) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")

    return emb


def normalize_instruction(instruction: str):
    instruction = instruction.strip()
    if len(instruction) > 0 and instruction[-1] in [';', ':', ',', '.']:
        return instruction[:-1]
    else:
        return instruction

def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


def full_contrastive_scores_and_labels(
        query: torch.Tensor,
        key: torch.Tensor,
        use_all_pairs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, '{} % {} > 0'.format(key.shape[0], query.shape[0])

    train_n_passages = key.shape[0] // query.shape[0]
    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)
    # labels = labels * train_n_passages

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    if not use_all_pairs:
        return qk, labels

    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    kq.fill_diagonal_(float('-inf'))

    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf'))

    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf'))

    scores = torch.cat([qk, kq, qq, kk], dim=-1)

    return scores, labels


if __name__ == '__main__':
    query = torch.randn(4, 16)
    key = torch.randn(4 * 3, 16)
    scores, labels = full_contrastive_scores_and_labels(query, key)
    print(scores.shape)
    print(labels)
