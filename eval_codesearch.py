import os
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np
import random
import argparse
import gzip

from functools import partial
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict


from utils import logger, pool, move_to_cuda

parser = argparse.ArgumentParser(description='zero shot text classification evaluation on sst2 dev')
parser.add_argument('--model-name-or-path', default='tmp-outputs/',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--l2-normalize', action='store_true', help='whether to l2 normalize embeddings')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prompt', default='', help='prompt')
parser.add_argument('--batch-size', type=int, default=16, help='default 16 for 560m on 16G v100')
parser.add_argument('--output-dir', default='.', type=str, metavar='N', help='output directory')

args = parser.parse_args()
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.output_dir, 'output_dir should be set'
os.makedirs(args.output_dir, exist_ok=True)
print("Pooling type:", args.pool_type)

def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    if args.prompt:
        examples['input_texts'] = [args.prompt + t for t in examples['input_texts']]
    batch_dict = tokenizer(examples['input_texts'],
                           max_length=512,
                           padding=True,
                           truncation=True)

    return batch_dict


class DenseEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.tokenizer.padding_side = 'right'
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()
        self.encoder.half()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                if args.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        results = np.concatenate(encoded_embeds, axis=0)
        return np.nan_to_num(results)


def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def load_data(lang: str):
    data = list()
    prefix = '/data01/lizehan/proqa/data/codesearch'
    filename = f'{prefix}/{lang}/final/jsonl/test/{lang}_test_0.jsonl.gz'
    with gzip.open(filename, 'rt', encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def evaluate(model, data):
    random.seed(42)
    random.shuffle(data)

    text_queries, code_corpus = list(), list()
    for item in data:
        text_queries.append(item['docstring'])
        code_corpus.append(item['code'])
    query_embeddings = model.encode(text_queries)
    corpus_embeddings = model.encode(code_corpus)

    ranks = list()
    for i in range(0, len(data), 1000):
        q_emb = query_embeddings[i: i+1000]
        c_emb = corpus_embeddings[i: i+1000]
        if c_emb.shape[0] < 1000:
            padding = corpus_embeddings[i + c_emb.shape[0] - 1000: i]
            c_emb = np.concatenate((c_emb, padding), axis=0)
        sim_mat = cos_sim(q_emb, c_emb)
        for j in range(q_emb.shape[0]):
            correct_score = sim_mat[j, j]
            rank = torch.sum(sim_mat[j] >= correct_score).item()
            ranks.append(rank)
    
    result = {
        "mrr@1000": torch.mean(1 / torch.tensor(ranks)).item()
    }
    return result


def main():
    model = DenseEncoder()

    langs = ['ruby', 'go', 'java', 'javascript', 'php', 'python']

    scores = 0

    for lang in langs:
        data = load_data(lang)
        result = evaluate(model, data)
        with open(os.path.join(args.output_dir, lang + '.json'), 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"{lang}: {json.dumps(result, indent=2)}")
        scores += result['mrr@1000']
    
    with open(os.path.join(args.output_dir, "avg.json"), 'w') as f:
        json.dump({"mrr@1000": scores / len(langs)}, f, indent=2)
    logger.info(f"average score: {scores/len(langs)}")
    


if __name__ == '__main__':
    main()
