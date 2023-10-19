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


def load_data(langs):
    prefix = '/data01/lizehan/proqa/CodeBERT/UniXcoder/downstream-tasks/zero-shot-search/dataset'

    def get_data(lang: str) -> List[Dict]:
        code_file = f'{prefix}/{lang}_with_func.jsonl'
        code_data = []
        with open(code_file) as f:
            for line in f:
                item = json.loads(line)
                code_data.append(item)
        return code_data

    return {
        lang: get_data(lang)
        for lang in langs
    }


def evaluate(model, data):
    all_data = data
    all_embeddings = {}
    for lang, code_data in all_data.items():
        code_snippets = [code['func'] for code in code_data]
        code_embeddings = model.encode(code_snippets)
        all_embeddings[lang] = code_embeddings

    langs = all_data.keys()
    
    def evaluate_lang(src_lang, tgt_lang):
        query_embeddings = all_embeddings[src_lang]
        query_data = all_data[src_lang]

        doc_embeddings = all_embeddings[tgt_lang]
        doc_data = all_data[tgt_lang]

        scores = query_embeddings @ doc_embeddings.T # N x C
    
        sort_ids = np.argsort(scores.cpu().numpy(), axis=-1, kind='quicksort', order=None)[:,::-1]
        MAP = []
        results = {}
        for i in range(scores.shape[0]):
            cont = 0
            query_label = query_data[i]['label']
            query_index = query_data[i]['index']
            Avep = []
            for rank, j in enumerate(list(sort_ids[i])):
                if query_index == doc_data[j]['index']:
                    cont += 1
                    continue
                if doc_data[j]['label'] == query_label:
                    Avep.append((len(Avep)+1) / (rank + 1 - cont))
            if len(Avep)!=0:
                MAP.append(sum(Avep)/len(Avep))
        return float(np.mean(MAP))
    
    result = {}
    scores = 0
    for src_lang in langs:
        for tgt_lang in langs:
            score = evaluate_lang(src_lang, tgt_lang)
            with open(os.path.join(args.output_dir, f"{src_lang}_{tgt_lang}.json"), 'w') as f:
                json.dump({f"map": score}, f, indent=2)
            scores += score
            print(f"{src_lang}-{tgt_lang}-map: {score}")
            result[f"{src_lang}-{tgt_lang}-map"] = score

    avg_score = scores / (len(langs) * len(langs))
    result["avg_map"] = avg_score
    with open(os.path.join(args.output_dir, f"avg.json"), 'w') as f:
        json.dump({"avg_map": avg_score}, f, indent=2)
    return result


def main():
    model = DenseEncoder()

    langs = ['ruby', 'python', 'java']

    scores = 0

    data = load_data(langs)
    result = evaluate(model, data)
    
    
    logger.info(f"{result}")
    


if __name__ == '__main__':
    main()
