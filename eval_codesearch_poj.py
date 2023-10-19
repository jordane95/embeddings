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


def load_data(lang):
    text_data = list()
    code_data = list()
    prefix = '/data01/lizehan/proqa/code/CodeXGLUE/Code-Code/Clone-detection-POJ-104/dataset'
    code_file = f'{prefix}/test.jsonl'
    for line in open(code_file):
        code_data.append(json.loads(line.strip()))
    return {
        "code": code_data,
    }


def evaluate(model, data):
    code_data = data["code"]

    code_snippets = [code['code'] for code in code_data]
    indexs = [example['index'] for example in code_data]
    labels = [example['label'] for example in code_data]

    label_count = {}
    for label in labels:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    code_embeddings = model.encode(code_snippets)

    scores = code_embeddings @ code_embeddings.T # N x N

    np.fill_diagonal(scores, -np.inf)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    
    MAP=[]

    for i, example in enumerate(code_data):
        label = example['label']
        Avep = []
        gold_num = label_count[label] - 1
        for j in range(gold_num): # of ground trouth solutions
            index = sort_ids[i, j]
            if code_data[index]['label'] == label:
                Avep.append((len(Avep)+1)/(j+1))
        MAP.append(sum(Avep)/gold_num)
          
    result = {
        "map": float(np.mean(MAP)),
    }

    return result


def main():
    model = DenseEncoder()

    langs = ['python']

    scores = 0

    for lang in langs:
        data = load_data(lang)
        result = evaluate(model, data)
        with open(os.path.join(args.output_dir, lang + '.json'), 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"{lang}: {json.dumps(result, indent=2)}")
        scores += result['map']
    
    with open(os.path.join(args.output_dir, "avg.json"), 'w') as f:
        json.dump({"map": scores / len(langs)}, f, indent=2)
    logger.info(f"average score: {scores/len(langs)}")
    

if __name__ == '__main__':
    main()
