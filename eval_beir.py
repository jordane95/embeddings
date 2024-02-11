import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoTokenizer
from models import AutoModelForSentenceEmbedding
from mteb import MTEB, AbsTaskRetrieval, DRESModel

from utils import logger, move_to_cuda


def get_args():
    parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
    parser.add_argument('--task-types', nargs='+', default=[], help='task types to evaluate')
    parser.add_argument('--output-dir', default='', type=str, metavar='N', help='output directory')
    parser.add_argument('--multilingual', action='store_true', help='whether to use multilingual model')

    parser.add_argument('--model-name-or-path', default='tmp-outputs/', type=str, metavar='N', help='which model to use')
    parser.add_argument('--pooling', default='mean', help='pool type')
    parser.add_argument('--normalize', action='store_true', help='normalize embeddings?')
    parser.add_argument('--add-pooler', default='dense', type=str, help='projection head type')
    parser.add_argument('--n-experts', default=8, type=int, help='number of experts')
    parser.add_argument('--residual-pooler', action='store_true', help='add residual conntection to pooler')

    args = parser.parse_args()

    logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    assert args.pooling in ['mean', 'last', 'weightedmean'], 'pool_type should be cls / avg / last'
    os.makedirs(args.output_dir, exist_ok=True)
    return args


class AutoModelForSentenceEmbeddingDP(AutoModelForSentenceEmbedding):
    def forward(self, input_ids, attention_mask):
        outputs = self.lm(input_ids, attention_mask)
        embeddings = self.compress(outputs, attention_mask)
        return embeddings


class RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, args):
        self.encoder = AutoModelForSentenceEmbeddingDP(
            model_name_or_path=args.model_name_or_path,
            pooling=args.pooling,
            normalize=args.normalize,
            add_pooler=args.add_pooler,
            n_experts=args.n_experts,
            residual_pooler=args.residual_pooler,
        ).load_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = queries
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 64 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = self.tokenizer(
                batch_input_texts,
                max_length=512,
                padding=True,
                pad_to_multiple_of=8,
                return_token_type_ids=False,
                truncation=True,
                return_tensors='pt'
            )
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                embeds = self.encoder(**batch_dict)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


if __name__ == "__main__":
    args = get_args()
    assert AbsTaskRetrieval.is_dres_compatible(RetrievalModel)
    model = RetrievalModel(args)

    task_names = [t.description["name"] for t in MTEB(task_types=['Retrieval'], task_langs=['en']).tasks]
    task_names = [t for t in task_names if t != 'MSMARCOv2']
    logger.info('Tasks: {}'.format(task_names))

    for task in task_names:
        # if args.dry_run and task not in ['SciFact', 'FiQA2018']:
        #     continue

        logger.info('Processing task: {}'.format(task))

        # if args.prefix_type == 'query_or_passage':
        #     args.doc_as_query = task in ['QuoraRetrieval']
        # else:
        #     task_def: str = get_task_def_by_task_name_and_type(task_name=task, task_type='Retrieval')
        #     prompt: str = get_detailed_instruct(task_def)
        #     model.set_prompt(prompt=prompt)
        #     logger.info('Set prompt: {}'.format(prompt))

        evaluation = MTEB(tasks=[task], task_langs=['en'])
        evaluation.run(model, eval_splits=["test" if task not in ['MSMARCO'] else 'dev'],
                       output_folder=args.output_dir)
